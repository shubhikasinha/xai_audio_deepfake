#!/usr/bin/env python3
"""
Evaluate explanation faithfulness (Deletion/Insertion AUC) and Explanation Consistency Score (ECS).
Saves results to results/faithfulness/.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.aasist import AASISTDetector
from src.models.wavlm_ecapa import WavLMECAPADetector
from src.data.dataset import ASVspoof2021DF
from src.data.degradation import DegradationPipeline
from src.evaluation.consistency_score import ExplanationConsistencyScore
from src.evaluation.faithfulness_metrics import (
    compute_deletion_auc,
    compute_insertion_auc,
)


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate explanation faithfulness.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="configs/degradation/conditions.yaml",
        help="Path to conditions config YAML.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    conditions_config = load_yaml(args.conditions)
    conditions = conditions_config.get("conditions", {})

    output_dir = Path(config["paths"]["results_dir"]) / "faithfulness"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = config["project"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA selected but not available. Falling back to CPU.")
        device = "cpu"

    print(f"Using device: {device}")

    # Load models
    models = {}
    for model_cfg in config["models"]:
        if model_cfg.get("enabled", False):
            name = model_cfg["name"]
            print(f"Initializing {name} detector...")
            if name == "aasist":
                chkpt = "checkpoints/aasist/AASIST.pth"
                models[name] = AASISTDetector(device=device, checkpoint_path=chkpt)
            elif name == "wavlm_ecapa":
                chkpt = "checkpoints/wavlm_ecapa/best_model.pth"
                models[name] = WavLMECAPADetector(device=device, checkpoint_path=chkpt)

    # Initialize degradation pipeline
    sample_rate = config["spectrogram"].get("sample_rate", 16000)
    degradation_pipeline = DegradationPipeline(sample_rate=sample_rate)

    # Load dataset
    data_root = config["paths"]["asvspoof2021_df"]
    # Same n_samples as used in XAI script
    n_xai_samples = 20
    
    print(f"Loading ASVspoof 2021 DF dataset from: {data_root}")
    try:
        dataset = ASVspoof2021DF(
            root_dir=data_root,
            sample_rate=sample_rate,
            max_duration_sec=config["spectrogram"].get("max_duration_sec", 4.0),
            max_samples=n_xai_samples,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        sys.exit(1)

    # Build dataset index map by utt_id for fast lookup
    dataset_map = {sample["utt_id"]: sample for sample in dataset}

    # Initialize ECS scorer
    ecs_cfg = config["ecs"]
    ecs_scorer = ExplanationConsistencyScore(
        alpha=ecs_cfg.get("alpha", 0.4),
        beta=ecs_cfg.get("beta", 0.3),
        gamma=ecs_cfg.get("gamma", 0.3),
        artifact_bands=ecs_cfg.get("artifact_bands"),
    )

    # Path to attributions
    attr_dir = Path(config["paths"]["results_dir"]) / "attributions"
    
    all_results = []

    for model_name, model in models.items():
        print(f"\nEvaluating faithfulness for model: {model_name.upper()}")
        model.eval()

        # Load clean attributions (C0) as baseline
        clean_attr_path = attr_dir / f"{model_name}_ig_C0.npy"
        clean_meta_path = attr_dir / f"{model_name}_ig_C0_meta.txt"

        if not clean_attr_path.exists() or not clean_meta_path.exists():
            print(f"  ✗ Clean attributions not found at {clean_attr_path}. Run scripts/run_xai.py first.")
            continue

        clean_attrs = np.load(clean_attr_path)
        with open(clean_meta_path, "r") as f:
            clean_meta = f.read().strip().split("\n")

        # Map clean meta index to attribution
        clean_attr_map = {utt_id: clean_attrs[i] for i, utt_id in enumerate(clean_meta)}

        # Load clean faithfulness metrics to avoid re-computation
        clean_metrics = {}

        # Loop over other conditions
        for cond_key, cond in conditions.items():
            cond_name = cond.get("name", cond_key)
            print(f"  -> Processing condition: {cond_key} ({cond_name})")

            attr_path = attr_dir / f"{model_name}_ig_{cond_key}.npy"
            meta_path = attr_dir / f"{model_name}_ig_{cond_key}_meta.txt"

            if not attr_path.exists() or not meta_path.exists():
                print(f"    Attributions for {cond_key} not found (skipping).")
                continue

            attrs = np.load(attr_path)
            with open(meta_path, "r") as f:
                meta = f.read().strip().split("\n")

            # Evaluate each sample
            for i, utt_id in enumerate(tqdm(meta, desc=f"ECS | {cond_key}")):
                if utt_id not in clean_attr_map:
                    continue

                # Load sample audio
                if utt_id not in dataset_map:
                    continue
                sample = dataset_map[utt_id]
                waveform = sample["waveform"]

                # Run clean model inference
                with torch.no_grad():
                    prob_clean = model.get_spoof_probability(waveform.to(device))[0].item()

                # Get clean metrics
                if utt_id not in clean_metrics:
                    attr_clean = clean_attr_map[utt_id]
                    
                    def model_fn(x):
                        return model.get_spoof_probability(x.to(device))[0].item()

                    # Compute deletion AUC on clean
                    del_auc_clean, _ = compute_deletion_auc(
                        model_fn=model_fn,
                        waveform=waveform.numpy(),
                        attribution=attr_clean,
                        n_steps=config["faithfulness"]["deletion"].get("n_steps", 20),
                        hop_length=config["spectrogram"].get("hop_length", 512),
                    )
                    clean_metrics[utt_id] = del_auc_clean
                else:
                    del_auc_clean = clean_metrics[utt_id]

                # Run degraded model inference
                degraded_waveform = waveform
                if cond.get("type") != "builtin":
                    degraded_waveform = degradation_pipeline.apply(waveform, cond)

                with torch.no_grad():
                    prob_degraded = model.get_spoof_probability(degraded_waveform.to(device))[0].item()

                # Compute deletion AUC on degraded
                attr_degraded = attrs[i]
                
                def model_fn_degraded(x):
                    return model.get_spoof_probability(x.to(device))[0].item()

                del_auc_degraded, _ = compute_deletion_auc(
                    model_fn=model_fn_degraded,
                    waveform=degraded_waveform.numpy(),
                    attribution=attr_degraded,
                    n_steps=config["faithfulness"]["deletion"].get("n_steps", 20),
                    hop_length=config["spectrogram"].get("hop_length", 512),
                )

                # Compute ECS
                ecs_result = ecs_scorer.compute(
                    attr_clean=clean_attr_map[utt_id],
                    attr_degraded=attr_degraded,
                    score_clean=prob_clean,
                    score_degraded=prob_degraded,
                    del_auc_clean=del_auc_clean,
                    del_auc_degraded=del_auc_degraded,
                )

                all_results.append({
                    "model": model_name,
                    "utt_id": utt_id,
                    "condition_id": cond_key,
                    "condition_name": cond_name,
                    "prob_clean": prob_clean,
                    "prob_degraded": prob_degraded,
                    "del_auc_clean": del_auc_clean,
                    "del_auc_degraded": del_auc_degraded,
                    "ecs": ecs_result["ecs"],
                    "stability": ecs_result["stability"],
                    "spectral_alignment": ecs_result["spectral_alignment"],
                    "faithfulness_preservation": ecs_result["faithfulness_preservation"],
                })

    if len(all_results) == 0:
        print("No results computed. Make sure XAI attributions exist.")
        sys.exit(1)

    # Save to CSV
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "faithfulness_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n-> Faithfulness evaluation complete. Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
