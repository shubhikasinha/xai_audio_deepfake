#!/usr/bin/env python3
"""
Evaluate detector robustness (detection performance) under controlled degradations.
Saves EER, min t-DCF, and other statistics to results/detection/.
"""

import os
import sys
import argparse
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
from src.evaluation.detection_metrics import compute_detection_metrics


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate detector robustness.")
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

    # Load configurations
    config = load_yaml(args.config)
    conditions_config = load_yaml(args.conditions)
    conditions = conditions_config.get("conditions", {})

    # Create output directory
    output_dir = Path(config["paths"]["results_dir"]) / "detection"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = config["project"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA selected but not available. Falling back to CPU.")
        device = "cpu"

    print(f"Using device: {device}")

    # Load models
    models_to_eval = {}
    for model_cfg in config["models"]:
        if model_cfg.get("enabled", False):
            name = model_cfg["name"]
            print(f"Initializing {name} detector...")
            if name == "aasist":
                chkpt = "checkpoints/aasist/AASIST.pth"
                models_to_eval[name] = AASISTDetector(device=device, checkpoint_path=chkpt)
            elif name == "wavlm_ecapa":
                chkpt = "checkpoints/wavlm_ecapa/best_model.pth"
                models_to_eval[name] = WavLMECAPADetector(device=device, checkpoint_path=chkpt)

    # Initialize degradation pipeline
    sample_rate = config["spectrogram"].get("sample_rate", 16000)
    degradation_pipeline = DegradationPipeline(sample_rate=sample_rate)

    # Load dataset
    data_root = config["paths"]["asvspoof2021_df"]
    n_eval_samples = config["evaluation"].get("n_eval_samples", None)
    
    print(f"Loading ASVspoof 2021 DF dataset from: {data_root}")
    # Initialize dataset loader
    try:
        dataset = ASVspoof2021DF(
            root_dir=data_root,
            sample_rate=sample_rate,
            max_duration_sec=config["spectrogram"].get("max_duration_sec", 4.0),
            max_samples=n_eval_samples,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check if the datasets are linked correctly.")
        sys.exit(1)

    if len(dataset) == 0:
        print("Error: Dataset is empty. Make sure trial metadata file exists.")
        sys.exit(1)

    print(f"Dataset loaded: {len(dataset)} samples for evaluation.")

    # Core evaluation loop
    all_results = []

    # Get condition keys
    cond_keys = list(conditions.keys())
    
    for model_name, model in models_to_eval.items():
        print(f"\nEvaluating model: {model_name.upper()}")
        model.eval()

        for cond_key in cond_keys:
            cond = conditions[cond_key]
            cond_name = cond.get("name", cond_key)
            print(f"  -> Testing condition: {cond_key} ({cond_name})")

            all_scores = []
            all_labels = []

            # We process samples sequentially or in small batches
            for idx in tqdm(range(len(dataset)), desc=f"{model_name} | {cond_key}"):
                sample = dataset[idx]
                waveform = sample["waveform"].to(device)
                label = sample["label"]

                # Handle built-in vs custom degradation
                # If the condition type is 'builtin', we load the pre-degraded file matching the codec
                # Otherwise, we apply the custom degradation pipeline
                if cond.get("type") == "builtin":
                    # For builtin codecs, check if the sample's codec matches.
                    # Since ASVspoof 2021 has mixed files, if we are evaluating a specific builtin condition
                    # (e.g. C1 = MP3 Low), we only keep samples belonging to that codec condition!
                    # Wait, let's filter by codec!
                    if sample.get("codec") != cond_key:
                        continue
                    degraded_waveform = waveform
                else:
                    # Apply custom degradation (C8, C9, N1, N2, etc.)
                    degraded_waveform = degradation_pipeline.apply(waveform, cond)

                # Get model prediction
                with torch.no_grad():
                    # waveform input should be [B, T] or [1, T]
                    if degraded_waveform.dim() == 1:
                        degraded_waveform = degraded_waveform.unsqueeze(0)
                    scores = model(degraded_waveform)
                    score = scores[0].item()

                all_scores.append(score)
                all_labels.append(label)

            if len(all_scores) == 0:
                print(f"    No samples matched condition {cond_key} (skipping).")
                continue

            # Compute metrics
            metrics = compute_detection_metrics(
                scores=torch.tensor(all_scores).numpy(),
                labels=torch.tensor(all_labels).numpy(),
            )

            print(f"    EER: {metrics['eer']:.4f} | min t-DCF: {metrics['min_tdcf']:.4f}")

            # Append to results
            all_results.append({
                "model": model_name,
                "condition_id": cond_key,
                "condition_name": cond_name,
                "type": cond.get("type"),
                "eer": metrics["eer"],
                "min_tdcf": metrics["min_tdcf"],
                "accuracy": metrics["accuracy_at_eer"],
                "n_samples": metrics["n_total"],
            })

    # Save to CSV
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "detection_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n-> Detection evaluation complete. Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
