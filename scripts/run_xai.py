#!/usr/bin/env script
"""
Compute XAI attributions (Integrated Gradients or Kernel SHAP) for clean and degraded samples.
Saves attributions to results/attributions/ as numpy files.
"""

import os
import sys
import argparse
import numpy as np
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
from src.xai.integrated_gradients import IntegratedGradientsExplainer
from src.xai.kernel_shap import KernelSHAPExplainer
from src.data.dataset import ASVspoof2021DF
from src.data.degradation import DegradationPipeline


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate XAI explanations.")
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
    parser.add_argument(
        "--method",
        type=str,
        choices=["ig", "shap"],
        default="ig",
        help="XAI attribution method to run.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    conditions_config = load_yaml(args.conditions)
    conditions = conditions_config.get("conditions", {})

    output_dir = Path(config["paths"]["results_dir"]) / "attributions"
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
            
            # For SHAP, skip WavLM if config dictates (since it is too slow)
            if args.method == "shap" and name != config["xai"]["kernel_shap"].get("subset_model", "aasist"):
                continue

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
    # We restrict to a smaller number of samples for XAI (e.g. 50 samples per condition)
    # to control computation time.
    n_xai_samples = 20  # Fast default for testing/running, can be increased by user
    
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

    print(f"Dataset loaded: {len(dataset)} samples for XAI computation.")

    # Initialize explainers
    explainers = {}
    for model_name, model in models.items():
        if args.method == "ig":
            ig_cfg = config["xai"]["integrated_gradients"]
            explainers[model_name] = IntegratedGradientsExplainer(
                model=model,
                device=device,
                n_steps=ig_cfg.get("n_steps", 50),
                baseline_type=ig_cfg.get("baseline", "zero"),
                n_mels=config["spectrogram"].get("n_mels", 128),
                n_fft=config["spectrogram"].get("n_fft", 2048),
                hop_length=config["spectrogram"].get("hop_length", 512),
                sample_rate=sample_rate,
            )
        elif args.method == "shap":
            shap_cfg = config["xai"]["kernel_shap"]
            explainers[model_name] = KernelSHAPExplainer(
                model=model,
                device=device,
                n_samples=shap_cfg.get("n_samples", 200),
                n_mels=config["spectrogram"].get("n_mels", 128),
                n_fft=config["spectrogram"].get("n_fft", 2048),
                hop_length=config["spectrogram"].get("hop_length", 512),
                sample_rate=sample_rate,
            )

    # Core XAI loop
    # Determine conditions to evaluate
    if args.method == "shap":
        cond_keys = config["xai"]["kernel_shap"].get("subset_conditions", ["C0", "C3", "C7", "C8", "N2"])
    else:
        cond_keys = list(conditions.keys())

    for model_name, explainer in explainers.items():
        print(f"\nComputing {args.method.upper()} attributions for model: {model_name.upper()}")

        for cond_key in cond_keys:
            cond = conditions[cond_key]
            cond_name = cond.get("name", cond_key)
            print(f"  -> Processing condition: {cond_key} ({cond_name})")

            # Store attributions for this condition
            # Shape will be: [N, n_mels, T']
            condition_attributions = []
            sample_ids = []

            for idx in tqdm(range(len(dataset)), desc=f"{model_name} | {cond_key}"):
                sample = dataset[idx]
                waveform = sample["waveform"]
                utt_id = sample["utt_id"]

                # Skip if built-in codec doesn't match sample condition
                if cond.get("type") == "builtin" and sample.get("codec") != cond_key:
                    continue

                # Apply custom degradation if needed
                if cond.get("type") != "builtin":
                    waveform = degradation_pipeline.apply(waveform, cond)

                # Compute XAI explanation
                try:
                    attr = explainer.explain(waveform)
                    condition_attributions.append(attr)
                    sample_ids.append(utt_id)
                except Exception as e:
                    print(f"    Error explaining sample {utt_id}: {e}")

            if len(condition_attributions) == 0:
                continue

            # Save attributions as numpy array
            attr_array = np.stack(condition_attributions)
            save_path = output_dir / f"{model_name}_{args.method}_{cond_key}.npy"
            np.save(save_path, attr_array)

            # Save sample metadata mappings
            meta_path = output_dir / f"{model_name}_{args.method}_{cond_key}_meta.txt"
            with open(meta_path, "w") as f:
                f.write("\n".join(sample_ids))

            print(f"    Saved attributions to: {save_path} (shape: {attr_array.shape})")

    print(f"\n-> XAI computation complete.")


if __name__ == "__main__":
    main()
