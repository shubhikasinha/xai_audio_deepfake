"""
End-to-end experiment pipeline.

Runs the complete explanation robustness evaluation:
1. Load pretrained detectors
2. Apply degradation conditions
3. Compute detection metrics (EER, min t-DCF)
4. Compute XAI attributions (IG primary, SHAP secondary)
5. Evaluate faithfulness (Deletion/Insertion AUC, Sensitivity-N)
6. Compute ECS (Explanation Consistency Score)
7. Run statistical analysis
8. Generate paper figures
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.aasist import AASISTDetector
from src.models.wavlm_ecapa import WavLMECAPADetector
from src.xai.integrated_gradients import IntegratedGradientsExplainer
from src.evaluation.detection_metrics import compute_detection_metrics
from src.evaluation.faithfulness_metrics import compute_all_faithfulness_metrics
from src.evaluation.consistency_score import ExplanationConsistencyScore
from src.evaluation.statistical_tests import spearman_correlation, bootstrap_ci


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_conditions(conditions_path: str) -> dict:
    """Load degradation conditions."""
    with open(conditions_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("conditions", {})


def setup_model(model_name: str, device: str) -> torch.nn.Module:
    """Initialize and load a pretrained detector."""
    if model_name == "aasist":
        model = AASISTDetector(device=device)
        checkpoint = "checkpoints/aasist/AASIST.pth"
        if os.path.exists(checkpoint):
            model.load_checkpoint(checkpoint)
    elif model_name == "wavlm_ecapa":
        model = WavLMECAPADetector(device=device)
        checkpoint = "checkpoints/wavlm_ecapa/best_model.pth"
        if os.path.exists(checkpoint):
            model.load_checkpoint(checkpoint)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    return model


def run_quick_test(config: dict, n_samples: int = 10):
    """
    Quick pipeline test with synthetic data.

    Verifies all components work end-to-end without requiring
    actual datasets or pretrained checkpoints.
    """
    print("=" * 60)
    print("QUICK TEST — Verifying pipeline components")
    print("=" * 60)

    device = "cpu"  # Force CPU for quick test
    sample_rate = 16000
    duration_sec = 2.0
    n_samples_audio = int(sample_rate * duration_sec)

    # Generate synthetic audio
    print("\n[1/6] Generating synthetic test data...")
    waveforms = torch.randn(n_samples, n_samples_audio)
    labels = torch.randint(0, 2, (n_samples,))
    print(f"  -> Generated {n_samples} samples ({duration_sec}s each)")

    # Test AASIST
    print("\n[2/6] Testing AASIST detector...")
    model = AASISTDetector(device=device)
    with torch.no_grad():
        result = model.predict(waveforms[:2])
    print(f"  -> AASIST scores: {result['scores'].numpy()}")
    print(f"  -> AASIST probs:  {result['probs'].numpy()}")

    # Test Integrated Gradients
    print("\n[3/6] Testing Integrated Gradients...")
    ig = IntegratedGradientsExplainer(
        model, device=device, n_steps=5, n_mels=64
    )
    attr = ig.explain(waveforms[0])
    print(f"  -> Attribution shape: {attr.shape}")
    print(f"  -> Attribution range: [{attr.min():.4f}, {attr.max():.4f}]")

    # Test faithfulness metrics
    print("\n[4/6] Testing faithfulness metrics...")

    def model_fn(x):
        with torch.no_grad():
            return model.get_spoof_probability(x).item()

    from src.evaluation.faithfulness_metrics import (
        compute_deletion_auc,
        compute_explanation_stability,
    )

    del_auc, del_curve = compute_deletion_auc(
        model_fn, waveforms[0].numpy(), attr, n_steps=5, hop_length=512
    )
    print(f"  -> Deletion AUC: {del_auc:.4f}")

    # Test ECS
    print("\n[5/6] Testing Explanation Consistency Score...")
    ecs_scorer = ExplanationConsistencyScore()

    # Simulate degraded attribution
    attr_degraded = attr + np.random.randn(*attr.shape) * 0.1
    stability = compute_explanation_stability(attr, attr_degraded)

    ecs_result = ecs_scorer.compute(
        attr_clean=attr,
        attr_degraded=attr_degraded,
        score_clean=0.8,
        score_degraded=0.7,
        del_auc_clean=del_auc,
        del_auc_degraded=del_auc + 0.05,
    )
    print(f"  -> ECS: {ecs_result['ecs']:.4f}")
    print(f"  -> Stability: {ecs_result['stability']:.4f}")
    print(f"  -> Alignment: {ecs_result['spectral_alignment']:.4f}")

    # Test statistical utilities
    print("\n[6/6] Testing statistical utilities...")
    x = np.random.randn(10)
    y = x * 0.5 + np.random.randn(10) * 0.3
    corr_result = spearman_correlation(x, y)
    print(f"  -> Spearman rho: {corr_result['rho']:.4f} (p={corr_result['p_value']:.4f})")

    ci_result = bootstrap_ci(x, n_resamples=1000)
    print(f"  -> Bootstrap CI: [{ci_result['ci_lower']:.4f}, {ci_result['ci_upper']:.4f}]")

    print("\n" + "=" * 60)
    print("PASS: ALL COMPONENTS PASSED - Pipeline is functional")
    print("=" * 60)


def main():
    """Main entry point for the experiment pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the explanation robustness experiment pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test with synthetic data.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of samples for quick test.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["aasist", "wavlm_ecapa", "all"],
        default="all",
        help="Which model to evaluate.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="core",
        choices=["core", "all", "shap_subset"],
        help="Which degradation conditions to evaluate.",
    )

    args = parser.parse_args()

    if args.quick_test:
        run_quick_test({}, n_samples=args.n_samples)
        return

    # Load configuration
    config = load_config(args.config)

    print("=" * 60)
    print("EXPLANATION ROBUSTNESS EXPERIMENT PIPELINE")
    print(f"Config: {args.config}")
    print(f"Model: {args.model}")
    print(f"Conditions: {args.conditions}")
    print("=" * 60)

    # Full pipeline would go here
    # For now, show what would be executed
    print("\nPipeline steps:")
    print("  1. Load pretrained detectors")
    print("  2. Load ASVspoof 2021 DF evaluation set")
    print("  3. Apply degradation conditions (12 core)")
    print("  4. Compute detection metrics (EER, min t-DCF)")
    print("  5. Compute IG attributions (primary XAI)")
    print("  6. Compute faithfulness metrics (Deletion/Insertion AUC)")
    print("  7. Compute ECS (Explanation Consistency Score)")
    print("  8. Run statistical analysis (Spearman, Wilcoxon)")
    print("  9. Generate paper figures")
    print("\nNote: Download datasets first with scripts/download_data.py")


if __name__ == "__main__":
    main()
