"""
Explanation Consistency Score (ECS).

Novel composite metric for quantifying explanation trustworthiness
under degradation. Core contribution of the paper.

ECS(x, d) = α · Stability(x, d)           # Cosine similarity of attributions
           + β · SpectralAlignment(x, d)    # Attribution in artifact bands
           + γ · FaithfulnessPreserv(x, d)  # Consistency of deletion AUC

where α + β + γ = 1
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from src.evaluation.faithfulness_metrics import (
    compute_explanation_stability,
    compute_spectral_band_alignment,
)


class ExplanationConsistencyScore:
    """
    Explanation Consistency Score (ECS) — our novel metric.

    Combines three dimensions of explanation robustness into a single
    interpretable score:

    1. **Stability** (α): How much does the attribution map change
       between clean and degraded versions of the same utterance?
    2. **Spectral Alignment** (β): Do attributions still concentrate
       in forensically meaningful frequency bands after degradation?
    3. **Faithfulness Preservation** (γ): Does the deletion/insertion AUC
       remain consistent between clean and degraded conditions?

    The ECS can serve as an early-warning signal: when ECS drops below
    a calibrated threshold, the detector's explanations should not be
    trusted even if its accuracy remains acceptable.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        artifact_bands: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        Initialize ECS with component weights.

        Args:
            alpha: Weight for explanation stability (cosine similarity).
            beta: Weight for spectral band alignment.
            gamma: Weight for faithfulness preservation (Δ deletion AUC).
            artifact_bands: Frequency bands (Hz) known to contain artifacts.
        """
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, \
            f"Weights must sum to 1.0, got {alpha + beta + gamma}"

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.artifact_bands = artifact_bands or [
            (0, 500),
            (2000, 4000),
            (4000, 8000),
        ]

    def compute(
        self,
        attr_clean: np.ndarray,
        attr_degraded: np.ndarray,
        score_clean: float,
        score_degraded: float,
        del_auc_clean: float,
        del_auc_degraded: float,
    ) -> Dict:
        """
        Compute ECS for a single sample under one degradation condition.

        Args:
            attr_clean: Attribution from clean audio [F, T'].
            attr_degraded: Attribution from degraded audio [F, T'].
            score_clean: Spoof probability on clean audio.
            score_degraded: Spoof probability on degraded audio.
            del_auc_clean: Deletion AUC on clean audio.
            del_auc_degraded: Deletion AUC on degraded audio.

        Returns:
            Dict with ECS score and component values.
        """
        # Component 1: Stability (cosine similarity)
        stability = compute_explanation_stability(attr_clean, attr_degraded)
        # Clamp to [0, 1] range (cosine sim can be negative)
        stability_score = max(0.0, stability)

        # Component 2: Spectral band alignment (degraded condition)
        alignment = compute_spectral_band_alignment(
            attr_degraded, score_degraded, self.artifact_bands
        )

        # Component 3: Faithfulness preservation
        faithfulness_delta = abs(del_auc_clean - del_auc_degraded)
        faithfulness_preservation = max(0.0, 1.0 - faithfulness_delta)

        # Composite ECS
        ecs = (
            self.alpha * stability_score
            + self.beta * alignment
            + self.gamma * faithfulness_preservation
        )

        return {
            "ecs": float(ecs),
            "stability": float(stability),
            "stability_score": float(stability_score),
            "spectral_alignment": float(alignment),
            "faithfulness_preservation": float(faithfulness_preservation),
            "faithfulness_delta": float(faithfulness_delta),
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }

    def compute_batch(
        self,
        attrs_clean: List[np.ndarray],
        attrs_degraded: List[np.ndarray],
        scores_clean: np.ndarray,
        scores_degraded: np.ndarray,
        del_aucs_clean: np.ndarray,
        del_aucs_degraded: np.ndarray,
    ) -> Dict:
        """
        Compute ECS for a batch of samples.

        Returns:
            Dict with per-sample ECS values and aggregate statistics.
        """
        n_samples = len(attrs_clean)

        ecs_values = np.zeros(n_samples)
        stability_values = np.zeros(n_samples)
        alignment_values = np.zeros(n_samples)
        faithfulness_values = np.zeros(n_samples)

        for i in range(n_samples):
            result = self.compute(
                attrs_clean[i], attrs_degraded[i],
                scores_clean[i], scores_degraded[i],
                del_aucs_clean[i], del_aucs_degraded[i],
            )
            ecs_values[i] = result["ecs"]
            stability_values[i] = result["stability_score"]
            alignment_values[i] = result["spectral_alignment"]
            faithfulness_values[i] = result["faithfulness_preservation"]

        return {
            "ecs_values": ecs_values,
            "ecs_mean": float(np.mean(ecs_values)),
            "ecs_std": float(np.std(ecs_values)),
            "ecs_median": float(np.median(ecs_values)),
            "stability_mean": float(np.mean(stability_values)),
            "alignment_mean": float(np.mean(alignment_values)),
            "faithfulness_mean": float(np.mean(faithfulness_values)),
        }

    def calibrate_threshold(
        self,
        ecs_values: np.ndarray,
        faithfulness_labels: np.ndarray,
        method: str = "roc",
    ) -> Tuple[float, float]:
        """
        Calibrate the ECS threshold for early-warning.

        Finds the ECS threshold that best separates "trustworthy"
        from "untrustworthy" explanations (based on ground-truth
        faithfulness labels).

        Args:
            ecs_values: ECS scores for each sample.
            faithfulness_labels: Binary labels (1 = faithful, 0 = unfaithful).
                Derived from deletion AUC being above/below a threshold.
            method: Calibration method ("roc" or "percentile").

        Returns:
            Tuple of (optimal threshold, ROC-AUC score).
        """
        from sklearn.metrics import roc_auc_score, roc_curve

        if method == "roc":
            roc_auc = roc_auc_score(faithfulness_labels, ecs_values)
            fpr, tpr, thresholds = roc_curve(faithfulness_labels, ecs_values)

            # Youden's J statistic
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = float(thresholds[optimal_idx])

            return optimal_threshold, float(roc_auc)

        elif method == "percentile":
            # Use 25th percentile of "faithful" ECS as threshold
            faithful_ecs = ecs_values[faithfulness_labels == 1]
            threshold = float(np.percentile(faithful_ecs, 25))
            roc_auc = roc_auc_score(faithfulness_labels, ecs_values)
            return threshold, float(roc_auc)

        raise ValueError(f"Unknown calibration method: {method}")

    def assess_trustworthiness(
        self,
        ecs_value: float,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Assess whether an explanation should be trusted.

        Args:
            ecs_value: Computed ECS score.
            threshold: Calibrated threshold.

        Returns:
            Dict with trustworthiness assessment.
        """
        is_trustworthy = ecs_value >= threshold

        if ecs_value >= 0.8:
            confidence = "high"
            message = "Explanation is highly stable and faithful under this degradation."
        elif ecs_value >= 0.6:
            confidence = "moderate"
            message = "Explanation is moderately reliable but shows some shift."
        elif ecs_value >= threshold:
            confidence = "low"
            message = "Explanation is marginally trustworthy. Interpret with caution."
        else:
            confidence = "untrustworthy"
            message = (
                "WARNING: Explanation has significantly degraded. "
                "The detector's reasoning may not reflect actual forensic evidence."
            )

        return {
            "ecs": ecs_value,
            "threshold": threshold,
            "is_trustworthy": is_trustworthy,
            "confidence": confidence,
            "message": message,
        }
