"""
Detection performance metrics.

Computes Equal Error Rate (EER) and minimum tandem Detection Cost
Function (min t-DCF) — the standard metrics for ASVspoof evaluation.
"""

import numpy as np
from typing import Dict, Tuple
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).

    The EER is the point where False Acceptance Rate (FAR) equals
    False Rejection Rate (FRR) on the ROC curve.

    Args:
        scores: Detection scores (higher = more likely spoof).
        labels: Ground truth labels (0 = bonafide, 1 = spoof).

    Returns:
        Tuple of (EER, threshold at EER).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Find EER: point where FPR ≈ FNR
    try:
        eer_threshold_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2)
        threshold = float(thresholds[eer_threshold_idx])
    except (ValueError, IndexError):
        # Fallback: interpolation
        try:
            eer_fn = interp1d(fpr, fnr)
            eer = float(brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0, 1))
            threshold = float(
                thresholds[np.nanargmin(np.abs(fpr - eer))]
            )
        except (ValueError, RuntimeError):
            eer = 0.5
            threshold = 0.0

    return eer, threshold


def compute_min_tdcf(
    cm_scores: np.ndarray,
    cm_labels: np.ndarray,
    asv_scores: np.ndarray = None,
    asv_labels: np.ndarray = None,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
    p_target: float = 0.05,
) -> float:
    """
    Compute minimum tandem Detection Cost Function (min t-DCF).

    Simplified version — when ASV scores are not available, falls back
    to a DCF-like metric using CM scores only.

    Args:
        cm_scores: Countermeasure scores.
        cm_labels: Ground truth (0 = bonafide, 1 = spoof).
        asv_scores: ASV system scores (optional).
        asv_labels: ASV ground truth (optional).
        c_miss: Cost of missing a spoof.
        c_fa: Cost of false alarm.
        p_target: Prior probability of target.

    Returns:
        min t-DCF value.
    """
    fpr, tpr, thresholds = roc_curve(cm_labels, cm_scores, pos_label=1)
    fnr = 1 - tpr

    # Compute DCF at each threshold
    dcf = c_miss * p_target * fnr + c_fa * (1 - p_target) * fpr

    # Normalize
    dcf_min = min(c_miss * p_target, c_fa * (1 - p_target))
    if dcf_min > 0:
        normalized_dcf = dcf / dcf_min
    else:
        normalized_dcf = dcf

    return float(np.min(normalized_dcf))


def compute_detection_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    """
    Compute all standard detection metrics.

    Args:
        scores: Detection scores [N].
        labels: Ground truth labels [N].

    Returns:
        Dict with EER, threshold, min t-DCF, and additional stats.
    """
    eer, threshold = compute_eer(scores, labels)
    min_tdcf = compute_min_tdcf(scores, labels)

    # Additional stats
    n_bonafide = int(np.sum(labels == 0))
    n_spoof = int(np.sum(labels == 1))

    # Accuracy at EER threshold
    predictions = (scores >= threshold).astype(int)
    accuracy = float(np.mean(predictions == labels))

    return {
        "eer": eer,
        "eer_threshold": threshold,
        "min_tdcf": min_tdcf,
        "accuracy_at_eer": accuracy,
        "n_bonafide": n_bonafide,
        "n_spoof": n_spoof,
        "n_total": len(labels),
    }
