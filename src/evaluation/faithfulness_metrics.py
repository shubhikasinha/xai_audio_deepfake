"""
Faithfulness metrics for XAI evaluation.

Quantifies how well attribution maps reflect the model's actual
decision process via deletion/insertion curves, Sensitivity-N,
and explanation stability.
"""

import numpy as np
from typing import Callable, Dict, Tuple

import torch


def compute_deletion_auc(
    model_fn: Callable,
    waveform: np.ndarray,
    attribution: np.ndarray,
    n_steps: int = 20,
    baseline_value: float = 0.0,
    sample_rate: int = 16000,
    hop_length: int = 512,
) -> Tuple[float, np.ndarray]:
    """
    Compute Deletion AUC.

    Progressively masks the most important features (according to the
    attribution map) and measures the drop in model confidence.

    Lower Deletion AUC = more faithful explanation (important features
    were correctly identified).

    Args:
        model_fn: Function that takes waveform [1, T] → spoof probability.
        waveform: Original waveform [T].
        attribution: Attribution map [F, T'] from XAI method.
        n_steps: Number of deletion steps.
        baseline_value: Value to replace deleted features with.
        sample_rate: Audio sample rate.
        hop_length: Spectrogram hop length.

    Returns:
        Tuple of (AUC value, per-step predictions array).
    """
    abs_attr = np.abs(attribution)
    # Flatten and rank features by importance
    flat_attr = abs_attr.flatten()
    n_features = len(flat_attr)

    # Sort by importance (highest first)
    importance_order = np.argsort(flat_attr)[::-1]

    # Compute predictions at each deletion step
    step_sizes = np.linspace(0, n_features, n_steps + 1, dtype=int)
    predictions = np.zeros(n_steps + 1)

    # Original prediction (0% deleted)
    with torch.no_grad():
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        predictions[0] = model_fn(waveform_tensor)

    for step_idx in range(1, n_steps + 1):
        # Determine which spectrogram features to mask
        n_to_mask = step_sizes[step_idx]
        mask_indices = importance_order[:n_to_mask]

        # Convert spectrogram mask to waveform mask
        masked_waveform = _apply_spectrogram_mask(
            waveform, attribution.shape, mask_indices,
            baseline_value, hop_length
        )

        with torch.no_grad():
            masked_tensor = torch.from_numpy(masked_waveform).float().unsqueeze(0)
            predictions[step_idx] = model_fn(masked_tensor)

    # Compute AUC using trapezoidal rule
    x = np.linspace(0, 1, n_steps + 1)
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auc = float(trapz_fn(predictions, x))

    return auc, predictions


def compute_insertion_auc(
    model_fn: Callable,
    waveform: np.ndarray,
    attribution: np.ndarray,
    n_steps: int = 20,
    baseline_value: float = 0.0,
    sample_rate: int = 16000,
    hop_length: int = 512,
) -> Tuple[float, np.ndarray]:
    """
    Compute Insertion AUC.

    Starts from a blank/baseline input and progressively adds the most
    important features. Measures the gain in model confidence.

    Higher Insertion AUC = more faithful (important features are sufficient).

    Args:
        model_fn: Function that takes waveform [1, T] → spoof probability.
        waveform: Original waveform [T].
        attribution: Attribution map [F, T'].
        n_steps: Number of insertion steps.
        baseline_value: Baseline input value.

    Returns:
        Tuple of (AUC value, per-step predictions array).
    """
    abs_attr = np.abs(attribution)
    flat_attr = abs_attr.flatten()
    n_features = len(flat_attr)

    importance_order = np.argsort(flat_attr)[::-1]

    step_sizes = np.linspace(0, n_features, n_steps + 1, dtype=int)
    predictions = np.zeros(n_steps + 1)

    # Baseline prediction (0% inserted = blank input)
    blank_waveform = np.full_like(waveform, baseline_value)
    with torch.no_grad():
        blank_tensor = torch.from_numpy(blank_waveform).float().unsqueeze(0)
        predictions[0] = model_fn(blank_tensor)

    for step_idx in range(1, n_steps + 1):
        n_to_insert = step_sizes[step_idx]
        insert_indices = importance_order[:n_to_insert]

        # Start from blank, add original values at important locations
        inserted_waveform = _apply_spectrogram_insertion(
            waveform, attribution.shape, insert_indices,
            baseline_value, hop_length
        )

        with torch.no_grad():
            inserted_tensor = torch.from_numpy(inserted_waveform).float().unsqueeze(0)
            predictions[step_idx] = model_fn(inserted_tensor)

    x = np.linspace(0, 1, n_steps + 1)
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auc = float(trapz_fn(predictions, x))

    return auc, predictions


def compute_sensitivity_n(
    model_fn: Callable,
    waveform: np.ndarray,
    attribution: np.ndarray,
    n_subsets: int = 100,
    subset_size_ratio: float = 0.1,
    hop_length: int = 512,
) -> float:
    """
    Compute Sensitivity-N metric.

    Measures correlation between the sum of attribution values in a
    random subset of features and the prediction change when those
    features are removed.

    Higher correlation = more faithful attributions.

    Args:
        model_fn: Function that takes waveform [1, T] → spoof probability.
        waveform: Original waveform [T].
        attribution: Attribution map [F, T'].
        n_subsets: Number of random subsets to test.
        subset_size_ratio: Fraction of features in each subset.

    Returns:
        Pearson correlation coefficient.
    """
    flat_attr = attribution.flatten()
    n_features = len(flat_attr)
    subset_size = max(1, int(n_features * subset_size_ratio))

    # Original prediction
    with torch.no_grad():
        orig_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        orig_pred = model_fn(orig_tensor)

    attribution_sums = np.zeros(n_subsets)
    prediction_changes = np.zeros(n_subsets)

    for i in range(n_subsets):
        # Random subset of features
        subset_indices = np.random.choice(n_features, subset_size, replace=False)

        # Sum of attributions in this subset
        attribution_sums[i] = np.sum(flat_attr[subset_indices])

        # Mask these features and measure prediction change
        masked_waveform = _apply_spectrogram_mask(
            waveform, attribution.shape, subset_indices,
            0.0, hop_length
        )

        with torch.no_grad():
            masked_tensor = torch.from_numpy(masked_waveform).float().unsqueeze(0)
            masked_pred = model_fn(masked_tensor)

        prediction_changes[i] = orig_pred - masked_pred

    # Pearson correlation
    if np.std(attribution_sums) == 0 or np.std(prediction_changes) == 0:
        return 0.0

    correlation = float(np.corrcoef(attribution_sums, prediction_changes)[0, 1])
    return correlation


def compute_explanation_stability(
    attr_clean: np.ndarray,
    attr_degraded: np.ndarray,
) -> float:
    """
    Compute explanation stability between clean and degraded attributions.

    Uses cosine similarity of flattened attribution vectors.

    Args:
        attr_clean: Attribution from clean audio [F, T'].
        attr_degraded: Attribution from degraded audio [F, T'].

    Returns:
        Cosine similarity in [-1, 1]. Higher = more stable.
    """
    flat_clean = attr_clean.flatten()
    flat_degraded = attr_degraded.flatten()

    # Handle size mismatch
    min_len = min(len(flat_clean), len(flat_degraded))
    flat_clean = flat_clean[:min_len]
    flat_degraded = flat_degraded[:min_len]

    norm_clean = np.linalg.norm(flat_clean)
    norm_degraded = np.linalg.norm(flat_degraded)

    if norm_clean == 0 or norm_degraded == 0:
        return 0.0

    return float(np.dot(flat_clean, flat_degraded) / (norm_clean * norm_degraded))


def compute_spectral_band_alignment(
    attribution: np.ndarray,
    detection_score: float,
    artifact_bands: list = None,
    n_mels: int = 128,
    sample_rate: int = 16000,
) -> float:
    """
    Compute spectral band alignment.

    Measures whether attributions are concentrated in frequency bands
    known to contain deepfake artifacts.

    Args:
        attribution: Attribution map [F, T'].
        detection_score: Model's spoof probability.
        artifact_bands: List of (low_hz, high_hz) artifact frequency ranges.
        n_mels: Number of mel bins.
        sample_rate: Audio sample rate.

    Returns:
        Alignment score (correlation) in [-1, 1].
    """
    if artifact_bands is None:
        artifact_bands = [
            (0, 500),       # Low-frequency artifacts
            (4000, 8000),   # High-frequency synthesis artifacts
            (2000, 4000),   # Formant region
        ]

    abs_attr = np.abs(attribution)
    freq_importance = np.mean(abs_attr, axis=1)  # [F]

    # Create artifact mask
    artifact_mask = np.zeros(n_mels)
    max_freq = sample_rate / 2
    for low_hz, high_hz in artifact_bands:
        low_bin = int(low_hz / max_freq * n_mels)
        high_bin = int(high_hz / max_freq * n_mels)
        low_bin = max(0, min(low_bin, n_mels - 1))
        high_bin = max(0, min(high_bin, n_mels))
        artifact_mask[low_bin:high_bin] = 1.0

    # Handle size mismatch
    min_len = min(len(freq_importance), len(artifact_mask))
    freq_importance = freq_importance[:min_len]
    artifact_mask = artifact_mask[:min_len]

    # Compute fraction of attribution mass in artifact bands
    total_mass = np.sum(freq_importance)
    if total_mass == 0:
        return 0.0

    artifact_mass = np.sum(freq_importance * artifact_mask)
    alignment = artifact_mass / total_mass

    return float(alignment)


def compute_all_faithfulness_metrics(
    model_fn: Callable,
    waveform: np.ndarray,
    attribution: np.ndarray,
    detection_score: float,
    attr_clean: np.ndarray = None,
    n_steps: int = 20,
    hop_length: int = 512,
) -> Dict:
    """
    Compute all faithfulness metrics for a single sample.

    Args:
        model_fn: Model prediction function.
        waveform: Audio waveform [T].
        attribution: Attribution map [F, T'].
        detection_score: Model's spoof probability.
        attr_clean: Attribution from clean version (for stability).
        n_steps: Steps for deletion/insertion.

    Returns:
        Dict with all faithfulness metrics.
    """
    results = {}

    # Deletion AUC
    del_auc, del_curve = compute_deletion_auc(
        model_fn, waveform, attribution, n_steps, hop_length=hop_length
    )
    results["deletion_auc"] = del_auc
    results["deletion_curve"] = del_curve

    # Insertion AUC
    ins_auc, ins_curve = compute_insertion_auc(
        model_fn, waveform, attribution, n_steps, hop_length=hop_length
    )
    results["insertion_auc"] = ins_auc
    results["insertion_curve"] = ins_curve

    # Sensitivity-N
    sensitivity = compute_sensitivity_n(
        model_fn, waveform, attribution, hop_length=hop_length
    )
    results["sensitivity_n"] = sensitivity

    # Spectral band alignment
    sba = compute_spectral_band_alignment(attribution, detection_score)
    results["spectral_band_alignment"] = sba

    # Explanation stability (if clean attribution provided)
    if attr_clean is not None:
        stability = compute_explanation_stability(attr_clean, attribution)
        results["explanation_stability"] = stability

    return results


# ---- Helper functions ----

def _apply_spectrogram_mask(
    waveform: np.ndarray,
    spec_shape: tuple,
    mask_indices: np.ndarray,
    baseline_value: float,
    hop_length: int,
) -> np.ndarray:
    """
    Apply a spectrogram-domain mask to the waveform.

    Maps spectrogram feature indices back to waveform samples
    and replaces them with the baseline value.
    """
    masked = waveform.copy()
    n_freq, n_time = spec_shape

    for idx in mask_indices:
        freq_bin = idx // n_time
        time_frame = idx % n_time

        # Map time frame to waveform samples
        start_sample = time_frame * hop_length
        end_sample = min(start_sample + hop_length, len(masked))

        # Apply mask (zero out the corresponding waveform segment)
        masked[start_sample:end_sample] *= 0.5  # Attenuate rather than zero

    return masked


def _apply_spectrogram_insertion(
    waveform: np.ndarray,
    spec_shape: tuple,
    insert_indices: np.ndarray,
    baseline_value: float,
    hop_length: int,
) -> np.ndarray:
    """
    Start from baseline, insert original values at specified positions.
    """
    inserted = np.full_like(waveform, baseline_value)
    n_freq, n_time = spec_shape

    for idx in insert_indices:
        time_frame = idx % n_time
        start_sample = time_frame * hop_length
        end_sample = min(start_sample + hop_length, len(waveform))
        inserted[start_sample:end_sample] = waveform[start_sample:end_sample]

    return inserted
