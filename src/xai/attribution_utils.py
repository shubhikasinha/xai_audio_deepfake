"""
Attribution processing utilities.

Common operations on attribution maps used across XAI methods
and evaluation metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def normalize_attribution(
    attribution: np.ndarray,
    method: str = "abs_max",
) -> np.ndarray:
    """
    Normalize attribution map.

    Args:
        attribution: Raw attribution map [F, T].
        method: Normalization method.
            - "abs_max": Divide by max absolute value
            - "sum": Divide by sum of absolute values
            - "minmax": Scale to [0, 1] range
            - "percentile": Clip to [1st, 99th] percentile, then minmax

    Returns:
        Normalized attribution map [F, T].
    """
    if method == "abs_max":
        max_val = np.max(np.abs(attribution))
        if max_val > 0:
            return attribution / max_val
        return attribution

    elif method == "sum":
        total = np.sum(np.abs(attribution))
        if total > 0:
            return attribution / total
        return attribution

    elif method == "minmax":
        min_val = np.min(attribution)
        max_val = np.max(attribution)
        if max_val > min_val:
            return (attribution - min_val) / (max_val - min_val)
        return np.zeros_like(attribution)

    elif method == "percentile":
        p1 = np.percentile(attribution, 1)
        p99 = np.percentile(attribution, 99)
        clipped = np.clip(attribution, p1, p99)
        if p99 > p1:
            return (clipped - p1) / (p99 - p1)
        return np.zeros_like(attribution)

    raise ValueError(f"Unknown normalization method: {method}")


def get_top_k_regions(
    attribution: np.ndarray,
    k: int = 5,
    freq_resolution_hz: float = 62.5,  # For 16kHz, 128 mel bins
    time_resolution_sec: float = 0.032,  # hop_length/sample_rate
) -> Dict:
    """
    Extract top-k most important spectral/temporal regions.

    Args:
        attribution: Attribution map [F, T].
        k: Number of top regions.
        freq_resolution_hz: Frequency resolution per bin (Hz).
        time_resolution_sec: Time resolution per frame (seconds).

    Returns:
        Dict with top regions, their frequency ranges, and time ranges.
    """
    abs_attr = np.abs(attribution)

    # Aggregate by frequency bands (sum over time)
    freq_importance = np.sum(abs_attr, axis=1)  # [F]
    top_freq_bins = np.argsort(freq_importance)[-k:][::-1]

    # Aggregate by time frames (sum over frequency)
    time_importance = np.sum(abs_attr, axis=0)  # [T]
    top_time_frames = np.argsort(time_importance)[-k:][::-1]

    # Convert to physical units
    top_freq_ranges = [
        (int(b * freq_resolution_hz), int((b + 1) * freq_resolution_hz))
        for b in top_freq_bins
    ]
    top_time_ranges = [
        (round(t * time_resolution_sec, 3), round((t + 1) * time_resolution_sec, 3))
        for t in top_time_frames
    ]

    return {
        "top_freq_bins": top_freq_bins,
        "top_freq_ranges_hz": top_freq_ranges,
        "freq_importance": freq_importance,
        "top_time_frames": top_time_frames,
        "top_time_ranges_sec": top_time_ranges,
        "time_importance": time_importance,
    }


def compute_attribution_concentration(attribution: np.ndarray) -> float:
    """
    Compute how concentrated the attribution is.

    Uses the Gini coefficient of absolute attribution values.
    Higher = more concentrated (few features dominate).
    Lower = more spread out.

    Returns:
        Gini coefficient in [0, 1].
    """
    abs_attr = np.abs(attribution).flatten()
    abs_attr = np.sort(abs_attr)
    n = len(abs_attr)

    if n == 0 or np.sum(abs_attr) == 0:
        return 0.0

    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * abs_attr) / (n * np.sum(abs_attr))) - (n + 1) / n
    return float(gini)


def compute_spectral_energy_distribution(
    attribution: np.ndarray,
    n_bands: int = 4,
) -> Dict:
    """
    Compute energy distribution across frequency bands.

    Args:
        attribution: Attribution map [F, T].
        n_bands: Number of frequency bands to divide into.

    Returns:
        Dict with per-band energy fractions and labels.
    """
    abs_attr = np.abs(attribution)
    n_freq = abs_attr.shape[0]
    band_size = n_freq // n_bands

    total_energy = np.sum(abs_attr)
    if total_energy == 0:
        return {
            "band_energies": [0.0] * n_bands,
            "band_fractions": [0.0] * n_bands,
            "band_labels": [f"Band {i}" for i in range(n_bands)],
        }

    band_energies = []
    band_fractions = []
    band_labels = []

    for i in range(n_bands):
        start = i * band_size
        end = start + band_size if i < n_bands - 1 else n_freq
        energy = float(np.sum(abs_attr[start:end]))
        band_energies.append(energy)
        band_fractions.append(energy / total_energy)
        band_labels.append(f"Band {i} ({start}-{end})")

    return {
        "band_energies": band_energies,
        "band_fractions": band_fractions,
        "band_labels": band_labels,
    }


def compute_cosine_similarity(
    attr1: np.ndarray,
    attr2: np.ndarray,
) -> float:
    """
    Compute cosine similarity between two attribution maps.

    Args:
        attr1: First attribution map [F, T].
        attr2: Second attribution map [F, T].

    Returns:
        Cosine similarity in [-1, 1].
    """
    flat1 = attr1.flatten()
    flat2 = attr2.flatten()

    # Handle size mismatch (different spectrogram lengths)
    min_len = min(len(flat1), len(flat2))
    flat1 = flat1[:min_len]
    flat2 = flat2[:min_len]

    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(flat1, flat2) / (norm1 * norm2))
