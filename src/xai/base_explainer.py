"""
Abstract base class for explainability methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import numpy as np


class BaseExplainer(ABC):
    """
    Abstract interface for XAI methods.

    All explainers must:
    1. Accept a detector model and an input waveform
    2. Return attribution maps over the spectrogram representation
    3. Support batch processing
    """

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.method_name = "base"

    @abstractmethod
    def explain(
        self,
        waveform: torch.Tensor,
        target_class: int = 1,  # 1 = spoof
    ) -> np.ndarray:
        """
        Compute attribution map for a single audio sample.

        Args:
            waveform: Audio tensor [1, T] or [T].
            target_class: Class index to explain (1 = spoof).

        Returns:
            Attribution map as numpy array [F, T'] (frequency × time).
        """
        pass

    def explain_batch(
        self,
        waveforms: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Compute attribution maps for a batch of samples.

        Args:
            waveforms: Audio tensor [B, T].
            target_class: Class index to explain.

        Returns:
            Attribution maps [B, F, T'].
        """
        attributions = []
        for i in range(waveforms.shape[0]):
            attr = self.explain(waveforms[i], target_class)
            attributions.append(attr)
        return np.stack(attributions)

    def get_top_k_features(
        self,
        attribution: np.ndarray,
        k: int = 10,
    ) -> Dict:
        """
        Extract top-k most important features from attribution map.

        Args:
            attribution: Attribution map [F, T'].
            k: Number of top features.

        Returns:
            Dict with top feature indices, values, and frequency/time info.
        """
        flat_attr = np.abs(attribution).flatten()
        top_indices = np.argsort(flat_attr)[-k:][::-1]

        top_freq_indices = top_indices // attribution.shape[1]
        top_time_indices = top_indices % attribution.shape[1]

        return {
            "indices": top_indices,
            "freq_indices": top_freq_indices,
            "time_indices": top_time_indices,
            "values": flat_attr[top_indices],
            "attribution_map": attribution,
        }
