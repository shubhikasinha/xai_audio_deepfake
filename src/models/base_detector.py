"""
Abstract base class for deepfake detectors.

All detector wrappers must implement this interface to ensure
consistent behavior across the XAI and evaluation pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseDetector(ABC, nn.Module):
    """
    Abstract interface for audio deepfake detectors.

    All detectors must:
    1. Accept raw waveform input [B, 1, T] or [B, T]
    2. Return detection scores (higher = more likely spoof)
    3. Support gradient computation for XAI methods
    4. Provide a method to extract spectrogram-level features
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__()
        self.model_name = model_name
        self.device_str = device

    @abstractmethod
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: raw waveform → detection score.

        Args:
            waveform: Audio tensor [B, T] or [B, 1, T].

        Returns:
            Detection scores [B] — higher = more likely spoof.
        """
        pass

    @abstractmethod
    def predict(self, waveform: torch.Tensor) -> Dict:
        """
        Full prediction with scores, labels, and probabilities.

        Args:
            waveform: Audio tensor [B, T] or [B, 1, T].

        Returns:
            Dict with keys:
                - "scores": Raw detection scores [B]
                - "probs": Spoof probability [B] (sigmoid/softmax of scores)
                - "labels": Binary labels [B] (0=bonafide, 1=spoof)
        """
        pass

    @abstractmethod
    def get_intermediate_features(
        self, waveform: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract intermediate features for XAI attribution.

        Returns spectrogram-level or layer-level features that can be
        used as the attribution target for IG/SHAP.

        Args:
            waveform: Audio tensor [B, T].

        Returns:
            Feature tensor suitable for attribution computation.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load pretrained weights from checkpoint file."""
        pass

    def get_spoof_probability(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Get spoof probability (convenience method).

        Args:
            waveform: Audio tensor [B, T].

        Returns:
            Spoof probability [B] in [0, 1].
        """
        result = self.predict(waveform)
        return result["probs"]

    def classify(
        self,
        waveform: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify audio as bonafide or spoof.

        Args:
            waveform: Audio tensor [B, T].
            threshold: Decision threshold for spoof probability.

        Returns:
            Tuple of (labels [B], probabilities [B]).
        """
        probs = self.get_spoof_probability(waveform)
        labels = (probs >= threshold).long()
        return labels, probs
