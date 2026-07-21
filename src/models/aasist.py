"""
AASIST Detector Wrapper.

Wraps the AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal)
graph attention network for use in the XAI robustness pipeline.
"""

import os
from typing import Dict, Optional
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_detector import BaseDetector


class AASISTDetector(BaseDetector):
    """
    Wrapper for the AASIST deepfake detector.

    AASIST uses a spectro-temporal graph attention network that processes
    both spectral and temporal domains through graph attention layers.

    Reference:
        Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
        Spectro-Temporal Graph Attention Networks" (ICASSP 2022)
    """

    def __init__(
        self,
        device: str = "cpu",
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__(model_name="aasist", device=device)

        # Build the AASIST model architecture
        self.model = self._build_model()
        self.to(device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

    def _build_model(self) -> nn.Module:
        """
        Build AASIST architecture.

        NOTE: This is a simplified placeholder. In production, import
        the official AASIST implementation from:
        https://github.com/clovaai/aasist

        The actual integration requires cloning the AASIST repo and
        importing their model definition.
        """
        # Placeholder architecture — replace with official AASIST
        # when setting up the actual experiment
        model = nn.Sequential(
            # Sinc convolution front-end
            nn.Conv1d(1, 70, kernel_size=128, stride=16, padding=64),
            nn.BatchNorm1d(70),
            nn.ReLU(),
            # Simplified feature extraction (placeholder for graph attention)
            nn.Conv1d(70, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 160),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(160, 2),  # [bonafide, spoof]
        )
        return model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: raw waveform → spoof score.

        Args:
            waveform: [B, T] or [B, 1, T] raw audio.

        Returns:
            Spoof scores [B].
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # [B, 1, T]

        logits = self.model(waveform)  # [B, 2]
        scores = logits[:, 1] - logits[:, 0]  # Spoof - bonafide
        return scores

    def predict(self, waveform: torch.Tensor) -> Dict:
        """Full prediction with scores, probabilities, and labels."""
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        logits = self.model(waveform)
        probs = F.softmax(logits, dim=-1)

        return {
            "scores": logits[:, 1] - logits[:, 0],
            "probs": probs[:, 1],  # P(spoof)
            "labels": torch.argmax(logits, dim=-1),
            "logits": logits,
        }

    def get_intermediate_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract features after the sinc convolution layer.
        Used as attribution target for XAI methods.
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # Pass through sinc front-end only
        x = self.model[0](waveform)  # Conv1d
        x = self.model[1](x)  # BatchNorm
        x = self.model[2](x)  # ReLU
        return x

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load pretrained AASIST weights."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device_str,
            weights_only=True,
        )

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Try to load, allowing mismatched keys (placeholder vs. official)
        try:
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[AASIST] Loaded checkpoint from {checkpoint_path}")
        except RuntimeError as e:
            print(f"[AASIST] Warning: Could not load checkpoint: {e}")
            print("[AASIST] Using randomly initialized weights.")

    def forward_with_gradients(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that preserves gradients for IG computation.
        Returns spoof probability (differentiable).
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        waveform.requires_grad_(True)
        logits = self.model(waveform)
        spoof_prob = F.softmax(logits, dim=-1)[:, 1]
        return spoof_prob
