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
        Imports the official AASIST implementation.
        """
        from src.models.aasist_official import Model as OfficialAASISTModel
        model_config = {
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0]
        }
        return OfficialAASISTModel(model_config)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: raw waveform → spoof score.

        Args:
            waveform: [B, T] or [B, 1, T] raw audio.

        Returns:
            Spoof scores [B].
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # Ensure [B, T]

        _, logits = self.model(waveform)  # [B, 2]
        scores = logits[:, 1] - logits[:, 0]  # Spoof - bonafide
        return scores

    def predict(self, waveform: torch.Tensor) -> Dict:
        """Full prediction with scores, probabilities, and labels."""
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        _, logits = self.model(waveform)
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
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        x = waveform.unsqueeze(1)
        x = self.model.conv_time(x)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.model.first_bn(x)
        x = self.model.selu(x)
        return x

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load pretrained AASIST weights."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device_str,
            weights_only=True,
        )

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint

        # Strip 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v

        try:
            self.model.load_state_dict(new_state_dict, strict=True)
            print(f"[AASIST] Loaded checkpoint from {checkpoint_path}")
        except RuntimeError as e:
            print(f"[AASIST] Warning: strict load failed, trying strict=False: {e}")
            try:
                self.model.load_state_dict(new_state_dict, strict=False)
                print(f"[AASIST] Loaded checkpoint (non-strict) from {checkpoint_path}")
            except Exception as ex:
                print(f"[AASIST] Error: Could not load checkpoint: {ex}")

    def forward_with_gradients(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that preserves gradients for IG computation.
        Returns spoof probability (differentiable).
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        waveform.requires_grad_(True)
        _, logits = self.model(waveform)
        spoof_prob = F.softmax(logits, dim=-1)[:, 1]
        return spoof_prob
