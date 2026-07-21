"""
WavLM + ECAPA-TDNN Detector Wrapper.

Combines Microsoft's WavLM self-supervised speech model as a front-end
feature extractor with an ECAPA-TDNN back-end for deepfake detection.
"""

import os
from typing import Dict, List, Optional
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_detector import BaseDetector


class WavLMECAPADetector(BaseDetector):
    """
    WavLM + ECAPA-TDNN deepfake detector.

    Architecture:
    - Front-end: WavLM-Large (frozen SSL transformer, feature extractor)
    - Weighted layer aggregation across selected transformer layers
    - Back-end: ECAPA-TDNN for utterance-level embedding
    - Classifier: Linear layers for bonafide/spoof classification
    """

    def __init__(
        self,
        device: str = "cpu",
        checkpoint_path: Optional[str] = None,
        freeze_frontend: bool = True,
        selected_layers: Optional[List[int]] = None,
    ):
        super().__init__(model_name="wavlm_ecapa", device=device)

        self.freeze_frontend = freeze_frontend
        self.selected_layers = selected_layers or [6, 12, 18, 24]

        # Build components
        self.frontend = self._build_frontend()
        self.layer_weights = nn.Parameter(
            torch.ones(len(self.selected_layers)) / len(self.selected_layers)
        )
        self.backend = self._build_backend()
        self.classifier = self._build_classifier()

        self.to(device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

    def _build_frontend(self) -> nn.Module:
        """
        Build WavLM front-end.

        NOTE: Requires `transformers` library. On first run, downloads
        WavLM-Large (~1.2GB) from HuggingFace.
        """
        try:
            from transformers import WavLMModel

            model = WavLMModel.from_pretrained("microsoft/wavlm-large")

            if self.freeze_frontend:
                for param in model.parameters():
                    param.requires_grad = False

            return model

        except ImportError:
            print("[WavLM] transformers not installed. Using placeholder.")
            return self._build_placeholder_frontend()

    def _build_placeholder_frontend(self) -> nn.Module:
        """Placeholder front-end for testing without transformers."""
        return nn.Sequential(
            nn.Conv1d(1, 1024, kernel_size=400, stride=320, padding=200),
            nn.LayerNorm([1024]),
            nn.GELU(),
        )

    def _build_backend(self) -> nn.Module:
        """
        Build ECAPA-TDNN back-end.

        Simplified version — for production, use SpeechBrain's ECAPA-TDNN.
        """
        input_dim = 1024  # WavLM-Large hidden size

        backend = nn.Sequential(
            # Frame-level processing
            nn.Conv1d(input_dim, 512, kernel_size=5, dilation=1, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # Channel attention + aggregation
            nn.Conv1d(512, 1536, kernel_size=1),
            nn.BatchNorm1d(1536),
            nn.ReLU(),
        )
        return backend

    def _build_classifier(self) -> nn.Module:
        """Build the classification head."""
        return nn.Sequential(
            nn.Linear(3072, 256),  # 1536 * 2 (mean + std pooling)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),  # [bonafide, spoof]
        )

    def _extract_wavlm_features(
        self, waveform: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract weighted features from selected WavLM layers.

        Args:
            waveform: [B, T] raw audio at 16kHz.

        Returns:
            Weighted features [B, T', 1024].
        """
        try:
            from transformers import WavLMModel

            if isinstance(self.frontend, WavLMModel):
                with torch.no_grad() if self.freeze_frontend else torch.enable_grad():
                    outputs = self.frontend(
                        waveform,
                        output_hidden_states=True,
                    )

                hidden_states = outputs.hidden_states  # Tuple of [B, T', 1024]

                # Select and weight specific layers
                weights = F.softmax(self.layer_weights, dim=0)
                weighted_features = torch.zeros_like(hidden_states[0])

                for i, layer_idx in enumerate(self.selected_layers):
                    if layer_idx < len(hidden_states):
                        weighted_features += weights[i] * hidden_states[layer_idx]

                return weighted_features

        except (ImportError, AttributeError):
            pass

        # Fallback for placeholder
        if waveform.dim() == 2:
            x = waveform.unsqueeze(1)
        else:
            x = waveform
        x = self.frontend(x)
        if x.dim() == 3 and x.shape[1] != 1024:
            x = x.transpose(1, 2)  # [B, 1024, T'] → [B, T', 1024]
        return x

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: raw waveform → spoof score.

        Args:
            waveform: [B, T] raw audio.

        Returns:
            Spoof scores [B].
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # [B, 1, T] → [B, T]

        # Front-end: WavLM features
        features = self._extract_wavlm_features(waveform)  # [B, T', 1024]

        # Back-end: ECAPA-TDNN
        x = features.transpose(1, 2)  # [B, 1024, T']
        x = self.backend(x)  # [B, 1536, T']

        # Statistical pooling (mean + std)
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        pooled = torch.cat([mean, std], dim=1)  # [B, 3072]

        # Classifier
        logits = self.classifier(pooled)  # [B, 2]
        scores = logits[:, 1] - logits[:, 0]

        return scores

    def predict(self, waveform: torch.Tensor) -> Dict:
        """Full prediction with scores, probabilities, and labels."""
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        features = self._extract_wavlm_features(waveform)
        x = features.transpose(1, 2)
        x = self.backend(x)

        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        pooled = torch.cat([mean, std], dim=1)

        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)

        return {
            "scores": logits[:, 1] - logits[:, 0],
            "probs": probs[:, 1],
            "labels": torch.argmax(logits, dim=-1),
            "logits": logits,
        }

    def get_intermediate_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract WavLM features for XAI attribution.
        Returns the weighted layer features before ECAPA processing.
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        return self._extract_wavlm_features(waveform)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load pretrained weights (back-end + classifier only if frontend frozen)."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device_str,
            weights_only=True,
        )

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get(
                "model_state_dict",
                checkpoint.get("state_dict", checkpoint),
            )
        else:
            state_dict = checkpoint

        try:
            self.load_state_dict(state_dict, strict=False)
            print(f"[WavLM+ECAPA] Loaded checkpoint from {checkpoint_path}")
        except RuntimeError as e:
            print(f"[WavLM+ECAPA] Warning: Partial load: {e}")
