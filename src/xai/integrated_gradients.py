"""
Integrated Gradients for audio deepfake detection.

Primary XAI method — axiomatically grounded (completeness, sensitivity).
Computes attributions over spectrogram representations.
"""

from typing import Optional
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.xai.base_explainer import BaseExplainer
from src.data.utils import compute_spectrogram


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Integrated Gradients attribution for deepfake detectors.

    Computes attributions by integrating gradients along a path from
    a baseline input to the actual input. Satisfies:
    - Completeness: attributions sum to (F(x) - F(baseline))
    - Sensitivity: non-zero attribution for features that cause change

    Reference:
        Sundararajan et al., "Axiomatic Attribution for Deep Networks"
        (ICML 2017)
    """

    def __init__(
        self,
        model,
        device: str = "cpu",
        n_steps: int = 50,
        baseline_type: str = "zero",
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 16000,
    ):
        super().__init__(model, device)
        self.method_name = "integrated_gradients"
        self.n_steps = n_steps
        self.baseline_type = baseline_type
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

    def explain(
        self,
        waveform: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Compute Integrated Gradients attribution over spectrogram.

        The attribution is computed in the waveform domain but projected
        onto the spectrogram for visualization and evaluation.

        Args:
            waveform: Audio tensor [T] or [1, T].
            target_class: Class to explain (1 = spoof).

        Returns:
            Attribution map [n_mels, T'] over the log-mel spectrogram.
        """
        self.model.eval()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # [1, 1, T]

        waveform = waveform.to(self.device)

        # Create baseline
        baseline = self._create_baseline(waveform)

        # Compute IG in waveform domain
        waveform_attr = self._compute_ig_waveform(
            waveform, baseline, target_class
        )

        # Project attributions onto spectrogram
        spec_attr = self._project_to_spectrogram(waveform_attr.squeeze())

        return spec_attr

    def _create_baseline(self, waveform: torch.Tensor) -> torch.Tensor:
        """Create baseline input for IG computation."""
        if self.baseline_type == "zero":
            return torch.zeros_like(waveform)
        elif self.baseline_type == "random":
            return torch.randn_like(waveform) * 0.01
        elif self.baseline_type == "mean":
            return torch.ones_like(waveform) * waveform.mean()
        else:
            return torch.zeros_like(waveform)

    def _compute_ig_waveform(
        self,
        waveform: torch.Tensor,
        baseline: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients in the waveform domain.

        Args:
            waveform: Input [1, 1, T].
            baseline: Baseline [1, 1, T].
            target_class: Target class index.

        Returns:
            Attribution tensor [1, 1, T].
        """
        # Generate interpolated inputs along the path
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=self.device)

        # Accumulate gradients
        integrated_grads = torch.zeros_like(waveform)

        for alpha in alphas:
            interpolated = baseline + alpha * (waveform - baseline)
            interpolated = interpolated.detach().requires_grad_(True)

            # Forward pass
            output = self.model(interpolated.squeeze(1))  # Model expects [B, T]

            # For binary classification, use the target class score
            if output.dim() == 1:
                target_score = output[0]
            else:
                # If model returns logits [B, 2]
                result = self.model.predict(interpolated.squeeze(1))
                target_score = result["probs"][0]

            # Backward pass
            self.model.zero_grad()
            target_score.backward(retain_graph=True)

            if interpolated.grad is not None:
                integrated_grads += interpolated.grad

        # Average the gradients and multiply by (input - baseline)
        avg_grads = integrated_grads / (self.n_steps + 1)
        attributions = (waveform - baseline) * avg_grads

        return attributions.detach()

    def _project_to_spectrogram(
        self,
        waveform_attr: torch.Tensor,
    ) -> np.ndarray:
        """
        Project waveform-domain attributions onto log-mel spectrogram.

        Uses the magnitude of attributions in each STFT frame and mel bin
        as a proxy for the spectrogram-level importance.

        Args:
            waveform_attr: Waveform attribution [T].

        Returns:
            Spectrogram attribution [n_mels, T'].
        """
        waveform_attr = waveform_attr.cpu()

        if waveform_attr.dim() > 1:
            waveform_attr = waveform_attr.squeeze()

        # Compute STFT of attributions
        stft_attr = torch.stft(
            waveform_attr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
        )
        magnitude_attr = torch.abs(stft_attr)  # [F, T']

        # Apply mel filterbank to get mel-scale attributions
        mel_fb = self._mel_filterbank()
        mel_attr = torch.matmul(mel_fb, magnitude_attr)  # [n_mels, T']

        return mel_attr.numpy()

    def _mel_filterbank(self) -> torch.Tensor:
        """Create mel filterbank matrix [n_mels, n_fft//2+1]."""
        import librosa

        mel_fb = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
        )
        return torch.from_numpy(mel_fb).float()

    def explain_with_captum(
        self,
        waveform: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Alternative implementation using the Captum library.

        Captum provides a well-tested IG implementation with
        additional features like convergence checking.
        """
        try:
            from captum.attr import IntegratedGradients as CaptumIG

            self.model.eval()

            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            waveform = waveform.to(self.device)
            baseline = self._create_baseline(waveform.unsqueeze(0)).squeeze(0)

            # Wrap model for Captum
            def forward_fn(x):
                result = self.model.predict(x)
                return result["probs"].unsqueeze(-1)

            ig = CaptumIG(forward_fn)
            attr = ig.attribute(
                waveform,
                baselines=baseline,
                n_steps=self.n_steps,
                return_convergence_delta=False,
            )

            # Project to spectrogram
            spec_attr = self._project_to_spectrogram(attr.squeeze())
            return spec_attr

        except ImportError:
            print("[IG] Captum not available, using custom implementation.")
            return self.explain(waveform, target_class)
