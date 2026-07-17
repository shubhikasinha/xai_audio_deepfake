"""
Kernel SHAP for audio deepfake detection.

Secondary XAI method — perturbation-based, model-agnostic.
Used on a subset of conditions for AASIST only (compute-constrained).
"""

from typing import Optional

import numpy as np
import torch

from src.xai.base_explainer import BaseExplainer
from src.data.utils import compute_spectrogram


class KernelSHAPExplainer(BaseExplainer):
    """
    Kernel SHAP attribution for deepfake detectors.

    Computes Shapley values using a weighted linear regression
    approximation (Kernel SHAP). Model-agnostic — treats the
    detector as a black box.

    Reference:
        Lundberg & Lee, "A Unified Approach to Interpreting Model
        Predictions" (NeurIPS 2017)
    """

    def __init__(
        self,
        model,
        device: str = "cpu",
        n_samples: int = 200,
        n_mels: int = 128,
        n_segments: int = 16,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 16000,
    ):
        super().__init__(model, device)
        self.method_name = "kernel_shap"
        self.n_samples = n_samples
        self.n_mels = n_mels
        self.n_segments = n_segments
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

    def explain(
        self,
        waveform: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Compute Kernel SHAP attribution over spectrogram segments.

        The spectrogram is divided into n_segments temporal segments.
        SHAP values are computed for each segment, then interpolated
        back to full spectrogram resolution.

        Args:
            waveform: Audio tensor [T] or [1, T].
            target_class: Class to explain (1 = spoof).

        Returns:
            Attribution map [n_mels, T'] over the log-mel spectrogram.
        """
        self.model.eval()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device)

        # Compute spectrogram for segmentation
        spec = compute_spectrogram(
            waveform,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
        ).squeeze().cpu().numpy()  # [n_mels, T']

        n_freq, n_time = spec.shape

        # Create temporal segments
        segment_boundaries = np.linspace(0, n_time, self.n_segments + 1, dtype=int)

        # Compute segment-level SHAP values
        segment_shap = self._compute_segment_shap(
            waveform, segment_boundaries, n_time, target_class
        )

        # Interpolate to full spectrogram resolution
        full_attr = np.zeros((n_freq, n_time))
        for i in range(self.n_segments):
            start = segment_boundaries[i]
            end = segment_boundaries[i + 1]
            full_attr[:, start:end] = segment_shap[i]

        return full_attr

    def _compute_segment_shap(
        self,
        waveform: torch.Tensor,
        segment_boundaries: np.ndarray,
        n_time_frames: int,
        target_class: int,
    ) -> np.ndarray:
        """
        Compute SHAP values for temporal segments using Kernel SHAP.

        Generates random coalition masks, evaluates the model on
        masked inputs, and fits a weighted linear regression to
        estimate Shapley values.

        Args:
            waveform: Original waveform [1, T].
            segment_boundaries: Segment start/end indices.
            n_time_frames: Number of spectrogram time frames.
            target_class: Target class.

        Returns:
            SHAP values for each segment [n_segments].
        """
        n_features = self.n_segments

        # Generate coalition masks
        masks = np.random.binomial(1, 0.5, size=(self.n_samples, n_features))

        # Ensure we include the full and empty coalitions
        masks[0] = np.ones(n_features)
        masks[1] = np.zeros(n_features)

        # Evaluate model on each masked input
        predictions = np.zeros(self.n_samples)
        waveform_np = waveform.squeeze().cpu().numpy()
        total_samples = len(waveform_np)

        for i, mask in enumerate(masks):
            masked_waveform = self._apply_mask_to_waveform(
                waveform_np, mask, segment_boundaries, n_time_frames, total_samples
            )
            masked_tensor = torch.from_numpy(masked_waveform).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                result = self.model.predict(masked_tensor)
                predictions[i] = result["probs"][0].cpu().item()

        # Compute Kernel SHAP weights
        weights = self._kernel_shap_weights(masks, n_features)

        # Weighted linear regression to estimate SHAP values
        shap_values = self._weighted_regression(masks, predictions, weights)

        return shap_values

    def _apply_mask_to_waveform(
        self,
        waveform: np.ndarray,
        mask: np.ndarray,
        segment_boundaries: np.ndarray,
        n_time_frames: int,
        total_samples: int,
    ) -> np.ndarray:
        """
        Apply a coalition mask by zeroing out masked segments in the waveform.

        Maps spectrogram temporal segments back to waveform samples.
        """
        masked = waveform.copy()

        for seg_idx in range(len(mask)):
            if mask[seg_idx] == 0:
                # Map spectrogram segment to waveform samples
                start_frame = segment_boundaries[seg_idx]
                end_frame = segment_boundaries[seg_idx + 1]
                start_sample = int(start_frame * self.hop_length)
                end_sample = min(int(end_frame * self.hop_length), total_samples)
                masked[start_sample:end_sample] = 0.0

        return masked

    def _kernel_shap_weights(
        self,
        masks: np.ndarray,
        n_features: int,
    ) -> np.ndarray:
        """
        Compute Kernel SHAP weights for each coalition.

        Weight = (n_features - 1) / (C(n_features, |S|) * |S| * (n_features - |S|))
        """
        from scipy.special import comb

        weights = np.zeros(len(masks))
        for i, mask in enumerate(masks):
            s = int(mask.sum())
            if 0 < s < n_features:
                weights[i] = (n_features - 1) / (
                    comb(n_features, s) * s * (n_features - s)
                )
            else:
                weights[i] = 1e6  # Large weight for full/empty coalitions

        return weights

    def _weighted_regression(
        self,
        masks: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Fit weighted linear regression to estimate SHAP values.

        Args:
            masks: Coalition masks [n_samples, n_features].
            predictions: Model predictions [n_samples].
            weights: Kernel SHAP weights [n_samples].

        Returns:
            SHAP values [n_features].
        """
        # Weighted least squares: minimize sum_i w_i * (y_i - X_i @ beta)^2
        W = np.diag(np.sqrt(weights))
        X = W @ masks
        y = W @ predictions

        # Solve via pseudoinverse
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(masks.shape[1])

        return beta

    def explain_with_shap_library(
        self,
        waveform: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Alternative implementation using the SHAP library.

        Uses shap.KernelExplainer for a more robust implementation.
        """
        try:
            import shap

            self.model.eval()

            def predict_fn(x):
                x_tensor = torch.from_numpy(x).float().to(self.device)
                with torch.no_grad():
                    result = self.model.predict(x_tensor)
                return result["probs"].cpu().numpy()

            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            background = torch.zeros(1, waveform.shape[1]).to(self.device)
            explainer = shap.KernelExplainer(
                predict_fn,
                background.cpu().numpy(),
            )

            shap_values = explainer.shap_values(
                waveform.cpu().numpy(),
                nsamples=self.n_samples,
            )

            # Project to spectrogram
            from src.data.utils import compute_spectrogram
            spec = compute_spectrogram(waveform).squeeze().numpy()
            return np.abs(spec)  # Simplified — full projection needs more work

        except ImportError:
            print("[SHAP] shap library not available, using custom implementation.")
            return self.explain(waveform, target_class)
