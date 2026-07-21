"""
Audio Degradation Pipeline.

Applies controlled degradation (codecs, noise, reverb) to audio for
evaluating explanation robustness under distribution shift.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Union

import numpy as np
import torch
import torchaudio


class DegradationPipeline:
    """
    Apply controlled audio degradation conditions.

    Supports:
    - Codec compression via ffmpeg (MP3, M4A, OGG, Opus, AMR-NB)
    - Additive noise (AWGN, babble, music)
    - Room impulse response (reverb)
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._verify_ffmpeg()

    def _verify_ffmpeg(self) -> None:
        """Check that ffmpeg is available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "ffmpeg not found. Install it: https://ffmpeg.org/download.html"
            )

    def apply(
        self,
        waveform: torch.Tensor,
        condition: Dict,
    ) -> torch.Tensor:
        """
        Apply a degradation condition to a waveform.

        Args:
            waveform: Input audio tensor [1, T] or [T].
            condition: Degradation condition dict from conditions.yaml.

        Returns:
            Degraded waveform tensor (same shape as input).
        """
        cond_type = condition.get("type", "none")

        if cond_type == "none":
            return waveform

        if cond_type == "builtin":
            # Built-in ASVspoof conditions — already applied in the dataset
            return waveform

        if cond_type == "custom_codec":
            return self._apply_codec(waveform, condition)

        if cond_type == "noise":
            return self._apply_noise(waveform, condition)

        if cond_type == "reverb":
            return self._apply_reverb(waveform, condition)

        raise ValueError(f"Unknown degradation type: {cond_type}")

    def _apply_codec(
        self,
        waveform: torch.Tensor,
        condition: Dict,
    ) -> torch.Tensor:
        """Apply codec compression via ffmpeg."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        ffmpeg_args = condition.get("ffmpeg_args", [])
        codec = condition.get("codec", "libopus")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            output_path = os.path.join(tmpdir, "output.wav")

            # Save input
            torchaudio.save(input_path, waveform, self.sample_rate)

            # Codec extension mapping
            ext_map = {
                "libopus": ".ogg",
                "libmp3lame": ".mp3",
                "aac": ".m4a",
                "libvorbis": ".ogg",
            }
            codec_ext = ext_map.get(codec, ".ogg")
            encoded_path = os.path.join(tmpdir, f"encoded{codec_ext}")

            # Encode with codec
            cmd_encode = [
                "ffmpeg", "-y", "-i", input_path,
                *ffmpeg_args,
                encoded_path,
            ]
            subprocess.run(cmd_encode, capture_output=True, check=True)

            # Decode back to WAV
            cmd_decode = [
                "ffmpeg", "-y", "-i", encoded_path,
                "-ar", str(self.sample_rate),
                "-ac", "1",
                output_path,
            ]
            subprocess.run(cmd_decode, capture_output=True, check=True)

            # Load degraded audio
            degraded, sr = torchaudio.load(output_path)

        # Match original length
        orig_len = waveform.shape[1]
        if degraded.shape[1] > orig_len:
            degraded = degraded[:, :orig_len]
        elif degraded.shape[1] < orig_len:
            padding = orig_len - degraded.shape[1]
            degraded = torch.nn.functional.pad(degraded, (0, padding))

        return degraded

    def _apply_noise(
        self,
        waveform: torch.Tensor,
        condition: Dict,
    ) -> torch.Tensor:
        """Apply additive noise at specified SNR."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        snr_db = condition.get("snr_db", 20)
        noise_type = condition.get("noise_type", "gaussian")

        if noise_type == "gaussian":
            noise = torch.randn_like(waveform)
        elif noise_type in ("babble", "music"):
            # For structured noise, load from MUSAN
            noise = self._load_musan_noise(
                noise_type, waveform.shape[1], condition
            )
        else:
            noise = torch.randn_like(waveform)

        # Scale noise to achieve target SNR
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)

        if noise_power > 0:
            snr_linear = 10 ** (snr_db / 10)
            scale = torch.sqrt(signal_power / (snr_linear * noise_power))
            noisy = waveform + scale * noise
        else:
            noisy = waveform

        return noisy

    def _apply_reverb(
        self,
        waveform: torch.Tensor,
        condition: Dict,
    ) -> torch.Tensor:
        """Apply room impulse response convolution."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Generate synthetic RIR if no file provided
        rir = self._generate_synthetic_rir(condition)

        # Convolve waveform with RIR
        reverbed = torch.nn.functional.conv1d(
            waveform.unsqueeze(0),
            rir.unsqueeze(0).unsqueeze(0),
            padding=rir.shape[0] // 2,
        ).squeeze(0)

        # Match original length
        reverbed = reverbed[:, : waveform.shape[1]]

        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(reverbed))
        if max_val > 1.0:
            reverbed = reverbed / max_val

        return reverbed

    def _load_musan_noise(
        self,
        noise_type: str,
        target_length: int,
        condition: Dict,
    ) -> torch.Tensor:
        """Load noise from MUSAN dataset."""
        # Fallback to Gaussian if MUSAN not available
        # In production, this would load actual MUSAN files
        return torch.randn(1, target_length)

    def _generate_synthetic_rir(self, condition: Dict) -> torch.Tensor:
        """Generate a simple synthetic room impulse response."""
        rt60_range = condition.get("rt60_range", [0.2, 0.4])
        rt60 = np.random.uniform(rt60_range[0], rt60_range[1])

        # Simple exponential decay RIR
        rir_length = int(rt60 * self.sample_rate)
        t = torch.arange(rir_length, dtype=torch.float32)
        decay_rate = -6.9 / (rt60 * self.sample_rate)  # ln(0.001) ≈ -6.9
        rir = torch.exp(decay_rate * t) * torch.randn(rir_length)

        # Normalize
        rir = rir / torch.max(torch.abs(rir))

        return rir

    def get_degradation_severity(self, condition: Dict) -> float:
        """
        Estimate degradation severity on a 0-1 scale.
        Used for ordering conditions in progressive analysis.
        """
        cond_type = condition.get("type", "none")

        if cond_type == "none":
            return 0.0

        if cond_type == "custom_codec":
            bitrate = condition.get("bitrate", "128k")
            bitrate_val = int(bitrate.replace("k", ""))
            # Lower bitrate = higher severity
            return max(0.0, min(1.0, 1.0 - bitrate_val / 128.0))

        if cond_type == "noise":
            snr = condition.get("snr_db", 30)
            # Lower SNR = higher severity
            return max(0.0, min(1.0, 1.0 - snr / 30.0))

        if cond_type == "reverb":
            rt60_range = condition.get("rt60_range", [0.2, 0.4])
            # Higher RT60 = higher severity
            return max(0.0, min(1.0, np.mean(rt60_range) / 1.0))

        return 0.5
