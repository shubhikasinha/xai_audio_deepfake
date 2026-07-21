"""
Audio I/O and preprocessing utilities.
"""

import os
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from typing import Optional, Tuple


def load_audio(
    filepath: str,
    sample_rate: int = 16000,
    max_duration_sec: Optional[float] = 4.0,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and preprocess it.

    Args:
        filepath: Path to the audio file.
        sample_rate: Target sample rate (Hz).
        max_duration_sec: Maximum duration in seconds. None for no limit.
        mono: If True, convert to mono.

    Returns:
        Tuple of (waveform tensor [1, T], sample_rate).
    """
    waveform, sr = torchaudio.load(filepath)

    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate

    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Truncate or pad to max_duration
    if max_duration_sec is not None:
        max_samples = int(max_duration_sec * sr)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            padding = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

    return waveform, sr


def save_audio(
    waveform: torch.Tensor,
    filepath: str,
    sample_rate: int = 16000,
) -> None:
    """Save a waveform tensor to an audio file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(filepath, waveform, sample_rate)


def compute_spectrogram(
    waveform: torch.Tensor,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Compute log-mel spectrogram from waveform.

    Args:
        waveform: Audio tensor [1, T] or [T].
        n_mels: Number of mel filterbank channels.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        sample_rate: Audio sample rate.

    Returns:
        Log-mel spectrogram tensor [1, n_mels, T'].
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    mel_spec = mel_transform(waveform)
    log_mel_spec = torch.log(mel_spec + 1e-9)

    return log_mel_spec


def compute_stft(
    waveform: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Compute STFT magnitude spectrogram.

    Args:
        waveform: Audio tensor [1, T] or [T].
        n_fft: FFT window size.
        hop_length: Hop length.

    Returns:
        STFT magnitude tensor [1, F, T'].
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    stft = torch.stft(
        waveform.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
    )

    magnitude = torch.abs(stft).unsqueeze(0)
    return magnitude


def get_audio_info(filepath: str) -> dict:
    """Get metadata about an audio file."""
    info = torchaudio.info(filepath)
    return {
        "sample_rate": info.sample_rate,
        "num_channels": info.num_channels,
        "num_frames": info.num_frames,
        "duration_sec": info.num_frames / info.sample_rate,
        "encoding": info.encoding,
        "bits_per_sample": info.bits_per_sample,
    }
