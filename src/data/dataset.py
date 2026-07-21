"""
ASVspoof Dataset Loaders.

Supports:
- ASVspoof 2019 LA (training)
- ASVspoof 2021 DF (evaluation with codec conditions)
- WaveFake (cross-dataset generalization)
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.data.utils import load_audio


class ASVspoof2019LA(Dataset):
    """
    ASVspoof 2019 Logical Access dataset.

    Structure:
        ASVspoof2019_LA/
        ├── ASVspoof2019_LA_train/flac/
        ├── ASVspoof2019_LA_dev/flac/
        ├── ASVspoof2019_LA_eval/flac/
        └── ASVspoof2019_LA_cm_protocols/
            ├── ASVspoof2019.LA.cm.train.trn.txt
            ├── ASVspoof2019.LA.cm.dev.trl.txt
            └── ASVspoof2019.LA.cm.eval.trl.txt
    """

    def __init__(
        self,
        root_dir: str,
        partition: str = "train",  # "train", "dev", "eval"
        sample_rate: int = 16000,
        max_duration_sec: float = 4.0,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.partition = partition
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec
        self.transform = transform

        self.samples = self._load_protocol()

    def _load_protocol(self) -> List[Dict]:
        """Load protocol file and build sample list."""
        protocol_dir = self.root_dir / "ASVspoof2019_LA_cm_protocols"

        protocol_files = {
            "train": "ASVspoof2019.LA.cm.train.trn.txt",
            "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
            "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
        }

        audio_dirs = {
            "train": "ASVspoof2019_LA_train/flac",
            "dev": "ASVspoof2019_LA_dev/flac",
            "eval": "ASVspoof2019_LA_eval/flac",
        }

        protocol_file = protocol_dir / protocol_files[self.partition]
        audio_dir = self.root_dir / audio_dirs[self.partition]

        samples = []
        if protocol_file.exists():
            with open(protocol_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        speaker_id = parts[0]
                        utt_id = parts[1]
                        # parts[2] is system_id or "-"
                        attack_type = parts[3]  # e.g., "A01", "-"
                        label = parts[4]  # "bonafide" or "spoof"

                        audio_path = audio_dir / f"{utt_id}.flac"
                        samples.append({
                            "speaker_id": speaker_id,
                            "utt_id": utt_id,
                            "attack_type": attack_type,
                            "label": label,
                            "label_int": 0 if label == "bonafide" else 1,
                            "audio_path": str(audio_path),
                        })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        waveform, sr = load_audio(
            sample["audio_path"],
            sample_rate=self.sample_rate,
            max_duration_sec=self.max_duration_sec,
        )

        if self.transform:
            waveform = self.transform(waveform)

        return {
            "waveform": waveform,
            "label": sample["label_int"],
            "label_str": sample["label"],
            "utt_id": sample["utt_id"],
            "attack_type": sample["attack_type"],
            "speaker_id": sample["speaker_id"],
        }


class ASVspoof2021DF(Dataset):
    """
    ASVspoof 2021 Deepfake dataset.

    Key feature: Built-in codec conditions (C1-C7) for each utterance.

    Structure:
        ASVspoof2021_DF/
        ├── ASVspoof2021_DF_eval/flac/
        └── keys/
            └── DF/
                └── CM/
                    └── trial_metadata.txt
    """

    def __init__(
        self,
        root_dir: str,
        codec_condition: Optional[str] = None,  # None = all, "C1", "C2", etc.
        sample_rate: int = 16000,
        max_duration_sec: float = 4.0,
        max_samples: Optional[int] = None,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.codec_condition = codec_condition
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec
        self.max_samples = max_samples
        self.transform = transform

        self.samples = self._load_metadata()

    def _load_metadata(self) -> List[Dict]:
        """Load trial metadata including codec conditions."""
        meta_file = self.root_dir / "keys" / "DF" / "CM" / "trial_metadata.txt"
        audio_dir = self.root_dir / "ASVspoof2021_DF_eval" / "flac"

        samples = []

        if meta_file.exists():
            with open(meta_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7:
                        speaker_id = parts[0]
                        utt_id = parts[1]
                        codec = parts[4]  # Codec condition identifier
                        label = parts[5]  # "bonafide" or "spoof"

                        # Filter by codec condition if specified
                        if self.codec_condition and codec != self.codec_condition:
                            continue

                        audio_path = audio_dir / f"{utt_id}.flac"
                        samples.append({
                            "speaker_id": speaker_id,
                            "utt_id": utt_id,
                            "codec": codec,
                            "label": label,
                            "label_int": 0 if label == "bonafide" else 1,
                            "audio_path": str(audio_path),
                        })

        # Apply max_samples limit if set
        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        waveform, sr = load_audio(
            sample["audio_path"],
            sample_rate=self.sample_rate,
            max_duration_sec=self.max_duration_sec,
        )

        if self.transform:
            waveform = self.transform(waveform)

        return {
            "waveform": waveform,
            "label": sample["label_int"],
            "label_str": sample["label"],
            "utt_id": sample["utt_id"],
            "codec": sample["codec"],
            "speaker_id": sample["speaker_id"],
        }

    def get_codec_conditions(self) -> List[str]:
        """Return list of unique codec conditions in the dataset."""
        return sorted(set(s["codec"] for s in self.samples))


class WaveFakeDataset(Dataset):
    """
    WaveFake dataset for cross-dataset generalization testing.

    Structure:
        WaveFake/
        ├── ljspeech_full_band_melgan/
        ├── ljspeech_hifiGAN/
        ├── ljspeech_melgan/
        ├── ljspeech_melgan_large/
        ├── ljspeech_multi_band_melgan/
        ├── ljspeech_parallel_wavegan/
        ├── ljspeech_waveglow/
        └── LJSpeech-1.1/wavs/  (real samples)
    """

    FAKE_DIRS = [
        "ljspeech_full_band_melgan",
        "ljspeech_hifiGAN",
        "ljspeech_melgan",
        "ljspeech_melgan_large",
        "ljspeech_multi_band_melgan",
        "ljspeech_parallel_wavegan",
        "ljspeech_waveglow",
    ]

    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000,
        max_duration_sec: float = 4.0,
        max_samples_per_class: Optional[int] = None,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec
        self.transform = transform

        self.samples = self._build_sample_list(max_samples_per_class)

    def _build_sample_list(self, max_per_class: Optional[int]) -> List[Dict]:
        """Build sample list from directory structure."""
        samples = []

        # Real samples from LJSpeech
        real_dir = self.root_dir / "LJSpeech-1.1" / "wavs"
        if real_dir.exists():
            real_files = sorted(real_dir.glob("*.wav"))
            if max_per_class:
                real_files = real_files[:max_per_class]
            for f in real_files:
                samples.append({
                    "audio_path": str(f),
                    "label": "bonafide",
                    "label_int": 0,
                    "attack_type": "real",
                    "utt_id": f.stem,
                })

        # Fake samples from each vocoder
        for fake_dir_name in self.FAKE_DIRS:
            fake_dir = self.root_dir / fake_dir_name
            if fake_dir.exists():
                fake_files = sorted(fake_dir.glob("*.wav"))
                if max_per_class:
                    per_vocoder = max_per_class // len(self.FAKE_DIRS)
                    fake_files = fake_files[:per_vocoder]
                for f in fake_files:
                    samples.append({
                        "audio_path": str(f),
                        "label": "spoof",
                        "label_int": 1,
                        "attack_type": fake_dir_name,
                        "utt_id": f.stem,
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        waveform, sr = load_audio(
            sample["audio_path"],
            sample_rate=self.sample_rate,
            max_duration_sec=self.max_duration_sec,
        )

        if self.transform:
            waveform = self.transform(waveform)

        return {
            "waveform": waveform,
            "label": sample["label_int"],
            "label_str": sample["label"],
            "utt_id": sample["utt_id"],
            "attack_type": sample["attack_type"],
        }
