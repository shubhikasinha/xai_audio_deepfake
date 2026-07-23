#!/usr/bin/env python3
"""
Download pretrained checkpoints for the deepfake detectors.
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def download_file(url: str, dest_path: Path):
    """Download a file with progress bar."""
    print(f"Downloading {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    def report_hook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 100 / total_size
            sys.stdout.write(f"\r  -> Progress: {percent:.1f}% ({read_so_far}/{total_size} bytes)")
        else:
            sys.stdout.write(f"\r  -> Progress: {read_so_far} bytes")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(dest_path), reporthook=report_hook)
        print("\n-> Download completed successfully!")
    except Exception as e:
        print(f"\n-> Error downloading file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download model checkpoints.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["aasist", "wavlm_ecapa", "all"],
        default="all",
        help="Which model checkpoint to download.",
    )
    args = parser.parse_args()

    checkpoints_dir = ROOT_DIR / "checkpoints"

    if args.model in ["aasist", "all"]:
        # Official AASIST checkpoint from ClovaAI
        aasist_url = "https://github.com/clovaai/aasist/raw/main/models/weights/AASIST.pth"
        aasist_dest = checkpoints_dir / "aasist" / "AASIST.pth"
        download_file(aasist_url, aasist_dest)

    if args.model in ["wavlm_ecapa", "all"]:
        # WavLM is loaded via HuggingFace's transformers library automatically.
        # But for the ECAPA-TDNN back-end trained on ASVspoof, since it's custom,
        # we check if it already exists or print instructions.
        wavlm_ecapa_dest = checkpoints_dir / "wavlm_ecapa" / "best_model.pth"
        if not wavlm_ecapa_dest.exists():
            print("\n[WavLM+ECAPA Checkpoint]")
            print(f"Note: The fine-tuned backend checkpoint should be placed at: {wavlm_ecapa_dest}")
            print("WavLM-Large front-end weights will be downloaded automatically from Hugging Face on first use.")
            print("To generate a baseline model checkpoint for testing, run:")
            print("  python scripts/generate_mock_checkpoints.py")


if __name__ == "__main__":
    main()
