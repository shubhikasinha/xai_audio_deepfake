#!/usr/bin/env python3
"""
Download and extract dataset metadata (keys/protocols) for the deepfake robustness pipeline.
Supports downloading ASVspoof 2021 DF evaluation keys.
"""

import os
import sys
import tarfile
import argparse
import urllib.request
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def download_and_extract(url: str, dest_dir: Path):
    """Download tar.gz and extract to destination directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    temp_archive = dest_dir / "temp_keys.tar.gz"

    print(f"Downloading keys from {url}...")
    
    def report_hook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 100 / total_size
            sys.stdout.write(f"\r  -> Progress: {percent:.1f}% ({read_so_far}/{total_size} bytes)")
        else:
            sys.stdout.write(f"\r  -> Progress: {read_so_far} bytes")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(temp_archive), reporthook=report_hook)
        print("\n-> Download completed. Extracting archive...")
        
        with tarfile.open(temp_archive, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        
        print("-> Extraction completed.")
        
        # Clean up temporary archive
        if temp_archive.exists():
            os.remove(temp_archive)
            
    except Exception as e:
        print(f"\n-> Error downloading or extracting: {e}")
        if temp_archive.exists():
            os.remove(temp_archive)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download dataset protocols/keys.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["asvspoof2021df", "asvspoof2019la"],
        required=True,
        help="Which dataset keys to download.",
    )
    args = parser.parse_args()

    data_dir = ROOT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "asvspoof2021df":
        # Official ASVspoof 2021 DF keys containing trial_metadata.txt
        keys_url = "https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz"
        # The target dir should align with configs/experiment.yaml: data/ASVspoof2021_DF
        dest_dir = data_dir / "ASVspoof2021_DF"
        download_and_extract(keys_url, dest_dir)
        
        # The tarball extracts a folder named 'keys'. We need to make sure it is at:
        # dest_dir / keys / DF / CM / trial_metadata.txt
        # Let's verify and log the path
        expected_meta = dest_dir / "keys" / "DF" / "CM" / "trial_metadata.txt"
        if expected_meta.exists():
            print(f"-> Success: ASVspoof 2021 DF trial metadata verified at: {expected_meta}")
        else:
            print(f"-> Warning: Could not find trial metadata at expected path: {expected_meta}")

    elif args.dataset == "asvspoof2019la":
        # ASVspoof 2019 LA protocols are typically packaged with the dataset,
        # but we download the separate metadata repository if needed.
        print("Note: ASVspoof 2019 LA protocols are typically loaded from the Edinburgh DataShare download.")
        print("Please ensure your data/ASVspoof2019_LA folder contains the ASVspoof2019_LA_cm_protocols directory.")


if __name__ == "__main__":
    main()
