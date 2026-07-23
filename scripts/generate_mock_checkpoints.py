#!/usr/bin/env python3
"""
Generate dummy/mock checkpoints for testing and verification of the pipeline.
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.aasist import AASISTDetector
from src.models.wavlm_ecapa import WavLMECAPADetector


def main():
    print("Generating mock checkpoints for development/testing...")
    checkpoints_dir = ROOT_DIR / "checkpoints"

    # 1. AASIST Mock Checkpoint
    aasist_dir = checkpoints_dir / "aasist"
    aasist_dir.mkdir(parents=True, exist_ok=True)
    aasist_path = aasist_dir / "AASIST.pth"
    
    if not aasist_path.exists():
        print(f"Creating mock AASIST checkpoint at {aasist_path}...")
        model = AASISTDetector(device="cpu")
        torch.save(model.model.state_dict(), str(aasist_path))
        print("-> Created AASIST mock checkpoint.")
    else:
        print("AASIST checkpoint already exists. Skipping.")

    # 2. WavLM+ECAPA Mock Checkpoint
    wavlm_dir = checkpoints_dir / "wavlm_ecapa"
    wavlm_dir.mkdir(parents=True, exist_ok=True)
    wavlm_path = wavlm_dir / "best_model.pth"
    
    if not wavlm_path.exists():
        print(f"Creating mock WavLM+ECAPA checkpoint at {wavlm_path}...")
        # Build WavLM+ECAPA model (use placeholder frontend to avoid 1.2GB download during mock creation)
        model = WavLMECAPADetector(device="cpu")
        
        # Save state dict
        torch.save(model.state_dict(), str(wavlm_path))
        print("-> Created WavLM+ECAPA mock checkpoint.")
    else:
        print("WavLM+ECAPA checkpoint already exists. Skipping.")

    print("\n-> Mock checkpoint generation complete.")


if __name__ == "__main__":
    main()
