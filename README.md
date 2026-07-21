# Explanation Robustness as an Early-Warning Signal

**Auditing Post-Hoc XAI in Audio Deepfake Detectors Under Real-World Codecs and Noise**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: AIST 2026](https://img.shields.io/badge/Paper-AIST%202026-green.svg)](https://www.aistconference.com)

---

## Overview

This repository contains the code, configurations, and experimental framework for our research paper submitted to the **8th International Conference on Artificial Intelligence and Speech Technology (AIST 2026)**, published in Springer CCIS (Scopus indexed).

### The Trust Gap Problem

Audio deepfake detectors achieve impressive accuracy on clean benchmarks, but what happens to their **explanations** when audio degrades through real-world codecs and noise? A detector might still correctly flag a spoofed sample — but its *explanation* of why could be pointing at entirely wrong spectral regions.

We systematically audit whether **explanation faithfulness degrades before, with, or after detection accuracy** under controlled degradation, and introduce the **Explanation Consistency Score (ECS)** as a practical early-warning signal for when to distrust a detector's reasoning.

### Key Contributions

1. **First systematic study** decoupling detection accuracy from explanation faithfulness under controlled real-world degradation on ASVspoof 2021 DF
2. **Explanation Consistency Score (ECS)**: A novel composite metric combining attribution stability, spectral band alignment, and faithfulness preservation
3. **Early-warning finding**: Empirical evidence on whether faithfulness degrades ahead of accuracy — with statistical backing (Spearman correlation, paired Wilcoxon, bootstrap CI)
4. **Forensic early-warning dashboard**: Prototype visualizations showing analysts when to distrust detector explanations
5. **Human-validated NL explanations**: LLM-generated natural language rationales evaluated for semantic consistency under degradation

---

## Research Questions

| RQ | Question | Key Finding |
|----|----------|-------------|
| **RQ1** | Does explanation faithfulness degrade predictably with — or ahead of — detection accuracy? | *[To be filled after experiments]* |
| **RQ2** | Which XAI method (IG vs. SHAP) and architecture (AASIST vs. WavLM+ECAPA) is more robust? | *[To be filled after experiments]* |
| **RQ3** | Can ECS reliably flag untrustworthy explanations for deployment? | *[To be filled after experiments]* |

---

## Architecture

```
Raw Audio (ASVspoof 2021 DF)
    │
    ▼
Degradation Pipeline (12 conditions: codecs, noise, reverb)
    │
    ▼
Pretrained Detector (AASIST / WavLM+ECAPA)
    │
    ├──▶ Detection Metrics (EER, min t-DCF)
    │
    ▼
XAI Engine (Integrated Gradients primary, Kernel SHAP secondary)
    │
    ▼
Faithfulness Evaluator
    ├── Deletion/Insertion AUC
    ├── Sensitivity-N
    └── Explanation Consistency Score (ECS) ← Novel
    │
    ▼
Statistical Analysis (Spearman ρ, Wilcoxon, Bootstrap CI, Cohen's d)
    │
    ├──▶ LLM Explanation Module (Gemini API → NL rationale)
    │     └── Semantic Consistency (Sentence-BERT)
    │
    └──▶ Forensic Early-Warning Dashboard
```

---

## Models

| Model | Architecture | Role | Source |
|-------|-------------|------|--------|
| **AASIST** | Graph Attention Network (spectral) | Primary detector | [clovaai/aasist](https://github.com/clovaai/aasist) |
| **WavLM + ECAPA-TDNN** | SSL Transformer + TDNN back-end | Secondary detector | [microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large) |

---

## Datasets

| Dataset | Role | Access |
|---------|------|--------|
| **ASVspoof 2019 LA** | Training | [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336) |
| **ASVspoof 2021 DF** | Primary evaluation (7 built-in codec conditions) | [Zenodo](https://zenodo.org/record/4835108) |
| **WaveFake** | Cross-dataset generalization | [Zenodo](https://zenodo.org/record/5642694) |
| **MUSAN** | Noise augmentation | [OpenSLR](https://www.openslr.org/17/) |

---

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- ffmpeg (for codec degradation)

### Installation

```bash
# Clone the repository
git clone https://github.com/<username>/deepfake-xai-robustness.git
cd deepfake-xai-robustness

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Download Data & Checkpoints

```bash
# Download datasets (requires ~40GB disk space)
python scripts/download_data.py --dataset asvspoof2021df
python scripts/download_data.py --dataset asvspoof2019la

# Download pretrained checkpoints
python scripts/download_checkpoints.py --model aasist
python scripts/download_checkpoints.py --model wavlm_ecapa
```

### Run Experiments

```bash
# Quick test (10 samples, verifies pipeline works)
python scripts/run_full_pipeline.py --quick-test --n-samples 10

# Full detection evaluation
python scripts/run_detection.py --config configs/experiment.yaml

# XAI attribution computation
python scripts/run_xai.py --config configs/experiment.yaml --method ig

# Faithfulness evaluation
python scripts/run_faithfulness.py --config configs/experiment.yaml

# Generate paper figures
python scripts/generate_paper_figures.py --output results/figures/
```

---

## Project Structure

```
deepfake/
├── deepfake.md                   # Research blueprint & design document
├── README.md                     # This file
├── requirements.txt              # Pinned Python dependencies
├── setup.py                      # Package setup
├── pyproject.toml                # Modern project config
├── .gitignore
├── configs/                      # YAML experiment configurations
│   ├── experiment.yaml           # Master config
│   ├── models/                   # Per-model configs
│   └── degradation/              # Degradation condition definitions
├── src/                          # Source code
│   ├── data/                     # Dataset loaders & degradation pipeline
│   ├── models/                   # Detector wrappers (AASIST, WavLM+ECAPA)
│   ├── xai/                      # XAI methods (IG, SHAP)
│   ├── evaluation/               # Metrics (detection, faithfulness, ECS)
│   ├── llm_explanation/          # LLM NL explanation module
│   └── visualization/            # Plots, dashboards, paper figures
├── scripts/                      # CLI scripts for running experiments
├── notebooks/                    # Jupyter analysis notebooks
├── tests/                        # Unit & integration tests
├── paper/                        # LaTeX source for submission
└── results/                      # Experiment outputs
```

---

## Degradation Conditions

| ID | Condition | Type |
|----|-----------|------|
| C0 | Clean (no degradation) | Baseline |
| C1–C7 | ASVspoof 2021 DF native (MP3, M4A, OGG) | Built-in codec |
| C8 | Opus @ 16kbps | Custom codec |
| C9 | Opus @ 6kbps | Custom codec |
| N1 | AWGN @ SNR 20dB | Noise |
| N2 | AWGN @ SNR 10dB | Noise |

---

## Explanation Consistency Score (ECS)

Our novel composite metric for quantifying explanation trustworthiness under degradation:

```
ECS(x, d) = α · CosSim(A_clean(x), A_degraded(x, d))       # Stability
           + β · Corr(A_artifact_bands(x,d), P(spoof|x,d))  # Alignment
           + γ · (1 - |DelAUC_clean - DelAUC_degraded|)      # Faithfulness

where α + β + γ = 1
```

Higher ECS → more trustworthy explanation under degradation condition `d`.

---

## Reproducibility

- All random seeds are fixed (42 by default, configurable)
- Experiment configs are versioned in `configs/`
- Results are logged as CSVs with full metadata
- Pretrained checkpoints are downloaded from official sources
- All degradation conditions are deterministic (given same ffmpeg version)

---

## Citation

```bibtex
@inproceedings{author2026explanation,
  title={Explanation Robustness as an Early-Warning Signal: Auditing Post-Hoc XAI in Audio Deepfake Detectors Under Real-World Codecs and Noise},
  author={[Authors]},
  booktitle={8th International Conference on Artificial Intelligence and Speech Technology (AIST 2026)},
  series={Communications in Computer and Information Science},
  publisher={Springer},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ASVspoof Challenge organizers for datasets and baselines
- AASIST authors (Clovaai) for pretrained models
- Microsoft for WavLM pretrained weights
- WaveFake dataset creators (RUB-SysSec)
