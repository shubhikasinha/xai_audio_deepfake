# Explanation Robustness as an Early-Warning Signal: Auditing Post-Hoc XAI in Audio Deepfake Detectors Under Real-World Codecs and Noise

> **Target**: AIST 2026 (Springer CCIS, Scopus) — Track 3: Explainable, Trustworthy AI for Speech  
> **Deadline**: 15 August 2026 | **Venue**: IGDTUW, New Delhi, 26–27 Nov 2026  
> **Acceptance Rate**: ~30% (71/235 at AIST 2023)  
> **Estimated Acceptance Probability**: 45–60% (post-redesign)  
> **Novelty Score**: 8/10 (post-redesign, up from 6.5/10)

---

## 1. Problem & Motivation

Current audio deepfake detectors achieve strong in-distribution accuracy, but their **explainability** — the post-hoc attribution maps (SHAP, Integrated Gradients) that tell analysts *why* a sample was flagged — has never been systematically stress-tested under the same distribution shifts (codecs, noise, channel effects) that degrade detection accuracy.

This creates a **trust gap**: a detector may still output a "spoof" label with high confidence under codec degradation, but its explanation may have silently shifted to highlight irrelevant spectral regions — making the system **confident but untrustworthy**.

We hypothesize that explanation faithfulness can degrade *faster* than detection accuracy under certain real-world degradations, and that this decoupling can serve as a **practical early-warning signal** for deployment trustworthiness.

### Why This Matters

- **For AIST 2026 Track 3**: Directly addresses "explainable, trustworthy, and responsible AI for speech"
- **For the field**: Fills a genuine gap — faithfulness-under-shift is discussed adjacently (Ge et al. ICASSP 2022; GATR 2025; shortcut-learning diagnostics 2024) but never systematically framed as decoupling detection robustness from explanation robustness
- **For practitioners**: Provides a concrete, deployable metric (Explanation Consistency Score) for monitoring when to distrust a detector's reasoning in production

---

## 2. Contributions (Sharp & Defensible)

1. **First systematic study** decoupling detection accuracy from explanation faithfulness under controlled real-world degradation on ASVspoof 2021 DF
2. **Explanation Consistency Score (ECS)**: A new composite metric (cross-condition attribution similarity + spectral band alignment) validated against deletion/insertion AUC
3. **Early-warning finding**: Empirical evidence on whether faithfulness degrades before, with, or after accuracy — with Spearman correlation and statistical backing
4. **Forensic early-warning dashboard**: Prototype visualization showing analysts when to distrust a detector's explanations
5. **Human-validated NL explanations**: Small-scale human study evaluating whether LLM-translated explanations remain useful under degradation

---

## 3. Research Questions (Focused — 3 Core)

| ID | Research Question | Novelty | Statistical Test |
|---|---|---|---|
| **RQ1** | Does explanation faithfulness degrade predictably with — or ahead of — detection accuracy under progressive codec and noise degradation? | **High** | Spearman ρ(ΔEER, ΔDeletion AUC) across conditions; paired Wilcoxon per condition |
| **RQ2** | Which XAI method (IG vs. SHAP) and detector architecture (AASIST vs. WavLM+ECAPA) produces more robust explanations under shift? | **Moderate–High** | Two-way ANOVA on faithfulness metrics × {method, model, condition} |
| **RQ3** *(Secondary)* | Can the Explanation Consistency Score reliably flag conditions where explanations have become untrustworthy, serving as a deployment early-warning? | **High** | ROC-AUC of ECS predicting faithfulness-below-threshold |

---

## 4. Models — 2 Architecturally Diverse (Reviewer-Optimized)

| Model | Architecture | Why Selected | Checkpoint Source |
|---|---|---|---|
| **AASIST** | Graph Attention Network (spectral domain) | SOTA on ASVspoof 2019/2021; interpretable graph attention weights; compact; official checkpoints | [clovaai/aasist](https://github.com/clovaai/aasist) |
| **WavLM + ECAPA-TDNN** | SSL Transformer + TDNN back-end | Represents modern SSL-based approach; captures both content and speaker artifacts; HuggingFace/SpeechBrain | [microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large) + SpeechBrain ECAPA |

**Why drop RawNet2?** The review correctly identifies that 3 models spread compute thin without proportional insight. AASIST (graph/spectral) vs. WavLM+ECAPA (SSL/transformer) gives maximum architectural diversity. RawNet2 (raw waveform CNN) is an intermediate that adds width, not depth, to the analysis.

**Why not HuBERT?** Too similar to WavLM — marginal additional insight vs. compute cost. WavLM outperforms on denoising tasks, making it more relevant to degradation robustness.

---

## 5. Datasets

### Primary (Core Results — Non-Negotiable)

| Dataset | Role | Access | Size |
|---|---|---|---|
| **ASVspoof 2019 LA** | Training / fine-tuning | Zenodo (open) | ~16GB |
| **ASVspoof 2021 DF** | Primary evaluation | Zenodo (open) | ~22GB; 7 built-in codec conditions (C1–C7) |

### Secondary (Cross-Dataset — Strengthens Paper)

| Dataset | Role | Access |
|---|---|---|
| **WaveFake** | Cross-dataset generalization test | Zenodo/Kaggle (CC-BY-SA) |
| **LibriSpeech** (clean-100) | Real speech for synthetic degradation experiments | OpenSLR (open) |

### Enrichment (Non-Blocking)

| Dataset | Role | Status |
|---|---|---|
| **MUSAN** | Noise sources (babble, music) for augmentation | Open |
| **RIR (MIT/ACE)** | Room impulse responses | Open |
| **ASVspoof 5** | Extended codec conditions (11 codecs incl. Encodec) | Attempt access in parallel; not a dependency |

---

## 6. Technical Architecture

### 6.1 Pipeline Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                    EXPLANATION ROBUSTNESS PIPELINE                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────┐    ┌─────────────────┐    ┌────────────────────┐         │
│  │ ASVspoof  │───▶│  Degradation    │───▶│ Pretrained Detector│         │
│  │ 2021 DF   │    │  Pipeline       │    │ (AASIST / WavLM)   │         │
│  └──────────┘    │  C0: Clean      │    └────────┬───────────┘         │
│                  │  C1–C7: Built-in│             │                     │
│                  │  C8: Opus 16k   │    ┌────────▼───────────┐         │
│                  │  N1–N3: Noise   │    │  XAI Engine         │         │
│                  │  R1: RIR        │    │  • IG (primary)     │         │
│                  └─────────────────┘    │  • SHAP (secondary) │         │
│                                        └────────┬───────────┘         │
│                                                  │                     │
│                          ┌───────────────────────▼──────────────┐     │
│                          │  Faithfulness Evaluator               │     │
│                          │  • Deletion/Insertion AUC             │     │
│                          │  • Sensitivity-N                      │     │
│                          │  • Explanation Consistency Score (NEW) │     │
│                          └───────────────────────┬──────────────┘     │
│                                                  │                     │
│                    ┌─────────────────────────────▼──────────────┐     │
│                    │  Statistical Engine                         │     │
│                    │  • Spearman ρ(ΔEER, ΔFaithfulness)         │     │
│                    │  • Paired Wilcoxon + Bonferroni             │     │
│                    │  • Bootstrap 95% CI + Cohen's d             │     │
│                    └─────────────────────────────┬──────────────┘     │
│                                                  │                     │
│              ┌───────────────────────────────────▼──────────────┐     │
│              │  LLM Explanation Module (Secondary)               │     │
│              │  • Attribution → NL Rationale (Gemini API)        │     │
│              │  • Semantic Consistency (Sentence-BERT)            │     │
│              │  • Human Study (20 samples × N raters)            │     │
│              └───────────────────────────────────┬──────────────┘     │
│                                                  │                     │
│              ┌───────────────────────────────────▼──────────────┐     │
│              │  Forensic Early-Warning Dashboard                 │     │
│              │  • ECS heatmaps per condition                     │     │
│              │  • "Trust zone" visualization                     │     │
│              │  • Accuracy vs. faithfulness scatter              │     │
│              └──────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Degradation Conditions (Focused)

| ID | Condition | Parameters | Source | Priority |
|---|---|---|---|---|
| **C0** | Clean (no degradation) | — | Baseline | Core |
| **C1–C7** | ASVspoof 2021 DF native codecs | MP3, M4A, OGG at varying bitrates | Built-in | Core |
| **C8** | Opus @ 16kbps | Low-bitrate modern codec | ffmpeg | Core |
| **C9** | Opus @ 6kbps | Extreme compression | ffmpeg | Core |
| **N1** | AWGN @ SNR 20dB | Mild noise | Custom | Core |
| **N2** | AWGN @ SNR 10dB | Moderate noise | Custom | Core |
| **N3** | Babble noise @ SNR 15dB | Realistic ambient | MUSAN | Extension |
| **R1** | Room impulse response (small) | Mild reverb | RIR dataset | Extension |

**Total core conditions**: 12 (C0–C9, N1–N2)  
**Experimental matrix**: 2 detectors × 2 XAI methods × 12 conditions = **48 core experiments**  
(Down from 72 — focused for depth over breadth)

### 6.3 XAI Methods

| Method | Type | Role | Compute |
|---|---|---|---|
| **Integrated Gradients (IG)** | Gradient-based | **Primary** — run on all conditions, both models | Low–moderate |
| **Kernel SHAP** | Perturbation-based | **Secondary** — run on subset (C0, C3, C7, C8, N2) for AASIST; validates IG findings | High (budget: 100–500 perturbations) |

**Why IG primary?** Axiomatically grounded (completeness, sensitivity), efficient on spectrograms, works natively with both architectures. SHAP validates that findings aren't method-specific but is too expensive to run exhaustively on WavLM.

### 6.4 Faithfulness Metrics

| Metric | Formula Intuition | Direction | Status |
|---|---|---|---|
| **Deletion AUC** | Mask top-k% important features → measure confidence drop | ↓ = more faithful | Standard |
| **Insertion AUC** | Add top-k% features to blank → measure confidence gain | ↑ = more faithful | Standard |
| **Sensitivity-N** | Correlation: subset removal ↔ predicted change | ↑ = more faithful | Standard |
| **Explanation Stability (ES)** | Cosine similarity of attribution vectors (clean vs. degraded) | ↑ = more stable | **New** |
| **Spectral Band Alignment (SBA)** | Attribution mass in known-artifact bands ↔ detection confidence correlation | ↑ = better aligned | **New** |
| **Explanation Consistency Score (ECS)** | Composite: α·ES + β·SBA + γ·ΔDeletionAUC (weighted) | ↑ = more trustworthy | **Novel contribution** |

**ECS Definition (Formal):**

```
ECS(x, d) = α · CosSim(A_clean(x), A_degraded(x, d))      # Stability
           + β · Corr(A_artifact_bands(x, d), P(spoof|x,d))  # Alignment
           + γ · (1 - |DelAUC_clean - DelAUC_degraded|)       # Faithfulness preservation

where α + β + γ = 1, tuned via grid search on validation split
```

### 6.5 LLM Explanation Module (Secondary — Not Core Contribution)

> **Critical constraint**: The LLM does **NOT** perform detection. It only converts detector outputs + XAI attribution maps into analyst-friendly natural language.

**Architecture:**
```
Detector Score + IG Attribution Map → Feature Extraction → Structured Prompt → Gemini API → NL Rationale
                                      ├── Top-5 freq bands
                                      ├── Top-3 temporal regions  
                                      ├── Attribution concentration
                                      └── Spectral energy stats
```

**Evaluation:**
- Semantic consistency: Sentence-BERT cosine similarity between clean and degraded explanations
- Claim preservation rate: % of forensic claims (e.g., "artifacts in 4–8kHz") that persist under degradation
- **Human study** (small scale): 20 sample pairs (clean vs. degraded), rated by 5–10 raters on a 5-point Likert scale for:
  - Informativeness
  - Technical accuracy (does the explanation match the heatmap?)
  - Actionability (would you trust this in a forensic report?)

---

## 7. Experimental Protocol

### 7.1 Evaluation Metrics Summary

| Category | Metric | Standard | Used For |
|---|---|---|---|
| Detection | Equal Error Rate (EER) | ASVspoof standard | Accuracy under degradation |
| Detection | min t-DCF | ASVspoof 2019/2021 | Cost-weighted accuracy |
| Faithfulness | Deletion AUC | Lower = more faithful | Core faithfulness |
| Faithfulness | Insertion AUC | Higher = more faithful | Core faithfulness |
| Faithfulness | Sensitivity-N (Pearson r) | Higher = more faithful | Proportionality check |
| Stability | Explanation Stability (ES) | Cosine similarity | Explanation shift |
| Alignment | Spectral Band Alignment (SBA) | Pearson r | Forensic relevance |
| Composite | Explanation Consistency Score (ECS) | Weighted composite | **Novel — deployment metric** |
| LLM | Semantic Consistency | Sentence-BERT cosine sim | NL explanation robustness |
| LLM | Claim Preservation Rate | % preserved | Forensic claim tracking |
| Human | Likert ratings (3 dimensions) | 5-point scale | Human validation |

### 7.2 Statistical Testing Plan

| Test | Purpose | Applied To |
|---|---|---|
| **Spearman rank correlation** | Core RQ1: correlate ΔEER with ΔFaithfulness across conditions | Primary analysis |
| **Paired Wilcoxon signed-rank** | Compare faithfulness clean vs. each degradation condition | Per-condition |
| **Two-way ANOVA** | RQ2: effects of {XAI method} × {model} on faithfulness | Method/model comparison |
| **Bonferroni correction** | Multiple comparison correction | All pairwise tests |
| **Bootstrap 95% CI** | Confidence intervals on all reported metrics | All results |
| **Cohen's d** | Effect size for practical significance | All significant results |
| **ROC-AUC** | RQ3: Can ECS predict "untrustworthy" explanations? | ECS validation |

### 7.3 Ablation Studies (Focused)

| Ablation | Purpose | Priority |
|---|---|---|
| **Spectrogram resolution** | 64 vs. 128 mel bins — effect on faithfulness? | High |
| **SHAP perturbation count** | 100 vs. 500 — compute–faithfulness tradeoff | High |
| **ECS weight sensitivity** | Vary α, β, γ — how robust is ECS? | High |
| **Attribution granularity** | Frame-level vs. utterance-level | Medium |
| **LLM prompt structure** | Structured vs. free-form — consistency impact | Low |

---

## 8. Cross-Dataset Generalization (Secondary)

> Explicitly **secondary** — the paper's spine is faithfulness-under-degradation on ASVspoof 2021 DF. Cross-dataset results strengthen but are not load-bearing.

**Protocol:**
1. Use detectors trained on ASVspoof 2019 LA (standard)
2. Evaluate on **WaveFake** without fine-tuning
3. Apply degradation pipeline + faithfulness analysis
4. Report whether the faithfulness-accuracy relationship generalizes OOD
5. If ECS early-warning signal holds cross-dataset, that's a bonus finding

---

## 9. Prior Art Positioning

| Paper | Year | Venue | Overlap | How We Differ |
|---|---|---|---|---|
| Ge, Patino, Todisco & Evans | 2022 | ICASSP | SHAP on spoofing CMs | We test faithfulness *under degradation*, not just clean; add stability metrics |
| Müller et al. "Does Deepfake Detection Generalize?" | 2022 | Interspeech | Cross-dataset benchmark | We add explanation robustness as a new axis |
| Müller et al. "Harder or Different?" | 2024 | — | Refined generalization | We decouple detection from explanation robustness |
| Grinberg et al. (time-domain XAI) | 2025 | ICASSP | XAI for audio forensics | We focus on faithfulness metrics under shift, not new XAI methods |
| Shortcut learning in spoofing CMs | 2024–25 | Various | Questions explanation trustworthiness | We test this empirically with controlled degradation + ECS metric |
| FT-GRPO / Acoustic CoT | 2026 | arXiv | LLM + audio forensics | We use LLM as explanation layer only; test semantic consistency |
| GATR (transformer relevancy) | 2025 | — | Advanced XAI comparison | We focus on faithfulness stability, not localization accuracy |
| Diffusion-based artifact localization | 2025 | — | Outperforms post-hoc XAI | We benchmark post-hoc methods specifically for robustness |

**Key differentiation sentence** (for Introduction):
> *While recent work has separately advanced deepfake detection robustness (Müller et al., 2022, 2024) and post-hoc explainability (Ge et al., 2022; Grinberg et al., 2025), no study has systematically investigated whether explanation faithfulness is coupled to or independent of detection accuracy under real-world degradation — a critical gap for trustworthy deployment of forensic AI systems.*

---

## 10. Project Structure

```
deepfake/
├── deepfake.md                        # This document — research blueprint
├── README.md                          # Project overview & reproduction guide
├── requirements.txt                   # Python dependencies (pinned versions)
├── setup.py                           # Package setup
├── pyproject.toml                     # Modern Python project config
├── .gitignore                         # Git ignore patterns
│
├── configs/
│   ├── experiment.yaml                # Master experiment configuration
│   ├── models/
│   │   ├── aasist.yaml               # AASIST model config
│   │   └── wavlm_ecapa.yaml          # WavLM+ECAPA config
│   └── degradation/
│       └── conditions.yaml            # All degradation conditions
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                # ASVspoof dataset loaders
│   │   ├── degradation.py            # Codec/noise degradation pipeline
│   │   └── utils.py                  # Audio I/O utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_detector.py          # Abstract detector interface
│   │   ├── aasist.py                 # AASIST wrapper
│   │   └── wavlm_ecapa.py           # WavLM+ECAPA wrapper
│   ├── xai/
│   │   ├── __init__.py
│   │   ├── base_explainer.py         # Abstract explainer interface
│   │   ├── integrated_gradients.py   # IG implementation
│   │   ├── kernel_shap.py            # Kernel SHAP implementation
│   │   └── attribution_utils.py      # Attribution processing utilities
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── detection_metrics.py      # EER, min t-DCF
│   │   ├── faithfulness_metrics.py   # Deletion/Insertion AUC, Sensitivity-N
│   │   ├── consistency_score.py      # Explanation Consistency Score (ECS)
│   │   └── statistical_tests.py      # Hypothesis testing
│   ├── llm_explanation/
│   │   ├── __init__.py
│   │   ├── prompt_templates.py       # Structured prompts
│   │   ├── explanation_generator.py  # LLM API integration
│   │   └── semantic_evaluator.py     # Sentence-BERT consistency
│   └── visualization/
│       ├── __init__.py
│       ├── attribution_plots.py      # Spectrogram heatmaps
│       ├── faithfulness_curves.py    # Deletion/insertion curves
│       ├── dashboard.py              # Forensic early-warning dashboard
│       └── paper_figures.py          # Publication-ready figures
│
├── scripts/
│   ├── download_data.sh              # Dataset download automation
│   ├── download_checkpoints.sh       # Pretrained model download
│   ├── run_detection.py              # Batch detection evaluation
│   ├── run_xai.py                    # Batch XAI computation
│   ├── run_faithfulness.py           # Faithfulness metric computation
│   ├── run_llm_explanations.py       # LLM explanation generation
│   ├── run_full_pipeline.py          # End-to-end experiment runner
│   └── generate_paper_figures.py     # All figures for the paper
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_detection_baselines.ipynb
│   ├── 03_xai_analysis.ipynb
│   ├── 04_faithfulness_analysis.ipynb
│   └── 05_paper_results.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_degradation.py
│   ├── test_metrics.py
│   ├── test_xai.py
│   └── test_ecs.py
│
├── paper/
│   ├── main.tex                       # Springer CCIS LaTeX source
│   ├── references.bib                 # Bibliography
│   └── figures/                       # Generated figures
│
└── results/
    ├── detection/                     # EER/t-DCF CSVs per condition
    ├── attributions/                  # Saved attribution maps (.npy)
    ├── faithfulness/                  # Faithfulness metric tables
    ├── explanations/                  # LLM-generated explanations
    └── figures/                       # Publication-ready plots
```

---

## 11. Workflow (10–12 Weeks)

### Phase 1: Infrastructure & Baselines (Weeks 1–2)
- [x] Initialize project repository with full structure
- [ ] Set up virtual environment with pinned dependencies
- [ ] Download ASVspoof 2019 LA + 2021 DF from Zenodo
- [ ] Download pretrained checkpoints (AASIST, WavLM-Large)
- [ ] Implement degradation pipeline (ffmpeg codecs, noise injection)
- [ ] Write dataset loaders with proper train/eval split handling
- [ ] Evaluate both detectors on ASVspoof 2021 DF clean (verify EER matches published)
- [ ] Evaluate under all built-in codec conditions (C1–C7)
- [ ] Apply custom conditions (C8, C9, N1, N2) and evaluate
- **Checkpoint**: Detection accuracy tables complete; match published baselines ±1% EER

### Phase 2: XAI Engine & Faithfulness (Weeks 3–5)
- [ ] Implement IG for AASIST (spectrogram-level attributions)
- [ ] Implement IG for WavLM+ECAPA (layer-wise feature attributions)
- [ ] Sanity check: visual inspection of clean attributions
- [ ] Run IG on all 12 conditions × 2 models
- [ ] Implement Kernel SHAP for AASIST (subset conditions only)
- [ ] Implement faithfulness metrics (Deletion AUC, Insertion AUC, Sensitivity-N)
- [ ] Implement Explanation Stability (cosine similarity clean vs. degraded)
- [ ] Implement Spectral Band Alignment
- [ ] Compute ECS composite metric
- [ ] Run faithfulness evaluation across all conditions
- **Checkpoint**: Core faithfulness-vs-accuracy data matrix complete

### Phase 3: Analysis & Early-Warning (Weeks 6–7)
- [ ] Statistical analysis: Spearman ρ(ΔEER, ΔFaithfulness) — core RQ1
- [ ] Paired Wilcoxon tests per condition with Bonferroni correction
- [ ] Two-way ANOVA for RQ2 (method × model effects)
- [ ] ROC-AUC analysis for ECS as early-warning signal (RQ3)
- [ ] Ablation studies (spectrogram resolution, SHAP count, ECS weights)
- [ ] Cross-dataset validation on WaveFake (if time)
- **Checkpoint**: All RQs answered with statistical backing; story is clear

### Phase 4: LLM Integration & Human Study (Week 8)
- [ ] Implement LLM explanation module (Gemini API)
- [ ] Generate NL explanations for clean vs. key degradation conditions
- [ ] Compute semantic consistency (Sentence-BERT)
- [ ] Design and run human evaluation study (20 samples, 5–10 raters)
- [ ] Analyze human study results
- **Checkpoint**: LLM + human results complement core findings

### Phase 5: Paper & Dashboard (Weeks 9–10)
- [ ] Generate all publication-ready figures (matplotlib/seaborn)
- [ ] Build forensic early-warning dashboard visualizations
- [ ] Write paper in Springer CCIS LaTeX template (10–12 pages)
- [ ] Internal review, revision, proofreading
- [ ] Prepare supplementary materials and reproducibility package
- [ ] Submit via Microsoft CMT
- **Checkpoint**: Paper submitted to AIST 2026

---

## 12. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Null result** (faithfulness tracks accuracy) | Medium | Medium | A clean null is publishable: "explanations are as robust as predictions under standard codecs." Frame as robustness validation. Include "when explanations fail first" case studies if any conditions show decoupling. |
| **Compute insufficient** for SHAP on WavLM | Medium | High | SHAP is secondary, run only on AASIST subset. IG is primary and efficient. |
| **Overclaiming** from small effect | Medium | High | Predefine faithfulness metric and significance test *before* running experiments. Report honestly even if effect is small. |
| **ASVspoof 5 not obtainable** | Medium | Low | Not a dependency. Core results use ASVspoof 2021 DF. |
| **Weak human study** | Medium | Medium | Keep scope small but rigorous: 20 samples, clear Likert scale, inter-rater agreement (Krippendorff's α). |
| **Timeline slippage** | Medium | High | Hard checkpoint at Week 5: if core faithfulness-vs-accuracy data doesn't show *any* signal, pivot to robustness validation framing. |
| **Degradation pipeline artifacts** | Low | High | Verify pipeline on LibriSpeech clean; measure SNR/PESQ before and after; document in paper. |

---

## 13. Ethical Considerations

1. **Dual-use risk**: Region-of-interest explanations could theoretically inform evasion. Mitigated by (a) using only existing pretrained models, (b) not releasing attack-specific artifact localization, (c) focusing on detector *auditing* rather than vulnerability mapping.

2. **Responsible deployment**: Our finding that explanations may be untrustworthy under degradation is itself a safety contribution — it cautions against deploying XAI-based forensic tools without degradation-aware validation.

3. **Data ethics**: All datasets are public benchmarks with established research licenses.

4. **Human study ethics**: If institutional IRB review is required for the Likert rating study, prepare application. Study involves no PII and presents only synthetic/spoofed audio.

---

## 14. Acceptance Assessment (Post-Redesign)

| Dimension | Score | Justification |
|---|---|---|
| **Novelty** | 8/10 | Sharpened claim + ECS metric + early-warning angle |
| **Technical Quality** | 8.5/10 | Rigorous, focused, proper stats |
| **Experimental Rigor** | 9/10 | Predefined metrics, ablations, public data |
| **Practical Impact** | 8/10 | Deployment trust, forensic dashboard |
| **Feasibility** | 9/10 | 2 models, IG primary, Colab-compatible |
| **Reproducibility** | 9.5/10 | Structured repo, pinned deps, seed fixing |
| **AIST Track Fit** | 9/10 | Directly addresses Track 3 call |
| **Estimated Acceptance** | **45–60%** | Exceeds ~30% historical rate for well-executed applied XAI |

### Top 5 Rejection Risks & Mitigations

1. **"Overlap with Ge et al. (2022)"** → Explicitly position as extension to degradation; cite as foundation; emphasize faithfulness metrics (not just visual SHAP) + ECS novelty
2. **"Results are null/expected"** → Predefine hypothesis; frame null as robustness finding; include case studies where decoupling occurs
3. **"Weak positioning against 2025 work"** → Include Grinberg et al. (ICASSP 2025), GATR, diffusion XAI in Related Work; state differentiation in one sentence
4. **"No human evaluation"** → Small but rigorous Likert study validates NL explanations
5. **"LLM module is tacked on"** → Explicitly label as secondary; evaluate with concrete metrics; don't overclaim

---

## 15. Paper Outline (Springer CCIS, 10–12 Pages)

1. **Introduction** (1.5 pages): Trust gap motivation; one-sentence contribution; RQ preview
2. **Related Work** (1.5 pages): Audio deepfake detection → XAI for speech → Faithfulness metrics → Robustness testing. Clear positioning against Ge et al., Müller et al., Grinberg et al.
3. **Methodology** (3 pages): Detectors (AASIST, WavLM+ECAPA); degradation pipeline; XAI methods (IG, SHAP); faithfulness metrics; ECS formulation; LLM module; human study design
4. **Experimental Setup** (1 page): Datasets; conditions; hyperparameters; statistical plan
5. **Results** (2.5 pages): Detection baselines under degradation; faithfulness analysis (RQ1, RQ2); ECS early-warning (RQ3); LLM semantic consistency; human study
6. **Discussion** (1 page): Implications for deployment; limitations; when to trust explanations
7. **Conclusion** (0.5 page): Key findings; practical recommendations; future work (intrinsic interpretability, neural codecs, accent fairness)
8. **References** (≤50 citations)
