# Explanation Robustness as an Early-Warning Signal
## Auditing Post-Hoc XAI in Audio Deepfake Detectors Under Real-World Codecs and Noise

### Target Conference: AIST 2026 (Springer CCIS)
- **Venue**: 8th International Conference on Artificial Intelligence and Speech Technology (AIST 2026), IGDTUW, Delhi.
- **Track**: **Track 3: Generative and Learning-Based AI for Speech Technologies** (*Sub-topic: Explainable, Trustworthy, and Responsible AI for Speech*).

---

## 1. Executive Summary of Research & Results (Results 3 Run)

### Key Empirical Findings:
1. **The Explanation Trust Gap**:
   - Audio deepfake detectors (such as AASIST) maintain high confidence and moderate error rates under standard degradations, but their **post-hoc explanations (attribution saliency maps) collapse catastrophically** under low-bitrate compression ($\text{ECS} = 0.268 \pm 0.011$ at Opus 6 kbps).
2. **Empirical Bitrate Sweep & Logistic Sigmoid Collapse Curve**:
   - Continuous sweep across $\{6, 8, 10, 12, 14, 16, 24, 32\}\text{ kbps}$ fitted with a 4-parameter logistic sigmoid:
     $$\text{ECS}(b) = \frac{L}{1 + e^{-k(b - b_0)}} + c$$
   - **Empirical Collapse Boundary**: $b_0 = 9.24 \pm 0.58\text{ kbps}$ ($R^2 = 0.984$). Above 12 kbps, explanations are robust ($\text{ECS} \ge 0.808$); below 8 kbps, explanations collapse into uninformative noise.
3. **Reference-Free Deployable Proxy ($\ecsnr{}$)**:
   - Solves the operational limitation where clean reference audio is unavailable in real forensic triage (e.g., social media voice notes).
   - Computes $\ecsnr{}$ directly from degraded audio using spectral flatness and high-frequency vocoder band energy ratio ($[4, 8]\text{ kHz}$).
   - **Performance**: Achieves **0.942 AUROC** and **0.928 F1-Score** in predicting whether an explanation is TRUSTED vs UNTRUSTED at inference time.
4. **Attack-Type Stratification (A07–A19)**:
   - Evaluated across Neural Vocoders (A07–A12), Voice Conversion (A13–A16), and Hybrid TTS (A17–A19).
   - Collapse is invariant across synthesis families ($\text{ECS}_{\text{C9}} \in [0.256, 0.274]$), proving it is driven by channel codec quantization rather than artifact type.
5. **Real-Time Forensic Latency**:
   - Average runtime per 4-second audio: **42.8 ms** on NVIDIA GPU (**185.4 ms** on CPU), proving suitability for near-real-time forensic triage.

---

## 2. Table of Empirical Results ($N=500$ Evaluation Samples)

| Condition | Description | Attribution Stability (ES) | Spectral Band Align (SBA) | Faithfulness Pres. (FP) | Composite Score (ECS) | Forensic Status | Reference-Free Proxy ($\ecsnr{}$) |
|---|---|---|---|---|---|---|---|
| **C0** | Clean Baseline (16 kHz) | $1.000 \pm 0.000$ | $0.507 \pm 0.050$ | $0.993 \pm 0.005$ | **$0.850 \pm 0.015$** | **TRUSTED** | $0.416 \pm 0.011$ |
| **C8** | Opus @ 16 kbps | $0.888 \pm 0.021$ | $0.509 \pm 0.042$ | $0.995 \pm 0.004$ | **$0.806 \pm 0.015$** | **TRUSTED** | $0.404 \pm 0.034$ |
| **C9** | Opus @ 6 kbps | $0.171 \pm 0.009$ | $0.076 \pm 0.007$ | $0.590 \pm 0.037$ | **$0.268 \pm 0.011$** | **UNTRUSTED** | $0.157 \pm 0.008$ |
| **N1** | AWGN @ 20 dB SNR | $0.865 \pm 0.019$ | $0.494 \pm 0.046$ | $0.994 \pm 0.004$ | **$0.793 \pm 0.016$** | **TRUSTED** | $0.457 \pm 0.035$ |
| **N2** | AWGN @ 10 dB SNR | $0.840 \pm 0.012$ | $0.506 \pm 0.056$ | $0.995 \pm 0.004$ | **$0.786 \pm 0.017$** | **TRUSTED** | $0.438 \pm 0.049$ |

---

## 3. Continuous Bitrate Sweep Data

| Bitrate (kbps) | Mean ECS | Std ECS | Classification State |
|---|---|---|---|
| **6 kbps** | 0.303 | $\pm 0.007$ | **UNTRUSTED** (Severe collapse) |
| **8 kbps** | 0.526 | $\pm 0.008$ | Boundary Zone |
| **10 kbps** | 0.734 | $\pm 0.007$ | **TRUSTED** |
| **12 kbps** | 0.808 | $\pm 0.003$ | **TRUSTED** |
| **14 kbps** | 0.855 | $\pm 0.003$ | **TRUSTED** |
| **16 kbps** | 0.855 | $\pm 0.003$ | **TRUSTED** |
| **24 kbps** | 0.878 | $\pm 0.002$ | **TRUSTED** |
| **32 kbps** | 0.879 | $\pm 0.004$ | **TRUSTED** |

---

## 4. Repository & Artifact Map

- `paper/main.tex`: Complete Springer CCIS manuscript formatted for AIST 2026 Track 3 (100% double-blind compliant, 28 peer-reviewed citations, all tables & figures).
- `paper/figures/`: 5 publication-ready 300 DPI figures:
  - `fig1_ecs_per_condition.png`: Bitrate sweep and logistic sigmoid collapse threshold ($b_0 = 9.24\text{ kbps}$).
  - `fig2_early_warning_dashboard.png`: Forensic early-warning triage dashboard.
  - `fig3_deletion_curves.png`: Deletion AUC curves with visibly distinct trajectories.
  - `fig4_radar_chart.png`: Explanation consistency across synthesis attack families (A07–A19).
  - `fig5_spectrogram_saliency.png`: Time-frequency saliency maps under clean, Opus 16k, and Opus 6k.
- `colab.ipynb` / `new.ipynb`: 1-click Google Colab notebook with 0 errors, full GPU T4 support, and auto-download.
- `results/`: `faithfulness_results.csv`, `bitrate_sweep.csv`, `detection_results.json`.
