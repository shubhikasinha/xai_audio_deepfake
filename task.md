- [x] Initialize project repository with full structure
- [x] Create README.md with project overview & reproduction guide
- [x] Create deepfake.md research blueprint (updated with reviewer feedback)
- [x] Create requirements.txt with pinned dependencies
- [x] Create pyproject.toml with project config
- [x] Create .gitignore
- [x] Create YAML configs (experiment, models, degradation conditions)
- [x] Implement data loaders (ASVspoof 2019 LA, 2021 DF, WaveFake)
- [x] Implement degradation pipeline (codecs, noise, reverb)
- [x] Implement audio utilities (load, save, spectrogram)
- [x] Implement base detector interface
- [x] Implement AASIST detector wrapper
- [x] Implement WavLM+ECAPA detector wrapper
- [x] Implement Integrated Gradients explainer
- [x] Implement Kernel SHAP explainer
- [x] Implement attribution utilities (normalization, top-k, concentration)
- [x] Implement detection metrics (EER, min t-DCF)
- [x] Implement faithfulness metrics (Deletion/Insertion AUC, Sensitivity-N, Stability)
- [x] Implement Explanation Consistency Score (ECS) — novel metric
- [x] Implement statistical tests (Spearman, Wilcoxon, Bonferroni, Bootstrap CI, Cohen's d)
- [x] Implement LLM prompt templates
- [x] Implement LLM explanation generator (Gemini/OpenAI)
- [x] Implement semantic evaluator (Sentence-BERT consistency)
- [x] Implement publication-ready figure generators
- [x] Create main pipeline script with quick-test mode
- [x] Create unit tests (ECS, metrics, XAI utilities)
- [x] Create setup.py and LICENSE

## Phase 1 — Remaining Tasks (Week 1-2)
- [ ] Set up virtual environment and install dependencies
- [ ] Run quick-test to verify pipeline (`python scripts/run_full_pipeline.py --quick-test`)
- [ ] Run unit tests (`python -m pytest tests/ -v`)
- [ ] Download ASVspoof 2019 LA from Zenodo
- [ ] Download ASVspoof 2021 DF from Zenodo
- [ ] Download AASIST pretrained checkpoint
- [ ] Download WavLM-Large from HuggingFace
- [ ] Verify data integrity and model loading
- [ ] Evaluate both detectors on clean ASVspoof 2021 DF (verify EER)

## Phase 2 — XAI & Faithfulness (Weeks 3-5)
- [ ] Run IG on AASIST (all 12 conditions)
- [ ] Run IG on WavLM+ECAPA (all 12 conditions)
- [ ] Run SHAP on AASIST (subset: C0, C3, C7, C8, N2)
- [ ] Compute faithfulness metrics across all experiments
- [ ] Compute ECS for all condition pairs

## Phase 3 — Analysis (Weeks 6-7)
- [ ] Spearman correlation: ΔEER vs ΔFaithfulness (RQ1)
- [ ] Per-condition Wilcoxon tests + Bonferroni
- [ ] Two-way ANOVA: method × model (RQ2)
- [ ] ECS ROC-AUC as early-warning (RQ3)
- [ ] Ablation studies
- [ ] Cross-dataset validation on WaveFake

## Phase 4 — LLM & Human Study (Week 8)
- [ ] Generate NL explanations (clean vs. degraded)
- [ ] Semantic consistency evaluation
- [ ] Design and run human evaluation study

## Phase 5 — Paper & Submission (Weeks 9-10)
- [ ] Generate all paper figures
- [ ] Write paper in Springer CCIS LaTeX
- [ ] Internal review cycle
- [ ] Submit to AIST 2026
