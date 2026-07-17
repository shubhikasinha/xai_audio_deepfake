"""
Structured prompt templates for LLM explanation generation.

The LLM does NOT perform detection — it only converts detector outputs
and XAI attribution maps into analyst-friendly natural language.
"""

FORENSIC_EXPLANATION_PROMPT = """You are an audio forensics analyst. Given the following detector output \
and explainability analysis, produce a concise, technically grounded \
explanation for why this audio sample was classified as {label}.

## Detector Information
- Model: {model_name}
- Confidence Score: {score:.4f}
- Classification: {label}
- Degradation Condition: {condition}

## Explainability Analysis (Integrated Gradients)
- Top-5 most important frequency bands (Hz): {top_freq_bands}
- Top-3 temporal regions of interest (seconds): {top_time_regions}
- Attribution concentration (Gini): {concentration:.3f}
- Spectral energy distribution: {energy_distribution}
- Explanation Consistency Score (ECS): {ecs:.3f}

## Task
Produce exactly three items:
1. **Verdict**: A one-sentence verdict summarizing the classification.
2. **Technical Rationale**: A 2-3 sentence technical explanation citing specific \
frequency ranges and temporal regions as evidence. Be precise about which \
spectral features support the classification.
3. **Confidence Qualifier**: A one-sentence assessment of how much to trust \
this explanation, based on the attribution concentration and ECS score. \
If ECS < 0.5, explicitly warn that the explanation may be unreliable.

Format your response as:
**Verdict:** [your verdict]
**Rationale:** [your technical rationale]
**Confidence:** [your confidence qualifier]
"""


COMPARATIVE_EXPLANATION_PROMPT = """You are comparing two explanations for the same audio sample — \
one from the original (clean) audio and one from a degraded version.

## Clean Audio Analysis
- Confidence: {score_clean:.4f}
- Classification: {label_clean}
- Top frequency bands (Hz): {freq_bands_clean}
- Top temporal regions (sec): {time_regions_clean}

## Degraded Audio Analysis ({condition})
- Confidence: {score_degraded:.4f}
- Classification: {label_degraded}
- Top frequency bands (Hz): {freq_bands_degraded}
- Top temporal regions (sec): {time_regions_degraded}

## Consistency Metrics
- Explanation Stability (cosine sim): {stability:.3f}
- ECS: {ecs:.3f}

## Task
In 3-4 sentences, describe:
1. What forensic evidence changed between clean and degraded versions?
2. Is the change consistent with the type of degradation applied?
3. Should an analyst trust the degraded explanation? Why or why not?
"""


def format_forensic_prompt(
    model_name: str,
    score: float,
    label: str,
    condition: str,
    top_freq_bands: list,
    top_time_regions: list,
    concentration: float,
    energy_distribution: dict,
    ecs: float,
) -> str:
    """Format the forensic explanation prompt with actual values."""
    freq_str = ", ".join(
        f"{low}-{high} Hz" for low, high in top_freq_bands[:5]
    )
    time_str = ", ".join(
        f"{start:.2f}-{end:.2f}s" for start, end in top_time_regions[:3]
    )
    energy_str = "; ".join(
        f"{label}: {frac:.1%}"
        for label, frac in zip(
            energy_distribution.get("band_labels", []),
            energy_distribution.get("band_fractions", []),
        )
    )

    return FORENSIC_EXPLANATION_PROMPT.format(
        model_name=model_name,
        score=score,
        label=label,
        condition=condition,
        top_freq_bands=freq_str,
        top_time_regions=time_str,
        concentration=concentration,
        energy_distribution=energy_str,
        ecs=ecs,
    )


def format_comparative_prompt(
    score_clean: float,
    label_clean: str,
    freq_bands_clean: list,
    time_regions_clean: list,
    score_degraded: float,
    label_degraded: str,
    freq_bands_degraded: list,
    time_regions_degraded: list,
    condition: str,
    stability: float,
    ecs: float,
) -> str:
    """Format the comparative explanation prompt."""
    return COMPARATIVE_EXPLANATION_PROMPT.format(
        score_clean=score_clean,
        label_clean=label_clean,
        freq_bands_clean=", ".join(f"{l}-{h} Hz" for l, h in freq_bands_clean[:5]),
        time_regions_clean=", ".join(f"{s:.2f}-{e:.2f}s" for s, e in time_regions_clean[:3]),
        score_degraded=score_degraded,
        label_degraded=label_degraded,
        freq_bands_degraded=", ".join(f"{l}-{h} Hz" for l, h in freq_bands_degraded[:5]),
        time_regions_degraded=", ".join(f"{s:.2f}-{e:.2f}s" for s, e in time_regions_degraded[:3]),
        condition=condition,
        stability=stability,
        ecs=ecs,
    )
