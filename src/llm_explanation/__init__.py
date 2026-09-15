"""
LLM-based natural language explanation generation for XAI outputs.

Translates numerical attribution maps and ECS scores into human-readable
forensic reports using a local LLM (via Ollama) or OpenAI API.

This module adds a human-interpretability layer on top of IG/SHAP attributions,
making the paper's contribution accessible to forensic practitioners who may
not be fluent in XAI terminology.

Novel contribution: Coupling quantitative XAI metrics (ECS, deletion AUC) with
qualitative natural-language summaries bridges the XAI-practitioner gap.
"""

from typing import Dict, Optional


def generate_forensic_report(
    ecs_result: Dict,
    condition_name: str,
    model_name: str = "AASIST",
    xai_method: str = "Integrated Gradients",
    use_api: bool = False,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate a natural-language forensic report from ECS results.

    Args:
        ecs_result: Dict from ExplanationConsistencyScore.compute()
        condition_name: Name of the degradation condition (e.g. "Opus 6 kbps")
        model_name: Name of the detector model
        xai_method: XAI attribution method used
        use_api: If True, use OpenAI API; otherwise use template
        api_key: OpenAI API key (required if use_api=True)

    Returns:
        Human-readable forensic report string.
    """
    ecs        = ecs_result.get("ecs", 0.0)
    stability  = ecs_result.get("stability_score", 0.0)
    alignment  = ecs_result.get("spectral_alignment", 0.0)
    faith_pres = ecs_result.get("faithfulness_preservation", 0.0)
    confidence = ecs_result.get("confidence", "unknown")

    if use_api and api_key:
        return _generate_via_api(ecs_result, condition_name, model_name, xai_method, api_key)
    else:
        return _generate_template_report(
            ecs, stability, alignment, faith_pres, confidence,
            condition_name, model_name, xai_method
        )


def _generate_template_report(
    ecs, stability, alignment, faith_pres, confidence,
    condition_name, model_name, xai_method
) -> str:
    """Generate report using structured templates (no API required)."""

    if ecs >= 0.8:
        trustworthiness_str = "HIGH — the explanation can be trusted for forensic use"
        action_str = "Proceed with attribution-based forensic analysis."
    elif ecs >= 0.6:
        trustworthiness_str = "MODERATE — the explanation shows some degradation but remains usable"
        action_str = "Use attributions with caution; corroborate with secondary evidence."
    elif ecs >= 0.5:
        trustworthiness_str = "LOW — the explanation has degraded significantly"
        action_str = "Do not rely on attribution maps alone. Seek higher-quality audio."
    else:
        trustworthiness_str = "CRITICAL — explanation collapse detected"
        action_str = "WARNING: Attribution maps are unreliable under this codec condition. " \
                     "Forensic conclusions drawn from these explanations may be invalid."

    report = f"""
+==============================================================+
|         FORENSIC XAI TRUSTWORTHINESS REPORT                  |
+==============================================================+
  Detector    : {model_name}
  XAI Method  : {xai_method}
  Condition   : {condition_name}
+--------------------------------------------------------------+
  ECS Score             : {ecs:.3f} / 1.000
  Trustworthiness       : {trustworthiness_str}
+--------------------------------------------------------------+
  Component Breakdown:
  |-- Explanation Stability      : {stability:.3f}  (attribution cosine similarity)
  |-- Spectral Band Alignment    : {alignment:.3f}  (attribution mass in artifact bands)
  \\-- Faithfulness Preservation  : {faith_pres:.3f}  (deletion-AUC consistency)
+--------------------------------------------------------------+
  RECOMMENDED ACTION:
  {action_str}
+==============================================================+
""".strip()
    return report


def _generate_via_api(ecs_result, condition_name, model_name, xai_method, api_key) -> str:
    """Generate report via OpenAI API (requires openai package)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = f"""You are a forensic AI expert. Based on these XAI metrics for an audio deepfake detector,
write a concise 3-sentence forensic report explaining whether the AI's explanations can be trusted.

Detector: {model_name}
XAI Method: {xai_method}
Audio Condition: {condition_name}
ECS Score: {ecs_result.get('ecs', 0):.3f} (0=untrustworthy, 1=fully trustworthy)
Explanation Stability: {ecs_result.get('stability_score', 0):.3f}
Spectral Alignment: {ecs_result.get('spectral_alignment', 0):.3f}
Faithfulness Preservation: {ecs_result.get('faithfulness_preservation', 0):.3f}

Write a practitioner-friendly report (no jargon, max 100 words):"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content

    except ImportError:
        return _generate_template_report(
            ecs_result.get('ecs', 0),
            ecs_result.get('stability_score', 0),
            ecs_result.get('spectral_alignment', 0),
            ecs_result.get('faithfulness_preservation', 0),
            ecs_result.get('confidence', 'unknown'),
            condition_name, model_name, xai_method
        )


def batch_reports(ecs_results_per_condition: Dict, **kwargs) -> Dict[str, str]:
    """Generate reports for all conditions."""
    return {
        cond: generate_forensic_report(result, condition_name=cond, **kwargs)
        for cond, result in ecs_results_per_condition.items()
    }
