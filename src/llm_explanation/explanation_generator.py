"""
LLM-based explanation generator.

Converts detector outputs and XAI attribution maps into
analyst-friendly natural language explanations.
"""

import json
from typing import Dict, Optional
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.llm_explanation.prompt_templates import (
    format_forensic_prompt,
    format_comparative_prompt,
)
from src.xai.attribution_utils import (
    get_top_k_regions,
    compute_attribution_concentration,
    compute_spectral_energy_distribution,
)


class ExplanationGenerator:
    """
    Generate natural language explanations from detector + XAI outputs.

    Supports Gemini (Google) and OpenAI APIs.
    """

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ):
        self.provider = provider
        self.model_name = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _init_client(self):
        """Initialize the LLM API client."""
        if self._client is not None:
            return

        if self.provider == "gemini":
            try:
                import google.generativeai as genai
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model_name)
            except ImportError:
                print("[LLM] google-generativeai not installed.")
                self._client = None

        elif self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("[LLM] openai not installed.")
                self._client = None

    def generate_explanation(
        self,
        model_name: str,
        score: float,
        label: str,
        attribution: np.ndarray,
        condition: str = "Clean",
        ecs: float = 1.0,
    ) -> Dict:
        """
        Generate a natural language explanation for a single sample.

        Args:
            model_name: Detector name (e.g., "AASIST").
            score: Spoof probability.
            label: Classification label ("bonafide" or "spoof").
            attribution: Attribution map [F, T'].
            condition: Degradation condition name.
            ecs: Explanation Consistency Score.

        Returns:
            Dict with explanation text and metadata.
        """
        # Extract structured features from attribution
        regions = get_top_k_regions(attribution, k=5)
        concentration = compute_attribution_concentration(attribution)
        energy = compute_spectral_energy_distribution(attribution)

        # Format the prompt
        prompt = format_forensic_prompt(
            model_name=model_name,
            score=score,
            label=label,
            condition=condition,
            top_freq_bands=regions["top_freq_ranges_hz"],
            top_time_regions=regions["top_time_ranges_sec"],
            concentration=concentration,
            energy_distribution=energy,
            ecs=ecs,
        )

        # Call LLM
        response_text = self._call_llm(prompt)

        return {
            "explanation": response_text,
            "prompt": prompt,
            "model_name": model_name,
            "score": score,
            "label": label,
            "condition": condition,
            "ecs": ecs,
            "concentration": concentration,
            "top_freq_bands": regions["top_freq_ranges_hz"],
            "top_time_regions": regions["top_time_ranges_sec"],
        }

    def generate_comparative_explanation(
        self,
        score_clean: float,
        label_clean: str,
        attr_clean: np.ndarray,
        score_degraded: float,
        label_degraded: str,
        attr_degraded: np.ndarray,
        condition: str,
        stability: float,
        ecs: float,
    ) -> Dict:
        """Generate a comparative explanation (clean vs. degraded)."""
        regions_clean = get_top_k_regions(attr_clean, k=5)
        regions_degraded = get_top_k_regions(attr_degraded, k=5)

        prompt = format_comparative_prompt(
            score_clean=score_clean,
            label_clean=label_clean,
            freq_bands_clean=regions_clean["top_freq_ranges_hz"],
            time_regions_clean=regions_clean["top_time_ranges_sec"],
            score_degraded=score_degraded,
            label_degraded=label_degraded,
            freq_bands_degraded=regions_degraded["top_freq_ranges_hz"],
            time_regions_degraded=regions_degraded["top_time_ranges_sec"],
            condition=condition,
            stability=stability,
            ecs=ecs,
        )

        response_text = self._call_llm(prompt)

        return {
            "explanation": response_text,
            "condition": condition,
            "stability": stability,
            "ecs": ecs,
        }

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API and return the response text."""
        self._init_client()

        if self._client is None:
            return self._generate_fallback(prompt)

        try:
            if self.provider == "gemini":
                response = self._client.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    },
                )
                return response.text

            elif self.provider == "openai":
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content

        except Exception as e:
            print(f"[LLM] API call failed: {e}")
            return self._generate_fallback(prompt)

    def _generate_fallback(self, prompt: str) -> str:
        """Generate a template-based fallback explanation (no API needed)."""
        return (
            "**Verdict:** [LLM API not configured — template explanation]\n"
            "**Rationale:** Attribution analysis completed. See structured "
            "metrics in the prompt for frequency/temporal evidence.\n"
            "**Confidence:** Review the ECS score to assess explanation "
            "trustworthiness."
        )
