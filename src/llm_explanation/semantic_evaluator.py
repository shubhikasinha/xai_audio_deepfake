"""
Semantic consistency evaluator for LLM-generated explanations.

Measures whether explanations remain semantically consistent
when the underlying attribution maps shift under degradation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class SemanticEvaluator:
    """
    Evaluate semantic consistency of NL explanations across conditions.

    Uses Sentence-BERT embeddings to compute cosine similarity between
    explanations generated from clean vs. degraded audio.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Load Sentence-BERT model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except ImportError:
            print("[SemanticEval] sentence-transformers not installed.")
            self._model = None

    def compute_similarity(
        self,
        explanation_clean: str,
        explanation_degraded: str,
    ) -> float:
        """
        Compute semantic similarity between two explanations.

        Args:
            explanation_clean: Explanation from clean audio.
            explanation_degraded: Explanation from degraded audio.

        Returns:
            Cosine similarity in [-1, 1].
        """
        self._load_model()

        if self._model is None:
            return self._fallback_similarity(
                explanation_clean, explanation_degraded
            )

        embeddings = self._model.encode(
            [explanation_clean, explanation_degraded],
            convert_to_numpy=True,
        )

        # Cosine similarity
        sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(sim)

    def compute_claim_preservation(
        self,
        explanation_clean: str,
        explanation_degraded: str,
        claim_keywords: Optional[List[str]] = None,
    ) -> Dict:
        """
        Measure what fraction of forensic claims are preserved.

        Checks whether key terms from the clean explanation appear
        in the degraded explanation.

        Args:
            explanation_clean: Clean explanation.
            explanation_degraded: Degraded explanation.
            claim_keywords: Optional list of specific keywords to track.

        Returns:
            Dict with preservation rate and details.
        """
        if claim_keywords is None:
            # Default forensic keywords to track
            claim_keywords = [
                "high-frequency", "low-frequency", "artifact",
                "formant", "spectral", "temporal", "harmonic",
                "noise", "compression", "distortion", "spoof",
                "synthesis", "vocoder", "natural", "unnatural",
            ]

        clean_lower = explanation_clean.lower()
        degraded_lower = explanation_degraded.lower()

        # Find claims present in clean explanation
        claims_in_clean = [k for k in claim_keywords if k in clean_lower]

        if not claims_in_clean:
            return {
                "preservation_rate": 1.0,
                "claims_in_clean": [],
                "claims_preserved": [],
                "claims_lost": [],
            }

        # Check which are preserved in degraded
        claims_preserved = [k for k in claims_in_clean if k in degraded_lower]
        claims_lost = [k for k in claims_in_clean if k not in degraded_lower]

        preservation_rate = len(claims_preserved) / len(claims_in_clean)

        return {
            "preservation_rate": float(preservation_rate),
            "claims_in_clean": claims_in_clean,
            "claims_preserved": claims_preserved,
            "claims_lost": claims_lost,
            "n_claims_clean": len(claims_in_clean),
            "n_claims_preserved": len(claims_preserved),
        }

    def evaluate_batch(
        self,
        explanations_clean: List[str],
        explanations_degraded: List[str],
    ) -> Dict:
        """
        Evaluate semantic consistency for a batch of explanation pairs.

        Returns:
            Dict with aggregate statistics.
        """
        n = len(explanations_clean)
        similarities = np.zeros(n)
        preservation_rates = np.zeros(n)

        for i in range(n):
            similarities[i] = self.compute_similarity(
                explanations_clean[i], explanations_degraded[i]
            )
            claim_result = self.compute_claim_preservation(
                explanations_clean[i], explanations_degraded[i]
            )
            preservation_rates[i] = claim_result["preservation_rate"]

        return {
            "semantic_similarity_mean": float(np.mean(similarities)),
            "semantic_similarity_std": float(np.std(similarities)),
            "semantic_similarity_median": float(np.median(similarities)),
            "claim_preservation_mean": float(np.mean(preservation_rates)),
            "claim_preservation_std": float(np.std(preservation_rates)),
            "n_samples": n,
        }

    def _fallback_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Simple word-overlap fallback when Sentence-BERT is unavailable."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)  # Jaccard similarity
