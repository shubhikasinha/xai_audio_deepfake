"""
Unit tests for the Explanation Consistency Score (ECS).
Tests the paper's novel contribution.
"""

import numpy as np
import pytest

from src.evaluation.consistency_score import ExplanationConsistencyScore
from src.evaluation.faithfulness_metrics import (
    compute_explanation_stability,
    compute_spectral_band_alignment,
)


class TestExplanationConsistencyScore:
    """Tests for the ECS metric."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ecs = ExplanationConsistencyScore(
            alpha=0.4, beta=0.3, gamma=0.3
        )
        np.random.seed(42)

        # Create synthetic attribution maps
        self.attr_clean = np.random.randn(128, 100)
        self.attr_similar = self.attr_clean + np.random.randn(128, 100) * 0.1
        self.attr_different = np.random.randn(128, 100)

    def test_weights_sum_to_one(self):
        """ECS weights must sum to 1.0."""
        assert abs(self.ecs.alpha + self.ecs.beta + self.ecs.gamma - 1.0) < 1e-6

    def test_invalid_weights_raises(self):
        """Weights not summing to 1 should raise."""
        with pytest.raises(AssertionError):
            ExplanationConsistencyScore(alpha=0.5, beta=0.5, gamma=0.5)

    def test_perfect_consistency(self):
        """Identical clean and degraded should give high ECS."""
        result = self.ecs.compute(
            attr_clean=self.attr_clean,
            attr_degraded=self.attr_clean,  # Same!
            score_clean=0.9,
            score_degraded=0.9,
            del_auc_clean=0.3,
            del_auc_degraded=0.3,
        )
        # Perfect stability + perfect faithfulness preservation
        assert result["ecs"] > 0.7
        assert result["stability_score"] > 0.99
        assert result["faithfulness_preservation"] > 0.99

    def test_similar_attributions_high_ecs(self):
        """Similar attributions should produce moderate-high ECS."""
        result = self.ecs.compute(
            attr_clean=self.attr_clean,
            attr_degraded=self.attr_similar,
            score_clean=0.9,
            score_degraded=0.85,
            del_auc_clean=0.3,
            del_auc_degraded=0.35,
        )
        assert result["ecs"] > 0.5

    def test_different_attributions_low_ecs(self):
        """Very different attributions should produce low ECS."""
        result = self.ecs.compute(
            attr_clean=self.attr_clean,
            attr_degraded=self.attr_different,
            score_clean=0.9,
            score_degraded=0.3,
            del_auc_clean=0.3,
            del_auc_degraded=0.8,
        )
        assert result["ecs"] < 0.5

    def test_ecs_range(self):
        """ECS should be in [0, 1]."""
        result = self.ecs.compute(
            attr_clean=self.attr_clean,
            attr_degraded=self.attr_similar,
            score_clean=0.9,
            score_degraded=0.8,
            del_auc_clean=0.3,
            del_auc_degraded=0.4,
        )
        assert 0.0 <= result["ecs"] <= 1.0

    def test_components_returned(self):
        """All component values should be in the result."""
        result = self.ecs.compute(
            attr_clean=self.attr_clean,
            attr_degraded=self.attr_similar,
            score_clean=0.8,
            score_degraded=0.7,
            del_auc_clean=0.3,
            del_auc_degraded=0.35,
        )
        assert "ecs" in result
        assert "stability" in result
        assert "spectral_alignment" in result
        assert "faithfulness_preservation" in result
        assert "alpha" in result

    def test_batch_computation(self):
        """Batch ECS should return aggregate statistics."""
        attrs_clean = [self.attr_clean] * 5
        attrs_degraded = [self.attr_similar] * 5
        scores_clean = np.array([0.9] * 5)
        scores_degraded = np.array([0.8] * 5)
        del_clean = np.array([0.3] * 5)
        del_degraded = np.array([0.35] * 5)

        result = self.ecs.compute_batch(
            attrs_clean, attrs_degraded,
            scores_clean, scores_degraded,
            del_clean, del_degraded,
        )
        assert "ecs_mean" in result
        assert "ecs_std" in result
        assert len(result["ecs_values"]) == 5

    def test_trustworthiness_assessment(self):
        """Trustworthiness assessment should return correct labels."""
        high = self.ecs.assess_trustworthiness(0.9, threshold=0.5)
        assert high["is_trustworthy"] is True
        assert high["confidence"] == "high"

        low = self.ecs.assess_trustworthiness(0.3, threshold=0.5)
        assert low["is_trustworthy"] is False
        assert low["confidence"] == "untrustworthy"


class TestExplanationStability:
    """Tests for the explanation stability metric."""

    def test_identical_attributions(self):
        """Identical attributions should have stability 1.0."""
        attr = np.random.randn(64, 50)
        stability = compute_explanation_stability(attr, attr)
        assert abs(stability - 1.0) < 1e-6

    def test_opposite_attributions(self):
        """Negated attributions should have stability -1.0."""
        attr = np.random.randn(64, 50)
        stability = compute_explanation_stability(attr, -attr)
        assert abs(stability - (-1.0)) < 1e-6

    def test_orthogonal_attributions(self):
        """Orthogonal attributions should have stability ~0."""
        attr1 = np.zeros((2, 2))
        attr1[0, 0] = 1.0
        attr2 = np.zeros((2, 2))
        attr2[1, 1] = 1.0
        stability = compute_explanation_stability(attr1, attr2)
        assert abs(stability) < 1e-6

    def test_zero_attribution(self):
        """Zero attribution should return 0 stability."""
        attr = np.random.randn(64, 50)
        zeros = np.zeros_like(attr)
        stability = compute_explanation_stability(attr, zeros)
        assert stability == 0.0


class TestSpectralBandAlignment:
    """Tests for spectral band alignment metric."""

    def test_concentrated_in_artifact_bands(self):
        """Attribution concentrated in artifact bands should have high alignment."""
        attr = np.zeros((128, 50))
        # Put all energy in the 4000-8000 Hz range (bins 64-128 for 8kHz Nyquist)
        attr[64:128, :] = 1.0
        alignment = compute_spectral_band_alignment(
            attr, 0.9,
            artifact_bands=[(4000, 8000)],
            n_mels=128, sample_rate=16000,
        )
        assert alignment > 0.5

    def test_zero_attribution(self):
        """Zero attribution should return 0 alignment."""
        attr = np.zeros((128, 50))
        alignment = compute_spectral_band_alignment(attr, 0.9)
        assert alignment == 0.0
