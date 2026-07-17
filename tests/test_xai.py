"""
Unit tests for XAI methods (Integrated Gradients, Kernel SHAP).
"""

import numpy as np
import torch
import pytest

from src.xai.attribution_utils import (
    normalize_attribution,
    get_top_k_regions,
    compute_attribution_concentration,
    compute_cosine_similarity,
)


class TestAttributionUtils:
    """Tests for attribution processing utilities."""

    def setup_method(self):
        np.random.seed(42)
        self.attr = np.random.randn(64, 50)

    def test_normalize_abs_max(self):
        """abs_max normalization should scale to [-1, 1]."""
        normed = normalize_attribution(self.attr, "abs_max")
        assert np.max(np.abs(normed)) <= 1.0 + 1e-6

    def test_normalize_minmax(self):
        """minmax normalization should scale to [0, 1]."""
        normed = normalize_attribution(self.attr, "minmax")
        assert normed.min() >= -1e-6
        assert normed.max() <= 1.0 + 1e-6

    def test_normalize_sum(self):
        """sum normalization should make abs values sum to 1."""
        normed = normalize_attribution(self.attr, "sum")
        assert abs(np.sum(np.abs(normed)) - 1.0) < 1e-6

    def test_top_k_regions(self):
        """Should return k frequency and time regions."""
        regions = get_top_k_regions(self.attr, k=5)
        assert len(regions["top_freq_bins"]) == 5
        assert len(regions["top_time_frames"]) == 5
        assert len(regions["top_freq_ranges_hz"]) == 5

    def test_concentration_uniform(self):
        """Uniform attribution should have low concentration (Gini ≈ 0)."""
        uniform = np.ones((10, 10))
        gini = compute_attribution_concentration(uniform)
        assert gini < 0.1

    def test_concentration_sparse(self):
        """Sparse attribution should have high concentration (Gini → 1)."""
        sparse = np.zeros((100, 100))
        sparse[0, 0] = 1.0
        gini = compute_attribution_concentration(sparse)
        assert gini > 0.9

    def test_cosine_similarity_identical(self):
        """Identical attributions should have cosine similarity 1.0."""
        sim = compute_cosine_similarity(self.attr, self.attr)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_opposite(self):
        """Negated attributions should have cosine similarity -1.0."""
        sim = compute_cosine_similarity(self.attr, -self.attr)
        assert abs(sim - (-1.0)) < 1e-6

    def test_cosine_similarity_zero(self):
        """Zero attribution should return 0."""
        zeros = np.zeros_like(self.attr)
        sim = compute_cosine_similarity(self.attr, zeros)
        assert sim == 0.0
