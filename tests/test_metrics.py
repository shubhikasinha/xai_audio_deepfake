"""
Unit tests for detection metrics (EER, min t-DCF).
"""

import numpy as np
import pytest

from src.evaluation.detection_metrics import (
    compute_eer,
    compute_min_tdcf,
    compute_detection_metrics,
)


class TestEER:
    """Tests for Equal Error Rate computation."""

    def test_perfect_separation(self):
        """Perfect scores should give EER ≈ 0."""
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7])
        labels = np.array([0, 0, 0, 1, 1, 1])
        eer, threshold = compute_eer(scores, labels)
        assert eer < 0.1

    def test_random_scores(self):
        """Random scores should give EER ≈ 0.5."""
        np.random.seed(42)
        scores = np.random.rand(1000)
        labels = np.random.randint(0, 2, 1000)
        eer, threshold = compute_eer(scores, labels)
        assert 0.3 < eer < 0.7

    def test_eer_range(self):
        """EER should be in [0, 1]."""
        scores = np.random.rand(100)
        labels = np.random.randint(0, 2, 100)
        eer, threshold = compute_eer(scores, labels)
        assert 0.0 <= eer <= 1.0

    def test_compute_detection_metrics(self):
        """Full metrics function should return all expected keys."""
        scores = np.random.rand(100)
        labels = np.random.randint(0, 2, 100)
        metrics = compute_detection_metrics(scores, labels)
        assert "eer" in metrics
        assert "min_tdcf" in metrics
        assert "n_bonafide" in metrics
        assert "n_spoof" in metrics


class TestMinTDCF:
    """Tests for minimum tandem DCF."""

    def test_min_tdcf_range(self):
        """min t-DCF should be non-negative."""
        scores = np.random.rand(100)
        labels = np.random.randint(0, 2, 100)
        tdcf = compute_min_tdcf(scores, labels)
        assert tdcf >= 0.0
