"""
Statistical testing utilities for the faithfulness study.

Implements all statistical tests specified in the experimental protocol:
- Spearman rank correlation (RQ1: ΔEER vs ΔFaithfulness)
- Paired Wilcoxon signed-rank (per-condition faithfulness comparison)
- Two-way ANOVA (RQ2: method × model effects)
- Bonferroni correction (multiple comparison)
- Bootstrap confidence intervals
- Cohen's d effect size
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats


def spearman_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> Dict:
    """
    Compute Spearman rank correlation with p-value.

    Primary test for RQ1: correlating accuracy drop with faithfulness drop.

    Args:
        x: First variable (e.g., ΔEER across conditions).
        y: Second variable (e.g., ΔDeletion AUC across conditions).

    Returns:
        Dict with rho, p-value, and interpretation.
    """
    rho, p_value = stats.spearmanr(x, y)

    # Interpret strength
    abs_rho = abs(rho)
    if abs_rho >= 0.7:
        strength = "strong"
    elif abs_rho >= 0.4:
        strength = "moderate"
    elif abs_rho >= 0.2:
        strength = "weak"
    else:
        strength = "negligible"

    return {
        "rho": float(rho),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "strength": strength,
        "n": len(x),
    }


def paired_wilcoxon(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "two-sided",
) -> Dict:
    """
    Paired Wilcoxon signed-rank test.

    Compares faithfulness metrics between clean and a specific
    degradation condition (paired by utterance).

    Args:
        x: Metric values under condition A (e.g., clean).
        y: Metric values under condition B (e.g., degraded).
        alternative: "two-sided", "greater", or "less".

    Returns:
        Dict with statistic, p-value, and effect size.
    """
    # Remove ties (identical pairs)
    diff = x - y
    non_zero = diff != 0

    if np.sum(non_zero) < 10:
        return {
            "statistic": np.nan,
            "p_value": 1.0,
            "significant": False,
            "effect_size": 0.0,
            "n_effective": int(np.sum(non_zero)),
            "warning": "Too few non-zero differences for reliable test.",
        }

    stat, p_value = stats.wilcoxon(
        x[non_zero], y[non_zero],
        alternative=alternative,
    )

    # Effect size (r = Z / sqrt(N))
    n_eff = int(np.sum(non_zero))
    z_score = stats.norm.ppf(1 - p_value / 2) if p_value < 1.0 else 0.0
    effect_size = z_score / np.sqrt(n_eff) if n_eff > 0 else 0.0

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "effect_size": float(effect_size),
        "n_effective": n_eff,
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
    }


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Dict:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of uncorrected p-values.
        alpha: Family-wise error rate threshold.

    Returns:
        Dict with corrected p-values and significance decisions.
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    corrected_pvalues = [min(p * n_tests, 1.0) for p in p_values]

    return {
        "corrected_p_values": corrected_pvalues,
        "corrected_alpha": corrected_alpha,
        "n_tests": n_tests,
        "significant": [p < corrected_alpha for p in p_values],
        "n_significant": sum(p < corrected_alpha for p in p_values),
    }


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn=np.mean,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Sample data.
        statistic_fn: Function to compute the statistic.
        confidence: Confidence level (default 0.95).
        n_resamples: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Dict with point estimate, CI bounds, and SE.
    """
    rng = np.random.RandomState(seed)
    n = len(data)

    boot_stats = np.zeros(n_resamples)
    for i in range(n_resamples):
        resample = data[rng.randint(0, n, size=n)]
        boot_stats[i] = statistic_fn(resample)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return {
        "estimate": float(statistic_fn(data)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "std_error": float(np.std(boot_stats)),
        "confidence": confidence,
        "n_resamples": n_resamples,
    }


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
) -> Dict:
    """
    Compute Cohen's d effect size.

    Args:
        group1: First group values.
        group2: Second group values.

    Returns:
        Dict with d, interpretation, and pooled SD.
    """
    n1, n2 = len(group1), len(group2)
    mean_diff = np.mean(group1) - np.mean(group2)

    # Pooled standard deviation
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0

    # Interpret
    abs_d = abs(d)
    if abs_d >= 0.8:
        interpretation = "large"
    elif abs_d >= 0.5:
        interpretation = "medium"
    elif abs_d >= 0.2:
        interpretation = "small"
    else:
        interpretation = "negligible"

    return {
        "d": float(d),
        "abs_d": float(abs_d),
        "interpretation": interpretation,
        "pooled_sd": float(pooled_sd),
        "mean_diff": float(mean_diff),
    }


def run_full_statistical_analysis(
    eer_clean: float,
    eer_conditions: Dict[str, float],
    faithfulness_clean: Dict[str, np.ndarray],
    faithfulness_conditions: Dict[str, Dict[str, np.ndarray]],
) -> Dict:
    """
    Run the complete statistical analysis for the paper.

    Args:
        eer_clean: EER on clean data.
        eer_conditions: EER per degradation condition.
        faithfulness_clean: Per-sample faithfulness on clean data.
        faithfulness_conditions: Per-sample faithfulness per condition.

    Returns:
        Dict with all statistical test results.
    """
    results = {}

    # RQ1: Spearman correlation between ΔEER and ΔFaithfulness
    delta_eer = np.array([
        eer_conditions[c] - eer_clean for c in eer_conditions
    ])

    if "deletion_auc" in faithfulness_clean:
        clean_del_auc = np.mean(faithfulness_clean["deletion_auc"])
        delta_del_auc = np.array([
            np.mean(faithfulness_conditions[c]["deletion_auc"]) - clean_del_auc
            for c in faithfulness_conditions
        ])

        results["rq1_spearman"] = spearman_correlation(delta_eer, delta_del_auc)

    # Per-condition Wilcoxon tests
    p_values = []
    condition_tests = {}
    for condition_name in faithfulness_conditions:
        if "deletion_auc" in faithfulness_clean and \
           "deletion_auc" in faithfulness_conditions[condition_name]:
            test_result = paired_wilcoxon(
                faithfulness_clean["deletion_auc"],
                faithfulness_conditions[condition_name]["deletion_auc"],
            )
            condition_tests[condition_name] = test_result
            p_values.append(test_result["p_value"])

    results["per_condition_tests"] = condition_tests

    # Bonferroni correction
    if p_values:
        results["bonferroni"] = bonferroni_correction(p_values)

    return results
