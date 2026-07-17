"""
Publication-ready figure generation for the paper.

Generates all figures needed for the Springer CCIS submission,
following the paper's figure plan.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Publication-quality settings
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def plot_attribution_heatmap(
    attribution: np.ndarray,
    title: str = "Attribution Map",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 3),
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """
    Plot spectrogram-level attribution heatmap.

    Args:
        attribution: Attribution map [F, T'].
        title: Plot title.
        save_path: If set, save figure to this path.
        figsize: Figure size.
        cmap: Colormap.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(
        attribution,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        interpolation="bilinear",
    )

    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Mel Bin")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Attribution")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    return fig


def plot_deletion_insertion_curves(
    deletion_curves: Dict[str, np.ndarray],
    insertion_curves: Dict[str, np.ndarray],
    condition_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Plot deletion and insertion curves across degradation conditions.

    Args:
        deletion_curves: {condition_name: prediction_array}.
        insertion_curves: {condition_name: prediction_array}.
        condition_names: Order of conditions to plot.
        save_path: If set, save figure.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0, 1, len(condition_names)))

    for i, cond in enumerate(condition_names):
        x = np.linspace(0, 1, len(deletion_curves.get(cond, [])))

        if cond in deletion_curves:
            ax1.plot(x, deletion_curves[cond], label=cond, color=colors[i], linewidth=1.5)
        if cond in insertion_curves:
            ax2.plot(x, insertion_curves[cond], label=cond, color=colors[i], linewidth=1.5)

    ax1.set_xlabel("Fraction of Features Removed")
    ax1.set_ylabel("Model Confidence")
    ax1.set_title("Deletion Curves")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Fraction of Features Added")
    ax2.set_ylabel("Model Confidence")
    ax2.set_title("Insertion Curves")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    return fig


def plot_accuracy_vs_faithfulness(
    eer_values: Dict[str, float],
    faithfulness_values: Dict[str, float],
    model_name: str = "AASIST",
    metric_name: str = "Deletion AUC",
    save_path: Optional[str] = None,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """
    Scatter plot of accuracy (EER) vs. faithfulness across conditions.

    This is the core visualization for RQ1.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    conditions = sorted(set(eer_values.keys()) & set(faithfulness_values.keys()))
    x = [eer_values[c] for c in conditions]
    y = [faithfulness_values[c] for c in conditions]

    ax.scatter(x, y, c="steelblue", s=60, edgecolors="navy", alpha=0.8, zorder=3)

    for i, cond in enumerate(conditions):
        ax.annotate(
            cond, (x[i], y[i]),
            textcoords="offset points", xytext=(5, 5),
            fontsize=7, alpha=0.8,
        )

    # Add trend line
    if len(x) > 2:
        from scipy import stats
        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(
            x_line, slope * x_line + intercept,
            "r--", alpha=0.5, linewidth=1,
            label=f"ρ={r_value:.2f}, p={p_value:.3f}",
        )

    ax.set_xlabel(f"EER (↑ = worse detection) — {model_name}")
    ax.set_ylabel(f"{metric_name} (faithfulness)")
    ax.set_title(f"Detection Accuracy vs. Explanation Faithfulness\n{model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    return fig


def plot_ecs_heatmap(
    ecs_matrix: np.ndarray,
    model_names: List[str],
    condition_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """
    Heatmap of ECS values across models and conditions.

    Core visualization for the forensic early-warning dashboard.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = sns.heatmap(
        ecs_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        xticklabels=condition_names,
        yticklabels=model_names,
        vmin=0, vmax=1,
        ax=ax,
        cbar_kws={"label": "ECS (↑ = more trustworthy)"},
        linewidths=0.5,
    )

    ax.set_xlabel("Degradation Condition")
    ax.set_ylabel("Detector")
    ax.set_title("Explanation Consistency Score — Early-Warning Dashboard")

    # Add warning zone
    for i in range(ecs_matrix.shape[0]):
        for j in range(ecs_matrix.shape[1]):
            if ecs_matrix[i, j] < 0.5:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    fill=False, edgecolor="red", linewidth=2,
                ))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    return fig


def plot_faithfulness_comparison(
    metrics_by_method: Dict[str, Dict[str, float]],
    conditions: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Bar chart comparing faithfulness metrics across XAI methods.

    Visualization for RQ2.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    methods = list(metrics_by_method.keys())
    x = np.arange(len(conditions))
    width = 0.35

    colors = ["#2196F3", "#FF9800"]

    for i, method in enumerate(methods):
        if "deletion_auc" in metrics_by_method[method]:
            del_values = [
                metrics_by_method[method]["deletion_auc"].get(c, 0)
                for c in conditions
            ]
            ax1.bar(x + i * width, del_values, width, label=method, color=colors[i], alpha=0.8)

        if "insertion_auc" in metrics_by_method[method]:
            ins_values = [
                metrics_by_method[method]["insertion_auc"].get(c, 0)
                for c in conditions
            ]
            ax2.bar(x + i * width, ins_values, width, label=method, color=colors[i], alpha=0.8)

    for ax, title, ylabel in [
        (ax1, "Deletion AUC (↓ better)", "Deletion AUC"),
        (ax2, "Insertion AUC (↑ better)", "Insertion AUC"),
    ]:
        ax.set_xlabel("Degradation Condition")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    return fig
