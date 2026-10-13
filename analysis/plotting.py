"""
Coverage matrix visualizations.

All plots saved to PLOTS_DIR as both PNG and PDF.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from config import PLOTS_DIR
from taxonomy import CATEGORIES, CATEGORY_INFO

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def plot_coverage_heatmap(
    matrix: pd.DataFrame,
    title: str = "Source x Phenomenon Coverage Matrix",
    filename: str = "coverage_heatmap",
    annot: bool = True,
):
    """
    Heatmap of the coverage matrix.
    Rows = sources, columns = categories, values = coverage fraction.
    """
    fig, ax = plt.subplots(figsize=(12, max(4, len(matrix) * 0.6)))

    # Pretty column labels
    col_labels = [CATEGORY_INFO.get(c, {}).get("label", c) for c in matrix.columns]

    sns.heatmap(
        matrix.values,
        xticklabels=col_labels,
        yticklabels=matrix.index.tolist(),
        annot=annot,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0, vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Phenomenon category")
    ax.set_ylabel("Data source")
    plt.xticks(rotation=35, ha="right")

    for ext in ["png", "pdf"]:
        fig.savefig(PLOTS_DIR / f"{filename}.{ext}")
    plt.close(fig)
    print(f"  Saved {filename}.png/pdf")


def plot_group_bars(
    group_matrix: pd.DataFrame,
    title: str = "Coverage by Source Group",
    filename: str = "group_coverage_bars",
):
    """
    Grouped bar chart: robot vs. ego vs. sim for each category.
    """
    cats = group_matrix.columns.tolist()
    groups = group_matrix.index.tolist()
    x = np.arange(len(cats))
    width = 0.25
    colors = {"robot": "#2196F3", "ego": "#FF9800", "sim": "#4CAF50"}

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, grp in enumerate(groups):
        vals = group_matrix.loc[grp].values
        color = colors.get(grp, f"C{i}")
        ax.bar(x + i * width, vals, width, label=grp, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    col_labels = [CATEGORY_INFO.get(c, {}).get("label", c) for c in cats]
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_ylabel("Coverage (fraction of clips)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="thin threshold")

    for ext in ["png", "pdf"]:
        fig.savefig(PLOTS_DIR / f"{filename}.{ext}")
    plt.close(fig)
    print(f"  Saved {filename}.png/pdf")


def plot_gap_analysis(
    gap_df: pd.DataFrame,
    filename: str = "gap_analysis",
):
    """
    Horizontal bar chart showing robot coverage gaps and available supplements.
    """
    if gap_df.empty:
        print("  No gaps found — skipping gap analysis plot.")
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(gap_df) * 0.8)))

    y = np.arange(len(gap_df))
    cat_labels = [CATEGORY_INFO.get(c, {}).get("label", c) for c in gap_df["category"]]

    ax.barh(y, gap_df["robot_coverage"], height=0.3, label="Robot", color="#2196F3", alpha=0.8)
    ax.barh(y + 0.3, gap_df["ego_coverage"], height=0.3, label="Ego", color="#FF9800", alpha=0.8)
    ax.barh(y + 0.6, gap_df["sim_coverage"], height=0.3, label="Sim", color="#4CAF50", alpha=0.8)

    ax.set_yticks(y + 0.3)
    ax.set_yticklabels(cat_labels)
    ax.set_xlabel("Coverage (fraction)")
    ax.set_title("Gap Analysis: Categories Where Robot Data Is Thin (<10%)")
    ax.legend(loc="lower right")
    ax.axvline(x=0.1, color="red", linestyle="--", alpha=0.5)

    for ext in ["png", "pdf"]:
        fig.savefig(PLOTS_DIR / f"{filename}.{ext}")
    plt.close(fig)
    print(f"  Saved {filename}.png/pdf")


def plot_domain_distances(
    distances: dict[str, float],
    per_category_distances: dict[str, dict[str, float]],
    filename: str = "domain_distances",
):
    """
    Bar chart of DINOv2 embedding distances between source groups,
    overall and per category.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distances
    ax = axes[0]
    pairs = list(distances.keys())
    vals = [distances[p] for p in pairs]
    ax.bar(pairs, vals, color=["#9C27B0", "#E91E63", "#00BCD4"], alpha=0.8)
    ax.set_ylabel("Cosine distance")
    ax.set_title("Overall Domain Gap (DINOv2)")
    ax.set_ylim(0, max(vals) * 1.3 if vals else 1)

    # Per-category distances (robot vs ego and robot vs sim)
    ax = axes[1]
    cats_with_data = [c for c in CATEGORIES if c in per_category_distances and per_category_distances[c]]
    if cats_with_data:
        x = np.arange(len(cats_with_data))
        width = 0.35
        ego_vals = [per_category_distances[c].get("robot_vs_ego", 0) for c in cats_with_data]
        sim_vals = [per_category_distances[c].get("robot_vs_sim", 0) for c in cats_with_data]
        cat_labels = [CATEGORY_INFO.get(c, {}).get("label", c) for c in cats_with_data]

        ax.bar(x - width / 2, ego_vals, width, label="Robot vs Ego", color="#FF9800", alpha=0.8)
        ax.bar(x + width / 2, sim_vals, width, label="Robot vs Sim", color="#4CAF50", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, rotation=35, ha="right")
        ax.set_ylabel("Cosine distance")
        ax.set_title("Per-Category Domain Gap")
        ax.legend()

    for ext in ["png", "pdf"]:
        fig.savefig(PLOTS_DIR / f"{filename}.{ext}")
    plt.close(fig)
    print(f"  Saved {filename}.png/pdf")


def plot_subcategory_coverage(
    sub_df: pd.DataFrame,
    filename: str = "subcategory_coverage",
):
    """Heatmap of subcategory-level coverage pivoted by source_group."""
    if sub_df.empty:
        return

    pivot = sub_df.pivot_table(
        index=["category", "subcategory"],
        columns="source_group",
        values="coverage",
        fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.4)))
    labels = [f"{cat} / {sub}" for cat, sub in pivot.index]

    sns.heatmap(
        pivot.values,
        xticklabels=pivot.columns.tolist(),
        yticklabels=labels,
        annot=True, fmt=".2f",
        cmap="YlOrRd", vmin=0, vmax=1,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Subcategory Coverage by Source Group")

    for ext in ["png", "pdf"]:
        fig.savefig(PLOTS_DIR / f"{filename}.{ext}")
    plt.close(fig)
    print(f"  Saved {filename}.png/pdf")
