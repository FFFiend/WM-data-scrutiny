"""
Build the source × phenomenon coverage matrix.

Primary normalization: per-clip (fraction of clips in each source
that exhibit each category).
Sensitivity check: per-hour (adjusts for variable clip lengths).
"""

import pandas as pd
import numpy as np

from taxonomy import CATEGORIES, CATEGORY_INFO


def build_coverage_matrix(
    tags_df: pd.DataFrame,
    normalize: str = "per_clip",
) -> pd.DataFrame:
    """
    Build a source × category coverage matrix.

    Args:
        tags_df: DataFrame with columns clip_id, source, source_group,
                 and {cat}_any (bool) for each category.
        normalize: "per_clip" (fraction of clips) or "per_hour"
                   (requires duration_s column).

    Returns:
        DataFrame with sources as rows, categories as columns,
        values as coverage fractions [0, 1].
    """
    bool_cols = [f"{cat}_any" for cat in CATEGORIES]

    if normalize == "per_clip":
        matrix = tags_df.groupby("source")[bool_cols].mean()
    elif normalize == "per_hour":
        if "duration_s" not in tags_df.columns:
            raise ValueError("per_hour normalization requires 'duration_s' column")
        # Weighted mean: each clip weighted by its duration
        def weighted_mean(group):
            weights = group["duration_s"].values
            total_hours = weights.sum() / 3600.0
            if total_hours == 0:
                return pd.Series(0.0, index=bool_cols)
            result = {}
            for col in bool_cols:
                weighted = (group[col].values * weights).sum() / 3600.0
                result[col] = weighted / total_hours
            return pd.Series(result)

        matrix = tags_df.groupby("source").apply(weighted_mean)
    else:
        raise ValueError(f"Unknown normalization: {normalize}")

    # Rename columns for readability
    matrix.columns = [col.replace("_any", "") for col in matrix.columns]

    return matrix


def build_group_coverage_matrix(
    tags_df: pd.DataFrame,
    normalize: str = "per_clip",
) -> pd.DataFrame:
    """Like build_coverage_matrix but grouped by source_group (robot/ego/sim)."""
    bool_cols = [f"{cat}_any" for cat in CATEGORIES]

    if normalize == "per_clip":
        matrix = tags_df.groupby("source_group")[bool_cols].mean()
    else:
        raise NotImplementedError("per_hour not yet implemented for group level")

    matrix.columns = [col.replace("_any", "") for col in matrix.columns]
    return matrix


def gap_analysis(
    matrix: pd.DataFrame,
    thin_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    For each category where robot data is thin (< threshold),
    check whether ego or sim data has better coverage.

    Returns DataFrame with columns:
        category, robot_coverage, ego_coverage, sim_coverage,
        best_supplement, supplement_coverage, gap_filled
    """
    # Need a group-level matrix
    groups = matrix.index.tolist()

    def get_group_val(group_name, cat):
        if group_name in matrix.index:
            return matrix.loc[group_name, cat]
        return 0.0

    results = []
    for cat in CATEGORIES:
        if cat not in matrix.columns:
            continue

        robot_cov = get_group_val("robot", cat)
        ego_cov = get_group_val("ego", cat)
        sim_cov = get_group_val("sim", cat)

        if robot_cov >= thin_threshold:
            continue  # robot coverage is sufficient

        # Find best supplement
        supplements = {"ego": ego_cov, "sim": sim_cov}
        best = max(supplements, key=supplements.get)
        best_cov = supplements[best]

        results.append({
            "category": cat,
            "robot_coverage": round(robot_cov, 3),
            "ego_coverage": round(ego_cov, 3),
            "sim_coverage": round(sim_cov, 3),
            "best_supplement": best if best_cov > robot_cov else "none",
            "supplement_coverage": round(best_cov, 3),
            "gap_filled": best_cov > thin_threshold,
        })

    return pd.DataFrame(results)


def subcategory_coverage(tags_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a finer-grained coverage table at the subcategory level.
    Returns source_group × subcategory fraction.
    """
    rows = []
    for cat in CATEGORIES:
        col = cat  # the pipe-separated column
        if col not in tags_df.columns:
            continue
        for sub in CATEGORY_INFO[cat]["subcategories"]:
            for group in ["robot", "ego", "sim"]:
                group_df = tags_df[tags_df["source_group"] == group]
                if len(group_df) == 0:
                    continue
                has_sub = group_df[col].apply(lambda x: sub in str(x).split("|"))
                frac = has_sub.mean()
                rows.append({
                    "category": cat,
                    "subcategory": sub,
                    "source_group": group,
                    "coverage": round(frac, 4),
                    "n_clips": int(has_sub.sum()),
                })
    return pd.DataFrame(rows)
