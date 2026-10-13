#!/usr/bin/env python3
"""
Analysis runner: coverage matrix, gap analysis, embedding distances, plots.

Usage:
    python scripts/run_analysis.py [--skip-embeddings] [--tags-file path]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from config import TAGS_DIR, RESULTS_DIR, PLOTS_DIR
from taxonomy import CATEGORIES, CATEGORY_INFO
from analysis.coverage_matrix import (
    build_coverage_matrix,
    build_group_coverage_matrix,
    gap_analysis,
    subcategory_coverage,
)
from analysis.embeddings import (
    embed_all_clips,
    compute_domain_distances,
    per_category_domain_gap,
)
from analysis.plotting import (
    plot_coverage_heatmap,
    plot_group_bars,
    plot_gap_analysis,
    plot_domain_distances,
    plot_subcategory_coverage,
)


def write_findings(
    group_matrix: pd.DataFrame,
    gap_df: pd.DataFrame,
    overall_distances: dict,
    cat_distances: dict,
) -> str:
    """Generate the 3-5 concrete findings as a text summary."""
    lines = [
        "=" * 60,
        "FINDINGS: World Model Data Coverage Analysis",
        "=" * 60,
        "",
    ]

    finding_num = 1

    # Finding 1: Overall coverage comparison
    lines.append(f"Finding {finding_num}: Overall Coverage Distribution")
    finding_num += 1
    for grp in group_matrix.index:
        mean_cov = group_matrix.loc[grp].mean()
        best_cat = group_matrix.loc[grp].idxmax()
        worst_cat = group_matrix.loc[grp].idxmin()
        best_label = CATEGORY_INFO.get(best_cat, {}).get("label", best_cat)
        worst_label = CATEGORY_INFO.get(worst_cat, {}).get("label", worst_cat)
        lines.append(
            f"  {grp}: mean coverage {mean_cov:.1%}, "
            f"strongest in {best_label} ({group_matrix.loc[grp, best_cat]:.1%}), "
            f"weakest in {worst_label} ({group_matrix.loc[grp, worst_cat]:.1%})"
        )
    lines.append("")

    # Finding 2: Specific gaps
    if not gap_df.empty:
        lines.append(f"Finding {finding_num}: Categories Where Robot Data Is Thin")
        finding_num += 1
        for _, row in gap_df.iterrows():
            cat_label = CATEGORY_INFO.get(row["category"], {}).get("label", row["category"])
            lines.append(
                f"  {cat_label}: robot {row['robot_coverage']:.1%}, "
                f"ego {row['ego_coverage']:.1%}, sim {row['sim_coverage']:.1%} "
                f"→ best supplement: {row['best_supplement']}"
            )
        lines.append("")

    # Finding 3: Domain gap
    if overall_distances:
        lines.append(f"Finding {finding_num}: Visual Domain Gap (DINOv2 cosine distance)")
        finding_num += 1
        for pair, dist in sorted(overall_distances.items()):
            lines.append(f"  {pair}: {dist:.3f}")
        lines.append("")

    # Finding 4: Per-category domain gap for thin categories
    if cat_distances and not gap_df.empty:
        lines.append(f"Finding {finding_num}: Domain Gap for Thin Categories")
        finding_num += 1
        for _, row in gap_df.iterrows():
            cat = row["category"]
            cat_label = CATEGORY_INFO.get(cat, {}).get("label", cat)
            dists = cat_distances.get(cat, {})
            if dists:
                parts = [f"{k}: {v:.3f}" for k, v in dists.items()]
                lines.append(f"  {cat_label}: {', '.join(parts)}")
                # Flag high-distance supplements
                best_src = row["best_supplement"]
                dist_key = f"robot_vs_{best_src}"
                if dist_key in dists and dists[dist_key] > 0.4:
                    lines.append(
                        f"    ⚠ High domain gap ({dists[dist_key]:.3f}) — "
                        f"coverage from {best_src} may transfer poorly"
                    )
        lines.append("")

    # Finding 5: Remediation recommendations
    lines.append(f"Finding {finding_num}: Gap → Remedy Mapping")
    finding_num += 1
    if not gap_df.empty:
        for _, row in gap_df.iterrows():
            cat = row["category"]
            cat_label = CATEGORY_INFO.get(cat, {}).get("label", cat)
            best = row["best_supplement"]
            dist = cat_distances.get(cat, {}).get(f"robot_vs_{best}", 0)

            if best == "sim" and dist < 0.3:
                remedy = "Sim generation (low domain gap, scalable)"
            elif best == "ego" and dist < 0.3:
                remedy = "Egocentric data (low domain gap, free labels from narrations)"
            elif best == "sim":
                remedy = "Sim + domain randomization (high domain gap needs bridging)"
            elif best == "ego":
                remedy = "Egocentric data + visual adaptation or targeted teleop collection"
            else:
                remedy = "Targeted teleop collection (no good supplement available)"

            lines.append(f"  {cat_label}: {remedy}")
    else:
        lines.append("  No significant gaps found — robot data covers all categories.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analysis: coverage analysis and deliverables")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--tags-file", type=str, default=None,
                        help="Path to merged tags parquet (default: merged_tags.parquet)")
    args = parser.parse_args()

    # Load tags
    tags_path = Path(args.tags_file) if args.tags_file else TAGS_DIR / "merged_tags.parquet"
    if not tags_path.exists():
        # Fall back to cheap signal tags
        tags_path = TAGS_DIR / "cheap_tags.parquet"
    if not tags_path.exists():
        print(f"ERROR: No tags file found. Run sampling/tagging first.")
        sys.exit(1)

    tags_df = pd.read_parquet(tags_path)
    print(f"Loaded {len(tags_df)} clips from {tags_path.name}")

    # ── Coverage matrix ────────────────────────────────────────────
    print("\n── Building coverage matrices ──")
    source_matrix = build_coverage_matrix(tags_df, normalize="per_clip")
    group_matrix = build_group_coverage_matrix(tags_df, normalize="per_clip")
    sub_df = subcategory_coverage(tags_df)

    print("\nSource-level matrix:")
    print(source_matrix.to_string())
    print("\nGroup-level matrix:")
    print(group_matrix.to_string())

    # Save matrices
    source_matrix.to_csv(RESULTS_DIR / "coverage_matrix_source.csv")
    group_matrix.to_csv(RESULTS_DIR / "coverage_matrix_group.csv")
    sub_df.to_csv(RESULTS_DIR / "subcategory_coverage.csv", index=False)

    # ── Gap analysis ───────────────────────────────────────────────
    print("\n── Gap analysis ──")
    gap_df = gap_analysis(group_matrix, thin_threshold=0.1)
    if gap_df.empty:
        print("No categories with thin robot coverage (<10%)")
    else:
        print(gap_df.to_string(index=False))
    gap_df.to_csv(RESULTS_DIR / "gap_analysis.csv", index=False)

    # ── Embeddings & domain gap ────────────────────────────────────
    overall_distances = {}
    cat_distances = {}

    if not args.skip_embeddings:
        print("\n── Computing DINOv2 embeddings ──")
        embeddings = embed_all_clips(
            clip_ids=tags_df["clip_id"].tolist(),
            sources=tags_df["source"].tolist(),
        )

        clip_groups = dict(zip(tags_df["clip_id"], tags_df["source_group"]))
        overall_distances = compute_domain_distances(embeddings, clip_groups)
        print(f"Overall domain distances: {overall_distances}")

        cat_distances = {}
        for cat in CATEGORIES:
            dists = per_category_domain_gap(embeddings, tags_df, cat)
            if dists:
                cat_distances[cat] = dists
        print(f"Per-category distances computed for {len(cat_distances)} categories")
    else:
        print("\n── Skipping embeddings (--skip-embeddings) ──")

    # ── Plots ──────────────────────────────────────────────────────
    print("\n── Generating plots ──")
    plot_coverage_heatmap(source_matrix, title="Per-Source Coverage Matrix")
    plot_coverage_heatmap(group_matrix, title="Per-Group Coverage Matrix",
                          filename="coverage_heatmap_group")
    plot_group_bars(group_matrix)
    plot_gap_analysis(gap_df)
    plot_subcategory_coverage(sub_df)
    if overall_distances:
        plot_domain_distances(overall_distances, cat_distances)

    # ── Findings ───────────────────────────────────────────────────
    print("\n── Writing findings ──")
    findings = write_findings(group_matrix, gap_df, overall_distances, cat_distances)
    findings_path = RESULTS_DIR / "findings.txt"
    findings_path.write_text(findings)
    print(findings)
    print(f"\nFindings saved to {findings_path}")

    # ── Summary of deliverables ────────────────────────────────────
    print("\n" + "=" * 60)
    print("DELIVERABLES")
    print("=" * 60)
    print(f"  Coverage matrix:     {RESULTS_DIR}/coverage_matrix_*.csv")
    print(f"  Gap analysis:        {RESULTS_DIR}/gap_analysis.csv")
    print(f"  Subcategory detail:  {RESULTS_DIR}/subcategory_coverage.csv")
    print(f"  Findings:            {RESULTS_DIR}/findings.txt")
    print(f"  Plots:               {PLOTS_DIR}/")
    print(f"  Tags (raw):          {TAGS_DIR}/")
    print(f"  Validation template: results/audit/annotation_sheet.json")


if __name__ == "__main__":
    main()
