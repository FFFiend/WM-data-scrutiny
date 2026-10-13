#!/usr/bin/env python3
"""
Tagging runner: VLM tagging, detector-based tagging, merge, and
generate validation annotation sheet.

Usage:
    python scripts/run_tagging.py [--skip-vlm] [--skip-detectors] [--validate-only annotation_sheet.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import TAGS_DIR, FRAMES_DIR, AUDIT_DIR
from taxonomy import CATEGORIES, ClipTags
from tagging.vlm_tagger import tag_all_clips
from tagging.detectors import run_detectors
from tagging.validation import select_validation_set, generate_annotation_sheet, compute_validation_metrics


def merge_tag_dicts(base: dict, new: dict) -> dict:
    """Merge new tags into base, deduplicating subcategories."""
    merged = dict(base)
    for cat, subs in new.items():
        if cat.startswith("_"):
            continue
        merged.setdefault(cat, [])
        for s in subs:
            if s not in merged[cat]:
                merged[cat].append(s)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Tagging: VLM and detector tagging")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM tagging (use existing)")
    parser.add_argument("--skip-detectors", action="store_true", help="Skip detector-based tagging")
    parser.add_argument("--validate-only", type=str, default=None,
                        help="Only run validation on a filled annotation sheet")
    args = parser.parse_args()

    # Validation-only mode
    if args.validate_only:
        print("Running validation metrics...")
        df = compute_validation_metrics(args.validate_only)
        return

    # Load cheap signal tags
    cheap_tags_path = TAGS_DIR / "cheap_tags.parquet"
    if not cheap_tags_path.exists():
        print(f"ERROR: {cheap_tags_path} not found. Run sampling first.")
        sys.exit(1)

    cheap_df = pd.read_parquet(cheap_tags_path)
    print(f"Loaded {len(cheap_df)} clips from cheap signal tags")

    # ── VLM tagging ────────────────────────────────────────────────
    vlm_tags_path = TAGS_DIR / "vlm_tags.json"
    if args.skip_vlm and vlm_tags_path.exists():
        print("Loading existing VLM tags...")
        with open(vlm_tags_path) as f:
            vlm_results = json.load(f)
    else:
        print("Running VLM tagging (via Modal)...")
        vlm_results = tag_all_clips(
            clip_ids=cheap_df["clip_id"].tolist(),
            sources=cheap_df["source"].tolist(),
        )
        with open(vlm_tags_path, "w") as f:
            json.dump(vlm_results, f, indent=2)
        print(f"  VLM tags saved to {vlm_tags_path}")

        # Report errors
        errors = [r for r in vlm_results if "_error" in r.get("tags", {})]
        if errors:
            print(f"  {len(errors)} clips had VLM errors")

    # ── Detector-based tagging ─────────────────────────────────────
    det_tags_path = TAGS_DIR / "detector_tags.json"
    if args.skip_detectors and det_tags_path.exists():
        print("Loading existing detector tags...")
        with open(det_tags_path) as f:
            det_results = json.load(f)
    else:
        print("Running detector-based tagging...")
        det_results = []
        for _, row in cheap_df.iterrows():
            clip_id = row["clip_id"]
            source = row["source"]
            source_group = row["source_group"]
            frame_dir = FRAMES_DIR / source / clip_id
            frame_paths = sorted(frame_dir.glob("frame_*.png")) if frame_dir.exists() else []

            if not frame_paths:
                det_results.append({"clip_id": clip_id, "source": source, "tags": {}})
                continue

            tags = run_detectors(
                frame_paths=[str(p) for p in frame_paths],
                source_group=source_group,
                use_grounding_dino=True,
            )
            det_results.append({"clip_id": clip_id, "source": source, "tags": tags})

        with open(det_tags_path, "w") as f:
            json.dump(det_results, f, indent=2)
        print(f"  Detector tags saved to {det_tags_path}")

    # ── Merge all tag sources ──────────────────────────────────────
    print("Merging cheap + VLM + detector tags...")
    vlm_lookup = {r["clip_id"]: r.get("tags", {}) for r in vlm_results}
    det_lookup = {r["clip_id"]: r.get("tags", {}) for r in det_results}

    merged_rows = []
    for _, row in cheap_df.iterrows():
        cid = row["clip_id"]
        # Start from cheap signal tags
        base = {}
        for cat in CATEGORIES:
            val = row.get(cat, "")
            base[cat] = val.split("|") if val else []

        # Merge VLM tags
        vlm = vlm_lookup.get(cid, {})
        base = merge_tag_dicts(base, vlm)

        # Merge detector tags
        det = det_lookup.get(cid, {})
        base = merge_tag_dicts(base, det)

        # Build final row
        ct = ClipTags(
            clip_id=cid,
            source=row["source"],
            source_group=row["source_group"],
        )
        for cat in CATEGORIES:
            setattr(ct, cat, base.get(cat, []))
        merged_rows.append(ct.to_row())

    merged_df = pd.DataFrame(merged_rows)
    out_path = TAGS_DIR / "merged_tags.parquet"
    merged_df.to_parquet(out_path, index=False)
    print(f"Merged tags saved to {out_path} ({len(merged_df)} clips)")

    # ── Generate validation set ────────────────────────────────────
    print("\nGenerating validation annotation sheet...")
    val_df = select_validation_set(merged_df)
    generate_annotation_sheet(val_df)
    print(f"\nNext step: hand-label the {len(val_df)} clips in {AUDIT_DIR}/annotation_sheet.json")
    print("Then run: python scripts/run_tagging.py --validate-only results/audit/annotation_sheet.json")


if __name__ == "__main__":
    main()
