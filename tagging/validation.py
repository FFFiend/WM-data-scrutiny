"""
Hand-label validation harness.

Sample 50 clips stratified by category (~6 per category), present
them for manual annotation, then compute per-category precision
and recall of the VLM tagger against the hand labels.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from config import TAGS_DIR, FRAMES_DIR, AUDIT_DIR, HAND_LABEL_COUNT, HAND_LABELS_PER_CATEGORY
from taxonomy import CATEGORIES, CATEGORY_INFO


def select_validation_set(
    tags_df: pd.DataFrame,
    n_total: int = HAND_LABEL_COUNT,
    n_per_cat: int = HAND_LABELS_PER_CATEGORY,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Select a stratified validation set ensuring minimum representation
    of each category.

    Strategy:
    1. For each category, pick n_per_cat clips that have that category
       active (according to VLM tags).
    2. Fill remaining slots randomly from the full set.
    3. Deduplicate.
    """
    rng = random.Random(seed)
    selected_ids: set[str] = set()

    for cat in CATEGORIES:
        col = f"{cat}_any"
        if col not in tags_df.columns:
            continue
        positive = tags_df[tags_df[col] == True]
        if len(positive) == 0:
            # Also try to find clips that *might* have this category
            # (ensures we don't miss categories entirely)
            continue
        sample_n = min(n_per_cat, len(positive))
        sampled = positive.sample(n=sample_n, random_state=seed)
        selected_ids.update(sampled["clip_id"].tolist())

    # Fill remaining slots
    remaining = n_total - len(selected_ids)
    if remaining > 0:
        available = tags_df[~tags_df["clip_id"].isin(selected_ids)]
        if len(available) > 0:
            extra = available.sample(n=min(remaining, len(available)), random_state=seed)
            selected_ids.update(extra["clip_id"].tolist())

    return tags_df[tags_df["clip_id"].isin(selected_ids)].copy()


def generate_annotation_sheet(val_df: pd.DataFrame) -> Path:
    """
    Generate a JSON annotation template for the validation set.

    Each entry has the clip_id, source, frame paths, and empty fields
    for each taxonomy category that the annotator fills in.
    """
    entries = []
    for _, row in val_df.iterrows():
        clip_id = row["clip_id"]
        source = row["source"]
        frame_dir = FRAMES_DIR / source / clip_id
        frame_paths = sorted(frame_dir.glob("frame_*.png")) if frame_dir.exists() else []

        entry = {
            "clip_id": clip_id,
            "source": source,
            "source_group": row.get("source_group", ""),
            "frame_paths": [str(p) for p in frame_paths],
            "vlm_tags": {cat: row.get(cat, "") for cat in CATEGORIES},
            "hand_labels": {cat: [] for cat in CATEGORIES},
            "annotator_notes": "",
        }
        entries.append(entry)

    out_path = AUDIT_DIR / "annotation_sheet.json"
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Annotation sheet saved to {out_path}")
    print(f"  {len(entries)} clips to annotate")
    print(f"  Fill in 'hand_labels' for each clip, then run validation.")
    return out_path


def compute_validation_metrics(
    annotation_path: Path | str,
) -> pd.DataFrame:
    """
    Compute per-category precision and recall of VLM tags
    against hand labels.

    Returns DataFrame with columns:
        category, precision, recall, f1, n_positive_hand, n_positive_vlm
    """
    with open(annotation_path) as f:
        entries = json.load(f)

    results = []
    for cat in CATEGORIES:
        tp = fp = fn = 0
        for entry in entries:
            vlm_subs = set(entry["vlm_tags"].get(cat, "").split("|")) - {""}
            hand_subs = set(entry["hand_labels"].get(cat, []))

            # Binary: does this category have any active subcategory?
            vlm_positive = len(vlm_subs) > 0
            hand_positive = len(hand_subs) > 0

            if vlm_positive and hand_positive:
                tp += 1
            elif vlm_positive and not hand_positive:
                fp += 1
            elif not vlm_positive and hand_positive:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            "category": cat,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "n_positive_hand": tp + fn,
            "n_positive_vlm": tp + fp,
            "tp": tp, "fp": fp, "fn": fn,
        })

    df = pd.DataFrame(results)

    # Also compute sub-category level metrics
    sub_results = []
    for cat in CATEGORIES:
        all_subs = CATEGORY_INFO[cat]["subcategories"]
        for sub in all_subs:
            tp = fp = fn = 0
            for entry in entries:
                vlm_subs = set(entry["vlm_tags"].get(cat, "").split("|")) - {""}
                hand_subs = set(entry["hand_labels"].get(cat, []))

                vlm_has = sub in vlm_subs
                hand_has = sub in hand_subs

                if vlm_has and hand_has:
                    tp += 1
                elif vlm_has and not hand_has:
                    fp += 1
                elif not vlm_has and hand_has:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            sub_results.append({
                "category": cat,
                "subcategory": sub,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "tp": tp, "fp": fp, "fn": fn,
            })

    sub_df = pd.DataFrame(sub_results)

    # Save both
    df.to_csv(AUDIT_DIR / "validation_category.csv", index=False)
    sub_df.to_csv(AUDIT_DIR / "validation_subcategory.csv", index=False)
    print(f"\nValidation metrics saved to {AUDIT_DIR}/")
    print(df.to_string(index=False))

    return df
