#!/usr/bin/env python3
"""
Sampling runner: sample clips, extract cheap signals, persist tags.

Usage:
    python scripts/run_sampling.py [--sources droid,bridge_v2] [--seed 42]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import SOURCES, TAGS_DIR, FRAMES_DIR, CLIPS_PER_SOURCE
from taxonomy import ClipTags, CATEGORIES
from sampling import sample_all_sources, uniform_frame_indices, extract_frames, save_clip_frames, download_clip_video
from signals.cheap_signals import compute_cheap_tags


def main():
    parser = argparse.ArgumentParser(description="Sampling: sample clips and extract cheap signals")
    parser.add_argument("--sources", type=str, default=None,
                        help="Comma-separated source names (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Override CLIPS_PER_SOURCE for quick testing")
    args = parser.parse_args()

    # Sample clips
    print("=" * 60)
    print("Sampling clips and extracting cheap signals")
    print("=" * 60)

    clips = sample_all_sources(seed=args.seed)

    if args.sources:
        keep = set(args.sources.split(","))
        clips = [c for c in clips if c.source in keep]
        print(f"Filtered to {len(clips)} clips from: {keep}")

    if args.max_clips:
        from itertools import groupby
        from operator import attrgetter
        filtered = []
        clips_sorted = sorted(clips, key=attrgetter("source"))
        for src, group in groupby(clips_sorted, key=attrgetter("source")):
            filtered.extend(list(group)[:args.max_clips])
        clips = filtered
        print(f"Capped to {args.max_clips} per source → {len(clips)} total")

    # Process each clip
    all_tags = []
    for i, clip in enumerate(clips):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing clip {i+1}/{len(clips)}: {clip.clip_id}")

        # Download video lazily (only for clips that survived filtering)
        download_clip_video(clip)

        # Extract frames if video exists
        frames = None
        if clip.video_path and Path(clip.video_path).exists():
            # LeRobot v3: videos are concatenated per chunk, so offset by global_start_frame
            global_start = clip.metadata.get("global_start_frame", 0)

            # 16 frames for dynamics signals (denser)
            local_indices = uniform_frame_indices(clip.n_frames, k=16)
            global_indices = [global_start + i for i in local_indices]
            frames = extract_frames(clip.video_path, global_indices)

            # 8 frames for VLM tagging (save to disk)
            vlm_local = uniform_frame_indices(clip.n_frames, k=8)
            vlm_global = [global_start + i for i in vlm_local]
            vlm_frames = extract_frames(clip.video_path, vlm_global)
            if vlm_frames:
                save_clip_frames(clip, vlm_frames)

        # Extract gripper states from proprioception
        # Skip for IPEC datasets (bridge_v2, fractal, kuka) — streaming through
        # the full dataset to find one episode is too slow. Only use for droid
        # which has a smaller dataset (droid_100).
        gripper_states = None
        if clip.has_proprioception and clip.metadata.get("hub_id") and clip.source == "droid":
            try:
                from sampling import extract_gripper_states
                gripper_states = extract_gripper_states(
                    clip.metadata["hub_id"],
                    clip.metadata.get("episode_index", 0),
                )
            except Exception as e:
                pass  # gripper extraction is best-effort

        # Sim contacts — only available if source has physics engine exports
        sim_contacts = None

        # Compute cheap tags
        tags_dict = compute_cheap_tags(
            clip=clip,
            gripper_states=gripper_states,
            sim_contacts=sim_contacts,
            frames=frames,
        )

        # Build ClipTags
        ct = ClipTags(
            clip_id=clip.clip_id,
            source=clip.source,
            source_group=clip.source_group,
        )
        for cat in CATEGORIES:
            if cat in tags_dict:
                setattr(ct, cat, tags_dict[cat])
        all_tags.append(ct)

    # Persist
    rows = [t.to_row() for t in all_tags]
    df = pd.DataFrame(rows)
    out_path = TAGS_DIR / "cheap_tags.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} clip tags to {out_path}")

    # Summary
    print("\n── Coverage summary (cheap signals) ──")
    for cat in CATEGORIES:
        col = f"{cat}_any"
        if col in df.columns:
            by_group = df.groupby("source_group")[col].mean()
            print(f"  {cat}:")
            for grp, frac in by_group.items():
                print(f"    {grp}: {frac:.1%}")

    # Also save raw JSON for debugging
    json_path = TAGS_DIR / "cheap_tags.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"  (debug JSON: {json_path})")


if __name__ == "__main__":
    main()
