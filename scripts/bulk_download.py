#!/usr/bin/env python3
"""
Bulk-download videos for sampled episodes from IPEC LeRobot datasets.
Run overnight — downloads are slow but parallelized.

Usage:
    python scripts/bulk_download.py [--workers 8] [--sources bridge_v2,fractal,kuka]
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download
from config import SOURCES, CLIPS_PER_SOURCE


def download_one(hub_id: str, rel_path: str) -> tuple[str, bool, str]:
    """Download a single file, return (rel_path, success, local_path_or_error)."""
    try:
        local = hf_hub_download(hub_id, rel_path, repo_type="dataset")
        return (rel_path, True, local)
    except Exception as e:
        return (rel_path, False, str(e)[:120])


def get_episode_video_paths(hub_id: str, seed: int = 42) -> list[tuple[str, str]]:
    """
    Return list of (hub_id, video_rel_path) for sampled episodes.
    """
    import json as _json

    info_path = hf_hub_download(hub_id, "meta/info.json", repo_type="dataset")
    with open(info_path) as f:
        ds_info = _json.load(f)

    video_path_template = ds_info.get("video_path", "")
    chunks_size = ds_info.get("chunks_size", 1000)
    version = ds_info.get("codebase_version", "v3.0")

    # Get episodes
    episodes = []
    for fmt in ["meta/episodes.jsonl", "meta/episodes.parquet"]:
        try:
            ep_path = hf_hub_download(hub_id, fmt, repo_type="dataset")
            if fmt.endswith(".jsonl"):
                with open(ep_path) as f:
                    episodes = [_json.loads(line) for line in f]
            else:
                import pandas as pd
                episodes = pd.read_parquet(ep_path).to_dict("records")
            break
        except Exception:
            continue

    if not episodes:
        return []

    # Determine video key
    features = ds_info.get("features", {})
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    preferred = None
    for vk in video_keys:
        if "exterior" in vk and "1" in vk:
            preferred = vk
            break
    if preferred is None:
        for vk in video_keys:
            if "exterior" in vk:
                preferred = vk
                break
    if preferred is None and video_keys:
        preferred = video_keys[0]
    if not preferred:
        return []

    # Sample episodes
    random.seed(seed)
    n_sample = min(CLIPS_PER_SOURCE, len(episodes))
    sampled = random.sample(episodes, n_sample)

    paths = []
    is_v2 = version.startswith("v2")
    seen_chunks = set()

    for ep in sampled:
        ep_idx = ep["episode_index"]
        chunk_idx = ep_idx // chunks_size

        if is_v2:
            rel_path = video_path_template.format(
                video_key=preferred,
                episode_chunk=chunk_idx,
                episode_index=ep_idx,
            )
            paths.append((hub_id, rel_path))
        else:
            # v3: one video per chunk, only download once
            if chunk_idx not in seen_chunks:
                rel_path = video_path_template.format(
                    video_key=preferred,
                    chunk_index=chunk_idx,
                    file_index=0,
                )
                paths.append((hub_id, rel_path))
                seen_chunks.add(chunk_idx)

    return paths


def main():
    parser = argparse.ArgumentParser(description="Bulk download IPEC dataset videos")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sources", type=str, default="bridge_v2,fractal,kuka")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_names = args.sources.split(",")
    all_downloads = []

    for name in source_names:
        cfg = SOURCES.get(name)
        if not cfg or not cfg.get("hub_id"):
            print(f"[SKIP] {name}: no hub_id configured")
            continue

        hub_id = cfg["hub_id"]
        print(f"Resolving episodes for {name} ({hub_id})...")
        paths = get_episode_video_paths(hub_id, seed=args.seed)
        print(f"  {len(paths)} videos to download")
        all_downloads.extend(paths)

    print(f"\nTotal: {len(all_downloads)} files to download with {args.workers} workers")
    print("Starting downloads...\n")

    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, hub_id, rel_path): (hub_id, rel_path)
            for hub_id, rel_path in all_downloads
        }
        for future in as_completed(futures):
            rel_path, success, result = future.result()
            done += 1
            if success:
                if done % 25 == 0 or done == len(all_downloads):
                    print(f"  [{done}/{len(all_downloads)}] downloaded")
            else:
                failed += 1
                print(f"  [{done}/{len(all_downloads)}] FAILED {rel_path}: {result}")

    print(f"\nDone. {done - failed}/{done} succeeded, {failed} failed.")
    print("Now re-run: python scripts/run_sampling.py")


if __name__ == "__main__":
    main()
