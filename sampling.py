"""
Clip sampling: draw CLIPS_PER_SOURCE clips from each data source,
extract VLM_FRAMES_PER_CLIP uniformly-spaced frames, and persist
both metadata and frame images to disk.

Each LeRobot/HF dataset stores per-frame rows with episode_index.
We group by episode, randomly sample episodes, and treat each
episode as one clip.
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from config import (
    CLIPS_PER_SOURCE,
    VLM_FRAMES_PER_CLIP,
    SOURCES,
    RAW_DIR,
    FRAMES_DIR,
)


@dataclass
class ClipRecord:
    clip_id: str
    source: str
    source_group: str
    n_frames: int
    fps: float
    duration_s: float
    language: str | None
    has_proprioception: bool
    proprioception_keys: list[str]
    video_path: str | None
    metadata: dict


def uniform_frame_indices(n_frames: int, k: int) -> list[int]:
    """Return k uniformly-spaced frame indices from [0, n_frames)."""
    if n_frames <= k:
        return list(range(n_frames))
    return [int(round(i * (n_frames - 1) / (k - 1))) for i in range(k)]


def extract_frames(video_path: str, indices: list[int]) -> list[np.ndarray]:
    """Read specific frame indices from a video file."""
    if not indices:
        return []
    cap = cv2.VideoCapture(video_path)
    frames = []
    max_idx = max(indices)
    idx_set = set(indices)
    frame_i = 0
    while cap.isOpened() and frame_i <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_i in idx_set:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_i += 1
    cap.release()
    return frames


def save_clip_frames(clip: ClipRecord, frames: list[np.ndarray]) -> Path:
    """Save extracted frames as PNGs and return the clip's frame directory."""
    clip_dir = FRAMES_DIR / clip.source / clip.clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        out_path = clip_dir / f"frame_{i:03d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    meta_path = clip_dir / "clip_meta.json"
    meta_path.write_text(json.dumps(asdict(clip), indent=2, default=str))
    return clip_dir


# ────────────────────────────────────────────────────────────────────
# Source-specific loaders
# ────────────────────────────────────────────────────────────────────

def _load_lerobot_hf(source_name: str, cfg: dict) -> Iterator[ClipRecord]:
    """
    Load clips from a LeRobot v3 HuggingFace dataset.

    LeRobot v3 stores all episodes in one concatenated video per chunk.
    We download the parquet to get global frame indices, then record the
    video path + frame offset so extract_frames_range() can seek correctly.
    """
    from huggingface_hub import hf_hub_download
    import json as _json
    import pandas as pd

    hub_id = cfg["hub_id"]

    # Get dataset info
    info_path = hf_hub_download(hub_id, "meta/info.json", repo_type="dataset")
    with open(info_path) as f:
        ds_info = _json.load(f)
    fps = ds_info.get("fps", 15)
    video_path_template = ds_info.get("video_path", "")
    chunks_size = ds_info.get("chunks_size", 1000)

    # Get episode metadata
    episodes = []
    for fmt in ["meta/episodes.jsonl", "meta/episodes.parquet"]:
        try:
            ep_path = hf_hub_download(hub_id, fmt, repo_type="dataset")
            if fmt.endswith(".jsonl"):
                with open(ep_path) as f:
                    episodes = [_json.loads(line) for line in f]
            else:
                episodes = pd.read_parquet(ep_path).to_dict("records")
            break
        except Exception:
            continue
    if not episodes:
        episodes = [{"episode_index": i, "length": 100}
                    for i in range(ds_info.get("total_episodes", 0))]

    # Get task metadata
    tasks = {}
    for fmt in ["meta/tasks.jsonl", "meta/tasks.parquet"]:
        try:
            tp = hf_hub_download(hub_id, fmt, repo_type="dataset")
            if fmt.endswith(".jsonl"):
                with open(tp) as f:
                    for line in f:
                        t = _json.loads(line)
                        tasks[t["task_index"]] = t.get("task", "")
            else:
                tdf = pd.read_parquet(tp)
                for _, row in tdf.iterrows():
                    idx = row.get("task_index", row.name)
                    tasks[idx] = str(row.iloc[0]) if "task" not in row else row["task"]
            break
        except Exception:
            continue

    # Detect format version
    version = ds_info.get("codebase_version", "v3.0")
    is_v2 = version.startswith("v2")

    # For v3 (concatenated videos): download parquet to get global frame offsets
    episode_offsets = {}
    if not is_v2:
        try:
            data_path_template = ds_info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
            parquet_path = hf_hub_download(
                hub_id,
                data_path_template.format(chunk_index=0, file_index=0),
                repo_type="dataset",
            )
            df = pd.read_parquet(parquet_path, columns=["episode_index", "index"])
            for ep_idx, grp in df.groupby("episode_index"):
                episode_offsets[ep_idx] = (int(grp["index"].min()), int(grp["index"].max()))
        except Exception as e:
            print(f"    [WARN] Could not load parquet for frame offsets: {e}")

    # Determine preferred video key
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

    # For v3 concatenated videos: cache downloaded chunk files
    chunk_videos = {}  # chunk_idx → local path

    # Sample episodes
    n_sample = min(CLIPS_PER_SOURCE, len(episodes))
    sampled = random.sample(episodes, n_sample)

    for ep in sampled:
        ep_idx = ep["episode_index"]
        ep_length = ep.get("length", 100)
        clip_id = f"{source_name}_{ep_idx:06d}"
        chunk_idx = ep_idx // chunks_size

        # Resolve task text
        ep_tasks = ep.get("tasks", [])
        language = None
        if ep_tasks:
            language = ep_tasks[0] if ep_tasks[0] else None
        if not language and tasks:
            language = tasks.get(0, None)

        # Build the video's HF relative path (downloaded lazily in the runner)
        video_rel_path = None
        if preferred and video_path_template:
            if is_v2:
                video_rel_path = video_path_template.format(
                    video_key=preferred,
                    episode_chunk=chunk_idx,
                    episode_index=ep_idx,
                )
            else:
                video_rel_path = video_path_template.format(
                    video_key=preferred,
                    chunk_index=chunk_idx,
                    file_index=0,
                )
        video_path = None  # downloaded lazily

        # Get global frame range (only matters for v3 concatenated videos)
        start_frame = 0  # v2: each video starts at 0
        end_frame = ep_length - 1
        if not is_v2:
            if ep_idx in episode_offsets:
                start_frame, end_frame = episode_offsets[ep_idx]
            else:
                start_frame = sum(e.get("length", 100) for e in episodes if e["episode_index"] < ep_idx)
                end_frame = start_frame + ep_length - 1

        yield ClipRecord(
            clip_id=clip_id,
            source=source_name,
            source_group=cfg["group"],
            n_frames=ep_length,
            fps=float(fps),
            duration_s=ep_length / fps if fps else 0.0,
            language=language,
            has_proprioception=cfg.get("has_proprioception", False),
            proprioception_keys=["observation.state", "action"],
            video_path=video_path,
            metadata={
                "hub_id": hub_id,
                "episode_index": ep_idx,
                "video_key": preferred,
                "video_rel_path": video_rel_path,
                "global_start_frame": start_frame,
                "global_end_frame": end_frame,
                "format_version": version,
            },
        )


def _load_epic_kitchens(source_name: str, cfg: dict) -> Iterator[ClipRecord]:
    """
    Load EPIC-KITCHENS-100 clips.
    Expects annotation CSVs and video segments under RAW_DIR/epic_kitchens/.
    """
    import csv

    ann_path = RAW_DIR / "epic_kitchens" / "EPIC_100_train.csv"
    video_dir = RAW_DIR / "epic_kitchens" / "videos"

    if not ann_path.exists():
        print(f"  [SKIP] {ann_path} not found. "
              "Download from https://github.com/epic-kitchens/epic-kitchens-100-annotations "
              "and place EPIC_100_train.csv under data/raw/epic_kitchens/")
        return

    rows = []
    with open(ann_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    sampled = random.sample(rows, min(CLIPS_PER_SOURCE, len(rows)))
    for i, row in enumerate(sampled):
        narration = row.get("narration", "")
        verb = row.get("verb", "")
        noun = row.get("noun", "")
        vid_id = row.get("video_id", "")
        start_frame = int(row.get("start_frame", 0))
        stop_frame = int(row.get("stop_frame", start_frame + 1))
        n_frames = stop_frame - start_frame
        fps = 60.0

        yield ClipRecord(
            clip_id=f"epic_{i:05d}",
            source=source_name,
            source_group="ego",
            n_frames=n_frames,
            fps=fps,
            duration_s=n_frames / fps,
            language=narration or f"{verb} {noun}",
            has_proprioception=False,
            proprioception_keys=[],
            video_path=str(video_dir / f"{vid_id}.MP4") if video_dir.exists() else None,
            metadata={"verb": verb, "noun": noun, "video_id": vid_id,
                       "start_frame": start_frame, "stop_frame": stop_frame},
        )


def _load_placeholder(source_name: str, cfg: dict) -> Iterator[ClipRecord]:
    """
    Fallback for sources that need custom download/setup.
    Scans RAW_DIR/<source_name>/ for video files.
    """
    local_dir = RAW_DIR / source_name
    if not local_dir.exists():
        print(f"  [SKIP] {local_dir} not found. "
              f"Place raw data under {local_dir}/ to enable {source_name}.")
        return
    videos = sorted(local_dir.glob("**/*.mp4")) + sorted(local_dir.glob("**/*.avi"))
    sampled = random.sample(videos, min(CLIPS_PER_SOURCE, len(videos))) if videos else []
    for i, vpath in enumerate(sampled):
        cap = cv2.VideoCapture(str(vpath))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        yield ClipRecord(
            clip_id=f"{source_name}_{i:05d}",
            source=source_name,
            source_group=cfg["group"],
            n_frames=n_frames,
            fps=fps,
            duration_s=n_frames / fps if fps else 0.0,
            language=None,
            has_proprioception=cfg.get("has_proprioception", False),
            proprioception_keys=[],
            video_path=str(vpath),
            metadata={},
        )


def _load_robocasa_hf(source_name: str, cfg: dict) -> Iterator[ClipRecord]:
    """
    Load RoboCasa pre-rendered videos from PEARLS-Lab/robocasa-composite-raw-videos.
    71 task folders, each with episode_NNNNNN.mp4 files.
    """
    from huggingface_hub import HfApi, hf_hub_download

    hub_id = cfg["hub_id"]
    api = HfApi()

    # List all task folders
    top_items = list(api.list_repo_tree(hub_id, repo_type="dataset", recursive=False))
    task_folders = [
        item.rpath if hasattr(item, "rpath") else item.path
        for item in top_items
        if not str(item.rpath if hasattr(item, "rpath") else item.path).startswith(".")
        and not str(item.rpath if hasattr(item, "rpath") else item.path).endswith((".md", ".txt", ".json"))
    ]

    # Collect all available (task, episode) pairs
    all_videos = []
    for task_folder in task_folders:
        try:
            files = list(api.list_repo_tree(hub_id, path_in_repo=task_folder, repo_type="dataset"))
            mp4s = [f for f in files if str(f.rpath if hasattr(f, "rpath") else f.path).endswith(".mp4")]
            for f in mp4s:
                fpath = f.rpath if hasattr(f, "rpath") else f.path
                all_videos.append((task_folder, str(fpath)))
        except Exception:
            continue

    if not all_videos:
        print(f"  [SKIP] No videos found in {hub_id}")
        return

    # Sample
    n_sample = min(CLIPS_PER_SOURCE, len(all_videos))
    sampled = random.sample(all_videos, n_sample)

    for i, (task_name, rel_path) in enumerate(sampled):
        clip_id = f"robocasa_{i:05d}"

        # Download video
        video_path = None
        try:
            video_path = hf_hub_download(hub_id, rel_path, repo_type="dataset")
        except Exception as e:
            print(f"    [WARN] Could not download {rel_path}: {e}")
            continue

        # Get video info
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # Task name is the language label (e.g. "AddIceCubes" → "Add Ice Cubes")
        import re as _re
        language = _re.sub(r'([A-Z])', r' \1', task_name).strip()

        yield ClipRecord(
            clip_id=clip_id,
            source=source_name,
            source_group="sim",
            n_frames=n_frames,
            fps=fps,
            duration_s=n_frames / fps if fps else 0.0,
            language=language,
            has_proprioception=False,
            proprioception_keys=[],
            video_path=video_path,
            metadata={
                "hub_id": hub_id,
                "task": task_name,
                "rel_path": rel_path,
                "global_start_frame": 0,
                "global_end_frame": n_frames - 1,
            },
        )


# ── Lazy video download ─────────────────────────────────────────────

def download_clip_video(clip: ClipRecord) -> str | None:
    """
    Download the video for a clip if not already on disk.
    Returns local path or None.
    """
    if clip.video_path and Path(clip.video_path).exists():
        return clip.video_path

    hub_id = clip.metadata.get("hub_id")
    rel_path = clip.metadata.get("video_rel_path") or clip.metadata.get("rel_path")
    if not hub_id or not rel_path:
        return clip.video_path  # nothing to download

    from huggingface_hub import hf_hub_download
    try:
        local_path = hf_hub_download(hub_id, rel_path, repo_type="dataset")
        clip.video_path = local_path
        return local_path
    except Exception as e:
        print(f"    [WARN] Could not download video for {clip.clip_id}: {e}")
        return None


# ── Gripper state extraction from LeRobot data ─────────────────────

def extract_gripper_states(hub_id: str, episode_index: int) -> np.ndarray | None:
    """
    Extract the gripper (last element of observation.state) for an episode.
    LeRobot convention: state is [joint0..jointN, gripper] where gripper
    is typically the last DOF.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(hub_id, split="train", streaming=True)
        states = []
        for row in ds:
            if row["episode_index"] != episode_index:
                if states:  # already past the episode
                    break
                continue
            state = row.get("observation.state")
            if state:
                states.append(state[-1])  # last element = gripper

        if states:
            arr = np.array(states, dtype=np.float32)
            # Normalize to [0, 1] if not already
            if arr.max() > 1.0:
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            return arr
    except Exception as e:
        print(f"    [WARN] Could not extract gripper states: {e}")
    return None


# ── Loader dispatch ─────────────────────────────────────────────────
LOADERS = {
    "lerobot_hf": _load_lerobot_hf,
    "epic_kitchens": _load_epic_kitchens,
    "ego4d": _load_placeholder,
    "hot3d": _load_placeholder,
    "maniskill3": _load_placeholder,
    "robocasa_hf": _load_robocasa_hf,
    "behavior_1k": _load_placeholder,
}


def sample_all_sources(seed: int = 42) -> list[ClipRecord]:
    """Sample clips from every configured source."""
    random.seed(seed)
    all_clips = []
    for name, cfg in SOURCES.items():
        loader_key = cfg["loader"]
        loader_fn = LOADERS.get(loader_key)
        if loader_fn is None:
            print(f"  [WARN] No loader for {loader_key}, skipping {name}")
            continue
        print(f"  Loading {name}...")
        clips = list(loader_fn(name, cfg))
        print(f"    {name}: sampled {len(clips)} clips")
        all_clips.extend(clips)
    print(f"Total: {len(all_clips)} clips from {len(SOURCES)} sources")
    return all_clips
