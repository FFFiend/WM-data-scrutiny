"""
DINOv2 embedding extraction and domain-gap analysis.

Embed each clip's frames with DINOv2 (via Modal), average per clip,
then measure inter-source distances.  Coverage that only exists in a
very different visual domain is weaker coverage.
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import DINO_EMBED_DIM, EMBEDDINGS_DIR, FRAMES_DIR


def embed_all_clips(
    clip_ids: list[str],
    sources: list[str],
    cache: bool = True,
) -> dict[str, np.ndarray]:
    """
    Embed all clips via Modal. Returns {clip_id: embedding}.
    Caches to EMBEDDINGS_DIR if cache=True.
    """
    from modal_client import get_embedder

    embedder = get_embedder()
    results = {}

    for clip_id, source in tqdm(zip(clip_ids, sources), total=len(clip_ids), desc="Embedding"):
        cache_path = EMBEDDINGS_DIR / f"{clip_id}.npy"
        if cache and cache_path.exists():
            results[clip_id] = np.load(cache_path)
            continue

        frame_dir = FRAMES_DIR / source / clip_id
        if not frame_dir.exists():
            results[clip_id] = np.zeros(DINO_EMBED_DIM, dtype=np.float32)
            continue

        frame_paths = sorted(frame_dir.glob("frame_*.png"))
        if not frame_paths:
            results[clip_id] = np.zeros(DINO_EMBED_DIM, dtype=np.float32)
            continue

        emb = embedder.embed_clip(frame_paths)
        results[clip_id] = emb

        if cache:
            np.save(cache_path, emb)

    return results


def compute_domain_distances(
    embeddings: dict[str, np.ndarray],
    clip_groups: dict[str, str],
) -> dict[str, float]:
    """
    Compute mean cosine distance between source groups.

    Returns dict like:
        {"robot_vs_ego": 0.35, "robot_vs_sim": 0.52, "ego_vs_sim": 0.41}
    """
    from scipy.spatial.distance import cosine

    group_embs: dict[str, list[np.ndarray]] = {"robot": [], "ego": [], "sim": []}
    for cid, emb in embeddings.items():
        grp = clip_groups.get(cid, "")
        if grp in group_embs:
            group_embs[grp].append(emb)

    centroids = {}
    for grp, embs in group_embs.items():
        if embs:
            centroids[grp] = np.mean(embs, axis=0)

    distances = {}
    pairs = [("robot", "ego"), ("robot", "sim"), ("ego", "sim")]
    for a, b in pairs:
        if a in centroids and b in centroids:
            distances[f"{a}_vs_{b}"] = float(cosine(centroids[a], centroids[b]))

    return distances


def per_category_domain_gap(
    embeddings: dict[str, np.ndarray],
    tags_df,
    category: str,
) -> dict[str, float]:
    """
    For a specific category, compute the embedding distance between
    robot clips that have this category and ego/sim clips that have it.
    """
    from scipy.spatial.distance import cosine

    col = f"{category}_any"
    if col not in tags_df.columns:
        return {}

    positive = tags_df[tags_df[col] == True]

    group_embs: dict[str, list[np.ndarray]] = {"robot": [], "ego": [], "sim": []}
    for _, row in positive.iterrows():
        cid = row["clip_id"]
        grp = row["source_group"]
        if cid in embeddings and grp in group_embs:
            emb = embeddings[cid]
            if np.linalg.norm(emb) > 0:
                group_embs[grp].append(emb)

    centroids = {}
    for grp, embs in group_embs.items():
        if embs:
            centroids[grp] = np.mean(embs, axis=0)

    distances = {}
    for other in ["ego", "sim"]:
        if "robot" in centroids and other in centroids:
            distances[f"robot_vs_{other}"] = float(cosine(centroids["robot"], centroids[other]))

    return distances
