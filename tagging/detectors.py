"""
Perception-based detectors for specific categories.

1. Hand-object contact detection (100DOH) for egocentric clips.
2. Grounding DINO for deformable and articulated object detection.
3. Scene complexity via object counting.

Grounding DINO runs on Modal. Hand-object contact uses a lightweight
local heuristic (skin detection) that doesn't need a GPU.
"""

from pathlib import Path

import cv2
import numpy as np


# ── 1. Hand-object contact detection (egocentric clips) ────────────

def detect_hand_object_contact(
    frame_paths: list[str | Path],
    contact_threshold: float = 0.5,
) -> dict[str, list[str]]:
    """
    Run hand-object contact detection on a set of frames.

    Uses a simplified approach: detect hands (via MediaPipe or a hand
    detector) and objects, then check spatial overlap / proximity.

    For production, swap in 100DOH (https://github.com/ddshan/hand_object_detector).
    Here we provide the skeleton and a lighter fallback.

    Returns:
        {"contact": [...], "embodiment": ["gripper_vs_hand:hand"]}
    """
    tags: dict[str, list[str]] = {}
    contact_frames = 0

    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        skin_ratio = skin_mask.sum() / (255.0 * skin_mask.size)
        if skin_ratio > 0.02:
            tags.setdefault("embodiment", [])
            if "gripper_vs_hand:hand" not in tags.get("embodiment", []):
                tags["embodiment"] = tags.get("embodiment", []) + ["gripper_vs_hand:hand"]
            h = skin_mask.shape[0]
            lower_skin_ratio = skin_mask[h // 2 :].sum() / (255.0 * skin_mask[h // 2 :].size)
            if lower_skin_ratio > 0.03:
                contact_frames += 1

    if contact_frames > len(frame_paths) * contact_threshold:
        tags.setdefault("contact", [])
        tags["contact"].append("grasp")

    return tags


# ── 2. Grounding DINO via Modal ─────────────────────────────────────

DEFAULT_QUERIES = [
    "cloth", "towel", "rope", "cable", "sponge",  # deformable
    "drawer", "door", "cabinet", "lid", "faucet",  # articulated
    "cup", "bottle", "bowl", "plate", "box",       # rigid (for counting)
]

DEFORMABLE_NAMES = {"cloth", "towel", "rope", "cable", "sponge", "dough"}
ARTICULATED_NAMES = {"drawer", "door", "cabinet", "lid", "faucet", "hinge"}


def detect_objects_grounding_dino(
    frame_paths: list[str | Path],
    text_queries: list[str] | None = None,
    confidence_threshold: float = 0.25,
    detector_client=None,
) -> dict[str, list[str]]:
    """
    Use Grounding DINO (via Modal) to detect object types and count.
    """
    from modal_client import get_detector

    if text_queries is None:
        text_queries = DEFAULT_QUERIES
    if detector_client is None:
        detector_client = get_detector()

    query_text = ". ".join(text_queries) + "."

    per_frame_results = detector_client.detect(
        frame_paths, query_text,
        box_threshold=confidence_threshold,
        text_threshold=confidence_threshold,
    )

    tags: dict[str, list[str]] = {"object_type": [], "scene_complexity": []}
    all_detected: set[str] = set()
    max_objects_in_frame = 0

    for fr in per_frame_results:
        labels = fr.get("labels", [])
        max_objects_in_frame = max(max_objects_in_frame, fr.get("n_objects", len(labels)))
        for label in labels:
            all_detected.add(label.lower().strip())

    if all_detected & DEFORMABLE_NAMES:
        tags["object_type"].append("deformable")
    if all_detected & ARTICULATED_NAMES:
        tags["object_type"].append("articulated")
    if all_detected - DEFORMABLE_NAMES - ARTICULATED_NAMES:
        tags["object_type"].append("rigid")

    if max_objects_in_frame >= 6:
        tags["scene_complexity"] = ["cluttered"]
    elif max_objects_in_frame >= 2:
        tags["scene_complexity"] = ["few_objects"]
    elif max_objects_in_frame >= 1:
        tags["scene_complexity"] = ["single_object"]

    return tags


# ── 3. Combine detector outputs ────────────────────────────────────

def run_detectors(
    frame_paths: list[str | Path],
    source_group: str,
    use_grounding_dino: bool = True,
    detector_client=None,
) -> dict[str, list[str]]:
    """
    Run all relevant detectors for a clip.
    Grounding DINO runs on Modal; hand-contact runs locally.
    """
    tags: dict[str, list[str]] = {}

    def merge(new: dict[str, list[str]]):
        for cat, subs in new.items():
            tags.setdefault(cat, [])
            for s in subs:
                if s not in tags[cat]:
                    tags[cat].append(s)

    if source_group == "ego":
        hoc_tags = detect_hand_object_contact(frame_paths)
        merge(hoc_tags)

    if use_grounding_dino:
        try:
            det_tags = detect_objects_grounding_dino(
                frame_paths, detector_client=detector_client,
            )
            merge(det_tags)
        except Exception as e:
            print(f"  [WARN] Grounding DINO failed: {e}")

    return tags
