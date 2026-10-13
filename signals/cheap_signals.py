"""
Cheap signals — no learned detectors required.

1. Gripper open/close events  (robot proprioception → contact proxy)
2. Sim ground-truth contacts   (physics engine export)
3. Keyword matching on language (regex dictionary)
4. Motion magnitude & idle fraction (frame differencing + optional RAFT)
5. Embodiment metadata         (from OXE / source config)

Each function takes a ClipRecord (and optionally raw data) and returns
partial tag dicts that get merged into the final ClipTags.
"""

import re
from typing import Any

import cv2
import numpy as np

from config import KEYWORD_PATTERNS, SOURCES, TEMPORAL_FPS
from sampling import ClipRecord


# ── 1. Gripper open/close events ───────────────────────────────────

def detect_gripper_events(
    gripper_states: np.ndarray,
    threshold: float = 0.5,
    min_gap_frames: int = 3,
) -> dict[str, list[int]]:
    """
    Detect grasp/release from a 1-D gripper-state signal.

    Args:
        gripper_states: shape (T,), normalized to [0, 1] where 0=closed, 1=open.
        threshold: crossing threshold for open/close.
        min_gap_frames: debounce — ignore transitions closer than this.

    Returns:
        {"grasp_frames": [...], "release_frames": [...]}
    """
    binary = (gripper_states > threshold).astype(int)
    diff = np.diff(binary)

    grasp_frames = np.where(diff == -1)[0].tolist()   # open → closed
    release_frames = np.where(diff == 1)[0].tolist()   # closed → open

    # Debounce
    def debounce(frames: list[int]) -> list[int]:
        if not frames:
            return frames
        out = [frames[0]]
        for f in frames[1:]:
            if f - out[-1] >= min_gap_frames:
                out.append(f)
        return out

    return {
        "grasp_frames": debounce(grasp_frames),
        "release_frames": debounce(release_frames),
    }


def gripper_to_contact_tags(gripper_states: np.ndarray | None) -> list[str]:
    """Return contact subcategory tags from gripper proprioception."""
    if gripper_states is None or len(gripper_states) == 0:
        return []
    events = detect_gripper_events(gripper_states)
    tags = []
    if events["grasp_frames"]:
        tags.append("grasp")
    if events["release_frames"]:
        tags.append("release")
    return tags


# ── 2. Sim ground-truth contacts ───────────────────────────────────

def sim_contacts_to_tags(contact_records: list[dict] | None) -> dict[str, list[str]]:
    """
    Parse sim-engine contact records into taxonomy tags.

    Expected contact record format (ManiSkill3-style):
        {"actor_a": str, "actor_b": str, "force": float, "type": str}

    Returns partial tag dict with 'contact' and optionally 'multi_object' keys.
    """
    if not contact_records:
        return {}

    tags: dict[str, list[str]] = {"contact": []}
    actors_in_contact: set[str] = set()

    for rec in contact_records:
        force = rec.get("force", 0.0)
        if force < 1e-4:
            continue
        actors_in_contact.add(rec.get("actor_a", ""))
        actors_in_contact.add(rec.get("actor_b", ""))
        ctype = rec.get("type", "")
        if "grasp" in ctype.lower():
            tags["contact"].append("grasp")
        elif "push" in ctype.lower():
            tags["contact"].append("push")
        else:
            tags["contact"].append("tool_contact")

    # Deduplicate
    tags["contact"] = list(set(tags["contact"]))

    # If multiple distinct non-robot actors were in contact → multi-object
    non_robot = {a for a in actors_in_contact if "robot" not in a.lower() and "gripper" not in a.lower()}
    if len(non_robot) >= 2:
        tags["multi_object"] = ["collision"]

    return tags


# ── 3. Keyword matching on language ─────────────────────────────────

def keyword_match(text: str | None) -> dict[str, list[str]]:
    """
    Match text against the keyword regex dictionary.

    Returns a dict mapping taxonomy subcategories to the patterns that matched.
    Multiple categories can match — this is multi-label by design.
    """
    if not text:
        return {}

    text_lower = text.lower()
    hits: dict[str, list[str]] = {}

    # Map subcategory → parent category
    sub_to_cat = {
        "grasp": "contact", "release": "contact", "push": "contact", "tool_contact": "contact",
        "articulated": "object_type", "deformable": "object_type", "fluid_granular": "object_type",
        "stacking": "multi_object", "insertion": "multi_object",
        "drop": "failure_recovery", "slip": "failure_recovery", "regrasp": "failure_recovery",
    }

    for sub, patterns in KEYWORD_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                cat = sub_to_cat.get(sub, sub)
                hits.setdefault(cat, [])
                if sub not in hits[cat]:
                    hits[cat].append(sub)
                break  # one match per subcategory is enough

    return hits


# ── 4. Motion magnitude & idle fraction ─────────────────────────────

def frame_difference_energy(frames: list[np.ndarray]) -> np.ndarray:
    """
    Compute per-frame motion energy via absolute frame differencing.

    Args:
        frames: list of (H, W, 3) uint8 arrays.

    Returns:
        Array of shape (len(frames)-1,) with mean absolute difference per pair.
    """
    energies = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i - 1])
        energies.append(diff.mean())
    return np.array(energies, dtype=np.float32)


def classify_dynamics(
    energies: np.ndarray,
    idle_threshold: float = 2.0,
    high_motion_threshold: float = 20.0,
) -> dict[str, Any]:
    """
    Classify clip dynamics from frame-difference energies.

    Returns:
        {
            "dynamics_tags": ["high_motion" | "low_motion" | "idle"],
            "idle_fraction": float,
            "mean_energy": float,
            "max_energy": float,
        }
    """
    if len(energies) == 0:
        return {"dynamics_tags": ["idle"], "idle_fraction": 1.0,
                "mean_energy": 0.0, "max_energy": 0.0}

    idle_frac = (energies < idle_threshold).mean()
    mean_e = float(energies.mean())
    max_e = float(energies.max())

    tags = []
    if max_e > high_motion_threshold:
        tags.append("high_motion")
    if idle_frac > 0.5:
        tags.append("idle")
    if not tags:
        tags.append("low_motion")

    return {
        "dynamics_tags": tags,
        "idle_fraction": float(idle_frac),
        "mean_energy": mean_e,
        "max_energy": max_e,
    }


# ── 5. Embodiment metadata ─────────────────────────────────────────

def get_embodiment_tags(source_name: str) -> list[str]:
    """
    Return embodiment subcategory tags from source config and known facts.
    """
    cfg = SOURCES.get(source_name, {})
    group = cfg.get("group", "")
    tags = []

    # Robot type
    robot_map = {
        "droid": "franka",
        "bridge_v2": "widowx",
        "fractal": "rt1_robot",
        "kuka": "kuka_iiwa",
    }
    if source_name in robot_map:
        tags.append(f"robot_type:{robot_map[source_name]}")
        tags.append("gripper_vs_hand:gripper")

    # Egocentric → hand
    if group == "ego":
        tags.append("gripper_vs_hand:hand")
        tags.append("camera_placement:head_mounted")

    # Camera placement for robots
    camera_map = {
        "droid": "third_person",    # DROID has external cameras
        "bridge_v2": "third_person",
        "fractal": "third_person",
        "kuka": "third_person",
    }
    if source_name in camera_map:
        tags.append(f"camera_placement:{camera_map[source_name]}")

    # Sim → varies, default to third_person
    if group == "sim":
        tags.append("gripper_vs_hand:gripper")
        tags.append("camera_placement:third_person")

    return tags


# ── Combine all cheap signals for a clip ────────────────────────────

def compute_cheap_tags(
    clip: ClipRecord,
    gripper_states: np.ndarray | None = None,
    sim_contacts: list[dict] | None = None,
    frames: list[np.ndarray] | None = None,
) -> dict[str, list[str]]:
    """
    Run all cheap signal extractors and merge into a single tag dict.

    Returns dict mapping category names → list of subcategory strings.
    """
    tags: dict[str, list[str]] = {}

    def merge(new: dict[str, list[str]]):
        for cat, subs in new.items():
            tags.setdefault(cat, [])
            for s in subs:
                if s not in tags[cat]:
                    tags[cat].append(s)

    # Gripper events
    contact_tags = gripper_to_contact_tags(gripper_states)
    if contact_tags:
        merge({"contact": contact_tags})

    # Sim contacts
    sim_tags = sim_contacts_to_tags(sim_contacts)
    merge(sim_tags)

    # Keyword matching
    kw_tags = keyword_match(clip.language)
    merge(kw_tags)

    # Motion / dynamics
    if frames and len(frames) > 1:
        energies = frame_difference_energy(frames)
        dyn = classify_dynamics(energies)
        merge({"dynamics": dyn["dynamics_tags"]})

    # Embodiment
    emb_tags = get_embodiment_tags(clip.source)
    if emb_tags:
        merge({"embodiment": emb_tags})

    return tags
