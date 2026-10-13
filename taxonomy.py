"""
Fixed taxonomy of phenomena a world model must learn.

Every clip is multi-labeled: it can (and usually does) belong to
multiple categories simultaneously.  The taxonomy is flat on purpose --
hierarchical nesting makes the coverage matrix harder to read.
"""

from dataclasses import dataclass, field

# ── Category IDs (stable keys used everywhere) ─────────────────────
CATEGORIES = [
    "contact",
    "object_type",
    "multi_object",
    "embodiment",
    "dynamics",
    "occlusion",
    "failure_recovery",
    "scene_complexity",
]

# ── Human-readable descriptions ────────────────────────────────────
CATEGORY_INFO: dict[str, dict] = {
    "contact": {
        "label": "Contact events",
        "subcategories": ["grasp", "release", "push", "tool_contact"],
        "description": "Moments where an agent makes or breaks physical contact with an object.",
    },
    "object_type": {
        "label": "Object types",
        "subcategories": ["rigid", "articulated", "deformable", "fluid_granular"],
        "description": "Material/kinematic class of the manipulated object.",
    },
    "multi_object": {
        "label": "Multi-object interaction",
        "subcategories": ["stacking", "insertion", "collision"],
        "description": "Interactions involving two or more distinct objects.",
    },
    "embodiment": {
        "label": "Embodiment",
        "subcategories": ["robot_type", "gripper_vs_hand", "camera_placement"],
        "description": "Physical form of the agent and viewpoint.",
        "camera_values": ["wrist", "third_person", "head_mounted"],
    },
    "dynamics": {
        "label": "Dynamics",
        "subcategories": ["high_motion", "low_motion", "idle"],
        "description": "Magnitude and distribution of motion in the clip.",
    },
    "occlusion": {
        "label": "Occlusion & re-appearance",
        "subcategories": ["partial_occlusion", "full_occlusion", "re_entry"],
        "description": "Objects leaving and re-entering the view; tests object permanence.",
    },
    "failure_recovery": {
        "label": "Failure / recovery",
        "subcategories": ["drop", "slip", "regrasp", "retry"],
        "description": "Unintended events and corrective actions (if labels exist).",
    },
    "scene_complexity": {
        "label": "Scene complexity",
        "subcategories": ["single_object", "few_objects", "cluttered"],
        "description": "Number of objects visible — affects prediction difficulty and occlusion likelihood.",
        "thresholds": {"single_object": 1, "few_objects": (2, 5), "cluttered": 6},
    },
}


@dataclass
class ClipTags:
    """Multi-label tag set for one clip, keyed by category."""

    clip_id: str
    source: str  # e.g. "droid", "epic_kitchens", "maniskill3"
    source_group: str  # "robot", "ego", "sim"

    # Each value is a list of active subcategory strings (possibly empty).
    contact: list[str] = field(default_factory=list)
    object_type: list[str] = field(default_factory=list)
    multi_object: list[str] = field(default_factory=list)
    embodiment: list[str] = field(default_factory=list)
    dynamics: list[str] = field(default_factory=list)
    occlusion: list[str] = field(default_factory=list)
    failure_recovery: list[str] = field(default_factory=list)
    scene_complexity: list[str] = field(default_factory=list)

    def active_categories(self) -> list[str]:
        """Return category IDs that have at least one active subcategory."""
        return [c for c in CATEGORIES if getattr(self, c)]

    def to_row(self) -> dict:
        """Flat dict suitable for a DataFrame row."""
        row = {
            "clip_id": self.clip_id,
            "source": self.source,
            "source_group": self.source_group,
        }
        for cat in CATEGORIES:
            subs = getattr(self, cat)
            row[cat] = "|".join(sorted(subs)) if subs else ""
            row[f"{cat}_any"] = bool(subs)
        return row
