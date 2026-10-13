"""
Central configuration: data sources, sampling parameters, paths.

Design decisions locked in here:
- Per-clip normalization is primary; per-hour is a sensitivity check.
- 8 uniformly-spaced frames per clip for VLM tagging.
- Full video (or 2 fps sub-sample) for temporal signals (flow, idle).
- HuggingFace / LeRobot mirrors preferred over TFDS.
- 50 hand-labeled clips for validation, stratified ~6 per category.
"""

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FRAMES_DIR = DATA_DIR / "frames"
TAGS_DIR = DATA_DIR / "tags"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
AUDIT_DIR = RESULTS_DIR / "audit"

for _d in [RAW_DIR, FRAMES_DIR, TAGS_DIR, EMBEDDINGS_DIR, RESULTS_DIR, PLOTS_DIR, AUDIT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Sampling ────────────────────────────────────────────────────────
CLIPS_PER_SOURCE = 300
VLM_FRAMES_PER_CLIP = 8          # uniformly spaced, for VLM tagging
TEMPORAL_FPS = 2                  # sub-sample rate for flow / dynamics
HAND_LABEL_COUNT = 50             # validation set size
HAND_LABELS_PER_CATEGORY = 6     # stratified minimum

# ── Data sources ────────────────────────────────────────────────────
SOURCES: dict[str, dict] = {
    # ── Robot ───────────────────────────────────────────────────────
    "droid": {
        "group": "robot",
        "hub_id": "lerobot/droid_100",
        "loader": "lerobot_hf",
        "has_proprioception": True,
        "has_language": True,
    },
    "bridge_v2": {
        "group": "robot",
        "hub_id": "IPEC-COMMUNITY/bridge_orig_lerobot",
        "loader": "lerobot_hf",
        "has_proprioception": True,
        "has_language": True,
    },
    "fractal": {
        "group": "robot",
        "hub_id": "IPEC-COMMUNITY/fractal20220817_data_lerobot",
        "loader": "lerobot_hf",
        "has_proprioception": True,
        "has_language": True,
        "note": "Google RT-1 robot, different embodiment from DROID/Bridge",
    },
    "kuka": {
        "group": "robot",
        "hub_id": "IPEC-COMMUNITY/kuka_lerobot",
        "loader": "lerobot_hf",
        "has_proprioception": True,
        "has_language": False,
        "note": "Industrial arm, good embodiment diversity",
    },
    # ── Egocentric ──────────────────────────────────────────────────
    "epic_kitchens": {
        "group": "ego",
        "hub_id": None,  # requires academic download
        "loader": "epic_kitchens",
        "has_proprioception": False,
        "has_language": True,  # verb/noun narrations
        "narration_format": "verb_noun",
    },
    "ego4d": {
        "group": "ego",
        "hub_id": None,
        "loader": "ego4d",
        "has_proprioception": False,
        "has_language": True,  # narrations available for a subset
    },
    "hot3d": {
        "group": "ego",
        "hub_id": None,
        "loader": "hot3d",
        "has_proprioception": False,
        "has_language": False,
        "has_3d_annotations": True,
    },
    # ── Simulation ──────────────────────────────────────────────────
    "maniskill3": {
        "group": "sim",
        "hub_id": "haosulab/ManiSkill",
        "loader": "maniskill3",
        "has_proprioception": True,
        "has_language": True,
        "has_contacts": True,  # dense physics-engine contacts
    },
    "robocasa": {
        "group": "sim",
        "hub_id": "PEARLS-Lab/robocasa-composite-raw-videos",
        "loader": "robocasa_hf",
        "has_proprioception": False,  # video-only dataset
        "has_language": True,  # task name is the language label
        "has_contacts": False,
    },
    "behavior_1k": {
        "group": "sim",
        "hub_id": None,
        "loader": "behavior_1k",
        "has_proprioception": True,
        "has_language": True,
        "has_contacts": False,  # limited contact info
        "note": "Prefer pre-rendered demos to avoid iGibson setup overhead",
    },
}

# Convenience groupings
ROBOT_SOURCES = [k for k, v in SOURCES.items() if v["group"] == "robot"]
EGO_SOURCES = [k for k, v in SOURCES.items() if v["group"] == "ego"]
SIM_SOURCES = [k for k, v in SOURCES.items() if v["group"] == "sim"]

# ── Keyword dictionaries for cheap tagging ──────────────────────────
# Each pattern list is matched against language instructions / narrations.
# Deliberately conservative: prefer precision over recall here;
# VLM tagging catches what keywords miss.
KEYWORD_PATTERNS: dict[str, list[str]] = {
    # contact
    "grasp": [r"\bgrasp\b", r"\bgrip\b", r"\bgrab\b", r"\bpick\s*up\b", r"\btake\b"],
    "release": [r"\brelease\b", r"\bput\s*down\b", r"\bplace\b", r"\blet\s*go\b", r"\bdrop\b"],
    "push": [r"\bpush\b", r"\bpress\b", r"\bslide\b"],
    "tool_contact": [r"\bknife\b", r"\bspatula\b", r"\btool\b", r"\bspoon\b", r"\bfork\b", r"\bbrush\b"],
    # object_type
    "articulated": [r"\bdrawer\b", r"\bdoor\b", r"\bhinge\b", r"\blid\b", r"\bcabinet\b", r"\bfaucet\b"],
    "deformable": [r"\bfold\b", r"\bcloth\b", r"\btowel\b", r"\brope\b", r"\bcable\b", r"\bwire\b", r"\bdough\b", r"\bsponge\b"],
    "fluid_granular": [r"\bpour\b", r"\bwater\b", r"\bliquid\b", r"\bsand\b", r"\brice\b", r"\bbeans\b", r"\bgranul"],
    # multi_object
    "stacking": [r"\bstack\b", r"\bpile\b", r"\btower\b"],
    "insertion": [r"\binsert\b", r"\bplug\b", r"\bpeg\b", r"\bsocket\b", r"\bfit\s*into\b"],
    # failure
    "drop": [r"\bdrop\b", r"\bfell\b", r"\bfall\b"],
    "slip": [r"\bslip\b", r"\bslid\b"],
    "regrasp": [r"\bregrasp\b", r"\bre-grasp\b", r"\btry\s*again\b", r"\bretry\b"],
}

# ── VLM config ──────────────────────────────────────────────────────
VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_MAX_TOKENS = 512
VLM_TEMPERATURE = 0.1  # low temp for structured output

# ── Embedding config ────────────────────────────────────────────────
DINO_MODEL = "dinov2_vitb14"
DINO_EMBED_DIM = 768
