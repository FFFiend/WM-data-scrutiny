"""
VLM-based structured tagging using Qwen2.5-VL.

For each clip, send VLM_FRAMES_PER_CLIP frames with a fixed prompt
that asks for taxonomy categories as structured JSON.  Parse the
response with strict schema validation — malformed responses get
flagged for manual review rather than silently dropped.

All inference runs on Modal (see modal_app.py / modal_client.py).
"""

import json
import re
from pathlib import Path

from config import FRAMES_DIR
from taxonomy import CATEGORIES, CATEGORY_INFO


# ── Schema for the VLM output ──────────────────────────────────────

VLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        cat: {
            "type": "array",
            "items": {
                "type": "string",
                "enum": CATEGORY_INFO[cat]["subcategories"],
            },
        }
        for cat in CATEGORIES
    },
    "required": CATEGORIES,
    "additionalProperties": False,
}

# ── Prompt construction ─────────────────────────────────────────────

def build_vlm_prompt() -> str:
    """
    Build the fixed system prompt that instructs the VLM to output
    structured JSON matching our taxonomy.
    """
    cat_descriptions = []
    for cat in CATEGORIES:
        info = CATEGORY_INFO[cat]
        subs = ", ".join(info["subcategories"])
        cat_descriptions.append(
            f'- "{cat}" ({info["label"]}): choose zero or more from [{subs}]. '
            f'{info["description"]}'
        )

    prompt = f"""You are a video analysis assistant. Given frames from a manipulation video clip, classify the clip into the following taxonomy categories. For each category, list which subcategories are present (can be empty if none apply).

Categories:
{chr(10).join(cat_descriptions)}

Special notes:
- "scene_complexity": count visible objects. single_object = 1, few_objects = 2-5, cluttered = 6+.
- "occlusion": only tag if an object visibly leaves and/or re-enters the frame across the shown frames.
- "failure_recovery": only tag if you see clear evidence of a drop, slip, or re-attempt.
- "dynamics": judge from apparent motion blur and position changes across frames. If frames look nearly identical, tag "idle".

Return ONLY a JSON object with the 8 category keys. Each value is a list of strings (subcategory names). No markdown, no explanation, just the JSON.

Example output:
{{"contact": ["grasp"], "object_type": ["rigid"], "multi_object": [], "embodiment": ["robot_type", "gripper_vs_hand", "camera_placement"], "dynamics": ["low_motion"], "occlusion": [], "failure_recovery": [], "scene_complexity": ["few_objects"]}}"""

    return prompt


# ── Response parsing ────────────────────────────────────────────────

def parse_vlm_response(response_text: str) -> dict[str, list[str]]:
    """
    Parse VLM response text into validated tag dict.
    Attempts to extract JSON even from slightly malformed responses.
    """
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not json_match:
        return {"_error": [f"No JSON found in response: {response_text[:200]}"]}

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        return {"_error": [f"JSON parse error: {e}. Raw: {response_text[:200]}"]}

    validated: dict[str, list[str]] = {}
    for cat in CATEGORIES:
        val = parsed.get(cat, [])
        if not isinstance(val, list):
            val = [val] if isinstance(val, str) else []
        known_subs = set(CATEGORY_INFO[cat]["subcategories"])
        clean = [s for s in val if isinstance(s, str) and s in known_subs]
        unknown = [s for s in val if isinstance(s, str) and s not in known_subs]
        validated[cat] = clean
        if unknown:
            validated.setdefault("_warnings", [])
            validated["_warnings"].append(f"{cat}: unknown subcategories {unknown}")

    return validated


# ── Batch tagging via Modal ─────────────────────────────────────────

def tag_clip_with_vlm(
    frame_paths: list[str | Path],
    vlm_client=None,
    prompt: str | None = None,
) -> dict[str, list[str]]:
    """
    Send frames to VLM via Modal and parse the structured response.
    """
    from modal_client import get_vlm

    if vlm_client is None:
        vlm_client = get_vlm()
    if prompt is None:
        prompt = build_vlm_prompt()

    response_text = vlm_client.tag_frames(frame_paths, prompt)
    return parse_vlm_response(response_text)


def tag_all_clips(
    clip_ids: list[str],
    sources: list[str],
) -> list[dict]:
    """
    Tag all clips that have extracted frames on disk.
    All inference routed to Modal.

    Returns list of dicts: {"clip_id": str, "source": str, "tags": {...}}.
    """
    from modal_client import get_vlm

    vlm = get_vlm()
    prompt = build_vlm_prompt()
    results = []
    total = len(clip_ids)

    for i, (clip_id, source) in enumerate(zip(clip_ids, sources)):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  VLM tagging clip {i+1}/{total}: {clip_id}", flush=True)
        frame_dir = FRAMES_DIR / source / clip_id
        if not frame_dir.exists():
            results.append({"clip_id": clip_id, "source": source,
                            "tags": {"_error": ["no frames on disk"]}})
            continue

        frame_paths = sorted(frame_dir.glob("frame_*.png"))
        if not frame_paths:
            results.append({"clip_id": clip_id, "source": source,
                            "tags": {"_error": ["no frame PNGs found"]}})
            continue

        tags = tag_clip_with_vlm(frame_paths, vlm_client=vlm, prompt=prompt)
        results.append({"clip_id": clip_id, "source": source, "tags": tags})

    return results
