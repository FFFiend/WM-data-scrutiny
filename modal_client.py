"""
Local client that routes all model calls to Modal.

Usage from any module:
    from modal_client import get_vlm, get_detector, get_embedder

    vlm = get_vlm()
    response_text = vlm.tag_frames(frame_bytes_list, prompt)

Each getter returns a thin wrapper whose methods call .remote()
under the hood, so the rest of the codebase doesn't need to know
about Modal at all.
"""

import io
from pathlib import Path

import modal
import numpy as np
from PIL import Image

APP_NAME = "wm-data-scrutiny"


def _frame_path_to_bytes(path: str | Path) -> bytes:
    """Read a frame PNG from disk and return raw bytes."""
    with open(path, "rb") as f:
        return f.read()


def _frames_to_bytes(frame_paths: list[str | Path]) -> list[bytes]:
    """Convert a list of frame paths to a list of PNG byte strings."""
    return [_frame_path_to_bytes(p) for p in frame_paths]


# ═══════════════════════════════════════════════════════════════════
# VLM client
# ═══════════════════════════════════════════════════════════════════

class VLMClient:
    """Wraps the remote QwenVLM Modal class."""

    def __init__(self):
        Cls = modal.Cls.from_name(APP_NAME, "QwenVLM")
        self._obj = Cls()

    def tag_frames(self, frame_paths: list[str | Path], prompt: str) -> str:
        """Send frames to Modal and return raw VLM response text."""
        frame_bytes = _frames_to_bytes(frame_paths)
        return self._obj.tag_frames.remote(frame_bytes, prompt)


def get_vlm() -> VLMClient:
    return VLMClient()


# ═══════════════════════════════════════════════════════════════════
# Grounding DINO client
# ═══════════════════════════════════════════════════════════════════

class DetectorClient:
    """Wraps the remote GroundingDINODetector Modal class."""

    def __init__(self):
        Cls = modal.Cls.from_name(APP_NAME, "GroundingDINODetector")
        self._obj = Cls()

    def detect(
        self,
        frame_paths: list[str | Path],
        query_text: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> list[dict]:
        """
        Run detection on frames.
        Returns per-frame results: [{"labels": [...], "scores": [...], "n_objects": int}]
        """
        frame_bytes = _frames_to_bytes(frame_paths)
        return self._obj.detect.remote(
            frame_bytes, query_text, box_threshold, text_threshold,
        )


def get_detector() -> DetectorClient:
    return DetectorClient()


# ═══════════════════════════════════════════════════════════════════
# DINOv2 embedding client
# ═══════════════════════════════════════════════════════════════════

class EmbedderClient:
    """Wraps the remote DINOv2Embed Modal class."""

    def __init__(self):
        Cls = modal.Cls.from_name(APP_NAME, "DINOv2Embed")
        self._obj = Cls()

    def embed_frames(self, frame_paths: list[str | Path]) -> list[np.ndarray]:
        """
        Embed frames and return list of numpy embedding vectors.
        """
        frame_bytes = _frames_to_bytes(frame_paths)
        raw = self._obj.embed_frames.remote(frame_bytes)
        return [np.array(v, dtype=np.float32) for v in raw]

    def embed_clip(self, frame_paths: list[str | Path]) -> np.ndarray:
        """
        Embed frames and return mean-pooled clip embedding.
        """
        embs = self.embed_frames(frame_paths)
        if not embs:
            return np.zeros(768, dtype=np.float32)
        return np.mean(embs, axis=0)


def get_embedder() -> EmbedderClient:
    return EmbedderClient()
