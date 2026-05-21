from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


CLASSES = [
    "drone",
    "bird",
    "airplane",
    "insect",
    "ground_object",
    "alignment_artifact",
    "background",
    "unknown",
]


@dataclass
class DetectionCandidate:
    bbox_xyxy: tuple[float, float, float, float]
    objectness: float
    source: str
    motion_score: float = 0.0
    alignment_quality: float = 0.0
    track_score: float = 0.0
    mode: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecognitionResult:
    crop_probs: dict[str, float]
    feature_probs: dict[str, float]
    temporal_probs: dict[str, float]
    final_probs: dict[str, float]
    disagreement: float
    predicted_class: str
    final_drone_score: float
    error_type: str | None
    diagnostic_cause: str | None = None


@dataclass
class AlignmentResult:
    transform: np.ndarray | None
    transform_type: str
    quality: float
    inlier_ratio: float
    num_inliers: int
    median_reproj_error: float
    photometric_residual: float
    blur_score: float
    debug: dict[str, Any]


@dataclass
class FrameResult:
    frame_id: int
    candidates: list[DetectionCandidate]
    recognitions: list[RecognitionResult]
    final_detections: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def uniform_probs(value: float = 1.0) -> dict[str, float]:
    p = float(value) / len(CLASSES)
    return {c: p for c in CLASSES}


def normalize_probs(probs: dict[str, float]) -> dict[str, float]:
    clean = {c: max(0.0, float(probs.get(c, 0.0))) for c in CLASSES}
    s = sum(clean.values())
    if s <= 1e-12:
        return uniform_probs()
    return {c: clean[c] / s for c in CLASSES}
