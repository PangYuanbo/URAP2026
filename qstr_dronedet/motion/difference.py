from __future__ import annotations

import cv2
import numpy as np

from qstr_dronedet.motion.alignment import estimate_best_alignment, preprocess_gray, warp_frame
from qstr_dronedet.types import AlignmentResult


def compute_motion_map(prev_frame: np.ndarray, curr_frame: np.ndarray, alignment_result: AlignmentResult, clean: bool = True) -> np.ndarray:
    curr_gray = preprocess_gray(curr_frame)
    prev_gray = preprocess_gray(prev_frame)
    h, w = curr_gray.shape[:2]
    aligned = prev_gray if alignment_result.transform is None else warp_frame(prev_gray, alignment_result, (w, h))
    diff = cv2.absdiff(curr_gray, aligned)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if clean:
        kernel = np.ones((3, 3), np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
    return diff


def compute_multik_motion(frame_buffer: list[np.ndarray], curr_index: int, k_values: tuple[int, ...] = (1, 2, 4)) -> dict:
    curr = frame_buffer[curr_index]
    h, w = curr.shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)
    weight = 0.0
    per_k: dict[int, dict] = {}
    best_quality = 0.0
    best_k: int | None = None
    for k in k_values:
        prev_idx = curr_index - int(k)
        if prev_idx < 0:
            per_k[k] = {"map": np.zeros((h, w), np.uint8), "quality": 0.0, "alignment": None}
            continue
        alignment = estimate_best_alignment(frame_buffer[prev_idx], curr)
        if alignment.quality > 0.3:
            motion = compute_motion_map(frame_buffer[prev_idx], curr, alignment)
            acc += motion.astype(np.float32) * alignment.quality
            weight += alignment.quality
        else:
            motion = np.zeros((h, w), np.uint8)
        if alignment.quality > best_quality:
            best_quality = alignment.quality
            best_k = k
        per_k[k] = {"map": motion, "quality": alignment.quality, "alignment": alignment}
    final = np.clip(acc / weight, 0, 255).astype(np.uint8) if weight > 1e-6 else np.zeros((h, w), np.uint8)
    return {"motion_map": final, "per_k": per_k, "best_quality": float(best_quality), "best_k": best_k}


def motion_score_in_bbox(motion_map: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> float:
    h, w = motion_map.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1i, y1i = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    x2i, y2i = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if x2i <= x1i or y2i <= y1i:
        return 0.0
    return float(np.mean(motion_map[y1i:y2i, x1i:x2i]) / 255.0)
