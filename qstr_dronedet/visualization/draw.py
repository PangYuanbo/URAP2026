from __future__ import annotations

import cv2
import numpy as np

from qstr_dronedet.types import DetectionCandidate, RecognitionResult


def draw_overlay(frame: np.ndarray, candidates: list[DetectionCandidate], recognitions: list[RecognitionResult] | None = None) -> np.ndarray:
    out = frame.copy()
    recognitions = recognitions or []
    for i, cand in enumerate(candidates):
        x1, y1, x2, y2 = [int(round(v)) for v in cand.bbox_xyxy]
        rec = recognitions[i] if i < len(recognitions) else None
        is_drone = rec is not None and rec.predicted_class == "drone" and rec.final_drone_score > 0.2
        color = (0, 0, 255) if is_drone else (0, 200, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        if rec:
            label = f"{rec.predicted_class} {rec.final_drone_score:.2f} qH={cand.alignment_quality:.2f} {cand.mode} {cand.source}"
        else:
            label = f"cand {cand.objectness:.2f} {cand.source}"
        cv2.putText(out, label, (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out


def make_side_by_side(frame: np.ndarray, motion_map: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    motion_bgr = cv2.cvtColor(motion_map, cv2.COLOR_GRAY2BGR) if motion_map.ndim == 2 else motion_map
    h, w = frame.shape[:2]
    motion_bgr = cv2.resize(motion_bgr, (w, h))
    overlay = cv2.resize(overlay, (w, h))
    return np.concatenate([frame, motion_bgr, overlay], axis=1)

