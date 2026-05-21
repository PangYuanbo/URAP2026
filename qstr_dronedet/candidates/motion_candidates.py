from __future__ import annotations

import cv2
import numpy as np

from qstr_dronedet.motion.difference import motion_score_in_bbox
from qstr_dronedet.types import DetectionCandidate


def candidates_from_motion(
    motion_map: np.ndarray,
    min_area: int = 3,
    max_area: int = 5000,
    threshold: int | None = None,
) -> list[DetectionCandidate]:
    if motion_map.size == 0 or int(motion_map.max()) == 0:
        return []
    if threshold is None:
        _, binary = cv2.threshold(motion_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if int(binary.sum()) == 0:
            threshold = int(np.percentile(motion_map, 98))
            _, binary = cv2.threshold(motion_map, threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(motion_map, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[DetectionCandidate] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        bbox = (float(x), float(y), float(x + w), float(y + h))
        score = motion_score_in_bbox(motion_map, bbox)
        candidates.append(DetectionCandidate(bbox, objectness=min(1.0, 0.25 + score), source="motion", motion_score=score))
    return candidates

