from __future__ import annotations
import math
import numpy as np


def _finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _iou_matrix(boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.zeros((0, 0), np.float32)
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = areas[:, None] + areas[None, :] - intersection
    return np.divide(intersection, np.maximum(union, 1e-9), dtype=np.float32)


def _nms_context(scores: np.ndarray, overlaps: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    count = len(scores)
    survivor = np.zeros(count, np.float32)
    cluster_rank = np.zeros(count, np.float32)
    order = np.argsort(scores)[::-1]
    leaders: list[int] = []
    for candidate in order:
        matched = next((rank for rank, leader in enumerate(leaders) if overlaps[candidate, leader] > threshold), None)
        if matched is None:
            leaders.append(int(candidate))
            survivor[candidate] = 1.0
            cluster_rank[candidate] = len(leaders) - 1
        else:
            cluster_rank[candidate] = matched
    denominator = max(1, len(leaders) - 1)
    return survivor, cluster_rank / denominator


def candidate_context_features(detections: list[dict]) -> np.ndarray:
    valid = [row for row in detections if isinstance(row.get('bbox'), list) and len(row['bbox']) == 4]
    if not valid:
        return np.zeros((0, 23), np.float32)
    scores = np.asarray([_finite(row.get('score')) for row in valid], np.float32)
    boxes = np.asarray([[ _finite(value) for value in row['bbox']] for row in valid], np.float32)
    overlaps = _iou_matrix(boxes)
    np.fill_diagonal(overlaps, 0.0)
    higher = scores[None, :] > scores[:, None]
    count = len(scores)
    normalizer = max(1, count - 1)
    sorted_scores = np.sort(scores)[::-1]
    top = np.pad(sorted_scores[:5], (0, max(0, 5 - len(sorted_scores))))
    frame = np.asarray([
        math.log1p(count) / 6.0,
        math.log1p(int((scores >= 0.05).sum())) / 6.0,
        math.log1p(int((scores >= 0.10).sum())) / 6.0,
        math.log1p(int((scores >= 0.25).sum())) / 6.0,
        math.log1p(int((scores >= 0.50).sum())) / 6.0,
        top[0], top[1], top[2], top[4],
    ], np.float32)
    features = []
    nms = [_nms_context(scores, overlaps, threshold) for threshold in (0.1, 0.3, 0.5)]
    for index, score in enumerate(scores):
        overlap = overlaps[index]
        overlap_mask = overlap > 0.1
        higher_overlap = higher[index] & overlap_mask
        higher_disjoint = higher[index] & ~overlap_mask
        cluster_top = float(np.max(scores[overlap_mask])) if overlap_mask.any() else 0.0
        nearest = int(np.argmax(overlap)) if overlap.size else index
        row = [
            float(higher_overlap.sum()) / normalizer,
            float(higher_disjoint.sum()) / normalizer,
            float((overlap > 0.1).sum()) / normalizer,
            float((overlap > 0.3).sum()) / normalizer,
            float(overlap.max(initial=0.0)),
            float(scores[nearest]) if nearest != index else 0.0,
            float(score - cluster_top),
            float(not higher_overlap.any()),
        ]
        for survivor, cluster_rank in nms:
            row.extend((float(survivor[index]), float(cluster_rank[index])))
        features.append(np.concatenate((frame, np.asarray(row, np.float32))))
    return np.stack(features).astype(np.float32)
