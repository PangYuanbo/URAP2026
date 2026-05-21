from __future__ import annotations

import math

import numpy as np

from qstr_dronedet.types import DetectionCandidate


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(1e-6, area_a + area_b - inter))


def center_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return float(math.hypot(acx - bcx, acy - bcy))


def nms_candidates(candidates: list[DetectionCandidate], iou_threshold: float = 0.5) -> list[DetectionCandidate]:
    ordered = sorted(candidates, key=lambda c: c.objectness, reverse=True)
    keep: list[DetectionCandidate] = []
    for cand in ordered:
        if all(bbox_iou(cand.bbox_xyxy, k.bbox_xyxy) < iou_threshold for k in keep):
            keep.append(cand)
    return keep


def merge_candidates(candidates: list[DetectionCandidate], iou_threshold: float = 0.35, center_threshold: float = 8.0) -> list[DetectionCandidate]:
    merged: list[DetectionCandidate] = []
    used = [False] * len(candidates)
    for i, cand in enumerate(candidates):
        if used[i]:
            continue
        group = [cand]
        used[i] = True
        for j in range(i + 1, len(candidates)):
            other = candidates[j]
            if used[j]:
                continue
            if bbox_iou(cand.bbox_xyxy, other.bbox_xyxy) >= iou_threshold or center_distance(cand.bbox_xyxy, other.bbox_xyxy) <= center_threshold:
                group.append(other)
                used[j] = True
        weights = np.array([max(1e-3, g.objectness) for g in group], dtype=np.float32)
        boxes = np.array([g.bbox_xyxy for g in group], dtype=np.float32)
        box = tuple((boxes * weights[:, None]).sum(axis=0) / weights.sum())
        best = max(group, key=lambda g: g.objectness)
        sources = sorted({g.source for g in group})
        extra = {"merged_sources": sources, "num_merged": len(group), **best.extra}
        track_member = next((g for g in group if "tracker" in g.source), None)
        if track_member is not None:
            for key, value in track_member.extra.items():
                if key.startswith("track_"):
                    extra[key] = value
            if "track_id" in track_member.extra:
                extra["track_id"] = track_member.extra["track_id"]
        merged.append(
            DetectionCandidate(
                bbox_xyxy=box, objectness=max(g.objectness for g in group), source="+".join(sources),
                motion_score=max(g.motion_score for g in group), alignment_quality=max(g.alignment_quality for g in group),
                track_score=max(g.track_score for g in group), extra=extra,
            )
        )
    return nms_candidates(merged, iou_threshold=0.7)
