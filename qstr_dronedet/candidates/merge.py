from __future__ import annotations

import math

import numpy as np

from qstr_dronedet.types import DetectionCandidate


def _source_parts(source: str) -> set[str]:
    return {part for part in str(source).split("+") if part}


def _is_detector_source(source: str) -> bool:
    parts = _source_parts(source)
    return any(part in {"yolo", "yolo_tile", "yolov5_dual", "zoom_redetect", "crop_yolo"} or part.startswith("yolo") for part in parts)


def _raw_objectness(candidate: DetectionCandidate) -> float:
    return float(candidate.extra.get("raw_objectness", candidate.objectness))


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
        detector_members = [g for g in group if _is_detector_source(g.source)]
        best_detector = max(detector_members, key=lambda g: (_raw_objectness(g), float(g.objectness))) if detector_members else None
        sources = sorted({g.source for g in group})
        extra = {"merged_sources": sources, "num_merged": len(group), **best.extra}
        if best_detector is not None:
            extra.update(
                {
                    "has_detector_member": True,
                    "detector_raw_objectness": _raw_objectness(best_detector),
                    "detector_bbox_xyxy": [float(v) for v in best_detector.bbox_xyxy],
                    "detector_source": best_detector.source,
                }
            )
        else:
            extra["has_detector_member"] = False
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
