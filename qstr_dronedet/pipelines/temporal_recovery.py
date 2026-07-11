from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal

import cv2
import numpy as np

from qstr_dronedet.camera_motion import estimate_background_homography, transform_bbox_xyxy
from qstr_dronedet.candidates.merge import bbox_iou, center_distance, merge_candidates
from qstr_dronedet.types import DetectionCandidate


BBox = tuple[float, float, float, float]
DetectorFn = Callable[[np.ndarray], list[DetectionCandidate]]


@dataclass
class TemporalRecoveryConfig:
    profile: Literal["default", "dji-tiny"] = "default"
    final_selection_score: Literal["temporal", "raw"] = "temporal"
    final_output_score: Literal["temporal", "raw"] = "temporal"
    memory_update_selection: Literal["final", "temporal"] = "final"
    apply_output_gate: bool = True
    top_k: int = 80
    max_center_distance: float = 64.0
    motion_weight: float = 0.42
    detector_weight: float = 0.58
    memory_iou_bonus: float = 0.18
    ncc_min_score: float = 0.62
    ncc_search_scale: float = 3.0
    ncc_score: float = 0.34
    miss_patience: int = 5
    zoom_trigger_misses: int = 1
    zoom_tiny_max_side: float = 24.0
    zoom_crop_scale: float = 5.0
    zoom_min_score: float = 0.04
    merge_iou_threshold: float = 0.35
    merge_center_threshold: float = 8.0
    hard_reset_min_score: float = 0.55
    hard_reset_min_distance: float = 96.0
    hard_reset_min_iou: float = 0.05
    memory_quality_min: float = 0.38
    memory_detector_min: float = 0.05
    memory_motion_min: float = 0.08
    memory_detector_quality_weight: float = 0.45
    memory_motion_quality_weight: float = 0.40
    memory_support_quality_weight: float = 0.15
    memory_bank_size: int = 24
    allow_support_only_output: bool = True
    support_only_output_min_quality: float = 0.72
    support_only_min_detector_updates: int = 2
    support_only_max_misses: int = 1
    camera_motion_compensation: bool = False
    camera_motion_max_size: int = 512
    camera_motion_min_tracks: int = 24
    camera_motion_min_inlier_ratio: float = 0.45
    camera_motion_ransac_threshold: float = 2.5
    camera_motion_max_forward_backward_error: float = 1.5
    residual_velocity_momentum: float = 0.65
    samurai_motion_iou_weight: float = 0.18
    memory_motion_iou_min: float = 0.0


@dataclass
class MotionMemoryObservation:
    bbox_xyxy: BBox
    source: str
    objectness: float
    motion_score: float
    support_score: float
    quality: float
    frame_id: int | None = None


@dataclass
class MotionMemoryTrack:
    bbox_xyxy: BBox
    velocity_xy: tuple[float, float] = (0.0, 0.0)
    score: float = 0.0
    age: int = 1
    misses: int = 0
    detector_updates: int = 1
    history: list[BBox] = field(default_factory=list)
    memory_bank: list[MotionMemoryObservation] = field(default_factory=list)

    def predict(
        self,
        frame_shape: tuple[int, int] | tuple[int, int, int],
        camera_previous_to_current: np.ndarray | None = None,
    ) -> BBox:
        dx, dy = self.velocity_xy
        camera_bbox = self.bbox_xyxy
        if camera_previous_to_current is not None:
            camera_bbox = transform_bbox_xyxy(self.bbox_xyxy, camera_previous_to_current)
        return clip_bbox(shift_bbox(camera_bbox, dx, dy), frame_shape)

    def update(
        self,
        cand: DetectionCandidate,
        frame_shape: tuple[int, int] | tuple[int, int, int],
        *,
        write_memory: bool = True,
        memory_quality: float | None = None,
        frame_id: int | None = None,
        memory_bank_size: int = 24,
        camera_previous_to_current: np.ndarray | None = None,
        residual_velocity_momentum: float = 0.65,
    ) -> None:
        camera_bbox = self.bbox_xyxy
        if camera_previous_to_current is not None:
            camera_bbox = transform_bbox_xyxy(self.bbox_xyxy, camera_previous_to_current)
        old_cx, old_cy = bbox_center(camera_bbox)
        new_cx, new_cy = bbox_center(cand.bbox_xyxy)
        momentum = min(1.0, max(0.0, float(residual_velocity_momentum)))
        self.velocity_xy = (
            momentum * self.velocity_xy[0] + (1.0 - momentum) * (new_cx - old_cx),
            momentum * self.velocity_xy[1] + (1.0 - momentum) * (new_cy - old_cy),
        )
        self.bbox_xyxy = clip_bbox(cand.bbox_xyxy, frame_shape)
        self.score = float(max(cand.objectness, 0.75 * self.score + 0.25 * cand.objectness))
        self.age += 1
        self.misses = 0
        if cand.source not in {"gray_ncc", "motion_memory_prediction"}:
            self.detector_updates += 1
        self.history.append(self.bbox_xyxy)
        self.history = self.history[-32:]
        if write_memory:
            self.record_memory(cand, memory_quality, frame_id, memory_bank_size)

    def mark_miss(self, camera_previous_to_current: np.ndarray | None = None) -> None:
        self.bbox_xyxy = self.predict((10**9, 10**9), camera_previous_to_current)
        self.score *= 0.82
        self.age += 1
        self.misses += 1

    def record_memory(
        self,
        cand: DetectionCandidate,
        memory_quality: float | None = None,
        frame_id: int | None = None,
        memory_bank_size: int = 24,
    ) -> None:
        self.memory_bank.append(
            MotionMemoryObservation(
                bbox_xyxy=clip_bbox(cand.bbox_xyxy, (10**9, 10**9)),
                source=cand.source,
                objectness=float(cand.extra.get("raw_objectness", cand.objectness)),
                motion_score=float(cand.extra.get("motion_memory_score", cand.motion_score)),
                support_score=_candidate_support_score(cand),
                quality=float(memory_quality if memory_quality is not None else cand.objectness),
                frame_id=frame_id,
            )
        )
        self.memory_bank = sorted(self.memory_bank, key=lambda obs: (obs.quality, obs.frame_id if obs.frame_id is not None else -1), reverse=True)[:memory_bank_size]


@dataclass
class TemporalRecoveryFrame:
    frame_id: int
    candidates: list[DetectionCandidate]
    selected: DetectionCandidate | None
    memory_bbox: BBox | None
    diagnostics: dict[str, object]


def bbox_center(bbox: BBox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def bbox_size(bbox: BBox) -> tuple[float, float]:
    return (max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1]))


def shift_bbox(bbox: BBox, dx: float, dy: float) -> BBox:
    return (bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy)


def clip_bbox(bbox: BBox, frame_shape: tuple[int, int] | tuple[int, int, int]) -> BBox:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x1 = min(max(0.0, float(bbox[0])), float(w))
    y1 = min(max(0.0, float(bbox[1])), float(h))
    x2 = min(max(0.0, float(bbox[2])), float(w))
    y2 = min(max(0.0, float(bbox[3])), float(h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def expand_bbox(bbox: BBox, scale: float, frame_shape: tuple[int, int] | tuple[int, int, int], min_side: float = 32.0) -> BBox:
    cx, cy = bbox_center(bbox)
    bw, bh = bbox_size(bbox)
    side_w = max(min_side, bw * scale)
    side_h = max(min_side, bh * scale)
    return clip_bbox((cx - side_w / 2.0, cy - side_h / 2.0, cx + side_w / 2.0, cy + side_h / 2.0), frame_shape)


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _crop(frame: np.ndarray, bbox: BBox) -> tuple[np.ndarray, tuple[int, int]]:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(frame.shape[1], x1))
    x2 = max(0, min(frame.shape[1], x2))
    y1 = max(0, min(frame.shape[0], y1))
    y2 = max(0, min(frame.shape[0], y2))
    return frame[y1:y2, x1:x2], (x1, y1)


def _with_extra(cand: DetectionCandidate, **extra: object) -> DetectionCandidate:
    merged_extra = dict(cand.extra)
    merged_extra.update(extra)
    return DetectionCandidate(
        bbox_xyxy=cand.bbox_xyxy,
        objectness=float(cand.objectness),
        source=cand.source,
        motion_score=float(cand.motion_score),
        alignment_quality=float(cand.alignment_quality),
        track_score=float(cand.track_score),
        mode=cand.mode,
        extra=merged_extra,
    )


def _with_objectness(cand: DetectionCandidate, objectness: float, **extra: object) -> DetectionCandidate:
    merged_extra = dict(cand.extra)
    merged_extra.update(extra)
    return DetectionCandidate(
        bbox_xyxy=cand.bbox_xyxy,
        objectness=float(objectness),
        source=cand.source,
        motion_score=float(cand.motion_score),
        alignment_quality=float(cand.alignment_quality),
        track_score=float(cand.track_score),
        mode=cand.mode,
        extra=merged_extra,
    )


def _with_bbox_objectness_source(
    cand: DetectionCandidate,
    bbox_xyxy: BBox,
    objectness: float,
    source: str,
    **extra: object,
) -> DetectionCandidate:
    merged_extra = dict(cand.extra)
    merged_extra.update(extra)
    return DetectionCandidate(
        bbox_xyxy=tuple(float(v) for v in bbox_xyxy),
        objectness=float(objectness),
        source=source,
        motion_score=float(cand.motion_score),
        alignment_quality=float(cand.alignment_quality),
        track_score=float(cand.track_score),
        mode=cand.mode,
        extra=merged_extra,
    )


def _detector_raw_objectness(cand: DetectionCandidate) -> float:
    return float(cand.extra.get("detector_raw_objectness", cand.extra.get("raw_objectness", cand.objectness)))


def _detector_bbox_xyxy(cand: DetectionCandidate) -> BBox:
    value = cand.extra.get("detector_bbox_xyxy")
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(float(v) for v in value)
    return cand.bbox_xyxy


def apply_final_output_score(cand: DetectionCandidate, config: TemporalRecoveryConfig | None = None) -> DetectionCandidate:
    cfg = config or TemporalRecoveryConfig()
    if cfg.final_output_score == "raw":
        raw = _detector_raw_objectness(cand)
        source = str(cand.extra.get("detector_source", cand.source))
        return _with_bbox_objectness_source(
            cand,
            _detector_bbox_xyxy(cand),
            raw,
            source,
            final_output_score=raw,
            final_output_bbox_source=source,
        )
    return _with_extra(cand, final_output_score=float(cand.objectness))


def _candidate_support_score(cand: DetectionCandidate) -> float:
    values = [cand.track_score, cand.extra.get("ncc_score", 0.0), cand.extra.get("zoom_agreement", 0.0)]
    scores: list[float] = []
    for value in values:
        try:
            scores.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else 0.0


def memory_quality_score(cand: DetectionCandidate, config: TemporalRecoveryConfig | None = None) -> float:
    cfg = config or TemporalRecoveryConfig()
    detector_score = _detector_raw_objectness(cand)
    motion_score = float(cand.extra.get("motion_memory_score", cand.motion_score))
    support_score = _candidate_support_score(cand)
    total_weight = cfg.memory_detector_quality_weight + cfg.memory_motion_quality_weight + cfg.memory_support_quality_weight
    if total_weight <= 0:
        return 0.0
    quality = (
        cfg.memory_detector_quality_weight * detector_score
        + cfg.memory_motion_quality_weight * motion_score
        + cfg.memory_support_quality_weight * support_score
    ) / total_weight
    return float(max(0.0, min(1.0, quality)))


def should_write_motion_memory(cand: DetectionCandidate, config: TemporalRecoveryConfig | None = None) -> tuple[bool, float, str]:
    cfg = config or TemporalRecoveryConfig()
    detector_score = _detector_raw_objectness(cand)
    motion_score = float(cand.extra.get("motion_memory_score", cand.motion_score))
    quality = memory_quality_score(cand, cfg)
    if bool(cand.extra.get("hard_reset", False)):
        return True, quality, "hard_reset"
    if detector_score < cfg.memory_detector_min:
        return False, quality, "detector_score_below_memory_gate"
    if motion_score < cfg.memory_motion_min and cand.source != "gray_ncc":
        return False, quality, "motion_score_below_memory_gate"
    motion_iou = float(cand.extra.get("samurai_motion_iou", 0.0))
    if motion_iou < cfg.memory_motion_iou_min and cand.source != "gray_ncc":
        return False, quality, "motion_iou_below_memory_gate"
    if quality < cfg.memory_quality_min:
        return False, quality, "hybrid_quality_below_memory_gate"
    return True, quality, "hybrid_quality_pass"


def has_detector_evidence(cand: DetectionCandidate) -> bool:
    parts = {part for part in cand.source.split("+") if part}
    if bool(cand.extra.get("has_detector_member", False)):
        return True
    return any(part in {"yolo", "yolo_tile", "yolov5_dual", "zoom_redetect", "crop_yolo"} or part.startswith("yolo") for part in parts)


def should_emit_detection(
    cand: DetectionCandidate,
    memory: MotionMemoryTrack | None,
    config: TemporalRecoveryConfig | None = None,
) -> tuple[bool, float, str]:
    cfg = config or TemporalRecoveryConfig()
    quality = memory_quality_score(cand, cfg)
    if has_detector_evidence(cand):
        return True, quality, "detector_evidence"
    if not cfg.allow_support_only_output:
        return False, quality, "support_only_output_disabled"
    if memory is None:
        return False, quality, "support_only_without_memory"
    if memory.detector_updates < cfg.support_only_min_detector_updates:
        return False, quality, "support_only_insufficient_detector_history"
    if memory.misses > cfg.support_only_max_misses:
        return False, quality, "support_only_after_too_many_misses"
    if quality < cfg.support_only_output_min_quality:
        return False, quality, "support_only_quality_below_output_gate"
    return True, quality, "support_only_quality_pass"


def score_candidates_with_motion_memory(
    candidates: Iterable[DetectionCandidate],
    memory: MotionMemoryTrack | None,
    frame_shape: tuple[int, int] | tuple[int, int, int],
    config: TemporalRecoveryConfig | None = None,
    camera_previous_to_current: np.ndarray | None = None,
) -> list[DetectionCandidate]:
    cfg = config or TemporalRecoveryConfig()
    top = sorted(candidates, key=lambda c: c.objectness, reverse=True)[: cfg.top_k]
    if memory is None:
        return [_with_extra(c, temporal_score=float(c.objectness), motion_memory_score=0.0) for c in top]

    pred = memory.predict(frame_shape, camera_previous_to_current)
    scored: list[DetectionCandidate] = []
    for cand in top:
        dist = center_distance(cand.bbox_xyxy, pred)
        dist_score = max(0.0, 1.0 - dist / max(1.0, cfg.max_center_distance))
        iou_score = bbox_iou(cand.bbox_xyxy, pred)
        memory_score = min(1.0, dist_score + cfg.memory_iou_bonus * iou_score)
        samurai_score = (
            (1.0 - cfg.samurai_motion_iou_weight) * memory_score
            + cfg.samurai_motion_iou_weight * iou_score
        )
        temporal_score = cfg.detector_weight * float(cand.objectness) + cfg.motion_weight * samurai_score
        scored.append(
            DetectionCandidate(
                bbox_xyxy=cand.bbox_xyxy,
                objectness=float(temporal_score),
                source=cand.source,
                motion_score=max(float(cand.motion_score), float(samurai_score)),
                alignment_quality=cand.alignment_quality,
                track_score=max(float(cand.track_score), float(memory.score)),
                mode=cand.mode,
                extra={
                    **cand.extra,
                    "raw_objectness": float(cand.objectness),
                    "motion_memory_score": float(samurai_score),
                    "samurai_motion_iou": float(iou_score),
                    "camera_compensated_prediction": list(pred),
                    "temporal_score": float(temporal_score),
                },
            )
        )
    return sorted(scored, key=lambda c: c.objectness, reverse=True)


def final_selection_score(cand: DetectionCandidate, config: TemporalRecoveryConfig | None = None) -> float:
    cfg = config or TemporalRecoveryConfig()
    if cfg.final_selection_score == "raw":
        return _detector_raw_objectness(cand)
    return float(cand.objectness)


def select_final_candidate(candidates: list[DetectionCandidate], config: TemporalRecoveryConfig | None = None) -> DetectionCandidate | None:
    if not candidates:
        return None
    cfg = config or TemporalRecoveryConfig()
    if cfg.final_selection_score == "temporal":
        return candidates[0]
    return max(candidates, key=lambda cand: (final_selection_score(cand, cfg), float(cand.objectness)))


def should_emit_output_detection(
    cand: DetectionCandidate,
    memory: MotionMemoryTrack | None,
    config: TemporalRecoveryConfig | None = None,
) -> tuple[bool, float, str]:
    cfg = config or TemporalRecoveryConfig()
    if cfg.apply_output_gate:
        return should_emit_detection(cand, memory, cfg)
    return True, memory_quality_score(cand, cfg), "output_gate_disabled"


def ncc_proposal_from_memory(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    memory: MotionMemoryTrack | None,
    config: TemporalRecoveryConfig | None = None,
    camera_previous_to_current: np.ndarray | None = None,
) -> DetectionCandidate | None:
    cfg = config or TemporalRecoveryConfig()
    if memory is None:
        return None
    prev_gray = _gray(prev_frame)
    curr_gray = _gray(curr_frame)
    template_bbox = clip_bbox(memory.bbox_xyxy, prev_gray.shape)
    template, _ = _crop(prev_gray, template_bbox)
    if template.size == 0 or template.shape[0] < 3 or template.shape[1] < 3:
        return None
    search_bbox = expand_bbox(memory.predict(curr_gray.shape, camera_previous_to_current), cfg.ncc_search_scale, curr_gray.shape, min_side=max(16.0, float(max(template.shape))))
    search, (ox, oy) = _crop(curr_gray, search_bbox)
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None
    res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if float(max_val) < cfg.ncc_min_score:
        return None
    x1 = float(ox + max_loc[0])
    y1 = float(oy + max_loc[1])
    x2 = x1 + float(template.shape[1])
    y2 = y1 + float(template.shape[0])
    return DetectionCandidate(
        bbox_xyxy=clip_bbox((x1, y1, x2, y2), curr_gray.shape),
        objectness=cfg.ncc_score,
        source="gray_ncc",
        motion_score=float(max_val),
        track_score=float(max_val),
        extra={"ncc_score": float(max_val), "support": "gray_ncc_tracker"},
    )


def zoom_in_redetect(
    frame: np.ndarray,
    predicted_bbox: BBox,
    crop_detector: DetectorFn | None,
    config: TemporalRecoveryConfig | None = None,
) -> list[DetectionCandidate]:
    cfg = config or TemporalRecoveryConfig()
    if crop_detector is None:
        return []
    crop_bbox = expand_bbox(predicted_bbox, cfg.zoom_crop_scale, frame.shape, min_side=64.0)
    crop, (ox, oy) = _crop(frame, crop_bbox)
    if crop.size == 0:
        return []
    out: list[DetectionCandidate] = []
    for cand in crop_detector(crop):
        if cand.objectness < cfg.zoom_min_score:
            continue
        x1, y1, x2, y2 = cand.bbox_xyxy
        out.append(
            DetectionCandidate(
                bbox_xyxy=clip_bbox((x1 + ox, y1 + oy, x2 + ox, y2 + oy), frame.shape),
                objectness=float(cand.objectness),
                source="zoom_redetect",
                motion_score=cand.motion_score,
                alignment_quality=cand.alignment_quality,
                track_score=cand.track_score,
                mode=cand.mode,
                extra={**cand.extra, "crop_bbox_xyxy": list(crop_bbox), "crop_origin_xy": [ox, oy]},
            )
        )
    return out


def hard_reset_bbox_correction(
    selected: DetectionCandidate | None,
    memory: MotionMemoryTrack | None,
    frame_shape: tuple[int, int] | tuple[int, int, int],
    config: TemporalRecoveryConfig | None = None,
    camera_previous_to_current: np.ndarray | None = None,
) -> DetectionCandidate | None:
    cfg = config or TemporalRecoveryConfig()
    if selected is None or memory is None:
        return selected
    pred = memory.predict(frame_shape, camera_previous_to_current)
    dist = center_distance(selected.bbox_xyxy, pred)
    iou = bbox_iou(selected.bbox_xyxy, pred)
    is_detector_source = selected.source in {"yolo", "yolo_tile", "yolov5_dual", "zoom_redetect"} or selected.source.startswith("yolo")
    reset_score = max(float(selected.objectness), _detector_raw_objectness(selected))
    is_strong_detector = is_detector_source and reset_score >= cfg.hard_reset_min_score
    if is_strong_detector and dist >= cfg.hard_reset_min_distance and iou <= cfg.hard_reset_min_iou:
        return _with_extra(selected, hard_reset=True, hard_reset_reason="strong_detector_far_from_stale_memory", stale_memory_bbox=list(pred))
    if selected.source == "gray_ncc" and memory.misses >= cfg.miss_patience:
        return None
    return _with_extra(selected, hard_reset=False, stale_memory_bbox=list(pred))


def _should_zoom(
    memory: MotionMemoryTrack | None,
    cfg: TemporalRecoveryConfig,
    frame_shape: tuple[int, int] | tuple[int, int, int],
    camera_previous_to_current: np.ndarray | None = None,
) -> bool:
    if cfg.profile != "dji-tiny" or memory is None:
        return False
    bw, bh = bbox_size(memory.predict(frame_shape, camera_previous_to_current))
    return memory.misses >= cfg.zoom_trigger_misses or max(bw, bh) <= cfg.zoom_tiny_max_side


def run_temporal_recovery_frames(
    frames: Iterable[np.ndarray],
    detector: DetectorFn,
    crop_detector: DetectorFn | None = None,
    config: TemporalRecoveryConfig | None = None,
) -> list[TemporalRecoveryFrame]:
    cfg = config or TemporalRecoveryConfig()
    memory: MotionMemoryTrack | None = None
    prev_frame: np.ndarray | None = None
    rows: list[TemporalRecoveryFrame] = []

    for frame_id, frame in enumerate(frames):
        memory_write = False
        memory_quality: float | None = None
        memory_write_reason: str | None = None
        emit_detection = False
        emit_reason: str | None = None
        detector_candidates = sorted(detector(frame), key=lambda c: c.objectness, reverse=True)[: cfg.top_k]
        support_candidates: list[DetectionCandidate] = []
        camera_previous_to_current: np.ndarray | None = None
        camera_motion_diagnostics: dict[str, object] = {
            "enabled": cfg.camera_motion_compensation,
            "valid": False,
        }
        if prev_frame is not None:
            if cfg.camera_motion_compensation:
                estimate = estimate_background_homography(
                    prev_frame,
                    frame,
                    max_size=cfg.camera_motion_max_size,
                    min_tracks=cfg.camera_motion_min_tracks,
                    min_inlier_ratio=cfg.camera_motion_min_inlier_ratio,
                    ransac_threshold=cfg.camera_motion_ransac_threshold,
                    max_forward_backward_error=cfg.camera_motion_max_forward_backward_error,
                )
                if estimate.valid:
                    camera_previous_to_current = estimate.matrix
                camera_motion_diagnostics = {
                    "enabled": True,
                    "valid": estimate.valid,
                    "tracked_points": estimate.tracked_points,
                    "inliers": estimate.inliers,
                    "inlier_ratio": estimate.inlier_ratio,
                    "median_reprojection_error": estimate.median_reprojection_error,
                }
            ncc = ncc_proposal_from_memory(
                prev_frame,
                frame,
                memory,
                cfg,
                camera_previous_to_current,
            )
            if ncc is not None:
                support_candidates.append(ncc)
        if _should_zoom(memory, cfg, frame.shape, camera_previous_to_current):
            assert memory is not None
            support_candidates.extend(
                zoom_in_redetect(
                    frame,
                    memory.predict(frame.shape, camera_previous_to_current),
                    crop_detector,
                    cfg,
                )
            )

        scored = score_candidates_with_motion_memory(
            detector_candidates + support_candidates,
            memory,
            frame.shape,
            cfg,
            camera_previous_to_current,
        )
        merged = merge_candidates(scored, iou_threshold=cfg.merge_iou_threshold, center_threshold=cfg.merge_center_threshold)
        output_selected = hard_reset_bbox_correction(
            select_final_candidate(merged, cfg),
            memory,
            frame.shape,
            cfg,
            camera_previous_to_current,
        )
        memory_selected = output_selected
        if cfg.memory_update_selection == "temporal":
            memory_selected = hard_reset_bbox_correction(
                merged[0] if merged else None,
                memory,
                frame.shape,
                cfg,
                camera_previous_to_current,
            )

        if output_selected is not None:
            emit_detection, memory_quality, emit_reason = should_emit_output_detection(output_selected, memory, cfg)
            if emit_detection:
                output_selected = apply_final_output_score(output_selected, cfg)
            else:
                output_selected = None

        memory_update_candidate = output_selected if cfg.memory_update_selection == "final" else memory_selected
        if cfg.memory_update_selection == "temporal" and memory_update_candidate is not None:
            memory_emit, _, _ = should_emit_detection(memory_update_candidate, memory, cfg)
            if not memory_emit:
                memory_update_candidate = None

        if memory_update_candidate is not None:
            if memory is None:
                memory_write = True
                memory_write_reason = "initial_seed"
                memory_quality = memory_quality_score(memory_update_candidate, cfg)
                memory_update_candidate = _with_extra(
                    memory_update_candidate,
                    memory_quality=memory_quality,
                    memory_write=memory_write,
                    memory_write_reason=memory_write_reason,
                )
                memory = MotionMemoryTrack(memory_update_candidate.bbox_xyxy, score=memory_update_candidate.objectness, history=[memory_update_candidate.bbox_xyxy])
                memory.record_memory(memory_update_candidate, memory_quality, frame_id, cfg.memory_bank_size)
            else:
                memory_write, memory_quality, memory_write_reason = should_write_motion_memory(memory_update_candidate, cfg)
                memory_update_candidate = _with_extra(
                    memory_update_candidate,
                    memory_quality=memory_quality,
                    memory_write=memory_write,
                    memory_write_reason=memory_write_reason,
                )
                memory.update(
                    memory_update_candidate,
                    frame.shape,
                    write_memory=memory_write,
                    memory_quality=memory_quality,
                    frame_id=frame_id,
                    memory_bank_size=cfg.memory_bank_size,
                    camera_previous_to_current=camera_previous_to_current,
                    residual_velocity_momentum=cfg.residual_velocity_momentum,
                )
        selected = output_selected
        if selected is not None:
            selected = _with_extra(
                selected,
                memory_quality=memory_quality,
                memory_write=memory_write,
                memory_write_reason=memory_write_reason,
                emit_detection=emit_detection,
                emit_reason=emit_reason,
            )
        elif memory_update_candidate is None and memory is not None:
            memory.mark_miss(camera_previous_to_current)
            if memory.misses > cfg.miss_patience:
                memory = None

        rows.append(
            TemporalRecoveryFrame(
                frame_id=frame_id,
                candidates=merged,
                selected=selected,
                memory_bbox=memory.bbox_xyxy if memory is not None else None,
                diagnostics={
                    "detector_candidates": len(detector_candidates),
                    "support_candidates": len(support_candidates),
                    "profile": cfg.profile,
                    "memory_alive": memory is not None,
                    "memory_misses": memory.misses if memory is not None else None,
                    "memory_bank_size": len(memory.memory_bank) if memory is not None else 0,
                    "memory_write": memory_write,
                    "memory_quality": memory_quality,
                    "memory_write_reason": memory_write_reason,
                    "emit_detection": emit_detection,
                    "emit_reason": emit_reason,
                    "camera_motion": camera_motion_diagnostics,
                },
            )
        )
        prev_frame = frame
    return rows
