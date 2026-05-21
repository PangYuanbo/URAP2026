from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qstr_dronedet.candidates.merge import bbox_iou, center_distance
from qstr_dronedet.fusion.modes import determine_mode
from qstr_dronedet.motion.difference import compute_multik_motion, motion_score_in_bbox
from qstr_dronedet.tracking.kalman import ConstantVelocityTracker
from qstr_dronedet.types import DetectionCandidate


def _read_frames(video: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _best_track_match(track_candidates: list[DetectionCandidate], gt_box: tuple[float, float, float, float]) -> tuple[DetectionCandidate | None, float, float]:
    best = None
    best_iou = 0.0
    best_dist = float("inf")
    for cand in track_candidates:
        iou = bbox_iou(cand.bbox_xyxy, gt_box)
        dist = center_distance(cand.bbox_xyxy, gt_box)
        if iou > best_iou or (iou == best_iou and dist < best_dist):
            best = cand
            best_iou = iou
            best_dist = dist
    return best, best_iou, best_dist


def run_tracker_oracle_benchmark(
    metadata_paths: list[str | Path],
    out_dir: str | Path,
    detection_stride: int = 5,
    max_frames: int | None = 80,
    match_iou: float = 0.1,
    match_center_px: float = 16.0,
    tracker_r0: float = 24.0,
    tracker_alpha: float = 1.5,
    tracker_beta: float = 20.0,
    tracker_reacquire: float = 18.0,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for meta_like in metadata_paths:
        meta_path = Path(meta_like)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        video = Path(meta["output"])
        if not video.exists():
            video = meta_path.parent / video.name
        frames = _read_frames(video)
        if max_frames is not None:
            frames = frames[:max_frames]
        boxes = {int(b["frame_id"]): tuple(float(v) for v in b["bbox_xyxy"]) for b in meta.get("boxes", [])}
        tracker = ConstantVelocityTracker(r0=tracker_r0, alpha=tracker_alpha, beta=tracker_beta, reacquire=tracker_reacquire)
        for frame_id, frame in enumerate(frames):
            if frame_id not in boxes:
                continue
            gt_box = boxes[frame_id]
            use_detection = frame_id % max(1, detection_stride) == 0
            detections = [DetectionCandidate(gt_box, objectness=0.9, source="oracle")] if use_detection else []
            if frame_id == 0:
                motion_map = np.zeros(frame.shape[:2], np.uint8)
                alignment_quality = 0.0
                blur_score = 0.0
            else:
                motion = compute_multik_motion(frames[: frame_id + 1], frame_id, (1, 2, 4))
                motion_map = motion["motion_map"]
                alignment_quality = float(motion["best_quality"])
                blur_score = 0.0
                if motion["best_k"] is not None:
                    aln = motion["per_k"][motion["best_k"]]["alignment"]
                    blur_score = float(getattr(aln, "blur_score", 0.0)) if aln is not None else 0.0
            tracker.update(detections, alignment_quality=alignment_quality)
            track_candidates = tracker.get_track_candidates()
            best, best_iou, best_dist = _best_track_match(track_candidates, gt_box)
            matched = best is not None and (best_iou >= match_iou or best_dist <= match_center_px)
            track_speed = tracker.compute_track_speed()
            track_conf = tracker.track_confidence()
            motion_score = motion_score_in_bbox(motion_map, gt_box)
            mode = determine_mode(motion_score, alignment_quality, track_speed, blur_score, track_conf)
            rows.append(
                {
                    "metadata": str(meta_path),
                    "video": str(video),
                    "frame_id": frame_id,
                    "gt_bbox": list(gt_box),
                    "used_detection": use_detection,
                    "matched": matched,
                    "best_iou": best_iou,
                    "best_center_dist": best_dist if np.isfinite(best_dist) else None,
                    "track_confidence": track_conf,
                    "track_speed": track_speed,
                    "motion_score": motion_score,
                    "alignment_quality": alignment_quality,
                    "mode": mode,
                    "num_tracks": len(track_candidates),
                }
            )
    total = len(rows)
    no_det_rows = [r for r in rows if not r["used_detection"]]
    matched = sum(1 for r in rows if r["matched"])
    matched_no_det = sum(1 for r in no_det_rows if r["matched"])
    summary = {
        "num_frames": total,
        "survival_rate": matched / max(1, total),
        "survival_without_detection_rate": matched_no_det / max(1, len(no_det_rows)),
        "mean_track_confidence": float(np.mean([r["track_confidence"] for r in rows])) if rows else 0.0,
        "mean_track_speed": float(np.mean([r["track_speed"] for r in rows])) if rows else 0.0,
        "mean_motion_score": float(np.mean([r["motion_score"] for r in rows])) if rows else 0.0,
        "mean_alignment_quality": float(np.mean([r["alignment_quality"] for r in rows])) if rows else 0.0,
        "mode_counts": dict(Counter(r["mode"] for r in rows)),
        "per_video": {},
    }
    for video, group in _group_by(rows, "video").items():
        g_no_det = [r for r in group if not r["used_detection"]]
        summary["per_video"][video] = {
            "frames": len(group),
            "survival_rate": sum(1 for r in group if r["matched"]) / max(1, len(group)),
            "survival_without_detection_rate": sum(1 for r in g_no_det if r["matched"]) / max(1, len(g_no_det)),
            "mean_track_speed": float(np.mean([r["track_speed"] for r in group])),
            "mean_motion_score": float(np.mean([r["motion_score"] for r in group])),
            "mode_counts": dict(Counter(r["mode"] for r in group)),
        }
    (out_dir / "tracker_oracle_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (out_dir / "tracker_oracle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row[key]), []).append(row)
    return out
