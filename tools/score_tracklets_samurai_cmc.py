from __future__ import annotations

import argparse
import json
import math
import pickle
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from qstr_dronedet.camera_motion import estimate_background_homography, transform_bbox_xyxy
from qstr_dronedet.candidates.merge import bbox_iou


BBox = tuple[float, float, float, float]


def center(bbox: BBox) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5


def shift(bbox: BBox, dx: float, dy: float) -> BBox:
    return bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy


def raw_score(row: dict[str, object]) -> float:
    for key in ("score", "objectness", "final_drone_score"):
        try:
            return float(row.get(key, 0.0))
        except (TypeError, ValueError):
            pass
    return 0.0


def bbox(row: dict[str, object]) -> BBox:
    values = row.get("bbox")
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("row missing bbox")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


@lru_cache(maxsize=4096)
def _sequence_frame_root(frame_root: str, seq: str, frame_id: int) -> Path:
    root = Path(frame_root)
    name = f"{seq}_{frame_id:05d}.png"
    if (root / name).is_file():
        return root
    for candidate in sorted(root.glob("part*/frames")):
        if (candidate / name).is_file():
            return candidate
    return root


def frame_path(frame_root: Path, seq: str, frame_id: int) -> Path:
    return _sequence_frame_root(str(frame_root), seq, frame_id) / f"{seq}_{frame_id:05d}.png"


@dataclass
class MotionState:
    frame_id: int
    bbox: BBox
    residual_velocity: tuple[float, float]
    quality: float


class HomographyCache:
    def __init__(self, frame_root: Path, cache_path: Path | None, max_size: int) -> None:
        self.frame_root = frame_root
        self.cache_path = cache_path
        self.max_size = max_size
        self.values: dict[tuple[str, int], dict[str, object]] = {}
        self.sequence_sizes: dict[str, tuple[int, int]] = {}
        if cache_path and cache_path.is_file():
            with cache_path.open("rb") as handle:
                self.values = pickle.load(handle)
        self.computed = 0

    def adjacent(self, seq: str, frame_id: int) -> tuple[np.ndarray, bool]:
        key = (seq, frame_id)
        cached = self.values.get(key)
        if cached is not None:
            size = cached.get("source_size") or cached.get("image_size")
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                self.sequence_sizes[seq] = (int(size[0]), int(size[1]))
            return np.asarray(cached["matrix"], dtype=np.float64), bool(cached["valid"])
        previous = cv2.imread(str(frame_path(self.frame_root, seq, frame_id)), cv2.IMREAD_COLOR)
        current = cv2.imread(str(frame_path(self.frame_root, seq, frame_id + 1)), cv2.IMREAD_COLOR)
        if previous is None or current is None:
            matrix = np.eye(3, dtype=np.float64)
            valid = False
            payload = {"matrix": matrix, "valid": valid, "inlier_ratio": 0.0, "error": float("inf")}
        else:
            height, width = previous.shape[:2]
            self.sequence_sizes[seq] = (int(width), int(height))
            estimate = estimate_background_homography(previous, current, max_size=self.max_size)
            matrix = estimate.matrix
            valid = estimate.valid
            payload = {
                "matrix": matrix,
                "valid": valid,
                "inlier_ratio": estimate.inlier_ratio,
                "error": estimate.median_reprojection_error,
                "source_size": self.sequence_sizes[seq],
            }
        self.values[key] = payload
        self.computed += 1
        return matrix, valid

    def between(self, seq: str, source_frame: int, target_frame: int) -> tuple[np.ndarray, float]:
        if source_frame == target_frame:
            return np.eye(3, dtype=np.float64), 1.0
        matrix = np.eye(3, dtype=np.float64)
        valid_count = 0
        steps = abs(target_frame - source_frame)
        if target_frame > source_frame:
            for frame_id in range(source_frame, target_frame):
                adjacent, valid = self.adjacent(seq, frame_id)
                matrix = adjacent @ matrix
                valid_count += int(valid)
        else:
            for frame_id in range(source_frame - 1, target_frame - 1, -1):
                adjacent, valid = self.adjacent(seq, frame_id)
                try:
                    inverse = np.linalg.inv(adjacent)
                except np.linalg.LinAlgError:
                    inverse = np.eye(3, dtype=np.float64)
                    valid = False
                matrix = inverse @ matrix
                valid_count += int(valid)
        matrix /= matrix[2, 2]
        return matrix, valid_count / max(steps, 1)

    def sequence_size(self, seq: str, frame_id: int) -> tuple[int, int] | None:
        known = self.sequence_sizes.get(seq)
        if known is not None:
            return known
        image = cv2.imread(str(frame_path(self.frame_root, seq, frame_id)), cv2.IMREAD_COLOR)
        if image is None:
            return None
        height, width = image.shape[:2]
        size = (int(width), int(height))
        self.sequence_sizes[seq] = size
        return size

    def save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as handle:
            pickle.dump(self.values, handle, protocol=pickle.HIGHEST_PROTOCOL)


def directional_scores(
    rows: list[dict[str, object]],
    seq: str,
    cache: HomographyCache,
    *,
    reverse: bool,
    detector_gate: float,
    motion_gate: float,
    velocity_momentum: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    count = len(rows)
    support = [0.5] * count
    motion_ious = [0.0] * count
    residual_speeds = [0.0] * count
    camera_validity = [0.0] * count
    order = range(count - 1, -1, -1) if reverse else range(count)
    state: MotionState | None = None
    for index in order:
        row = rows[index]
        current_bbox = bbox(row)
        current_frame = int(float(row.get("frame_id", 0)))
        detector = raw_score(row)
        if state is None:
            if detector >= detector_gate:
                state = MotionState(current_frame, current_bbox, (0.0, 0.0), detector)
            continue
        transform, valid_ratio = cache.between(seq, state.frame_id, current_frame)
        camera_bbox = transform_bbox_xyxy(state.bbox, transform)
        gap = max(1, abs(current_frame - state.frame_id))
        predicted = shift(
            camera_bbox,
            state.residual_velocity[0] * gap,
            state.residual_velocity[1] * gap,
        )
        overlap = bbox_iou(current_bbox, predicted)
        current_center = center(current_bbox)
        camera_center = center(camera_bbox)
        residual_dx = (current_center[0] - camera_center[0]) / gap
        residual_dy = (current_center[1] - camera_center[1]) / gap
        residual_speed = math.hypot(residual_dx, residual_dy)
        side = max(4.0, current_bbox[2] - current_bbox[0], current_bbox[3] - current_bbox[1])
        residual_error = math.hypot(
            residual_dx - state.residual_velocity[0],
            residual_dy - state.residual_velocity[1],
        )
        residual_consistency = math.exp(-residual_error / max(2.0, side))
        motion = 0.72 * overlap + 0.28 * residual_consistency
        evidence = math.sqrt(max(0.0, min(1.0, detector * state.quality)))
        support[index] = 0.5 + (motion - 0.5) * evidence
        motion_ious[index] = overlap
        residual_speeds[index] = residual_speed
        camera_validity[index] = valid_ratio
        if detector >= detector_gate and motion >= motion_gate:
            momentum = min(1.0, max(0.0, velocity_momentum))
            velocity = (
                momentum * state.residual_velocity[0] + (1.0 - momentum) * residual_dx,
                momentum * state.residual_velocity[1] + (1.0 - momentum) * residual_dy,
            )
            quality = 0.55 * detector + 0.45 * motion
            state = MotionState(current_frame, current_bbox, velocity, quality)
    return support, motion_ious, residual_speeds, camera_validity


def attach_causal_camera_motion(
    rows: list[dict[str, object]],
    seq: str,
    cache: HomographyCache,
) -> None:
    if not rows:
        return
    sequence_size = cache.sequence_size(seq, int(float(rows[0].get("frame_id", 0))))
    if sequence_size is not None:
        for row in rows:
            row["image_width"], row["image_height"] = sequence_size
    rows[0]["camera_dx"] = 0.0
    rows[0]["camera_dy"] = 0.0
    rows[0]["camera_motion_validity"] = 0.0
    rows[0]["camera_motion_normalized"] = False
    rows[0]["camera_motion_gap_frames"] = 0
    for previous, current in zip(rows, rows[1:]):
        previous_frame = int(float(previous.get("frame_id", 0)))
        current_frame = int(float(current.get("frame_id", 0)))
        transform, valid_ratio = cache.between(seq, previous_frame, current_frame)
        previous_bbox = bbox(previous)
        camera_bbox = transform_bbox_xyxy(previous_bbox, transform)
        previous_center = center(previous_bbox)
        camera_center = center(camera_bbox)
        current["camera_dx"] = float(camera_center[0] - previous_center[0])
        current["camera_dy"] = float(camera_center[1] - previous_center[1])
        current["camera_motion_validity"] = float(valid_ratio)
        current["camera_motion_normalized"] = False
        current["camera_motion_gap_frames"] = abs(current_frame - previous_frame)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tracklets", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output-tracklets", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--homography-cache", type=Path)
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--detector-gate", type=float, default=0.03)
    parser.add_argument("--motion-gate", type=float, default=0.28)
    parser.add_argument("--velocity-momentum", type=float, default=0.65)
    parser.add_argument("--score-field", default="samurai_cmc_score")
    parser.add_argument("--causal-only", action="store_true", help="Use only past-to-current CMC and omit backward scoring.")
    parser.add_argument("--cache-save-every", type=int, default=10000)
    args = parser.parse_args()

    cache = HomographyCache(args.frame_root, args.homography_cache, args.max_size)
    args.output_tracklets.parent.mkdir(parents=True, exist_ok=True)
    total_tracklets = sum(1 for line in args.input_tracklets.open("r", encoding="utf-8-sig") if line.strip())
    tracklets = 0
    rows_scored = 0
    score_values: list[float] = []
    valid_values: list[float] = []
    with args.input_tracklets.open("r", encoding="utf-8-sig") as source, args.output_tracklets.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            meta = item.get("meta") or {}
            rows = item.get("rows") or []
            seq = str(meta.get("seq") or (rows[0].get("seq") if rows else ""))
            rows.sort(key=lambda row: int(float(row.get("frame_id", 0))))
            attach_causal_camera_motion(rows, seq, cache)
            forward, forward_iou, forward_speed, forward_valid = directional_scores(
                rows,
                seq,
                cache,
                reverse=False,
                detector_gate=args.detector_gate,
                motion_gate=args.motion_gate,
                velocity_momentum=args.velocity_momentum,
            )
            if args.causal_only:
                backward = forward
                backward_iou = forward_iou
                backward_speed = forward_speed
                backward_valid = forward_valid
            else:
                backward, backward_iou, backward_speed, backward_valid = directional_scores(
                    rows,
                    seq,
                    cache,
                    reverse=True,
                    detector_gate=args.detector_gate,
                    motion_gate=args.motion_gate,
                    velocity_momentum=args.velocity_momentum,
                )
            for index, row in enumerate(rows):
                score = forward[index] if args.causal_only else math.sqrt(max(1e-6, forward[index] * backward[index]))
                row[args.score_field] = score
                row["samurai_cmc_forward_score"] = forward[index]
                row["samurai_cmc_forward_iou"] = forward_iou[index]
                if not args.causal_only:
                    row["samurai_cmc_backward_score"] = backward[index]
                    row["samurai_cmc_backward_iou"] = backward_iou[index]
                row["samurai_cmc_residual_speed"] = forward_speed[index] if args.causal_only else 0.5 * (forward_speed[index] + backward_speed[index])
                row["samurai_cmc_camera_validity"] = forward_valid[index] if args.causal_only else 0.5 * (forward_valid[index] + backward_valid[index])
                score_values.append(score)
                valid_values.append(row["samurai_cmc_camera_validity"])
                rows_scored += 1
            if rows:
                track_scores = [float(row[args.score_field]) for row in rows]
                meta["samurai_cmc_track_score"] = float(np.mean(track_scores))
                meta["samurai_cmc_track_score_max"] = float(np.max(track_scores))
            target.write(json.dumps(item, separators=(",", ":")) + "\n")
            tracklets += 1
            if tracklets % 1000 == 0 or tracklets == total_tracklets:
                progress = {
                    "stage": "score",
                    "done": tracklets,
                    "total": total_tracklets,
                    "rows": rows_scored,
                    "homographies": len(cache.values),
                }
                if args.progress_json:
                    args.progress_json.parent.mkdir(parents=True, exist_ok=True)
                    args.progress_json.write_text(json.dumps(progress), encoding="utf-8")
                if tracklets % max(1, args.cache_save_every) == 0 or tracklets == total_tracklets:
                    cache.save()
                print(json.dumps(progress), flush=True)
    cache.save()
    summary = {
        "input": str(args.input_tracklets),
        "output": str(args.output_tracklets),
        "tracklets": tracklets,
        "rows": rows_scored,
        "homographies": len(cache.values),
        "new_homographies": cache.computed,
        "mean_score": float(np.mean(score_values)) if score_values else None,
        "mean_camera_validity": float(np.mean(valid_values)) if valid_values else None,
        "score_field": args.score_field,
        "causal_only": args.causal_only,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



