from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, hypot, log
from typing import Iterable

import numpy as np

from qstr_dronedet.camera_motion import transform_bbox_xyxy
from qstr_dronedet.tracking.action_bank import box_iou

EPS = 1e-6
BBox = tuple[float, float, float, float]


def _center(box: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _size(box: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return max(EPS, x2 - x1), max(EPS, y2 - y1)


def _shift_scale(box: BBox, dx: float, dy: float, dlogw: float, dlogh: float) -> BBox:
    cx, cy = _center(box)
    width, height = _size(box)
    width *= exp(float(np.clip(dlogw, -2.0, 2.0)))
    height *= exp(float(np.clip(dlogh, -2.0, 2.0)))
    cx += dx
    cy += dy
    return cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height


def _weighted(values: list[tuple[float, float]], now: float, seconds: float) -> float:
    eligible = [(timestamp, value) for timestamp, value in values if now - timestamp <= seconds + EPS]
    if not eligible:
        return 0.0
    weights = np.asarray([exp(-2.0 * max(0.0, now - timestamp) / seconds) for timestamp, _ in eligible])
    return float(np.average(np.asarray([value for _, value in eligible]), weights=weights))


@dataclass(frozen=True)
class ActionChunkCandidateScore:
    score: float
    predicted_iou: float
    center_similarity: float
    direction_similarity: float
    scale_similarity: float
    track_quality: float
    track_age_seconds: float
    acceleration_similarity: float = 0.0
    motion_stability: float = 0.0


@dataclass
class ActionChunkTrack:
    frame_id: int
    timestamp: float
    bbox: BBox
    quality: float
    observations: int = 1
    velocities_x: list[tuple[float, float]] = field(default_factory=list)
    velocities_y: list[tuple[float, float]] = field(default_factory=list)
    scale_w: list[tuple[float, float]] = field(default_factory=list)
    scale_h: list[tuple[float, float]] = field(default_factory=list)
    accelerations_x: list[tuple[float, float]] = field(default_factory=list)
    accelerations_y: list[tuple[float, float]] = field(default_factory=list)
    track_id: int = -1
    born_timestamp: float | None = None

    def __post_init__(self) -> None:
        if self.born_timestamp is None:
            self.born_timestamp = self.timestamp

    def clone(self) -> ActionChunkTrack:
        return ActionChunkTrack(
            self.frame_id, self.timestamp, self.bbox, self.quality, self.observations,
            list(self.velocities_x), list(self.velocities_y), list(self.scale_w), list(self.scale_h),
            list(self.accelerations_x), list(self.accelerations_y), self.track_id, self.born_timestamp,
        )

    def _motion(self, timestamp: float, short_seconds: float, long_seconds: float) -> tuple[float, float, float, float, float, float]:
        banks = (self.velocities_x, self.velocities_y, self.scale_w, self.scale_h, self.accelerations_x, self.accelerations_y)
        short = [_weighted(values, timestamp, short_seconds) for values in banks]
        long = [_weighted(values, timestamp, long_seconds) for values in banks]
        return tuple(0.72 * short[index] + 0.28 * long[index] for index in range(6))

    def candidate_action_tokens(
        self,
        candidate: BBox,
        timestamp: float,
        camera_transform: np.ndarray,
        seconds: float,
        token_count: int,
    ) -> list[float]:
        """Return causal, uniformly time-binned [valid, compatibility] action tokens."""
        token_count = max(1, int(token_count))
        bin_width = max(EPS, seconds / token_count)
        camera_box = tuple(float(value) for value in transform_bbox_xyxy(self.bbox, camera_transform))
        camera_center = _center(camera_box)
        candidate_center = _center(candidate)
        camera_width, camera_height = _size(camera_box)
        candidate_width, candidate_height = _size(candidate)
        dt = max(EPS, timestamp - self.timestamp)
        observed_vx = (candidate_center[0] - camera_center[0]) / dt
        observed_vy = (candidate_center[1] - camera_center[1]) / dt
        observed_sw = log(candidate_width / camera_width) / dt
        observed_sh = log(candidate_height / camera_height) / dt
        reference_side = max(5.0, 0.5 * (camera_width + camera_height + candidate_width + candidate_height))
        velocity_scale = max(5.0 / dt, reference_side / dt)
        series = (self.velocities_x, self.velocities_y, self.scale_w, self.scale_h, self.accelerations_x, self.accelerations_y)
        tokens: list[float] = []
        for index in range(token_count):
            newest_age = index * bin_width
            oldest_age = (index + 1) * bin_width
            newest = timestamp - newest_age
            oldest = timestamp - oldest_age
            values: list[float] = []
            valid = False
            for samples in series:
                selected = [value for stamp, value in samples if oldest - EPS <= stamp < newest + EPS]
                values.append(float(np.mean(selected)) if selected else 0.0)
                valid = valid or bool(selected)
            if not valid:
                tokens.extend((0.0, 0.0))
                continue
            vx, vy, scale_w, scale_h, acceleration_x, acceleration_y = values
            expected_vx = vx + acceleration_x * dt
            expected_vy = vy + acceleration_y * dt
            velocity_error = hypot(observed_vx - expected_vx, observed_vy - expected_vy)
            velocity_similarity = exp(-velocity_error / velocity_scale)
            observed_norm = hypot(observed_vx, observed_vy)
            expected_norm = hypot(expected_vx, expected_vy)
            if observed_norm < EPS or expected_norm < EPS:
                direction_similarity = velocity_similarity
            else:
                cosine = (observed_vx * expected_vx + observed_vy * expected_vy) / (observed_norm * expected_norm)
                direction_similarity = 0.5 + 0.5 * float(np.clip(cosine, -1.0, 1.0))
            scale_similarity = exp(-abs(observed_sw - scale_w) * dt - abs(observed_sh - scale_h) * dt)
            compatibility = 0.50 * velocity_similarity + 0.30 * direction_similarity + 0.20 * scale_similarity
            tokens.extend((1.0, float(np.clip(compatibility, 0.0, 1.0))))
        return tokens

    def candidate_motion_tokens(
        self,
        candidate: BBox,
        timestamp: float,
        camera_transform: np.ndarray,
        seconds: float,
        token_count: int,
        detector_score: float,
    ) -> list[float]:
        """Return causal Action Chunk tokens using real elapsed time."""
        token_count = max(1, int(token_count))
        bin_width = max(EPS, seconds / token_count)
        camera_box = tuple(float(value) for value in transform_bbox_xyxy(self.bbox, camera_transform))
        camera_center = _center(camera_box)
        candidate_center = _center(candidate)
        camera_width, camera_height = _size(camera_box)
        candidate_width, candidate_height = _size(candidate)
        reference_side = max(5.0, 0.5 * (camera_width + camera_height + candidate_width + candidate_height))
        dt = max(EPS, timestamp - self.timestamp)
        residual_dx = candidate_center[0] - camera_center[0]
        residual_dy = candidate_center[1] - camera_center[1]
        observed_vx = residual_dx / dt
        observed_vy = residual_dy / dt
        observed_sw = log(candidate_width / camera_width) / dt
        observed_sh = log(candidate_height / camera_height) / dt
        series = (self.velocities_x, self.velocities_y, self.scale_w, self.scale_h, self.accelerations_x, self.accelerations_y)
        tokens: list[float] = []
        for index in range(token_count):
            newest = timestamp - index * bin_width
            oldest = timestamp - (index + 1) * bin_width
            values: list[float] = []
            valid = False
            for samples in series:
                selected = [value for stamp, value in samples if oldest - EPS <= stamp < newest + EPS]
                values.append(float(np.mean(selected)) if selected else 0.0)
                valid = valid or bool(selected)
            if not valid:
                tokens.extend((0.0,) * 12)
                continue
            vx, vy, scale_w, scale_h, acceleration_x, acceleration_y = values
            expected_vx = vx + acceleration_x * dt
            expected_vy = vy + acceleration_y * dt
            predicted = _shift_scale(
                camera_box,
                expected_vx * dt,
                expected_vy * dt,
                scale_w * dt,
                scale_h * dt,
            )
            motion_iou = box_iou(candidate, predicted)
            velocity_error_x = (observed_vx - expected_vx) / reference_side
            velocity_error_y = (observed_vy - expected_vy) / reference_side
            scale_error_w = observed_sw - scale_w
            scale_error_h = observed_sh - scale_h
            velocity_similarity = exp(-hypot(velocity_error_x, velocity_error_y))
            scale_similarity = exp(-abs(scale_error_w) * dt - abs(scale_error_h) * dt)
            compatibility = 0.50 * motion_iou + 0.30 * velocity_similarity + 0.20 * scale_similarity
            tokens.extend((
                1.0,
                float(np.tanh(residual_dx / reference_side)),
                float(np.tanh(residual_dy / reference_side)),
                float(np.tanh(velocity_error_x)),
                float(np.tanh(velocity_error_y)),
                float(np.tanh(acceleration_x / reference_side)),
                float(np.tanh(acceleration_y / reference_side)),
                float(np.tanh(scale_error_w)),
                float(np.tanh(scale_error_h)),
                float(np.clip(motion_iou, 0.0, 1.0)),
                float(np.clip(detector_score, 0.0, 1.0)),
                float(np.clip(compatibility, 0.0, 1.0)),
            ))
        return tokens

    def predict(self, timestamp: float, camera_transform: np.ndarray, short_seconds: float, long_seconds: float) -> BBox:
        camera_box = tuple(float(value) for value in transform_bbox_xyxy(self.bbox, camera_transform))
        dt = max(EPS, timestamp - self.timestamp)
        velocity_x, velocity_y, scale_w, scale_h, acceleration_x, acceleration_y = self._motion(timestamp, short_seconds, long_seconds)
        width, height = _size(camera_box)
        base_dx, base_dy = velocity_x * dt, velocity_y * dt
        acceleration_dx = float(np.clip(0.5 * acceleration_x * dt * dt, -max(2.0 * width, 1.5 * abs(base_dx)), max(2.0 * width, 1.5 * abs(base_dx))))
        acceleration_dy = float(np.clip(0.5 * acceleration_y * dt * dt, -max(2.0 * height, 1.5 * abs(base_dy)), max(2.0 * height, 1.5 * abs(base_dy))))
        return _shift_scale(camera_box, base_dx + acceleration_dx, base_dy + acceleration_dy, scale_w * dt, scale_h * dt)

    def score_candidate(
        self,
        candidate: BBox,
        timestamp: float,
        camera_transform: np.ndarray,
        camera_validity: float,
        short_seconds: float = 1.0,
        long_seconds: float = 3.0,
    ) -> ActionChunkCandidateScore:
        predicted = self.predict(timestamp, camera_transform, short_seconds, long_seconds)
        predicted_iou = box_iou(predicted, candidate)
        pred_center = _center(predicted)
        candidate_center = _center(candidate)
        pred_width, pred_height = _size(predicted)
        candidate_width, candidate_height = _size(candidate)
        center_error = hypot(candidate_center[0] - pred_center[0], candidate_center[1] - pred_center[1])
        reference_side = max(5.0, 0.5 * (pred_width + pred_height + candidate_width + candidate_height))
        center_similarity = exp(-center_error / reference_side)
        scale_error = abs(log(candidate_width / pred_width)) + abs(log(candidate_height / pred_height))
        scale_similarity = exp(-scale_error)
        dt = max(EPS, timestamp - self.timestamp)
        camera_box = tuple(float(value) for value in transform_bbox_xyxy(self.bbox, camera_transform))
        camera_center = _center(camera_box)
        observed_velocity = ((candidate_center[0] - camera_center[0]) / dt, (candidate_center[1] - camera_center[1]) / dt)
        motion_state = self._motion(timestamp, short_seconds, long_seconds)
        expected_velocity = (motion_state[0] + motion_state[4] * dt, motion_state[1] + motion_state[5] * dt)
        observed_norm = hypot(*observed_velocity)
        expected_norm = hypot(*expected_velocity)
        if observed_norm < EPS or expected_norm < EPS:
            direction_similarity = center_similarity
        else:
            cosine = (observed_velocity[0] * expected_velocity[0] + observed_velocity[1] * expected_velocity[1]) / (observed_norm * expected_norm)
            direction_similarity = 0.5 + 0.5 * float(np.clip(cosine, -1.0, 1.0))
        velocity_error = hypot(observed_velocity[0] - expected_velocity[0], observed_velocity[1] - expected_velocity[1])
        velocity_scale = max(5.0 / dt, reference_side / dt)
        acceleration_similarity = exp(-velocity_error / velocity_scale)
        recent_velocity = [value for stamp, value in self.velocities_x if timestamp - stamp <= long_seconds + EPS] + [value for stamp, value in self.velocities_y if timestamp - stamp <= long_seconds + EPS]
        if len(recent_velocity) >= 4:
            velocity_mean = float(np.mean(np.abs(recent_velocity)))
            motion_stability = exp(-float(np.std(recent_velocity)) / max(1.0, velocity_mean))
        else:
            motion_stability = 0.5
        maturity = min(1.0, max(0.0, self.observations - 1) / 8.0)
        quality = float(np.clip(self.quality * (0.08 + 0.92 * maturity) * (0.75 + 0.25 * camera_validity), 0.0, 1.0))
        motion = 0.42 * predicted_iou + 0.23 * center_similarity + 0.15 * scale_similarity + 0.10 * direction_similarity + 0.10 * acceleration_similarity
        quality *= 0.85 + 0.15 * motion_stability
        score = 0.5 + (motion - 0.5) * quality
        return ActionChunkCandidateScore(float(np.clip(score, 0.0, 1.0)), predicted_iou, center_similarity, direction_similarity, scale_similarity, quality, max(0.0, timestamp - self.timestamp), acceleration_similarity, motion_stability)

    def update(
        self,
        frame_id: int,
        timestamp: float,
        candidate: BBox,
        detector_score: float,
        motion_score: float,
        camera_transform: np.ndarray,
        long_seconds: float = 3.0,
    ) -> None:
        dt = max(EPS, timestamp - self.timestamp)
        camera_box = tuple(float(value) for value in transform_bbox_xyxy(self.bbox, camera_transform))
        camera_center = _center(camera_box)
        candidate_center = _center(candidate)
        camera_width, camera_height = _size(camera_box)
        candidate_width, candidate_height = _size(candidate)
        observed_velocity_x = (candidate_center[0] - camera_center[0]) / dt
        observed_velocity_y = (candidate_center[1] - camera_center[1]) / dt
        previous_velocity_x = _weighted(self.velocities_x, self.timestamp, long_seconds) if self.velocities_x else observed_velocity_x
        previous_velocity_y = _weighted(self.velocities_y, self.timestamp, long_seconds) if self.velocities_y else observed_velocity_y
        self.velocities_x.append((timestamp, observed_velocity_x))
        self.velocities_y.append((timestamp, observed_velocity_y))
        self.accelerations_x.append((timestamp, (observed_velocity_x - previous_velocity_x) / dt))
        self.accelerations_y.append((timestamp, (observed_velocity_y - previous_velocity_y) / dt))
        self.scale_w.append((timestamp, log(candidate_width / camera_width) / dt))
        self.scale_h.append((timestamp, log(candidate_height / camera_height) / dt))
        cutoff = timestamp - long_seconds
        self.velocities_x = [item for item in self.velocities_x if item[0] >= cutoff]
        self.velocities_y = [item for item in self.velocities_y if item[0] >= cutoff]
        self.scale_w = [item for item in self.scale_w if item[0] >= cutoff]
        self.scale_h = [item for item in self.scale_h if item[0] >= cutoff]
        self.accelerations_x = [item for item in self.accelerations_x if item[0] >= cutoff]
        self.accelerations_y = [item for item in self.accelerations_y if item[0] >= cutoff]
        evidence = float(np.clip(0.58 * detector_score + 0.42 * motion_score, 0.0, 1.0))
        self.quality = float(np.clip(0.78 * self.quality + 0.22 * evidence, 0.0, 1.0))
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.bbox = candidate
        self.observations += 1
