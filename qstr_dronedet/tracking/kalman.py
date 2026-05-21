from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qstr_dronedet.candidates.merge import bbox_iou, center_distance
from qstr_dronedet.types import DetectionCandidate


@dataclass
class Track:
    state: np.ndarray
    confidence: float = 0.8
    age: int = 0
    misses: int = 0
    track_id: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)
    last_update_center: tuple[float, float] | None = None
    last_detector_center: tuple[float, float] | None = None
    last_detector_source: str = ""
    frames_since_detector_update: int = 0
    detector_updates: int = 0
    evidence_history: list[dict[str, float]] = field(default_factory=list)

    def bbox(self) -> tuple[float, float, float, float]:
        cx, cy, w, h, *_ = self.state.tolist()
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class ConstantVelocityTracker:
    def __init__(
        self,
        r0: float = 24.0,
        alpha: float = 1.5,
        beta: float = 20.0,
        reacquire: float = 18.0,
        fallback_bonus: float = 36.0,
        tiny_bonus: float = 18.0,
        max_spawn_per_frame: int = 12,
        high_score_threshold: float = 0.25,
        low_score_threshold: float = 0.05,
        evidence_window: int = 8,
    ) -> None:
        self.tracks: list[Track] = []
        self.next_id = 1
        self.r0 = r0
        self.alpha = alpha
        self.beta = beta
        self.reacquire = reacquire
        self.fallback_bonus = fallback_bonus
        self.tiny_bonus = tiny_bonus
        self.max_spawn_per_frame = max_spawn_per_frame
        self.high_score_threshold = high_score_threshold
        self.low_score_threshold = low_score_threshold
        self.evidence_window = evidence_window

    def predict(self) -> None:
        for tr in self.tracks:
            tr.state[0] += tr.state[4]
            tr.state[1] += tr.state[5]
            tr.age += 1
            tr.misses += 1
            tr.frames_since_detector_update += 1
            tr.confidence *= 0.92

    def update(self, detections: list[DetectionCandidate], alignment_quality: float = 1.0) -> None:
        self.predict()
        high_indices = {i for i, det in enumerate(detections) if self._is_high_score_detection(det)}
        low_indices = {i for i, det in enumerate(detections) if i not in high_indices and self._is_low_score_recovery_detection(det)}
        unmatched_high = set(high_indices)
        unmatched_tracks = self._associate_stage(detections, unmatched_high, list(range(len(self.tracks))), alignment_quality, low_stage=False)
        unmatched_low = set(low_indices)
        self._associate_stage(detections, unmatched_low, unmatched_tracks, alignment_quality, low_stage=True)
        matched = high_indices - unmatched_high
        low_matched = low_indices - unmatched_low
        matched_all = matched | low_matched
        unmatched = set(range(len(detections))) - matched_all
        spawnable = [detections[j] for j in unmatched if self._can_spawn(detections[j]) and self._is_high_score_detection(detections[j])]
        spawnable = sorted(spawnable, key=lambda d: (0 if "fallback" in d.source else 1, -d.objectness))
        for det in spawnable[: self.max_spawn_per_frame]:
            self._spawn(det)
        self.tracks = [t for t in self.tracks if t.misses <= 8 and t.confidence > 0.05]

    def _associate_stage(
        self,
        detections: list[DetectionCandidate],
        unmatched_detections: set[int],
        track_indices: list[int],
        alignment_quality: float,
        low_stage: bool,
    ) -> list[int]:
        still_unmatched_tracks: list[int] = []
        for tr_idx in track_indices:
            if tr_idx >= len(self.tracks):
                continue
            tr = self.tracks[tr_idx]
            best_j = None
            best_score = -1e9
            radius_base = self.r0 + self.alpha * self.compute_track_speed(tr) + self.beta * (1.0 - alignment_quality) + self.reacquire * max(0, tr.misses - 1)
            for j in list(unmatched_detections):
                det = detections[j]
                radius = radius_base + self._association_bonus(det)
                if low_stage:
                    radius += 0.5 * self.reacquire
                dist = center_distance(tr.bbox(), det.bbox_xyxy)
                if dist > radius:
                    continue
                source_bonus = 0.15 if "fallback" in det.source else 0.0
                score = 2.0 * bbox_iou(tr.bbox(), det.bbox_xyxy) - dist / max(radius, 1e-6) + source_bonus
                if score > best_score:
                    best_score, best_j = score, j
            if best_j is not None:
                self._update_track(tr, detections[best_j])
                unmatched_detections.remove(best_j)
            else:
                still_unmatched_tracks.append(tr_idx)
        return still_unmatched_tracks

    def _is_high_score_detection(self, det: DetectionCandidate) -> bool:
        if not self._can_spawn(det):
            return False
        if "fallback" in det.source:
            return det.objectness >= max(0.12, self.low_score_threshold)
        return det.objectness >= self.high_score_threshold or "oracle" in det.source or "seed" in det.source

    def _is_low_score_recovery_detection(self, det: DetectionCandidate) -> bool:
        if not self._is_detector_source(det.source):
            return False
        if "tracker" in det.source and det.source == "tracker":
            return False
        if "fallback" in det.source:
            return det.objectness >= self.low_score_threshold
        x1, y1, x2, y2 = det.bbox_xyxy
        side = max(float(x2 - x1), float(y2 - y1))
        return side <= 128.0 and det.objectness >= self.low_score_threshold

    def _association_bonus(self, det: DetectionCandidate) -> float:
        x1, y1, x2, y2 = det.bbox_xyxy
        side = max(float(x2 - x1), float(y2 - y1))
        bonus = 0.0
        if "fallback" in det.source:
            bonus += self.fallback_bonus
        if side <= 128.0:
            bonus += self.tiny_bonus
        return bonus

    def _can_spawn(self, det: DetectionCandidate) -> bool:
        if not self._is_detector_source(det.source):
            return False
        if "tracker" in det.source and det.source == "tracker":
            return False
        if "fallback" in det.source:
            return det.objectness >= 0.08
        if "oracle" in det.source:
            return True
        if "motion" in det.source and det.objectness < 0.15:
            return False
        return det.objectness >= 0.05

    def _spawn(self, det: DetectionCandidate) -> None:
        x1, y1, x2, y2 = det.bbox_xyxy
        state = np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(1, x2 - x1), max(1, y2 - y1), 0.0, 0.0], dtype=np.float32)
        center = (float(state[0]), float(state[1]))
        is_detector = self._is_detector_source(det.source)
        self.tracks.append(
            Track(
                state=state,
                confidence=max(0.3 if "fallback" in det.source else 0.2, det.objectness),
                track_id=self.next_id,
                history=[center],
                last_update_center=center,
                last_detector_center=center if is_detector else None,
                last_detector_source=det.source if is_detector else "",
                frames_since_detector_update=0 if is_detector else 999,
                detector_updates=1 if is_detector else 0,
            )
        )
        self.next_id += 1

    def _update_track(self, tr: Track, det: DetectionCandidate) -> None:
        old = tr.state.copy()
        missed = tr.misses
        x1, y1, x2, y2 = det.bbox_xyxy
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, max(1, x2 - x1), max(1, y2 - y1)
        measurement = np.array([cx, cy, w, h], dtype=np.float32)
        if missed > 1 and tr.last_update_center is not None:
            dt = float(missed + 1)
            vx = (float(cx) - tr.last_update_center[0]) / dt
            vy = (float(cy) - tr.last_update_center[1]) / dt
            tr.state[:4] = measurement
            tr.state[4] = 0.3 * tr.state[4] + 0.7 * vx
            tr.state[5] = 0.3 * tr.state[5] + 0.7 * vy
        else:
            tr.state[:4] = 0.35 * tr.state[:4] + 0.65 * measurement
            tr.state[4] = tr.state[0] - old[0]
            tr.state[5] = tr.state[1] - old[1]
        tr.confidence = min(1.0, 0.6 * tr.confidence + 0.4 * det.objectness + 0.1)
        tr.misses = 0
        tr.last_update_center = (float(cx), float(cy))
        if self._is_detector_source(det.source):
            tr.last_detector_center = (float(cx), float(cy))
            tr.last_detector_source = det.source
            tr.frames_since_detector_update = 0
            tr.detector_updates += 1
        tr.history.append((float(tr.state[0]), float(tr.state[1])))
        tr.history = tr.history[-20:]

    def get_track_candidates(self) -> list[DetectionCandidate]:
        out = []
        for tr in self.tracks:
            if tr.confidence < 0.25:
                continue
            drift = self.track_drift(tr)
            out.append(
                DetectionCandidate(
                    tr.bbox(),
                    objectness=0.35 * tr.confidence,
                    source="tracker",
                    track_score=tr.confidence,
                    extra={
                        "track_id": tr.track_id,
                        "track_age": tr.age,
                        "track_history_len": len(tr.history),
                        "track_detector_updates": tr.detector_updates,
                        "track_last_detector_source": tr.last_detector_source,
                        "track_frames_since_detector_update": tr.frames_since_detector_update,
                        "track_drift": drift,
                        "track_speed": self.compute_track_speed(tr),
                        "track_validated": self.is_validated_track(tr),
                        **self.track_evidence_summary(tr),
                    },
                )
            )
        return out

    @staticmethod
    def _is_detector_source(source: str) -> bool:
        parts = {s for s in source.split("+") if s}
        return any(("yolo" in s or "motion" in s or "seed" in s or "fallback" in s or "oracle" in s) for s in parts)

    def track_drift(self, track: Track) -> float:
        if track.last_detector_center is None:
            return float("inf")
        cx, cy = float(track.state[0]), float(track.state[1])
        return float(np.hypot(cx - track.last_detector_center[0], cy - track.last_detector_center[1]))

    def is_validated_track(
        self,
        track: Track,
        max_frames_since_detector_update: int = 3,
        min_detector_updates: int = 1,
        max_drift: float = 48.0,
        min_history_len: int = 2,
    ) -> bool:
        return (
            track.detector_updates >= min_detector_updates
            and track.frames_since_detector_update <= max_frames_since_detector_update
            and self.track_drift(track) <= max_drift
            and len(track.history) >= min_history_len
        )

    def update_evidence(
        self,
        track_id: int | None,
        crop_probs: dict[str, float],
        temporal_probs: dict[str, float],
        final_probs: dict[str, float],
    ) -> None:
        if track_id is None:
            return
        for tr in self.tracks:
            if tr.track_id == int(track_id):
                tr.evidence_history.append(
                    {
                        "crop_drone": float(crop_probs.get("drone", 0.0)),
                        "crop_background": float(crop_probs.get("background", 0.0)),
                        "temporal_drone": float(temporal_probs.get("drone", 0.0)),
                        "temporal_background": float(temporal_probs.get("background", 0.0)),
                        "final_drone": float(final_probs.get("drone", 0.0)),
                        "final_background": float(final_probs.get("background", 0.0)),
                    }
                )
                tr.evidence_history = tr.evidence_history[-self.evidence_window :]
                return

    def track_evidence_summary(self, track: Track) -> dict[str, float | int | bool]:
        hist = track.evidence_history[-self.evidence_window :]
        if not hist:
            return {
                "track_evidence_len": 0,
                "track_crop_drone_mean": 0.0,
                "track_temporal_drone_mean": 0.0,
                "track_background_mean": 1.0,
                "track_temporal_gain_rate": 0.0,
                "track_recognition_confirmed": False,
            }
        crop_mean = float(np.mean([e["crop_drone"] for e in hist]))
        temporal_mean = float(np.mean([e["temporal_drone"] for e in hist]))
        bg_mean = float(np.mean([max(e["crop_background"], e["temporal_background"], e["final_background"]) for e in hist]))
        gain_rate = float(np.mean([e["temporal_drone"] > e["crop_drone"] + 0.05 for e in hist]))
        confirmed = len(hist) >= 2 and temporal_mean >= 0.55 and crop_mean >= 0.38 and bg_mean <= 0.68 and gain_rate >= 0.5
        return {
            "track_evidence_len": len(hist),
            "track_crop_drone_mean": crop_mean,
            "track_temporal_drone_mean": temporal_mean,
            "track_background_mean": bg_mean,
            "track_temporal_gain_rate": gain_rate,
            "track_recognition_confirmed": confirmed,
        }

    def compute_track_speed(self, track: Track | None = None) -> float:
        if track is None:
            return max([self.compute_track_speed(t) for t in self.tracks], default=0.0)
        return float(np.hypot(track.state[4], track.state[5]))

    def track_confidence(self) -> float:
        return max([t.confidence for t in self.tracks], default=0.0)
