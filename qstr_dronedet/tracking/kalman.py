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

    def bbox(self) -> tuple[float, float, float, float]:
        cx, cy, w, h, *_ = self.state.tolist()
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class ConstantVelocityTracker:
    def __init__(self, r0: float = 24.0, alpha: float = 1.5, beta: float = 20.0, reacquire: float = 18.0) -> None:
        self.tracks: list[Track] = []
        self.next_id = 1
        self.r0 = r0
        self.alpha = alpha
        self.beta = beta
        self.reacquire = reacquire

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
        unmatched = set(range(len(detections)))
        for tr in self.tracks:
            best_j = None
            best_score = -1e9
            radius = self.r0 + self.alpha * self.compute_track_speed(tr) + self.beta * (1.0 - alignment_quality) + self.reacquire * max(0, tr.misses - 1)
            for j in list(unmatched):
                det = detections[j]
                dist = center_distance(tr.bbox(), det.bbox_xyxy)
                if dist > radius:
                    continue
                score = 2.0 * bbox_iou(tr.bbox(), det.bbox_xyxy) - dist / max(radius, 1e-6)
                if score > best_score:
                    best_score, best_j = score, j
            if best_j is not None:
                self._update_track(tr, detections[best_j])
                unmatched.remove(best_j)
        for j in unmatched:
            self._spawn(detections[j])
        self.tracks = [t for t in self.tracks if t.misses <= 8 and t.confidence > 0.05]

    def _spawn(self, det: DetectionCandidate) -> None:
        x1, y1, x2, y2 = det.bbox_xyxy
        state = np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(1, x2 - x1), max(1, y2 - y1), 0.0, 0.0], dtype=np.float32)
        center = (float(state[0]), float(state[1]))
        is_detector = self._is_detector_source(det.source)
        self.tracks.append(
            Track(
                state=state,
                confidence=max(0.2, det.objectness),
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
                    },
                )
            )
        return out

    @staticmethod
    def _is_detector_source(source: str) -> bool:
        parts = {s for s in source.split("+") if s}
        return any(("yolo" in s or "motion" in s or "seed" in s or "fallback" in s) for s in parts)

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

    def compute_track_speed(self, track: Track | None = None) -> float:
        if track is None:
            return max([self.compute_track_speed(t) for t in self.tracks], default=0.0)
        return float(np.hypot(track.state[4], track.state[5]))

    def track_confidence(self) -> float:
        return max([t.confidence for t in self.tracks], default=0.0)
