from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from qstr_dronedet.camera_motion import estimate_background_homography


def action_chunk_frame_path(frame_root: Path, sequence: str, frame_id: int) -> Path:
    stem = f"{sequence}_{frame_id:05d}"
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = frame_root / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return frame_root / f"{stem}.png"


class ActionChunkCameraMotionCache:
    def __init__(self, frame_root: Path, cache_path: Path | None, max_size: int = 320) -> None:
        self.frame_root = frame_root
        self.cache_path = cache_path
        self.max_size = max_size
        self.values: dict[tuple[str, int], dict[str, object]] = {}
        self.sequence_sizes: dict[str, tuple[int, int]] = {}
        if cache_path and cache_path.is_file():
            with cache_path.open("rb") as handle:
                self.values = pickle.load(handle)
        self.computed = 0

    def adjacent(self, sequence: str, frame_id: int) -> tuple[np.ndarray, bool]:
        key = (sequence, frame_id)
        cached = self.values.get(key)
        if cached is not None:
            size = cached.get("source_size") or cached.get("image_size")
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                self.sequence_sizes[sequence] = (int(size[0]), int(size[1]))
            return np.asarray(cached["matrix"], dtype=np.float64), bool(cached["valid"])
        previous = cv2.imread(str(action_chunk_frame_path(self.frame_root, sequence, frame_id)), cv2.IMREAD_COLOR)
        current = cv2.imread(str(action_chunk_frame_path(self.frame_root, sequence, frame_id + 1)), cv2.IMREAD_COLOR)
        if previous is None or current is None:
            matrix = np.eye(3, dtype=np.float64)
            valid = False
            payload: dict[str, object] = {"matrix": matrix, "valid": valid, "inlier_ratio": 0.0, "error": float("inf")}
        else:
            height, width = previous.shape[:2]
            self.sequence_sizes[sequence] = (int(width), int(height))
            estimate = estimate_background_homography(previous, current, max_size=self.max_size)
            matrix = estimate.matrix
            valid = estimate.valid
            payload = {"matrix": matrix, "valid": valid, "inlier_ratio": estimate.inlier_ratio, "error": estimate.median_reprojection_error, "source_size": self.sequence_sizes[sequence]}
        self.values[key] = payload
        self.computed += 1
        return matrix, valid

    def between(self, sequence: str, source_frame: int, target_frame: int) -> tuple[np.ndarray, float]:
        if source_frame == target_frame:
            return np.eye(3, dtype=np.float64), 1.0
        matrix = np.eye(3, dtype=np.float64)
        valid_count = 0
        steps = abs(target_frame - source_frame)
        if target_frame > source_frame:
            for frame_id in range(source_frame, target_frame):
                adjacent, valid = self.adjacent(sequence, frame_id)
                matrix = adjacent @ matrix
                valid_count += int(valid)
        else:
            for frame_id in range(source_frame - 1, target_frame - 1, -1):
                adjacent, valid = self.adjacent(sequence, frame_id)
                try:
                    inverse = np.linalg.inv(adjacent)
                except np.linalg.LinAlgError:
                    inverse = np.eye(3, dtype=np.float64)
                    valid = False
                matrix = inverse @ matrix
                valid_count += int(valid)
        matrix /= matrix[2, 2]
        return matrix, valid_count / max(steps, 1)

    def sequence_size(self, sequence: str, frame_id: int) -> tuple[int, int] | None:
        known = self.sequence_sizes.get(sequence)
        if known is not None:
            return known
        image = cv2.imread(str(action_chunk_frame_path(self.frame_root, sequence, frame_id)), cv2.IMREAD_COLOR)
        if image is None:
            return None
        height, width = image.shape[:2]
        size = (int(width), int(height))
        self.sequence_sizes[sequence] = size
        return size

    def save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as handle:
            pickle.dump(self.values, handle, protocol=pickle.HIGHEST_PROTOCOL)
