from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from qstr_dronedet.features.roi import crop_with_context
from qstr_dronedet.tracking.action_chunk import actions_from_boxes, reconstruct_boxes


@dataclass(frozen=True)
class VideoActionPolicyResult:
    out_path: Path
    summary: dict[str, Any]


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _row_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    value = row.get("bbox") or row.get("bbox_xyxy")
    if value is None:
        value = [row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")]
    if value is None or len(value) != 4:
        raise ValueError("row must contain bbox/bbox_xyxy or x1/y1/x2/y2")
    x1, y1, x2, y2 = [float(v) for v in value]
    return x1, y1, x2, y2


def _row_score(row: dict[str, Any]) -> float:
    return float(max(row.get("objectness", 0.0), row.get("final_drone_score", 0.0), row.get("score", 0.0)))


def _row_visible(row: dict[str, Any]) -> float:
    return float(row.get("visible", True))


def _row_image_size(row: dict[str, Any]) -> tuple[int, int] | None:
    width = row.get("image_width")
    height = row.get("image_height")
    if width is None or height is None:
        size = row.get("image_size")
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            width, height = size[0], size[1]
    if width is None or height is None:
        return None
    width_i = int(float(width))
    height_i = int(float(height))
    if width_i <= 0 or height_i <= 0:
        return None
    return width_i, height_i


def _normalize_box(box: tuple[float, float, float, float], image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = image_size
    x1, y1, x2, y2 = box
    return x1 / max(1, width), y1 / max(1, height), x2 / max(1, width), y2 / max(1, height)


def _constant_velocity_actions_from_past(past_boxes: np.ndarray, future_len: int) -> np.ndarray:
    boxes = np.asarray(past_boxes, dtype=np.float32)
    if len(boxes) >= 2:
        step = actions_from_boxes(boxes[-2:])[0]
    else:
        step = np.zeros((4,), dtype=np.float32)
    return np.tile(step.reshape(1, 4), (future_len, 1)).astype(np.float32)


def _motion_feature_vector(norm_boxes: np.ndarray, scores: np.ndarray, visible: np.ndarray) -> np.ndarray:
    boxes = np.asarray(norm_boxes, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    visible_arr = np.asarray(visible, dtype=np.float32).reshape(-1)
    centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5))
    sizes = np.column_stack((np.maximum(0.0, boxes[:, 2] - boxes[:, 0]), np.maximum(0.0, boxes[:, 3] - boxes[:, 1])))
    areas = sizes[:, 0] * sizes[:, 1]
    aspect = sizes[:, 0] / np.maximum(sizes[:, 1], 1e-6)
    if len(boxes) >= 2:
        center_delta = np.diff(centers, axis=0)
        size_delta = np.diff(sizes, axis=0)
        speed = np.linalg.norm(center_delta, axis=1)
        last_center_delta = center_delta[-1]
        last_size_delta = size_delta[-1]
    else:
        center_delta = np.zeros((1, 2), dtype=np.float32)
        size_delta = np.zeros((1, 2), dtype=np.float32)
        speed = np.zeros((1,), dtype=np.float32)
        last_center_delta = np.zeros((2,), dtype=np.float32)
        last_size_delta = np.zeros((2,), dtype=np.float32)
    if len(center_delta) >= 2:
        accel = np.diff(center_delta, axis=0)
        accel_norm = np.linalg.norm(accel, axis=1)
        last_accel = accel[-1]
    else:
        accel_norm = np.zeros((1,), dtype=np.float32)
        last_accel = np.zeros((2,), dtype=np.float32)
    score_delta = np.diff(scores_arr) if len(scores_arr) >= 2 else np.zeros((1,), dtype=np.float32)
    features = np.asarray(
        [
            centers[-1, 0],
            centers[-1, 1],
            sizes[-1, 0],
            sizes[-1, 1],
            areas[-1],
            aspect[-1],
            last_center_delta[0],
            last_center_delta[1],
            float(speed[-1]) if len(speed) else 0.0,
            float(np.mean(speed)) if len(speed) else 0.0,
            float(np.std(speed)) if len(speed) else 0.0,
            last_accel[0],
            last_accel[1],
            float(np.mean(accel_norm)) if len(accel_norm) else 0.0,
            last_size_delta[0],
            last_size_delta[1],
            float(np.mean(areas)),
            float(np.std(areas)),
            float(scores_arr[-1]) if len(scores_arr) else 0.0,
            float(np.mean(scores_arr)) if len(scores_arr) else 0.0,
            float(np.max(scores_arr)) if len(scores_arr) else 0.0,
            float(np.std(scores_arr)) if len(scores_arr) else 0.0,
            float(np.mean(score_delta)) if len(score_delta) else 0.0,
            float(np.mean(visible_arr)) if len(visible_arr) else 0.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def _row_float(row: dict[str, Any], names: tuple[str, ...], default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return float(default)


def _ego_motion_feature_vector(rows: list[dict[str, Any]], norm_boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(norm_boxes, dtype=np.float32)
    centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5))
    if len(centers) >= 2:
        apparent_delta = np.diff(centers, axis=0)
        apparent_speed = np.linalg.norm(apparent_delta, axis=1)
        last_apparent_delta = apparent_delta[-1]
    else:
        apparent_delta = np.zeros((1, 2), dtype=np.float32)
        apparent_speed = np.zeros((1,), dtype=np.float32)
        last_apparent_delta = np.zeros((2,), dtype=np.float32)
    if len(apparent_delta) >= 2:
        apparent_accel = np.diff(apparent_delta, axis=0)
        last_apparent_accel = apparent_accel[-1]
        apparent_accel_norm = np.linalg.norm(apparent_accel, axis=1)
    else:
        last_apparent_accel = np.zeros((2,), dtype=np.float32)
        apparent_accel_norm = np.zeros((1,), dtype=np.float32)

    camera_dx = np.asarray([_row_float(row, ("camera_dx", "ego_dx", "global_dx", "flow_dx"), 0.0) for row in rows], dtype=np.float32)
    camera_dy = np.asarray([_row_float(row, ("camera_dy", "ego_dy", "global_dy", "flow_dy"), 0.0) for row in rows], dtype=np.float32)
    camera_speed = np.asarray(
        [
            _row_float(
                row,
                ("camera_speed", "ego_speed", "global_speed"),
                float(np.hypot(camera_dx[index], camera_dy[index])),
            )
            for index, row in enumerate(rows)
        ],
        dtype=np.float32,
    )
    warp_error = np.asarray([_row_float(row, ("warp_error", "ego_warp_error", "global_warp_error"), 0.0) for row in rows], dtype=np.float32)
    camera_quality = np.asarray([_row_float(row, ("camera_motion_quality", "ego_quality", "warp_quality"), 1.0) for row in rows], dtype=np.float32)
    if len(camera_dx) >= 2:
        camera_accel_x = np.diff(camera_dx)
        camera_accel_y = np.diff(camera_dy)
        camera_accel = np.hypot(camera_accel_x, camera_accel_y)
        last_camera_accel = np.asarray([camera_accel_x[-1], camera_accel_y[-1]], dtype=np.float32)
    else:
        camera_accel = np.zeros((1,), dtype=np.float32)
        last_camera_accel = np.zeros((2,), dtype=np.float32)

    camera_signal = float(np.mean(np.abs(camera_dx)) + np.mean(np.abs(camera_dy)) + np.mean(np.abs(warp_error)))
    fallback_ego_speed = float(np.mean(apparent_speed)) if camera_signal <= 1e-8 else float(np.mean(camera_speed))
    fallback_ego_accel = float(np.mean(apparent_accel_norm)) if camera_signal <= 1e-8 else float(np.mean(camera_accel))
    fallback_warp_error = float(np.mean(apparent_accel_norm)) if float(np.mean(warp_error)) <= 1e-8 else float(np.mean(warp_error))
    features = np.asarray(
        [
            float(camera_dx[-1]) if len(camera_dx) else 0.0,
            float(camera_dy[-1]) if len(camera_dy) else 0.0,
            float(np.mean(camera_dx)) if len(camera_dx) else 0.0,
            float(np.mean(camera_dy)) if len(camera_dy) else 0.0,
            fallback_ego_speed,
            float(np.std(camera_speed)) if camera_signal > 1e-8 and len(camera_speed) else float(np.std(apparent_speed)),
            float(last_camera_accel[0]) if camera_signal > 1e-8 else float(last_apparent_accel[0]),
            float(last_camera_accel[1]) if camera_signal > 1e-8 else float(last_apparent_accel[1]),
            fallback_ego_accel,
            fallback_warp_error,
            float(np.mean(camera_quality)) if len(camera_quality) else 1.0,
            float(last_apparent_delta[0]),
            float(last_apparent_delta[1]),
            float(np.mean(apparent_speed)) if len(apparent_speed) else 0.0,
            float(np.std(apparent_speed)) if len(apparent_speed) else 0.0,
            float(np.mean(apparent_accel_norm)) if len(apparent_accel_norm) else 0.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def _remap_missing_direct_frame_path(path: Path, frame_root: str | Path | None) -> Path:
    if frame_root is None or path.exists():
        return path
    root = Path(frame_root)
    candidates = [root / path.name]
    parts_lower = [part.lower() for part in path.parts]
    for split_name in ("train", "val", "test"):
        if split_name in parts_lower:
            split_index = parts_lower.index(split_name)
            tail = path.parts[split_index + 1 :]
            if tail:
                candidates.append(root.joinpath(*tail))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_frame_path(row: dict[str, Any], frame_root: str | Path | None, image_name_template: str) -> Path | None:
    direct = row.get("frame_path") or row.get("image_path")
    if direct:
        path = Path(str(direct))
        if frame_root is not None and not path.is_absolute():
            path = Path(frame_root) / path
        return _remap_missing_direct_frame_path(path, frame_root)
    if frame_root is None:
        return None
    values = dict(row)
    try:
        frame_id = int(float(values.get("frame_id", 0) or 0))
    except (TypeError, ValueError):
        frame_id = 0
    values["frame_id_int"] = frame_id
    values["frame_id_05d"] = f"{frame_id:05d}"
    values["frame_id_06d"] = f"{frame_id:06d}"
    name = image_name_template.format(**values)
    return Path(frame_root) / name


def _load_frame(path: Path | None, allow_missing_images: bool, fallback_size: tuple[int, int]) -> np.ndarray:
    if path is not None:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is not None:
            return frame
    if not allow_missing_images:
        raise FileNotFoundError(f"could not read frame: {path}")
    width, height = fallback_size
    return np.zeros((height, width, 3), dtype=np.uint8)


class VideoActionTrackletDataset(Dataset):
    def __init__(
        self,
        tracklet_jsonl: str | Path,
        frame_root: str | Path | None = None,
        image_name_template: str = "{seq}_{frame_id_05d}.png",
        past_len: int = 4,
        future_len: int = 2,
        crop_size: int = 64,
        crop_scale: float = 4.0,
        image_size: tuple[int, int] | None = None,
        min_tracklet_rows: int = 0,
        max_samples: int | None = None,
        allow_missing_images: bool = False,
        frame_cache_size: int = 8,
        use_crops: bool = True,
    ) -> None:
        if past_len <= 0 or future_len <= 0:
            raise ValueError("past_len and future_len must be positive")
        self.frame_root = frame_root
        self.image_name_template = image_name_template
        self.past_len = past_len
        self.future_len = future_len
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.image_size = image_size
        self.allow_missing_images = allow_missing_images
        self.frame_cache_size = max(0, int(frame_cache_size))
        self.use_crops = bool(use_crops)
        self._frame_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.items = _read_jsonl(tracklet_jsonl)
        self._rows_by_item: list[list[dict[str, Any]]] = []
        self.samples: list[tuple[int, int]] = []
        total = past_len + future_len
        for item_index, item in enumerate(self.items):
            rows = sorted(list(item.get("rows") or []), key=lambda row: int(float(row.get("frame_id", 0) or 0)))
            self._rows_by_item.append(rows)
            if min_tracklet_rows > 0 and len(rows) < min_tracklet_rows:
                continue
            for start in range(0, len(rows) - total + 1):
                self.samples.append((item_index, start))
                if max_samples is not None and len(self.samples) >= max_samples:
                    return

    def __len__(self) -> int:
        return len(self.samples)

    def _window(self, index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        item_index, start = self.samples[index]
        item = self.items[item_index]
        rows = self._rows_by_item[item_index]
        return dict(item.get("meta") or {}), rows[start : start + self.past_len + self.future_len]

    def _load_frame_cached(self, path: Path | None, fallback_size: tuple[int, int]) -> np.ndarray:
        if path is None or self.frame_cache_size <= 0:
            return _load_frame(path, self.allow_missing_images, fallback_size=fallback_size)
        key = str(path)
        cached = self._frame_cache.get(key)
        if cached is not None:
            self._frame_cache.move_to_end(key)
            return cached
        frame = _load_frame(path, self.allow_missing_images, fallback_size=fallback_size)
        self._frame_cache[key] = frame
        self._frame_cache.move_to_end(key)
        while len(self._frame_cache) > self.frame_cache_size:
            self._frame_cache.popitem(last=False)
        return frame

    def __getitem__(self, index: int) -> dict[str, Any]:
        meta, rows = self._window(index)
        past_rows = rows[: self.past_len]
        boxes = [_row_box(row) for row in rows]
        image_size = _row_image_size(past_rows[-1]) or self.image_size
        frame = None
        if image_size is None:
            frame_path = _resolve_frame_path(past_rows[-1], self.frame_root, self.image_name_template)
            if self.use_crops:
                frame = self._load_frame_cached(frame_path, fallback_size=(256, 256))
                image_size = (int(frame.shape[1]), int(frame.shape[0]))
            else:
                image_size = (256, 256)
        norm_boxes = np.asarray([_normalize_box(box, image_size) for box in boxes], dtype=np.float32)
        crops = []
        if self.use_crops:
            for row, box in zip(past_rows, boxes[: self.past_len]):
                path = _resolve_frame_path(row, self.frame_root, self.image_name_template)
                row_frame = frame if path is None and frame is not None else self._load_frame_cached(path, fallback_size=image_size)
                crop = crop_with_context(row_frame, box, scale=self.crop_scale, out_size=self.crop_size)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                crops.append(np.transpose(rgb, (2, 0, 1)))
        else:
            crops = [np.zeros((3, self.crop_size, self.crop_size), dtype=np.float32) for _ in past_rows]
        past_scores = np.asarray([_row_score(row) for row in past_rows], dtype=np.float32).reshape(self.past_len, 1)
        past_visible = np.asarray([_row_visible(row) for row in past_rows], dtype=np.float32).reshape(self.past_len, 1)
        state = np.concatenate([norm_boxes[: self.past_len], past_scores, past_visible], axis=1).astype(np.float32)
        motion_features = _motion_feature_vector(norm_boxes[: self.past_len], past_scores, past_visible)
        ego_motion_features = _ego_motion_feature_vector(past_rows, norm_boxes[: self.past_len])
        target_actions = actions_from_boxes(norm_boxes[self.past_len - 1 :]).astype(np.float32)
        cv_actions = _constant_velocity_actions_from_past(norm_boxes[: self.past_len], self.future_len)
        target_action_residual = (target_actions - cv_actions).astype(np.float32)
        ego_instability = float(np.clip(abs(ego_motion_features[8]) + abs(ego_motion_features[9]), 0.0, 1.0))
        target_action_weight = float(np.clip(1.0 - ego_instability, 0.15, 1.0))
        window_scores = np.asarray([_row_score(row) for row in rows], dtype=np.float32)
        target_confidence_mean = float(np.mean(window_scores)) if len(window_scores) else 0.0
        target_confidence_max = float(np.max(window_scores)) if len(window_scores) else 0.0
        label = int(float(meta.get("label", 0)))
        return {
            "crops": torch.tensor(np.stack(crops), dtype=torch.float32),
            "state": torch.tensor(state, dtype=torch.float32),
            "motion_features": torch.tensor(motion_features, dtype=torch.float32),
            "ego_motion_features": torch.tensor(ego_motion_features, dtype=torch.float32),
            "future_actions": torch.tensor(target_actions, dtype=torch.float32),
            "future_action_residual": torch.tensor(target_action_residual, dtype=torch.float32),
            "target_action_weight": torch.tensor(target_action_weight, dtype=torch.float32),
            "target_confidence_mean": torch.tensor(target_confidence_mean, dtype=torch.float32),
            "target_confidence_max": torch.tensor(target_confidence_max, dtype=torch.float32),
            "target_motion_action": torch.tensor(float(label), dtype=torch.float32),
            "past_boxes": torch.tensor(norm_boxes[: self.past_len], dtype=torch.float32),
            "future_boxes": torch.tensor(norm_boxes[self.past_len :], dtype=torch.float32),
            "seq": str(meta.get("seq", past_rows[0].get("seq", ""))),
            "track_id": str(meta.get("track_id", past_rows[0].get("track_id", ""))),
            "raw_track_id": str(meta.get("raw_track_id", past_rows[0].get("raw_track_id", past_rows[0].get("track_id", "")))),
            "anchor_frame": int(float(past_rows[-1].get("frame_id", 0) or 0)),
            "label": label,
            "bucket": str(meta.get("bucket", "")),
            "dataset_source": str(meta.get("dataset_source", "")),
        }


class VideoActionChunkTransformer(torch.nn.Module):
    def __init__(self, past_len: int, future_len: int, d_model: int = 96, nhead: int = 4, num_layers: int = 2, crop_size: int = 64) -> None:
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        self.crop_size = crop_size
        self.visual = torch.nn.Sequential(
            torch.nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(48, d_model),
        )
        self.state = torch.nn.Sequential(torch.nn.Linear(6, d_model), torch.nn.ReLU(inplace=True), torch.nn.Linear(d_model, d_model))
        self.pos = torch.nn.Parameter(torch.zeros(1, past_len, d_model))
        layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True, dropout=0.0)
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, future_len * 4))

    def forward(self, crops: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        bsz, steps, channels, height, width = crops.shape
        visual = self.visual(crops.reshape(bsz * steps, channels, height, width)).reshape(bsz, steps, -1)
        tokens = visual + self.state(state) + self.pos[:, :steps, :]
        encoded = self.encoder(tokens)
        return self.head(encoded[:, -1, :]).reshape(bsz, self.future_len, 4)


class VideoActionMultiHeadTransformer(torch.nn.Module):
    def __init__(self, past_len: int, future_len: int, d_model: int = 96, nhead: int = 4, num_layers: int = 2, crop_size: int = 64) -> None:
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        self.crop_size = crop_size
        self.visual = torch.nn.Sequential(
            torch.nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(48, d_model),
        )
        self.state = torch.nn.Sequential(torch.nn.Linear(6, d_model), torch.nn.ReLU(inplace=True), torch.nn.Linear(d_model, d_model))
        self.pos = torch.nn.Parameter(torch.zeros(1, past_len, d_model))
        layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True, dropout=0.0)
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        self.action_head = torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, future_len * 4))
        self.confidence_head = torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, 1))

    def encode(self, crops: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        bsz, steps, channels, height, width = crops.shape
        visual = self.visual(crops.reshape(bsz * steps, channels, height, width)).reshape(bsz, steps, -1)
        tokens = visual + self.state(state) + self.pos[:, :steps, :]
        encoded = self.encoder(tokens)
        return encoded[:, -1, :]

    def forward(self, crops: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.encode(crops, state)
        actions = self.action_head(token).reshape(crops.shape[0], self.future_len, 4)
        confidence = torch.sigmoid(self.confidence_head(token)).squeeze(1)
        return actions, confidence


class VATDMotionActionTransformer(torch.nn.Module):
    """Octo-style no-language video-action model for short drone-like motion scoring."""

    def __init__(
        self,
        past_len: int,
        future_len: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        crop_size: int = 64,
        motion_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        self.crop_size = crop_size
        self.motion_feature_dim = int(motion_feature_dim)
        self.visual = torch.nn.Sequential(
            torch.nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(48, d_model),
        )
        self.state = torch.nn.Sequential(torch.nn.Linear(6, d_model), torch.nn.ReLU(inplace=True), torch.nn.Linear(d_model, d_model))
        self.pos = torch.nn.Parameter(torch.zeros(1, past_len, d_model))
        self.motion_readout = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        self.action_readout = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True, dropout=0.0)
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        if self.motion_feature_dim > 0:
            self.motion_features = torch.nn.Sequential(
                torch.nn.LayerNorm(self.motion_feature_dim),
                torch.nn.Linear(self.motion_feature_dim, d_model),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(d_model, d_model),
            )
        else:
            self.motion_features = None
        self.motion_action_head = torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, 1))
        self.action_residual_head = torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, future_len * 4))

    def forward(self, crops: torch.Tensor, state: torch.Tensor, motion_features: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, steps, channels, height, width = crops.shape
        visual = self.visual(crops.reshape(bsz * steps, channels, height, width)).reshape(bsz, steps, -1)
        tokens = visual + self.state(state) + self.pos[:, :steps, :]
        readouts = torch.cat(
            [
                self.motion_readout.expand(bsz, -1, -1),
                self.action_readout.expand(bsz, -1, -1),
            ],
            dim=1,
        )
        encoded = self.encoder(torch.cat([tokens, readouts], dim=1))
        motion_token = encoded[:, steps, :]
        if self.motion_features is not None:
            if motion_features is None:
                motion_features = torch.zeros((bsz, self.motion_feature_dim), dtype=motion_token.dtype, device=motion_token.device)
            motion_token = motion_token + self.motion_features(motion_features)
        action_token = encoded[:, steps + 1, :]
        motion_logits = self.motion_action_head(motion_token).squeeze(1)
        action_residual = self.action_residual_head(action_token).reshape(bsz, self.future_len, 4)
        return motion_logits, action_residual


class EgoAdaptiveVATDTransformer(torch.nn.Module):
    """Camera-conditioned multi-horizon VATD with soft adaptive action chunks."""

    def __init__(
        self,
        past_len: int,
        future_len: int,
        horizons: tuple[int, ...] = (3, 5, 7),
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        crop_size: int = 64,
        motion_feature_dim: int = 24,
        ego_feature_dim: int = 16,
    ) -> None:
        super().__init__()
        if past_len <= 0 or future_len <= 0:
            raise ValueError("past_len and future_len must be positive")
        valid_horizons = tuple(sorted({int(h) for h in horizons if 0 < int(h) <= int(past_len)}))
        if not valid_horizons:
            raise ValueError("horizons must contain at least one value in [1, past_len]")
        self.past_len = int(past_len)
        self.future_len = int(future_len)
        self.horizons = valid_horizons
        self.crop_size = int(crop_size)
        self.motion_feature_dim = int(motion_feature_dim)
        self.ego_feature_dim = int(ego_feature_dim)
        self.visual = torch.nn.Sequential(
            torch.nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(48, d_model),
        )
        self.state = torch.nn.Sequential(torch.nn.Linear(6, d_model), torch.nn.ReLU(inplace=True), torch.nn.Linear(d_model, d_model))
        self.pos = torch.nn.Parameter(torch.zeros(1, past_len, d_model))
        layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True, dropout=0.0)
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        self.motion_features = torch.nn.Sequential(
            torch.nn.LayerNorm(self.motion_feature_dim),
            torch.nn.Linear(self.motion_feature_dim, d_model),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(d_model, d_model),
        )
        self.ego_features = torch.nn.Sequential(
            torch.nn.LayerNorm(self.ego_feature_dim),
            torch.nn.Linear(self.ego_feature_dim, d_model),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(d_model, d_model),
        )
        self.router = torch.nn.Sequential(
            torch.nn.LayerNorm(self.ego_feature_dim),
            torch.nn.Linear(self.ego_feature_dim, d_model),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(d_model, len(self.horizons)),
        )
        self.motion_heads = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, 1)) for _ in self.horizons])
        self.action_heads = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, future_len * 4)) for _ in self.horizons])

    def forward(
        self,
        crops: torch.Tensor,
        state: torch.Tensor,
        motion_features: torch.Tensor,
        ego_motion_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, steps, channels, height, width = crops.shape
        if steps != self.past_len:
            raise ValueError(f"expected {self.past_len} steps, got {steps}")
        visual = self.visual(crops.reshape(bsz * steps, channels, height, width)).reshape(bsz, steps, -1)
        encoded = self.encoder(visual + self.state(state) + self.pos[:, :steps, :])
        motion_embed = self.motion_features(motion_features)
        ego_embed = self.ego_features(ego_motion_features)
        router_weights = torch.softmax(self.router(ego_motion_features), dim=1)
        motion_logits_per_horizon = []
        action_residual_per_horizon = []
        for head_index, horizon in enumerate(self.horizons):
            token = encoded[:, steps - horizon : steps, :].mean(dim=1) + motion_embed + ego_embed
            motion_logits_per_horizon.append(self.motion_heads[head_index](token).squeeze(1))
            action_residual_per_horizon.append(self.action_heads[head_index](token).reshape(bsz, self.future_len, 4))
        motion_logits_h = torch.stack(motion_logits_per_horizon, dim=1)
        action_residual_h = torch.stack(action_residual_per_horizon, dim=1)
        motion_logits = torch.sum(router_weights * motion_logits_h, dim=1)
        action_residual = torch.sum(router_weights[:, :, None, None] * action_residual_h, dim=1)
        return motion_logits, action_residual, router_weights, motion_logits_h


def _center_error(pred_boxes: np.ndarray, target_boxes: np.ndarray) -> float:
    if len(pred_boxes) == 0:
        return 0.0
    pred_centers = np.column_stack(((pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0, (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0))
    target_centers = np.column_stack(((target_boxes[:, 0] + target_boxes[:, 2]) / 2.0, (target_boxes[:, 1] + target_boxes[:, 3]) / 2.0))
    return float(np.mean(np.linalg.norm(pred_centers - target_centers, axis=1)))


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "crops": torch.stack([item["crops"] for item in batch]),
        "state": torch.stack([item["state"] for item in batch]),
        "motion_features": torch.stack([item["motion_features"] for item in batch]),
        "ego_motion_features": torch.stack([item["ego_motion_features"] for item in batch]),
        "future_actions": torch.stack([item["future_actions"] for item in batch]),
        "future_action_residual": torch.stack([item["future_action_residual"] for item in batch]),
        "target_action_weight": torch.stack([item["target_action_weight"] for item in batch]),
        "target_confidence_mean": torch.stack([item["target_confidence_mean"] for item in batch]),
        "target_confidence_max": torch.stack([item["target_confidence_max"] for item in batch]),
        "target_motion_action": torch.stack([item["target_motion_action"] for item in batch]),
        "past_boxes": torch.stack([item["past_boxes"] for item in batch]),
        "future_boxes": torch.stack([item["future_boxes"] for item in batch]),
        "meta": [{k: item[k] for k in ["seq", "track_id", "raw_track_id", "anchor_frame", "label", "bucket", "dataset_source"]} for item in batch],
    }


def train_video_action_chunk_policy(
    tracklet_jsonl: str | Path,
    out: str | Path,
    frame_root: str | Path | None = None,
    image_name_template: str = "{seq}_{frame_id_05d}.png",
    past_len: int = 4,
    future_len: int = 2,
    crop_size: int = 64,
    crop_scale: float = 4.0,
    image_size: tuple[int, int] | None = None,
    min_tracklet_rows: int = 0,
    max_samples: int | None = None,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 16,
    d_model: int = 96,
    nhead: int = 4,
    num_layers: int = 2,
    allow_missing_images: bool = False,
    verbose: bool = False,
) -> Path:
    dataset = VideoActionTrackletDataset(
        tracklet_jsonl,
        frame_root=frame_root,
        image_name_template=image_name_template,
        past_len=past_len,
        future_len=future_len,
        crop_size=crop_size,
        crop_scale=crop_scale,
        image_size=image_size,
        min_tracklet_rows=min_tracklet_rows,
        max_samples=max_samples,
        allow_missing_images=allow_missing_images,
    )
    if len(dataset) == 0:
        raise ValueError("video-action dataset has no samples")
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, collate_fn=_collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    model = VideoActionChunkTransformer(past_len=past_len, future_len=future_len, d_model=d_model, nhead=nhead, num_layers=num_layers, crop_size=crop_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.SmoothL1Loss()
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for batch in loader:
            crops = batch["crops"].to(device)
            state = batch["state"].to(device)
            target = batch["future_actions"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(crops, state), target)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(crops.shape[0])
        total += int(crops.shape[0])
        epoch_row = {"epoch": epoch + 1, "loss": total_loss / max(1, total)}
        epoch_row["device"] = str(device)
        epoch_row["cuda_name"] = cuda_name
        epoch_row["cuda_memory_allocated_mb"] = (
            round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3) if device.type == "cuda" else 0.0
        )
        history.append(epoch_row)
        if verbose:
            print(json.dumps({"video_action_train": epoch_row}, ensure_ascii=False), flush=True)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "past_len": past_len,
            "future_len": future_len,
            "crop_size": crop_size,
            "crop_scale": crop_scale,
            "image_size": list(image_size) if image_size is not None else None,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "num_samples": len(dataset),
            "history": history,
            "model_type": "video_action_chunk_transformer",
        },
        out_path,
    )
    return out_path


def train_video_action_multihead_policy(
    tracklet_jsonl: str | Path,
    out: str | Path,
    frame_root: str | Path | None = None,
    image_name_template: str = "{seq}_{frame_id_05d}.png",
    past_len: int = 4,
    future_len: int = 2,
    crop_size: int = 64,
    crop_scale: float = 4.0,
    image_size: tuple[int, int] | None = None,
    min_tracklet_rows: int = 0,
    max_samples: int | None = None,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 16,
    d_model: int = 96,
    nhead: int = 4,
    num_layers: int = 2,
    confidence_target: str = "max",
    confidence_loss_weight: float = 1.0,
    allow_missing_images: bool = False,
    num_workers: int = 0,
    frame_cache_size: int = 8,
    verbose: bool = False,
) -> Path:
    if confidence_target not in {"mean", "max"}:
        raise ValueError("confidence_target must be 'mean' or 'max'")
    dataset = VideoActionTrackletDataset(
        tracklet_jsonl,
        frame_root=frame_root,
        image_name_template=image_name_template,
        past_len=past_len,
        future_len=future_len,
        crop_size=crop_size,
        crop_scale=crop_scale,
        image_size=image_size,
        min_tracklet_rows=min_tracklet_rows,
        max_samples=max_samples,
        allow_missing_images=allow_missing_images,
        frame_cache_size=frame_cache_size,
    )
    if len(dataset) == 0:
        raise ValueError("video-action dataset has no samples")
    loader_batch_size = min(batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=loader_batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    model = VideoActionMultiHeadTransformer(
        past_len=past_len,
        future_len=future_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        crop_size=crop_size,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    action_loss_fn = torch.nn.SmoothL1Loss()
    confidence_loss_fn = torch.nn.MSELoss()
    history = []
    target_key = f"target_confidence_{confidence_target}"
    if verbose:
        print(
            json.dumps(
                {
                    "kind": "video_action_multihead_train_start",
                    "device": str(device),
                    "cuda_name": cuda_name,
                    "samples_total": len(dataset),
                    "batches_per_epoch": len(loader),
                    "epochs": epochs,
                    "batch_size": loader_batch_size,
                    "num_workers": num_workers,
                    "frame_cache_size": frame_cache_size,
                }
            ),
            flush=True,
        )
    for epoch in range(epochs):
        total_action_loss = 0.0
        total_confidence_loss = 0.0
        total_loss = 0.0
        total = 0
        progress_interval = max(1, min(25, len(loader) // 20))
        for batch_index, batch in enumerate(loader, start=1):
            crops = batch["crops"].to(device, non_blocking=device.type == "cuda")
            state = batch["state"].to(device, non_blocking=device.type == "cuda")
            target_actions = batch["future_actions"].to(device, non_blocking=device.type == "cuda")
            target_confidence = batch[target_key].to(device, non_blocking=device.type == "cuda")
            opt.zero_grad(set_to_none=True)
            pred_actions, pred_confidence = model(crops, state)
            action_loss = action_loss_fn(pred_actions, target_actions)
            confidence_loss = confidence_loss_fn(pred_confidence, target_confidence)
            loss = action_loss + float(confidence_loss_weight) * confidence_loss
            loss.backward()
            opt.step()
            batch_size_actual = int(crops.shape[0])
            total_action_loss += float(action_loss.item()) * batch_size_actual
            total_confidence_loss += float(confidence_loss.item()) * batch_size_actual
            total_loss += float(loss.item()) * batch_size_actual
            total += batch_size_actual
            if verbose and (batch_index == 1 or batch_index % progress_interval == 0 or batch_index == len(loader)):
                print(
                    json.dumps(
                        {
                            "kind": "video_action_multihead_train_progress",
                            "epoch": epoch + 1,
                            "epochs": epochs,
                            "batches_done": batch_index,
                            "batches_total": len(loader),
                            "samples_done": min(batch_index * loader_batch_size, len(dataset)),
                            "samples_total": len(dataset),
                            "loss_running": total_loss / max(1, total),
                            "device": str(device),
                            "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3)
                            if device.type == "cuda"
                            else 0.0,
                        }
                    ),
                    flush=True,
                )
        epoch_row = {
            "epoch": epoch + 1,
            "loss": total_loss / max(1, total),
            "action_loss": total_action_loss / max(1, total),
            "confidence_loss": total_confidence_loss / max(1, total),
        }
        epoch_row["device"] = str(device)
        epoch_row["cuda_name"] = cuda_name
        epoch_row["cuda_memory_allocated_mb"] = (
            round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3) if device.type == "cuda" else 0.0
        )
        history.append(epoch_row)
        if verbose:
            print(json.dumps({"video_action_multihead_train": epoch_row}, ensure_ascii=False), flush=True)
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        epoch_path = out_path.with_name(f"{out_path.stem}_epoch{epoch + 1:03d}{out_path.suffix}")
        torch.save(
            {
                "state_dict": model.cpu().state_dict(),
                "past_len": past_len,
                "future_len": future_len,
                "crop_size": crop_size,
                "crop_scale": crop_scale,
                "image_size": list(image_size) if image_size is not None else None,
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "num_samples": len(dataset),
                "history": list(history),
                "confidence_target": confidence_target,
                "confidence_loss_weight": confidence_loss_weight,
                "num_workers": num_workers,
                "frame_cache_size": frame_cache_size,
                "model_type": "video_action_multihead_transformer",
                "epoch_checkpoint": epoch + 1,
            },
            epoch_path,
        )
        model.to(device)
        if verbose:
            print(json.dumps({"kind": "video_action_multihead_epoch_checkpoint", "epoch": epoch + 1, "path": str(epoch_path)}, ensure_ascii=False), flush=True)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "past_len": past_len,
            "future_len": future_len,
            "crop_size": crop_size,
            "crop_scale": crop_scale,
            "image_size": list(image_size) if image_size is not None else None,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "num_samples": len(dataset),
            "history": history,
            "confidence_target": confidence_target,
            "confidence_loss_weight": confidence_loss_weight,
            "num_workers": num_workers,
            "frame_cache_size": frame_cache_size,
            "model_type": "video_action_multihead_transformer",
        },
        out_path,
    )
    return out_path


def train_vatd_motion_action_policy(
    tracklet_jsonl: str | Path,
    out: str | Path,
    frame_root: str | Path | None = None,
    image_name_template: str = "{seq}_{frame_id_05d}.png",
    past_len: int = 4,
    future_len: int = 2,
    crop_size: int = 64,
    crop_scale: float = 4.0,
    image_size: tuple[int, int] | None = None,
    min_tracklet_rows: int = 0,
    max_samples: int | None = None,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 16,
    d_model: int = 96,
    nhead: int = 4,
    num_layers: int = 2,
    action_loss_weight: float = 0.25,
    allow_missing_images: bool = False,
    num_workers: int = 0,
    frame_cache_size: int = 8,
    motion_pos_weight: float | str | None = "auto",
    pin_memory: bool | None = None,
    shuffle: bool = True,
    use_crops: bool = True,
    verbose: bool = False,
) -> Path:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    dataset = VideoActionTrackletDataset(
        tracklet_jsonl,
        frame_root=frame_root,
        image_name_template=image_name_template,
        past_len=past_len,
        future_len=future_len,
        crop_size=crop_size,
        crop_scale=crop_scale,
        image_size=image_size,
        min_tracklet_rows=min_tracklet_rows,
        max_samples=max_samples,
        allow_missing_images=allow_missing_images,
        frame_cache_size=frame_cache_size,
        use_crops=use_crops,
    )
    if len(dataset) == 0:
        raise ValueError("VATD video-action dataset has no samples")
    sample_labels = []
    for item_index, _start in dataset.samples:
        meta = dict(dataset.items[item_index].get("meta") or {})
        sample_labels.append(int(float(meta.get("label", 0))) > 0)
    positives = int(sum(sample_labels))
    negatives = int(len(sample_labels) - positives)
    if motion_pos_weight == "auto":
        pos_weight_value = float(negatives / max(1, positives)) if positives > 0 else 1.0
    elif motion_pos_weight is None:
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(motion_pos_weight)
    if pos_weight_value <= 0:
        raise ValueError("motion_pos_weight must be positive, 'auto', or None")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    use_pin_memory = (device.type == "cuda") if pin_memory is None else bool(pin_memory)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )
    model = VATDMotionActionTransformer(
        past_len=past_len,
        future_len=future_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        crop_size=crop_size,
        motion_feature_dim=24,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    motion_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device))
    action_loss_fn = torch.nn.SmoothL1Loss()
    history = []
    if verbose:
        print(
            json.dumps(
                {
                    "kind": "vatd_motion_action_train_start",
                    "device": str(device),
                    "cuda_name": cuda_name,
                    "epochs": epochs,
                    "batches_per_epoch": len(loader),
                    "samples_total": len(dataset),
                    "batch_size": min(batch_size, len(dataset)),
                    "num_workers": num_workers,
                    "frame_cache_size": frame_cache_size,
                    "pin_memory": use_pin_memory,
                    "shuffle": bool(shuffle),
                    "action_loss_weight": float(action_loss_weight),
                    "motion_pos_weight": float(pos_weight_value),
                    "positive_samples": positives,
                    "negative_samples": negatives,
                    "use_crops": bool(use_crops),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    for epoch in range(epochs):
        total_motion_loss = 0.0
        total_action_loss = 0.0
        total_loss = 0.0
        total = 0
        for batch_index, batch in enumerate(loader, start=1):
            non_blocking = device.type == "cuda" and use_pin_memory
            crops = batch["crops"].to(device, non_blocking=non_blocking)
            state = batch["state"].to(device, non_blocking=non_blocking)
            motion_features = batch["motion_features"].to(device, non_blocking=non_blocking)
            target_motion = batch["target_motion_action"].to(device, non_blocking=non_blocking)
            target_residual = batch["future_action_residual"].to(device, non_blocking=non_blocking)
            opt.zero_grad(set_to_none=True)
            motion_logits, pred_residual = model(crops, state, motion_features)
            motion_loss = motion_loss_fn(motion_logits, target_motion)
            action_loss = action_loss_fn(pred_residual, target_residual)
            loss = motion_loss + float(action_loss_weight) * action_loss
            loss.backward()
            opt.step()
            batch_size_actual = int(crops.shape[0])
            total_motion_loss += float(motion_loss.item()) * batch_size_actual
            total_action_loss += float(action_loss.item()) * batch_size_actual
            total_loss += float(loss.item()) * batch_size_actual
            total += batch_size_actual
            progress_interval = max(1, min(10, len(loader) // 20))
            if verbose and (batch_index == 1 or batch_index % progress_interval == 0 or batch_index == len(loader)):
                print(
                    json.dumps(
                        {
                            "kind": "vatd_motion_action_train_progress",
                            "epoch": epoch + 1,
                            "epochs": epochs,
                            "batches_done": batch_index,
                            "batches_total": len(loader),
                            "samples_done": min(batch_index * int(loader.batch_size or 1), len(dataset)),
                            "samples_total": len(dataset),
                            "loss": total_loss / max(1, total),
                            "motion_action_loss": total_motion_loss / max(1, total),
                            "action_residual_loss": total_action_loss / max(1, total),
                            "device": str(device),
                            "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3)
                            if device.type == "cuda"
                            else 0.0,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        epoch_row = {
            "epoch": epoch + 1,
            "loss": total_loss / max(1, total),
            "motion_action_loss": total_motion_loss / max(1, total),
            "action_residual_loss": total_action_loss / max(1, total),
            "device": str(device),
            "cuda_name": cuda_name,
            "num_workers": num_workers,
            "frame_cache_size": frame_cache_size,
            "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3) if device.type == "cuda" else 0.0,
        }
        history.append(epoch_row)
        if verbose:
            print(json.dumps({"vatd_motion_action_train": epoch_row}, ensure_ascii=False), flush=True)
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        epoch_path = out_path.with_name(f"{out_path.stem}_epoch{epoch + 1:03d}{out_path.suffix}")
        torch.save(
            {
                "state_dict": model.cpu().state_dict(),
                "past_len": past_len,
                "future_len": future_len,
                "crop_size": crop_size,
                "crop_scale": crop_scale,
                "image_size": list(image_size) if image_size is not None else None,
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "motion_feature_dim": 24,
                "num_samples": len(dataset),
                "positive_samples": positives,
                "negative_samples": negatives,
                "history": list(history),
                "action_loss_weight": float(action_loss_weight),
                "motion_pos_weight": float(pos_weight_value),
                "num_workers": num_workers,
                "frame_cache_size": frame_cache_size,
                "shuffle": bool(shuffle),
                "use_crops": bool(use_crops),
                "model_type": "vatd_motion_action_transformer",
                "epoch_checkpoint": epoch + 1,
            },
            epoch_path,
        )
        model.to(device)
        if verbose:
            print(json.dumps({"kind": "vatd_motion_action_epoch_checkpoint", "epoch": epoch + 1, "path": str(epoch_path)}, ensure_ascii=False), flush=True)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "past_len": past_len,
            "future_len": future_len,
            "crop_size": crop_size,
            "crop_scale": crop_scale,
            "image_size": list(image_size) if image_size is not None else None,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "motion_feature_dim": 24,
            "num_samples": len(dataset),
            "positive_samples": positives,
            "negative_samples": negatives,
            "history": history,
            "action_loss_weight": float(action_loss_weight),
            "motion_pos_weight": float(pos_weight_value),
            "num_workers": num_workers,
            "frame_cache_size": frame_cache_size,
            "use_crops": bool(use_crops),
            "model_type": "vatd_motion_action_transformer",
        },
        out_path,
    )
    return out_path


def train_ego_adaptive_vatd_policy(
    tracklet_jsonl: str | Path,
    out: str | Path,
    frame_root: str | Path | None = None,
    image_name_template: str = "{seq}_{frame_id_05d}.png",
    past_len: int = 7,
    future_len: int = 2,
    horizons: tuple[int, ...] = (3, 5, 7),
    crop_size: int = 64,
    crop_scale: float = 4.0,
    image_size: tuple[int, int] | None = None,
    min_tracklet_rows: int = 0,
    max_samples: int | None = None,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 16,
    d_model: int = 96,
    nhead: int = 4,
    num_layers: int = 2,
    action_loss_weight: float = 0.25,
    allow_missing_images: bool = False,
    num_workers: int = 0,
    frame_cache_size: int = 8,
    motion_pos_weight: float | str | None = "auto",
    pin_memory: bool | None = None,
    shuffle: bool = True,
    use_crops: bool = True,
    verbose: bool = False,
) -> Path:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    valid_horizons = tuple(sorted({int(h) for h in horizons if 0 < int(h) <= int(past_len)}))
    if not valid_horizons:
        raise ValueError("horizons must contain at least one value in [1, past_len]")
    dataset = VideoActionTrackletDataset(
        tracklet_jsonl,
        frame_root=frame_root,
        image_name_template=image_name_template,
        past_len=past_len,
        future_len=future_len,
        crop_size=crop_size,
        crop_scale=crop_scale,
        image_size=image_size,
        min_tracklet_rows=min_tracklet_rows,
        max_samples=max_samples,
        allow_missing_images=allow_missing_images,
        frame_cache_size=frame_cache_size,
        use_crops=use_crops,
    )
    if len(dataset) == 0:
        raise ValueError("Ego-adaptive VATD dataset has no samples")
    sample_labels = []
    for item_index, _start in dataset.samples:
        meta = dict(dataset.items[item_index].get("meta") or {})
        sample_labels.append(int(float(meta.get("label", 0))) > 0)
    positives = int(sum(sample_labels))
    negatives = int(len(sample_labels) - positives)
    if motion_pos_weight == "auto":
        pos_weight_value = float(negatives / max(1, positives)) if positives > 0 else 1.0
    elif motion_pos_weight is None:
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(motion_pos_weight)
    if pos_weight_value <= 0:
        raise ValueError("motion_pos_weight must be positive, 'auto', or None")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    use_pin_memory = (device.type == "cuda") if pin_memory is None else bool(pin_memory)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )
    model = EgoAdaptiveVATDTransformer(
        past_len=past_len,
        future_len=future_len,
        horizons=valid_horizons,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        crop_size=crop_size,
        motion_feature_dim=24,
        ego_feature_dim=16,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    motion_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device))
    action_loss_fn = torch.nn.SmoothL1Loss(reduction="none")
    history = []
    if verbose:
        print(
            json.dumps(
                {
                    "kind": "ego_adaptive_vatd_train_start",
                    "device": str(device),
                    "cuda_name": cuda_name,
                    "epochs": epochs,
                    "batches_per_epoch": len(loader),
                    "samples_total": len(dataset),
                    "batch_size": min(batch_size, len(dataset)),
                    "horizons": list(valid_horizons),
                    "action_loss_weight": float(action_loss_weight),
                    "motion_pos_weight": float(pos_weight_value),
                    "positive_samples": positives,
                    "negative_samples": negatives,
                    "use_crops": bool(use_crops),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    for epoch in range(epochs):
        total_motion_loss = 0.0
        total_action_loss = 0.0
        total_loss = 0.0
        total = 0
        router_sum = np.zeros((len(valid_horizons),), dtype=np.float64)
        for batch_index, batch in enumerate(loader, start=1):
            non_blocking = device.type == "cuda" and use_pin_memory
            crops = batch["crops"].to(device, non_blocking=non_blocking)
            state = batch["state"].to(device, non_blocking=non_blocking)
            motion_features = batch["motion_features"].to(device, non_blocking=non_blocking)
            ego_motion_features = batch["ego_motion_features"].to(device, non_blocking=non_blocking)
            target_motion = batch["target_motion_action"].to(device, non_blocking=non_blocking)
            target_residual = batch["future_action_residual"].to(device, non_blocking=non_blocking)
            target_action_weight = batch["target_action_weight"].to(device, non_blocking=non_blocking)
            opt.zero_grad(set_to_none=True)
            motion_logits, pred_residual, router_weights, _motion_logits_h = model(crops, state, motion_features, ego_motion_features)
            motion_loss = motion_loss_fn(motion_logits, target_motion)
            action_loss_raw = action_loss_fn(pred_residual, target_residual).mean(dim=(1, 2))
            action_loss = torch.mean(action_loss_raw * target_action_weight)
            loss = motion_loss + float(action_loss_weight) * action_loss
            loss.backward()
            opt.step()
            batch_size_actual = int(crops.shape[0])
            total_motion_loss += float(motion_loss.item()) * batch_size_actual
            total_action_loss += float(action_loss.item()) * batch_size_actual
            total_loss += float(loss.item()) * batch_size_actual
            total += batch_size_actual
            router_sum += router_weights.detach().cpu().numpy().sum(axis=0)
            progress_interval = max(1, min(10, len(loader) // 20))
            if verbose and (batch_index == 1 or batch_index % progress_interval == 0 or batch_index == len(loader)):
                print(
                    json.dumps(
                        {
                            "kind": "ego_adaptive_vatd_train_progress",
                            "epoch": epoch + 1,
                            "epochs": epochs,
                            "batches_done": batch_index,
                            "batches_total": len(loader),
                            "samples_done": min(batch_index * int(loader.batch_size or 1), len(dataset)),
                            "samples_total": len(dataset),
                            "loss": total_loss / max(1, total),
                            "motion_action_loss": total_motion_loss / max(1, total),
                            "action_residual_loss": total_action_loss / max(1, total),
                            "router_mean": (router_sum / max(1, total)).tolist(),
                            "device": str(device),
                            "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3)
                            if device.type == "cuda"
                            else 0.0,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        epoch_row = {
            "epoch": epoch + 1,
            "loss": total_loss / max(1, total),
            "motion_action_loss": total_motion_loss / max(1, total),
            "action_residual_loss": total_action_loss / max(1, total),
            "router_mean": (router_sum / max(1, total)).tolist(),
            "device": str(device),
            "cuda_name": cuda_name,
            "num_workers": num_workers,
            "frame_cache_size": frame_cache_size,
        }
        history.append(epoch_row)
        if verbose:
            print(json.dumps({"ego_adaptive_vatd_train": epoch_row}, ensure_ascii=False), flush=True)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "past_len": past_len,
            "future_len": future_len,
            "horizons": list(valid_horizons),
            "crop_size": crop_size,
            "crop_scale": crop_scale,
            "image_size": list(image_size) if image_size is not None else None,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "motion_feature_dim": 24,
            "ego_feature_dim": 16,
            "num_samples": len(dataset),
            "positive_samples": positives,
            "negative_samples": negatives,
            "history": history,
            "action_loss_weight": float(action_loss_weight),
            "motion_pos_weight": float(pos_weight_value),
            "num_workers": num_workers,
            "frame_cache_size": frame_cache_size,
            "use_crops": bool(use_crops),
            "model_type": "ego_adaptive_vatd_transformer",
        },
        out_path,
    )
    return out_path


def score_tracklets_with_video_action_policy(
    tracklet_jsonl: str | Path,
    weights: str | Path,
    out: str | Path,
    frame_root: str | Path | None = None,
    image_name_template: str = "{seq}_{frame_id_05d}.png",
    error_scale: float = 0.02,
    min_tracklet_rows: int = 0,
    max_samples: int | None = None,
    allow_missing_images: bool = False,
) -> VideoActionPolicyResult:
    ckpt = torch.load(weights, map_location="cpu")
    use_crops = bool(ckpt.get("use_crops", True))
    image_size_raw = ckpt.get("image_size")
    image_size = tuple(image_size_raw) if image_size_raw else None
    dataset = VideoActionTrackletDataset(
        tracklet_jsonl,
        frame_root=frame_root,
        image_name_template=image_name_template,
        past_len=int(ckpt["past_len"]),
        future_len=int(ckpt["future_len"]),
        crop_size=int(ckpt["crop_size"]),
        crop_scale=float(ckpt["crop_scale"]),
        image_size=image_size,  # type: ignore[arg-type]
        min_tracklet_rows=min_tracklet_rows,
        max_samples=max_samples,
        allow_missing_images=allow_missing_images,
    )
    if len(dataset) == 0:
        raise ValueError("video-action dataset has no samples")
    model = VideoActionChunkTransformer(
        past_len=int(ckpt["past_len"]),
        future_len=int(ckpt["future_len"]),
        d_model=int(ckpt["d_model"]),
        nhead=int(ckpt["nhead"]),
        num_layers=int(ckpt["num_layers"]),
        crop_size=int(ckpt["crop_size"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=_collate)
    out_rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["crops"], batch["state"]).cpu().numpy()
            past_boxes = batch["past_boxes"].cpu().numpy()
            future_boxes = batch["future_boxes"].cpu().numpy()
            for index, meta in enumerate(batch["meta"]):
                pred_boxes = reconstruct_boxes(past_boxes[index, -1], pred[index])
                err = _center_error(pred_boxes, future_boxes[index])
                score = float(np.exp(-err / max(float(error_scale), 1e-6)))
                row = {
                    **meta,
                    "num_rows": int(ckpt["past_len"]) + int(ckpt["future_len"]),
                    "anchor_frame": int(meta["anchor_frame"]),
                    "video_action_center_error": err,
                    "dynamics_score_mode": "video_action_chunk_transformer",
                    "dynamics_score": score,
                    "predicted_actions": pred[index].tolist(),
                }
                out_rows.append(row)
                grouped.setdefault((str(meta["seq"]), str(meta["track_id"])), []).append(row)
    tracklet_rows = []
    for (seq, track_id), rows in grouped.items():
        scores = [float(row["dynamics_score"]) for row in rows]
        errors = [float(row["video_action_center_error"]) for row in rows]
        first = rows[0]
        tracklet_rows.append(
            {
                "seq": seq,
                "track_id": track_id,
                "raw_track_id": first.get("raw_track_id", track_id),
                "label": int(float(first.get("label", 0))),
                "bucket": first.get("bucket", ""),
                "dataset_source": first.get("dataset_source", ""),
                "num_action_windows": len(rows),
                "mean_video_action_center_error": float(np.mean(errors)),
                "median_video_action_center_error": float(np.median(errors)),
                "dynamics_score_mode": "video_action_chunk_transformer",
                "dynamics_score": float(np.mean(scores)),
                "sample_scores": rows,
            }
        )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in tracklet_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "weights": str(weights),
        "out": str(out_path),
        "samples": len(out_rows),
        "tracklets": len(tracklet_rows),
        "error_scale": error_scale,
        "mean_video_action_center_error": float(np.mean([row["video_action_center_error"] for row in out_rows])) if out_rows else 0.0,
        "mean_dynamics_score": float(np.mean([row["dynamics_score"] for row in tracklet_rows])) if tracklet_rows else 0.0,
    }
    return VideoActionPolicyResult(out_path=out_path, summary=summary)


def score_tracklets_with_vatd_motion_action_policy(
    tracklet_jsonl: str | Path,
    weights: str | Path,
    out: str | Path,
    frame_root: str | Path | None = None,
    image_name_template: str = "{seq}_{frame_id_05d}.png",
    error_scale: float = 0.02,
    min_tracklet_rows: int = 0,
    max_samples: int | None = None,
    fusion_mode: str = "motion_action",
    allow_missing_images: bool = False,
    batch_size: int = 16,
    num_workers: int = 0,
    frame_cache_size: int = 8,
    use_crops: bool | None = None,
) -> VideoActionPolicyResult:
    if fusion_mode not in {"motion_action", "motion_times_action_consistency"}:
        raise ValueError("fusion_mode must be 'motion_action' or 'motion_times_action_consistency'")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    ckpt = torch.load(weights, map_location="cpu")
    resolved_use_crops = bool(ckpt.get("use_crops", True)) if use_crops is None else bool(use_crops)
    image_size_raw = ckpt.get("image_size")
    image_size = tuple(image_size_raw) if image_size_raw else None
    dataset = VideoActionTrackletDataset(
        tracklet_jsonl,
        frame_root=frame_root,
        image_name_template=image_name_template,
        past_len=int(ckpt["past_len"]),
        future_len=int(ckpt["future_len"]),
        crop_size=int(ckpt["crop_size"]),
        crop_scale=float(ckpt["crop_scale"]),
        image_size=image_size,  # type: ignore[arg-type]
        min_tracklet_rows=min_tracklet_rows,
        max_samples=max_samples,
        allow_missing_images=allow_missing_images,
        frame_cache_size=frame_cache_size,
        use_crops=resolved_use_crops,
    )
    if len(dataset) == 0:
        raise ValueError("VATD video-action dataset has no samples")
    model = VATDMotionActionTransformer(
        past_len=int(ckpt["past_len"]),
        future_len=int(ckpt["future_len"]),
        d_model=int(ckpt["d_model"]),
        nhead=int(ckpt["nhead"]),
        num_layers=int(ckpt["num_layers"]),
        crop_size=int(ckpt["crop_size"]),
        motion_feature_dim=int(ckpt.get("motion_feature_dim", 0)),
    )
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    model.to(device)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    out_rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    print(
        json.dumps(
            {
                "kind": "vatd_motion_action_score_start",
                "device": str(device),
                "cuda_name": cuda_name,
                "batches_total": len(loader),
                "samples_total": len(dataset),
                "batch_size": batch_size,
                "num_workers": num_workers,
                "frame_cache_size": frame_cache_size,
                "fusion_mode": fusion_mode,
                "use_crops": resolved_use_crops,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            crops = batch["crops"].to(device, non_blocking=device.type == "cuda")
            state = batch["state"].to(device, non_blocking=device.type == "cuda")
            motion_features = batch["motion_features"].to(device, non_blocking=device.type == "cuda")
            motion_logits, pred_residual = model(crops, state, motion_features)
            motion_scores = torch.sigmoid(motion_logits).cpu().numpy()
            pred_residual_np = pred_residual.cpu().numpy()
            past_boxes = batch["past_boxes"].cpu().numpy()
            future_boxes = batch["future_boxes"].cpu().numpy()
            target_motion = batch["target_motion_action"].cpu().numpy()
            for index, meta in enumerate(batch["meta"]):
                cv_actions = _constant_velocity_actions_from_past(past_boxes[index], int(ckpt["future_len"]))
                pred_actions = cv_actions + pred_residual_np[index]
                pred_boxes = reconstruct_boxes(past_boxes[index, -1], pred_actions)
                err = _center_error(pred_boxes, future_boxes[index])
                action_consistency = float(np.exp(-err / max(float(error_scale), 1e-6)))
                motion_score = float(motion_scores[index])
                fused_score = motion_score if fusion_mode == "motion_action" else motion_score * action_consistency
                row = {
                    **meta,
                    "num_rows": int(ckpt["past_len"]) + int(ckpt["future_len"]),
                    "anchor_frame": int(meta["anchor_frame"]),
                    "vatd_action_residual_center_error": err,
                    "vatd_action_consistency_score": action_consistency,
                    "motion_action_score": motion_score,
                    "target_motion_action": float(target_motion[index]),
                    "vatd_score": fused_score,
                    "vatd_fusion_mode": fusion_mode,
                    "dynamics_score_mode": "vatd_motion_action_transformer",
                    "predicted_action_residual": pred_residual_np[index].tolist(),
                }
                out_rows.append(row)
                grouped.setdefault((str(meta["seq"]), str(meta["track_id"])), []).append(row)
            progress_interval = max(1, min(10, len(loader) // 20))
            if batch_index == 1 or batch_index % progress_interval == 0 or batch_index == len(loader):
                print(
                    json.dumps(
                        {
                            "kind": "vatd_motion_action_score_progress",
                            "batches_done": batch_index,
                            "batches_total": len(loader),
                            "samples_done": min(batch_index * int(loader.batch_size or 1), len(dataset)),
                            "samples_total": len(dataset),
                            "tracklets_scored_so_far": len(grouped),
                            "device": str(device),
                            "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3)
                            if device.type == "cuda"
                            else 0.0,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    tracklet_rows = []
    for (seq, track_id), rows in grouped.items():
        action_scores = [float(row["vatd_action_consistency_score"]) for row in rows]
        motion_scores = [float(row["motion_action_score"]) for row in rows]
        fused_scores = [float(row["vatd_score"]) for row in rows]
        errors = [float(row["vatd_action_residual_center_error"]) for row in rows]
        first = rows[0]
        tracklet_rows.append(
            {
                "seq": seq,
                "track_id": track_id,
                "raw_track_id": first.get("raw_track_id", track_id),
                "label": int(float(first.get("label", 0))),
                "bucket": first.get("bucket", ""),
                "dataset_source": first.get("dataset_source", ""),
                "num_action_windows": len(rows),
                "mean_vatd_action_residual_center_error": float(np.mean(errors)),
                "median_vatd_action_residual_center_error": float(np.median(errors)),
                "vatd_action_consistency_score": float(np.mean(action_scores)),
                "motion_action_score": float(np.mean(motion_scores)),
                "vatd_score": float(np.mean(fused_scores)),
                "vatd_fusion_mode": fusion_mode,
                "dynamics_score_mode": "vatd_motion_action_transformer",
                "sample_scores": rows,
            }
        )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in tracklet_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "weights": str(weights),
        "out": str(out_path),
        "samples": len(out_rows),
        "tracklets": len(tracklet_rows),
        "error_scale": error_scale,
        "fusion_mode": fusion_mode,
        "device": str(device),
        "cuda_name": cuda_name,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "frame_cache_size": frame_cache_size,
        "mean_vatd_action_residual_center_error": float(np.mean([row["vatd_action_residual_center_error"] for row in out_rows])) if out_rows else 0.0,
        "mean_motion_action_score": float(np.mean([row["motion_action_score"] for row in tracklet_rows])) if tracklet_rows else 0.0,
        "mean_vatd_score": float(np.mean([row["vatd_score"] for row in tracklet_rows])) if tracklet_rows else 0.0,
    }
    return VideoActionPolicyResult(out_path=out_path, summary=summary)


def score_tracklets_with_ego_adaptive_vatd_policy(
    tracklet_jsonl: str | Path,
    weights: str | Path,
    out: str | Path,
    frame_root: str | Path | None = None,
    image_name_template: str = "{seq}_{frame_id_05d}.png",
    error_scale: float = 0.02,
    min_tracklet_rows: int = 0,
    max_samples: int | None = None,
    fusion_mode: str = "motion_action",
    allow_missing_images: bool = False,
    batch_size: int = 16,
    num_workers: int = 0,
    frame_cache_size: int = 8,
    use_crops: bool | None = None,
) -> VideoActionPolicyResult:
    if fusion_mode not in {"motion_action", "motion_times_action_consistency"}:
        raise ValueError("fusion_mode must be 'motion_action' or 'motion_times_action_consistency'")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    ckpt = torch.load(weights, map_location="cpu")
    if ckpt.get("model_type") != "ego_adaptive_vatd_transformer":
        raise ValueError(f"expected ego_adaptive_vatd_transformer checkpoint, got {ckpt.get('model_type')}")
    resolved_use_crops = bool(ckpt.get("use_crops", True)) if use_crops is None else bool(use_crops)
    image_size_raw = ckpt.get("image_size")
    image_size = tuple(image_size_raw) if image_size_raw else None
    dataset = VideoActionTrackletDataset(
        tracklet_jsonl,
        frame_root=frame_root,
        image_name_template=image_name_template,
        past_len=int(ckpt["past_len"]),
        future_len=int(ckpt["future_len"]),
        crop_size=int(ckpt["crop_size"]),
        crop_scale=float(ckpt["crop_scale"]),
        image_size=image_size,  # type: ignore[arg-type]
        min_tracklet_rows=min_tracklet_rows,
        max_samples=max_samples,
        allow_missing_images=allow_missing_images,
        frame_cache_size=frame_cache_size,
        use_crops=resolved_use_crops,
    )
    if len(dataset) == 0:
        raise ValueError("Ego-adaptive VATD dataset has no samples")
    model = EgoAdaptiveVATDTransformer(
        past_len=int(ckpt["past_len"]),
        future_len=int(ckpt["future_len"]),
        horizons=tuple(int(v) for v in ckpt["horizons"]),
        d_model=int(ckpt["d_model"]),
        nhead=int(ckpt["nhead"]),
        num_layers=int(ckpt["num_layers"]),
        crop_size=int(ckpt["crop_size"]),
        motion_feature_dim=int(ckpt.get("motion_feature_dim", 24)),
        ego_feature_dim=int(ckpt.get("ego_feature_dim", 16)),
    )
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    model.to(device)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    out_rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    horizons = [int(v) for v in ckpt["horizons"]]
    print(
        json.dumps(
            {
                "kind": "ego_adaptive_vatd_score_start",
                "device": str(device),
                "cuda_name": cuda_name,
                "batches_total": len(loader),
                "samples_total": len(dataset),
                "batch_size": batch_size,
                "num_workers": num_workers,
                "frame_cache_size": frame_cache_size,
                "fusion_mode": fusion_mode,
                "horizons": horizons,
                "use_crops": resolved_use_crops,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    router_weight_rows = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            crops = batch["crops"].to(device, non_blocking=device.type == "cuda")
            state = batch["state"].to(device, non_blocking=device.type == "cuda")
            motion_features = batch["motion_features"].to(device, non_blocking=device.type == "cuda")
            ego_motion_features = batch["ego_motion_features"].to(device, non_blocking=device.type == "cuda")
            motion_logits, pred_residual, router_weights, motion_logits_h = model(crops, state, motion_features, ego_motion_features)
            motion_scores = torch.sigmoid(motion_logits).cpu().numpy()
            horizon_scores = torch.sigmoid(motion_logits_h).cpu().numpy()
            router_np = router_weights.cpu().numpy()
            pred_residual_np = pred_residual.cpu().numpy()
            past_boxes = batch["past_boxes"].cpu().numpy()
            future_boxes = batch["future_boxes"].cpu().numpy()
            target_motion = batch["target_motion_action"].cpu().numpy()
            ego_np = batch["ego_motion_features"].cpu().numpy()
            for index, meta in enumerate(batch["meta"]):
                cv_actions = _constant_velocity_actions_from_past(past_boxes[index], int(ckpt["future_len"]))
                pred_actions = cv_actions + pred_residual_np[index]
                pred_boxes = reconstruct_boxes(past_boxes[index, -1], pred_actions)
                err = _center_error(pred_boxes, future_boxes[index])
                action_consistency = float(np.exp(-err / max(float(error_scale), 1e-6)))
                motion_score = float(motion_scores[index])
                fused_score = motion_score if fusion_mode == "motion_action" else motion_score * action_consistency
                router_row = {f"h{horizons[i]}": float(router_np[index, i]) for i in range(len(horizons))}
                router_weight_rows.append(router_np[index])
                row = {
                    **meta,
                    "num_rows": int(ckpt["past_len"]) + int(ckpt["future_len"]),
                    "anchor_frame": int(meta["anchor_frame"]),
                    "vatd_action_residual_center_error": err,
                    "vatd_action_consistency_score": action_consistency,
                    "motion_action_score": motion_score,
                    "target_motion_action": float(target_motion[index]),
                    "vatd_score": fused_score,
                    "vatd_fusion_mode": fusion_mode,
                    "dynamics_score_mode": "ego_adaptive_vatd_transformer",
                    "adaptive_horizons": horizons,
                    "adaptive_router_weights": router_row,
                    "adaptive_horizon_motion_scores": {f"h{horizons[i]}": float(horizon_scores[index, i]) for i in range(len(horizons))},
                    "ego_motion_features": ego_np[index].tolist(),
                    "predicted_action_residual": pred_residual_np[index].tolist(),
                }
                out_rows.append(row)
                grouped.setdefault((str(meta["seq"]), str(meta["track_id"])), []).append(row)
            progress_interval = max(1, min(10, len(loader) // 20))
            if batch_index == 1 or batch_index % progress_interval == 0 or batch_index == len(loader):
                print(
                    json.dumps(
                        {
                            "kind": "ego_adaptive_vatd_score_progress",
                            "batches_done": batch_index,
                            "batches_total": len(loader),
                            "samples_done": min(batch_index * int(loader.batch_size or 1), len(dataset)),
                            "samples_total": len(dataset),
                            "tracklets_scored_so_far": len(grouped),
                            "device": str(device),
                            "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3)
                            if device.type == "cuda"
                            else 0.0,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    tracklet_rows = []
    for (seq, track_id), rows in grouped.items():
        action_scores = [float(row["vatd_action_consistency_score"]) for row in rows]
        motion_scores = [float(row["motion_action_score"]) for row in rows]
        fused_scores = [float(row["vatd_score"]) for row in rows]
        errors = [float(row["vatd_action_residual_center_error"]) for row in rows]
        first = rows[0]
        mean_router = {f"h{h}": float(np.mean([float(row["adaptive_router_weights"][f"h{h}"]) for row in rows])) for h in horizons}
        tracklet_rows.append(
            {
                "seq": seq,
                "track_id": track_id,
                "raw_track_id": first.get("raw_track_id", track_id),
                "label": int(float(first.get("label", 0))),
                "bucket": first.get("bucket", ""),
                "dataset_source": first.get("dataset_source", ""),
                "num_action_windows": len(rows),
                "mean_vatd_action_residual_center_error": float(np.mean(errors)),
                "median_vatd_action_residual_center_error": float(np.median(errors)),
                "vatd_action_consistency_score": float(np.mean(action_scores)),
                "motion_action_score": float(np.mean(motion_scores)),
                "vatd_score": float(np.mean(fused_scores)),
                "vatd_fusion_mode": fusion_mode,
                "dynamics_score_mode": "ego_adaptive_vatd_transformer",
                "adaptive_horizons": horizons,
                "adaptive_router_weights": mean_router,
                "sample_scores": rows,
            }
        )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in tracklet_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    router_mean = np.mean(np.stack(router_weight_rows), axis=0).tolist() if router_weight_rows else [0.0 for _ in horizons]
    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "weights": str(weights),
        "out": str(out_path),
        "samples": len(out_rows),
        "tracklets": len(tracklet_rows),
        "error_scale": error_scale,
        "fusion_mode": fusion_mode,
        "device": str(device),
        "cuda_name": cuda_name,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "frame_cache_size": frame_cache_size,
        "horizons": horizons,
        "router_mean": router_mean,
        "mean_vatd_action_residual_center_error": float(np.mean([row["vatd_action_residual_center_error"] for row in out_rows])) if out_rows else 0.0,
        "mean_motion_action_score": float(np.mean([row["motion_action_score"] for row in tracklet_rows])) if tracklet_rows else 0.0,
        "mean_vatd_score": float(np.mean([row["vatd_score"] for row in tracklet_rows])) if tracklet_rows else 0.0,
    }
    return VideoActionPolicyResult(out_path=out_path, summary=summary)


def attach_vatd_scores_to_tracklets(
    tracklet_jsonl: str | Path,
    vatd_scores_jsonl: str | Path,
    out: str | Path,
    score_field: str = "vatd_score",
) -> VideoActionPolicyResult:
    scores: dict[tuple[str, str], dict[str, Any]] = {}
    with Path(vatd_scores_jsonl).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            seq = str(row.get("seq", ""))
            track_id = str(row.get("track_id", ""))
            raw_track_id = str(row.get("raw_track_id", ""))
            if seq and track_id:
                scores[(seq, track_id)] = row
            if seq and raw_track_id:
                scores[(seq, raw_track_id)] = row

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    attached = 0
    missing = 0
    score_values: list[float] = []
    copied_fields = (
        "vatd_score",
        "motion_action_score",
        "vatd_action_consistency_score",
        "mean_vatd_action_residual_center_error",
        "median_vatd_action_residual_center_error",
        "vatd_fusion_mode",
        "dynamics_score_mode",
        "num_action_windows",
    )
    with Path(tracklet_jsonl).open("r", encoding="utf-8-sig") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            item = json.loads(line)
            meta = dict(item.get("meta") or {})
            rows = [dict(row) for row in (item.get("rows") or [])]
            seq = str(meta.get("seq") or (rows[0].get("seq") if rows else ""))
            track_id = str(meta.get("track_id") or (rows[0].get("track_id") if rows else ""))
            raw_track_id = str(meta.get("raw_track_id") or (rows[0].get("raw_track_id") if rows else ""))
            score = scores.get((seq, track_id)) or scores.get((seq, raw_track_id))
            if score is None:
                missing += 1
            else:
                attached += 1
                for field in copied_fields:
                    if field in score:
                        meta[field] = score[field]
                if score_field in score:
                    try:
                        score_values.append(float(score[score_field]))
                    except (TypeError, ValueError):
                        pass
                for row in rows:
                    for field in copied_fields:
                        if field in score:
                            row[field] = score[field]
            item["meta"] = meta
            item["rows"] = rows
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "vatd_scores_jsonl": str(vatd_scores_jsonl),
        "out": str(out_path),
        "score_field": score_field,
        "total_tracklets": total,
        "attached_tracklets": attached,
        "missing_score_tracklets": missing,
        "score_rows_indexed": len(scores),
        "mean_score": float(np.mean(score_values)) if score_values else 0.0,
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return VideoActionPolicyResult(out_path=out_path, summary=summary)


def score_tracklets_with_video_action_multihead_policy(
    tracklet_jsonl: str | Path,
    weights: str | Path,
    out: str | Path,
    frame_root: str | Path | None = None,
    image_name_template: str = "{seq}_{frame_id_05d}.png",
    error_scale: float = 0.02,
    min_tracklet_rows: int = 0,
    max_samples: int | None = None,
    fusion_mode: str = "predicted_confidence",
    allow_missing_images: bool = False,
    batch_size: int = 16,
    num_workers: int = 0,
    frame_cache_size: int = 8,
) -> VideoActionPolicyResult:
    if fusion_mode not in {"predicted_confidence", "dynamics_times_predicted_confidence"}:
        raise ValueError("fusion_mode must be 'predicted_confidence' or 'dynamics_times_predicted_confidence'")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    ckpt = torch.load(weights, map_location="cpu")
    image_size_raw = ckpt.get("image_size")
    image_size = tuple(image_size_raw) if image_size_raw else None
    dataset = VideoActionTrackletDataset(
        tracklet_jsonl,
        frame_root=frame_root,
        image_name_template=image_name_template,
        past_len=int(ckpt["past_len"]),
        future_len=int(ckpt["future_len"]),
        crop_size=int(ckpt["crop_size"]),
        crop_scale=float(ckpt["crop_scale"]),
        image_size=image_size,  # type: ignore[arg-type]
        min_tracklet_rows=min_tracklet_rows,
        max_samples=max_samples,
        allow_missing_images=allow_missing_images,
        frame_cache_size=frame_cache_size,
    )
    if len(dataset) == 0:
        raise ValueError("video-action dataset has no samples")
    model = VideoActionMultiHeadTransformer(
        past_len=int(ckpt["past_len"]),
        future_len=int(ckpt["future_len"]),
        d_model=int(ckpt["d_model"]),
        nhead=int(ckpt["nhead"]),
        num_layers=int(ckpt["num_layers"]),
        crop_size=int(ckpt["crop_size"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    model.to(device)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    out_rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    print(
        json.dumps(
            {
                "kind": "video_action_multihead_score_start",
                "device": str(device),
                "cuda_name": cuda_name,
                "batches_total": len(loader),
                "samples_total": len(dataset),
                "batch_size": batch_size,
                "num_workers": num_workers,
                "frame_cache_size": frame_cache_size,
            }
        ),
        flush=True,
    )
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            crops = batch["crops"].to(device, non_blocking=device.type == "cuda")
            state = batch["state"].to(device, non_blocking=device.type == "cuda")
            pred_actions, pred_confidence = model(crops, state)
            pred = pred_actions.cpu().numpy()
            pred_conf = pred_confidence.cpu().numpy()
            past_boxes = batch["past_boxes"].cpu().numpy()
            future_boxes = batch["future_boxes"].cpu().numpy()
            target_conf = batch[f"target_confidence_{ckpt.get('confidence_target', 'max')}"].cpu().numpy()
            for index, meta in enumerate(batch["meta"]):
                pred_boxes = reconstruct_boxes(past_boxes[index, -1], pred[index])
                err = _center_error(pred_boxes, future_boxes[index])
                dynamics_score = float(np.exp(-err / max(float(error_scale), 1e-6)))
                confidence_score = float(pred_conf[index])
                fused_score = confidence_score if fusion_mode == "predicted_confidence" else dynamics_score * confidence_score
                row = {
                    **meta,
                    "num_rows": int(ckpt["past_len"]) + int(ckpt["future_len"]),
                    "anchor_frame": int(meta["anchor_frame"]),
                    "video_action_center_error": err,
                    "dynamics_score_mode": "video_action_multihead_transformer",
                    "dynamics_score": dynamics_score,
                    "predicted_confidence_score": confidence_score,
                    "target_confidence_score": float(target_conf[index]),
                    "video_action_model_fusion_score": fused_score,
                    "video_action_model_fusion_mode": fusion_mode,
                    "predicted_actions": pred[index].tolist(),
                }
                out_rows.append(row)
                grouped.setdefault((str(meta["seq"]), str(meta["track_id"])), []).append(row)
            progress_interval = max(1, min(10, len(loader) // 20))
            if batch_index == 1 or batch_index % progress_interval == 0 or batch_index == len(loader):
                print(
                    json.dumps(
                        {
                            "kind": "video_action_multihead_score_progress",
                            "batches_done": batch_index,
                            "batches_total": len(loader),
                            "samples_done": min(batch_index * int(loader.batch_size or 1), len(dataset)),
                            "samples_total": len(dataset),
                            "tracklets_scored_so_far": len(grouped),
                            "device": str(device),
                            "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3)
                            if device.type == "cuda"
                            else 0.0,
                        }
                    ),
                    flush=True,
                )
    tracklet_rows = []
    for (seq, track_id), rows in grouped.items():
        dynamics_scores = [float(row["dynamics_score"]) for row in rows]
        confidence_scores = [float(row["predicted_confidence_score"]) for row in rows]
        fused_scores = [float(row["video_action_model_fusion_score"]) for row in rows]
        errors = [float(row["video_action_center_error"]) for row in rows]
        first = rows[0]
        tracklet_rows.append(
            {
                "seq": seq,
                "track_id": track_id,
                "raw_track_id": first.get("raw_track_id", track_id),
                "label": int(float(first.get("label", 0))),
                "bucket": first.get("bucket", ""),
                "dataset_source": first.get("dataset_source", ""),
                "num_action_windows": len(rows),
                "mean_video_action_center_error": float(np.mean(errors)),
                "median_video_action_center_error": float(np.median(errors)),
                "dynamics_score_mode": "video_action_multihead_transformer",
                "dynamics_score": float(np.mean(dynamics_scores)),
                "predicted_confidence_score": float(np.mean(confidence_scores)),
                "video_action_model_fusion_score": float(np.mean(fused_scores)),
                "video_action_model_fusion_mode": fusion_mode,
                "sample_scores": rows,
            }
        )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in tracklet_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "weights": str(weights),
        "out": str(out_path),
        "samples": len(out_rows),
        "tracklets": len(tracklet_rows),
        "error_scale": error_scale,
        "fusion_mode": fusion_mode,
        "confidence_target": ckpt.get("confidence_target", "max"),
        "device": str(device),
        "cuda_name": cuda_name,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "frame_cache_size": frame_cache_size,
        "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3) if device.type == "cuda" else 0.0,
        "mean_video_action_center_error": float(np.mean([row["video_action_center_error"] for row in out_rows])) if out_rows else 0.0,
        "mean_dynamics_score": float(np.mean([row["dynamics_score"] for row in tracklet_rows])) if tracklet_rows else 0.0,
        "mean_predicted_confidence_score": float(np.mean([row["predicted_confidence_score"] for row in tracklet_rows])) if tracklet_rows else 0.0,
        "mean_video_action_model_fusion_score": float(np.mean([row["video_action_model_fusion_score"] for row in tracklet_rows])) if tracklet_rows else 0.0,
    }
    return VideoActionPolicyResult(out_path=out_path, summary=summary)
