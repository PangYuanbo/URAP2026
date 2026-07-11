from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, log
from typing import Any, Iterable, Sequence

import numpy as np
import torch

EPS = 1e-6
ACTION_TOKEN_DIM = 18


@dataclass(frozen=True)
class ActionBankConfig:
    short_seconds: float = 1.0
    long_seconds: float = 3.0
    short_tokens: int = 12
    long_tokens: int = 18
    fps_fallback: float = 29.97
    max_gap_seconds: float = 0.5
    sequence_fps: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.short_seconds <= 0 or self.long_seconds < self.short_seconds:
            raise ValueError("expected 0 < short_seconds <= long_seconds")
        if self.short_tokens <= 0 or self.long_tokens <= 0:
            raise ValueError("token counts must be positive")
        if self.fps_fallback <= 0 or self.max_gap_seconds <= 0:
            raise ValueError("fps_fallback and max_gap_seconds must be positive")


@dataclass(frozen=True)
class TimedActionToken:
    start_time: float
    end_time: float
    values: np.ndarray
    reliable: bool

    @property
    def dt(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True)
class ActionBankSnapshot:
    anchor_time: float
    short_tokens: np.ndarray
    short_mask: np.ndarray
    long_tokens: np.ndarray
    long_mask: np.ndarray
    raw_actions: tuple[TimedActionToken, ...]
    config: ActionBankConfig


@dataclass(frozen=True)
class CandidateMotionScore:
    score: float
    predicted_iou: float
    velocity_similarity: float
    direction_similarity: float
    scale_similarity: float
    confidence: float


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if isfinite(result) else float(default)


def _row_float(row: dict[str, Any], names: Sequence[str], default: float = 0.0) -> float:
    for name in names:
        if row.get(name) is not None:
            return _finite_float(row[name], default)
    return float(default)


def row_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    value = row.get("bbox") or row.get("bbox_xyxy")
    if value is None:
        value = [row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")]
    if value is None or len(value) != 4:
        raise ValueError("row must contain bbox/bbox_xyxy or x1/y1/x2/y2")
    x1, y1, x2, y2 = [_finite_float(item) for item in value]
    return x1, y1, max(x1 + EPS, x2), max(y1 + EPS, y2)


def row_score(row: dict[str, Any]) -> float:
    return float(np.clip(max(_row_float(row, ("objectness",)), _row_float(row, ("final_drone_score",)), _row_float(row, ("score",))), 0.0, 1.0))


def row_timestamp(row: dict[str, Any], fps_fallback: float = 29.97, sequence_fps: dict[str, float] | None = None) -> float:
    for name in ("timestamp_sec", "time_sec", "timestamp", "pts_time"):
        if row.get(name) is not None:
            return _finite_float(row[name])
    frame_id = _row_float(row, ("frame_id", "frame_index", "frame"), 0.0)
    seq = str(row.get("seq", row.get("sequence", row.get("video", ""))))
    mapped_fps = (sequence_fps or {}).get(seq, fps_fallback)
    fps = _row_float(row, ("fps", "frame_rate", "video_fps"), mapped_fps)
    return frame_id / max(EPS, fps)


def configured_timestamp(row: dict[str, Any], config: ActionBankConfig) -> float:
    return row_timestamp(row, config.fps_fallback, config.sequence_fps)


def _image_size(row: dict[str, Any]) -> tuple[float, float]:
    width = _row_float(row, ("image_width", "width"), 0.0)
    height = _row_float(row, ("image_height", "height"), 0.0)
    size = row.get("image_size")
    if (width <= 0 or height <= 0) and isinstance(size, (list, tuple)) and len(size) >= 2:
        width, height = _finite_float(size[0], width), _finite_float(size[1], height)
    return max(1.0, width), max(1.0, height)


def _cxcywh(box: Iterable[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    width, height = max(EPS, x2 - x1), max(EPS, y2 - y1)
    return x1 + 0.5 * width, y1 + 0.5 * height, width, height


def box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in left]
    bx1, by1, bx2, by2 = [float(value) for value in right]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(EPS, area_a + area_b - inter))


def _camera_displacement(row: dict[str, Any], width: float, height: float) -> tuple[float, float]:
    dx = _row_float(row, ("camera_dx", "ego_dx", "global_dx", "flow_dx"), 0.0)
    dy = _row_float(row, ("camera_dy", "ego_dy", "global_dy", "flow_dy"), 0.0)
    return (dx, dy) if bool(row.get("camera_motion_normalized", False)) else (dx / width, dy / height)


def action_token(previous: dict[str, Any], current: dict[str, Any], config: ActionBankConfig) -> TimedActionToken:
    start_time = configured_timestamp(previous, config)
    end_time = configured_timestamp(current, config)
    dt = max(EPS, end_time - start_time)
    width, height = _image_size(current)
    prev_cx, prev_cy, prev_w, prev_h = _cxcywh(row_box(previous))
    curr_cx, curr_cy, curr_w, curr_h = _cxcywh(row_box(current))
    apparent_dx, apparent_dy = (curr_cx - prev_cx) / width, (curr_cy - prev_cy) / height
    camera_dx, camera_dy = _camera_displacement(current, width, height)
    residual_dx, residual_dy = apparent_dx - camera_dx, apparent_dy - camera_dy
    vx, vy = residual_dx / dt, residual_dy / dt
    dlogw = log(max(EPS, curr_w) / max(EPS, prev_w))
    dlogh = log(max(EPS, curr_h) / max(EPS, prev_h))
    previous_vx = _row_float(previous, ("bank_vx", "residual_vx"), vx)
    previous_vy = _row_float(previous, ("bank_vy", "residual_vy"), vy)
    ax, ay = (vx - previous_vx) / dt, (vy - previous_vy) / dt
    motion_iou = _row_float(current, ("motion_iou", "samurai_iou", "samurai_cmc_forward_iou", "samurai_cmc_backward_iou", "predicted_iou"), box_iou(row_box(previous), row_box(current)))
    reliable = dt <= config.max_gap_seconds and bool(current.get("visible", True))
    values = np.asarray([dt, residual_dx, residual_dy, vx, vy, ax, ay, dlogw / dt, dlogh / dt, row_score(current), np.clip(motion_iou, 0.0, 1.0), camera_dx / dt, camera_dy / dt, curr_w / width, curr_h / height, float(reliable), apparent_dx / dt, apparent_dy / dt], dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0)
    return TimedActionToken(start_time=start_time, end_time=end_time, values=values, reliable=reliable)


def actions_from_rows(rows: Sequence[dict[str, Any]], config: ActionBankConfig | None = None) -> tuple[TimedActionToken, ...]:
    cfg = config or ActionBankConfig()
    ordered = sorted(rows, key=lambda row: configured_timestamp(row, cfg))
    return tuple(action_token(ordered[index - 1], ordered[index], cfg) for index in range(1, len(ordered)))


def _sample_recent_actions(actions: Sequence[TimedActionToken], anchor_time: float, seconds: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros((count, ACTION_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((count,), dtype=np.float32)
    eligible = [action for action in actions if anchor_time - seconds - EPS <= action.end_time <= anchor_time + EPS]
    if not eligible:
        return output, mask
    targets = np.linspace(anchor_time - seconds, anchor_time, count, dtype=np.float64)
    action_times = np.asarray([action.end_time for action in eligible], dtype=np.float64)
    used: set[int] = set()
    for output_index, target in enumerate(targets):
        for candidate_index in np.argsort(np.abs(action_times - target)):
            candidate = int(candidate_index)
            if candidate not in used or len(eligible) < count:
                used.add(candidate)
                action = eligible[candidate]
                output[output_index] = action.values
                output[output_index, 0] = float(np.clip((anchor_time - action.end_time) / seconds, 0.0, 1.0))
                mask[output_index] = float(action.reliable)
                break
    return output, mask


def _compress_long_actions(actions: Sequence[TimedActionToken], anchor_time: float, seconds: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros((count, ACTION_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((count,), dtype=np.float32)
    edges = np.linspace(anchor_time - seconds, anchor_time, count + 1, dtype=np.float64)
    for index in range(count):
        bucket = [action for action in actions if edges[index] - EPS <= action.end_time <= edges[index + 1] + EPS]
        if not bucket:
            continue
        values = np.stack([action.values for action in bucket], axis=0)
        weights = np.asarray([max(EPS, action.dt) * (1.0 if action.reliable else 0.25) for action in bucket], dtype=np.float32)
        summary = np.average(values, axis=0, weights=weights).astype(np.float32)
        summary[0] = float(np.clip((anchor_time - np.mean([action.end_time for action in bucket])) / seconds, 0.0, 1.0))
        summary[5:7] = values[-1, 3:5] - values[0, 3:5]
        summary[15] = float(np.mean([action.reliable for action in bucket]))
        output[index] = summary
        mask[index] = 1.0
    return output, mask


def build_action_bank(rows: Sequence[dict[str, Any]], anchor_time: float | None = None, config: ActionBankConfig | None = None) -> ActionBankSnapshot:
    cfg = config or ActionBankConfig()
    if not rows:
        raise ValueError("rows must not be empty")
    ordered = sorted(rows, key=lambda row: configured_timestamp(row, cfg))
    actions = actions_from_rows(ordered, cfg)
    resolved_anchor = configured_timestamp(ordered[-1], cfg) if anchor_time is None else float(anchor_time)
    short_tokens, short_mask = _sample_recent_actions(actions, resolved_anchor, cfg.short_seconds, cfg.short_tokens)
    long_tokens, long_mask = _compress_long_actions(actions, resolved_anchor, cfg.long_seconds, cfg.long_tokens)
    return ActionBankSnapshot(resolved_anchor, short_tokens, short_mask, long_tokens, long_mask, actions, cfg)


def _weighted_recent_velocity(snapshot: ActionBankSnapshot) -> tuple[float, float, float, float]:
    valid = snapshot.short_tokens[snapshot.short_mask > 0]
    if len(valid) == 0:
        return 0.0, 0.0, 0.0, 0.0
    weights = np.linspace(0.5, 1.0, len(valid), dtype=np.float32) * np.clip(valid[:, 15], 0.1, 1.0)
    return tuple(float(np.average(valid[:, index], weights=weights)) for index in (3, 4, 7, 8))


def predict_box(previous_row: dict[str, Any], candidate_time: float, snapshot: ActionBankSnapshot) -> tuple[float, float, float, float]:
    previous_time = snapshot.anchor_time
    dt = max(0.0, float(candidate_time) - previous_time)
    width, height = _image_size(previous_row)
    cx, cy, box_width, box_height = _cxcywh(row_box(previous_row))
    vx, vy, scale_w, scale_h = _weighted_recent_velocity(snapshot)
    predicted_width = box_width * exp(float(np.clip(scale_w * dt, -2.0, 2.0)))
    predicted_height = box_height * exp(float(np.clip(scale_h * dt, -2.0, 2.0)))
    predicted_cx, predicted_cy = cx + vx * dt * width, cy + vy * dt * height
    return predicted_cx - predicted_width * 0.5, predicted_cy - predicted_height * 0.5, predicted_cx + predicted_width * 0.5, predicted_cy + predicted_height * 0.5


def score_candidate(previous_row: dict[str, Any], candidate_row: dict[str, Any], snapshot: ActionBankSnapshot, samurai_iou: float | None = None) -> CandidateMotionScore:
    predicted_iou = box_iou(predict_box(previous_row, configured_timestamp(candidate_row, snapshot.config), snapshot), row_box(candidate_row))
    candidate_action = action_token(previous_row, candidate_row, snapshot.config)
    expected_vx, expected_vy, expected_scale_w, expected_scale_h = _weighted_recent_velocity(snapshot)
    actual_vx, actual_vy = float(candidate_action.values[3]), float(candidate_action.values[4])
    expected_speed, actual_speed = float(np.hypot(expected_vx, expected_vy)), float(np.hypot(actual_vx, actual_vy))
    velocity_error = float(np.hypot(actual_vx - expected_vx, actual_vy - expected_vy))
    velocity_similarity = exp(-velocity_error / max(0.01, expected_speed + 0.01))
    if expected_speed < EPS or actual_speed < EPS:
        direction_similarity = 1.0 if max(expected_speed, actual_speed) < 0.01 else 0.5
    else:
        cosine = (expected_vx * actual_vx + expected_vy * actual_vy) / (expected_speed * actual_speed)
        direction_similarity = float(np.clip((cosine + 1.0) * 0.5, 0.0, 1.0))
    scale_error = abs(float(candidate_action.values[7]) - expected_scale_w) + abs(float(candidate_action.values[8]) - expected_scale_h)
    scale_similarity = exp(-scale_error)
    confidence = row_score(candidate_row)
    iou_signal = predicted_iou if samurai_iou is None else 0.5 * predicted_iou + 0.5 * float(np.clip(samurai_iou, 0.0, 1.0))
    score = 0.38 * iou_signal + 0.24 * velocity_similarity + 0.18 * direction_similarity + 0.10 * scale_similarity + 0.10 * confidence
    return CandidateMotionScore(float(np.clip(score, 0.0, 1.0)), float(predicted_iou), float(velocity_similarity), float(direction_similarity), float(scale_similarity), float(confidence))


class DualTimeActionBankTransformer(torch.nn.Module):
    def __init__(self, token_dim: int = ACTION_TOKEN_DIM, short_tokens: int = 12, long_tokens: int = 18, d_model: int = 128, nhead: int = 4, num_layers: int = 2, future_steps: int = 1) -> None:
        super().__init__()
        self.short_tokens = int(short_tokens)
        self.long_tokens = int(long_tokens)
        self.future_steps = int(future_steps)
        self.short_projection = torch.nn.Linear(token_dim, d_model)
        self.long_projection = torch.nn.Linear(token_dim, d_model)
        self.short_position = torch.nn.Parameter(torch.zeros(1, self.short_tokens, d_model))
        self.long_position = torch.nn.Parameter(torch.zeros(1, self.long_tokens, d_model))
        short_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True, norm_first=True)
        long_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.short_encoder = torch.nn.TransformerEncoder(short_layer, num_layers=num_layers)
        self.long_encoder = torch.nn.TransformerEncoder(long_layer, num_layers=num_layers)
        self.fusion = torch.nn.Sequential(torch.nn.LayerNorm(d_model * 2), torch.nn.Linear(d_model * 2, d_model), torch.nn.GELU())
        self.motion_head = torch.nn.Linear(d_model, 1)
        self.future_head = torch.nn.Linear(d_model, self.future_steps * 4)
        self.reliability_head = torch.nn.Linear(d_model, 1)

    @staticmethod
    def _pool(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(self, short_tokens: torch.Tensor, short_mask: torch.Tensor, long_tokens: torch.Tensor, long_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        short = self.short_projection(short_tokens) + self.short_position[:, : short_tokens.shape[1]]
        long = self.long_projection(long_tokens) + self.long_position[:, : long_tokens.shape[1]]
        short_encoded = self.short_encoder(short, src_key_padding_mask=short_mask <= 0)
        long_encoded = self.long_encoder(long, src_key_padding_mask=long_mask <= 0)
        fused = self.fusion(torch.cat([self._pool(short_encoded, short_mask), self._pool(long_encoded, long_mask)], dim=1))
        motion_logits = self.motion_head(fused).squeeze(1)
        future_motion = self.future_head(fused).reshape(fused.shape[0], self.future_steps, 4)
        reliability = torch.sigmoid(self.reliability_head(fused)).squeeze(1)
        return motion_logits, future_motion, reliability


def attach_action_bank_scores(tracklet_jsonl: str, out_jsonl: str, config: ActionBankConfig | None = None, min_history_actions: int = 2) -> dict[str, Any]:
    import json
    from pathlib import Path

    cfg = config or ActionBankConfig()
    input_path, output_path = Path(tracklet_jsonl), Path(out_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracklets = rows_scored = 0
    score_values: list[float] = []
    with input_path.open("r", encoding="utf-8-sig") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            item = __import__("json").loads(line)
            rows = sorted(list(item.get("rows") or []), key=lambda row: configured_timestamp(row, cfg))
            row_scores: list[CandidateMotionScore] = []
            for index, row in enumerate(rows):
                if index < min_history_actions + 1:
                    row["action_bank_score"] = 0.0
                    row["action_bank_ready"] = False
                    continue
                history = rows[:index]
                snapshot = build_action_bank(history, config=cfg)
                samurai_iou = row.get("samurai_iou", row.get("samurai_cmc_forward_iou", row.get("samurai_cmc_backward_iou", row.get("predicted_iou", row.get("motion_iou")))))
                result = score_candidate(history[-1], row, snapshot, None if samurai_iou is None else _finite_float(samurai_iou))
                row.update(
                    {
                        "action_bank_score": result.score,
                        "action_bank_predicted_iou": result.predicted_iou,
                        "action_bank_velocity_similarity": result.velocity_similarity,
                        "action_bank_direction_similarity": result.direction_similarity,
                        "action_bank_scale_similarity": result.scale_similarity,
                        "action_bank_ready": True,
                        "action_bank_short_valid": int(snapshot.short_mask.sum()),
                        "action_bank_long_valid": int(snapshot.long_mask.sum()),
                    }
                )
                row_scores.append(result)
                score_values.append(result.score)
                rows_scored += 1
            meta = dict(item.get("meta") or {})
            if row_scores:
                meta.update(
                    {
                        "action_bank_score": float(np.mean([score.score for score in row_scores])),
                        "mean_action_bank_predicted_iou": float(np.mean([score.predicted_iou for score in row_scores])),
                        "mean_action_bank_velocity_similarity": float(np.mean([score.velocity_similarity for score in row_scores])),
                        "mean_action_bank_direction_similarity": float(np.mean([score.direction_similarity for score in row_scores])),
                        "mean_action_bank_scale_similarity": float(np.mean([score.scale_similarity for score in row_scores])),
                        "num_action_bank_windows": len(row_scores),
                        "action_bank_short_seconds": cfg.short_seconds,
                        "action_bank_long_seconds": cfg.long_seconds,
                    }
                )
            else:
                meta.update({"action_bank_score": 0.0, "num_action_bank_windows": 0})
            item["meta"], item["rows"] = meta, rows
            target.write(json.dumps(item, ensure_ascii=False) + "\n")
            tracklets += 1
    return {
        "input": str(input_path),
        "output": str(output_path),
        "tracklets": tracklets,
        "rows_scored": rows_scored,
        "mean_action_bank_score": float(np.mean(score_values)) if score_values else 0.0,
        "config": cfg.__dict__,
    }


@dataclass
class ActionBankTrack:
    track_id: int
    rows: list[dict[str, Any]]
    last_time: float
    missed_seconds: float = 0.0


class OnlineActionBankTracker:
    def __init__(self, config: ActionBankConfig | None = None, match_threshold: float = 0.35, max_dormant_seconds: float = 3.0) -> None:
        self.config = config or ActionBankConfig()
        self.match_threshold = float(match_threshold)
        self.max_dormant_seconds = float(max_dormant_seconds)
        self.tracks: dict[int, ActionBankTrack] = {}
        self.next_track_id = 1

    def _pair_score(self, track: ActionBankTrack, candidate: dict[str, Any]) -> float:
        if len(track.rows) >= 3:
            snapshot = build_action_bank(track.rows, config=self.config)
            samurai_iou = candidate.get("samurai_iou", candidate.get("samurai_cmc_forward_iou", candidate.get("samurai_cmc_backward_iou", candidate.get("predicted_iou", candidate.get("motion_iou")))))
            return score_candidate(track.rows[-1], candidate, snapshot, None if samurai_iou is None else _finite_float(samurai_iou)).score
        overlap = box_iou(track.rows[-1]["bbox"], candidate["bbox"])
        return float(np.clip(0.75 * overlap + 0.25 * row_score(candidate), 0.0, 1.0))

    @staticmethod
    def _greedy_assignment(scores: np.ndarray, threshold: float) -> list[tuple[int, int]]:
        matches: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_columns: set[int] = set()
        for flat_index in np.argsort(scores, axis=None)[::-1]:
            row_index, column_index = np.unravel_index(int(flat_index), scores.shape)
            if scores[row_index, column_index] < threshold:
                break
            if row_index not in used_rows and column_index not in used_columns:
                used_rows.add(row_index)
                used_columns.add(column_index)
                matches.append((row_index, column_index))
        return matches

    def update(self, candidates: Sequence[dict[str, Any]], timestamp: float | None = None) -> list[dict[str, Any]]:
        candidate_rows = [dict(candidate) for candidate in candidates]
        if not candidate_rows:
            if timestamp is not None:
                self._expire(float(timestamp))
            return []
        current_time = configured_timestamp(candidate_rows[0], self.config) if timestamp is None else float(timestamp)
        self._expire(current_time)
        active_tracks = list(self.tracks.values())
        scores = np.zeros((len(active_tracks), len(candidate_rows)), dtype=np.float32)
        for track_index, track in enumerate(active_tracks):
            for candidate_index, candidate in enumerate(candidate_rows):
                scores[track_index, candidate_index] = self._pair_score(track, candidate)
        matches = self._greedy_assignment(scores, self.match_threshold) if scores.size else []
        matched_candidates: set[int] = set()
        output: list[dict[str, Any]] = []
        for track_index, candidate_index in matches:
            track = active_tracks[track_index]
            candidate = candidate_rows[candidate_index]
            candidate["action_bank_track_id"] = track.track_id
            candidate["action_bank_association_score"] = float(scores[track_index, candidate_index])
            candidate["action_bank_reidentified"] = track.missed_seconds > 0.0
            track.rows.append(candidate)
            cutoff = current_time - self.config.long_seconds - self.config.max_gap_seconds
            track.rows = [row for row in track.rows if configured_timestamp(row, self.config) >= cutoff]
            track.last_time, track.missed_seconds = current_time, 0.0
            matched_candidates.add(candidate_index)
            output.append(candidate)
        for candidate_index, candidate in enumerate(candidate_rows):
            if candidate_index in matched_candidates:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            candidate["action_bank_track_id"] = track_id
            candidate["action_bank_association_score"] = row_score(candidate)
            candidate["action_bank_reidentified"] = False
            self.tracks[track_id] = ActionBankTrack(track_id=track_id, rows=[candidate], last_time=current_time)
            output.append(candidate)
        output.sort(key=lambda row: int(row["action_bank_track_id"]))
        return output

    def _expire(self, current_time: float) -> None:
        expired = []
        for track_id, track in self.tracks.items():
            track.missed_seconds = max(0.0, current_time - track.last_time)
            if track.missed_seconds > self.max_dormant_seconds:
                expired.append(track_id)
        for track_id in expired:
            del self.tracks[track_id]


class ActionBankWindowDataset(torch.utils.data.Dataset):
    def __init__(self, tracklet_jsonl: str, config: ActionBankConfig | None = None, min_history_actions: int = 2, max_samples: int | None = None) -> None:
        import json
        from pathlib import Path

        self.config = config or ActionBankConfig()
        self.source_path = Path(tracklet_jsonl).resolve()
        self.items: list[tuple[list[dict[str, Any]], int, int]] = []
        self.arrays: dict[str, np.ndarray] | None = None
        with self.source_path.open("r", encoding="utf-8-sig") as source:
            for line in source:
                if not line.strip():
                    continue
                item = json.loads(line)
                rows = sorted(list(item.get("rows") or []), key=lambda row: configured_timestamp(row, self.config))
                label = int(float((item.get("meta") or {}).get("label", 0)))
                for anchor_index in range(min_history_actions, len(rows) - 1):
                    self.items.append((rows, anchor_index, label))
        if max_samples is not None and len(self.items) > max_samples:
            positives = [item for item in self.items if item[2] > 0]
            negatives = [item for item in self.items if item[2] <= 0]
            positive_target = min(len(positives), max(1, int(round(max_samples * len(positives) / len(self.items)))))
            negative_target = min(len(negatives), max_samples - positive_target)
            positive_indices = np.linspace(0, len(positives) - 1, positive_target, dtype=np.int64) if positive_target else []
            negative_indices = np.linspace(0, len(negatives) - 1, negative_target, dtype=np.int64) if negative_target else []
            self.items = [positives[int(index)] for index in positive_indices] + [negatives[int(index)] for index in negative_indices]
        self.sample_count = len(self.items)
        self.positive_count = sum(label > 0 for _, _, label in self.items)

    def __len__(self) -> int:
        return self.sample_count

    def label_counts(self) -> tuple[int, int]:
        return self.positive_count, self.sample_count - self.positive_count

    def _sample_arrays(self, index: int) -> dict[str, np.ndarray | float]:
        rows, anchor_index, label = self.items[index]
        history = rows[: anchor_index + 1]
        snapshot = build_action_bank(history, config=self.config)
        future = action_token(rows[anchor_index], rows[anchor_index + 1], self.config)
        return {
            "short_tokens": snapshot.short_tokens,
            "short_mask": snapshot.short_mask,
            "long_tokens": snapshot.long_tokens,
            "long_mask": snapshot.long_mask,
            "target_motion": float(label),
            "future_motion": future.values[[3, 4, 7, 8]].astype(np.float32).reshape(1, 4),
            "future_reliable": float(future.reliable),
        }

    def _cache_manifest(self) -> dict[str, Any]:
        stat = self.source_path.stat()
        return {
            "source": str(self.source_path),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "samples": self.sample_count,
            "positives": self.positive_count,
            "config": self.config.__dict__,
        }

    def materialize(self, cache_dir: str | None = None, progress_interval: int = 5000) -> None:
        import json
        from pathlib import Path

        if self.arrays is not None:
            return
        cache = Path(cache_dir) if cache_dir else self.source_path.parent / ".action_bank_cache"
        cache.mkdir(parents=True, exist_ok=True)
        manifest_path = cache / "complete.json"
        expected = self._cache_manifest()
        names = ("short_tokens", "short_mask", "long_tokens", "long_mask", "target_motion", "future_motion", "future_reliable")
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing == expected and all((cache / f"{name}.npy").exists() for name in names):
                self.arrays = {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in names}
                self.items.clear()
                print(json.dumps({"kind": "action_bank_cache_ready", "done": self.sample_count, "total": self.sample_count, "cache": str(cache), "reused": True}), flush=True)
                return
        manifest_path.unlink(missing_ok=True)
        arrays = {
            "short_tokens": np.lib.format.open_memmap(cache / "short_tokens.npy", mode="w+", dtype=np.float16, shape=(self.sample_count, self.config.short_tokens, ACTION_TOKEN_DIM)),
            "short_mask": np.lib.format.open_memmap(cache / "short_mask.npy", mode="w+", dtype=np.uint8, shape=(self.sample_count, self.config.short_tokens)),
            "long_tokens": np.lib.format.open_memmap(cache / "long_tokens.npy", mode="w+", dtype=np.float16, shape=(self.sample_count, self.config.long_tokens, ACTION_TOKEN_DIM)),
            "long_mask": np.lib.format.open_memmap(cache / "long_mask.npy", mode="w+", dtype=np.uint8, shape=(self.sample_count, self.config.long_tokens)),
            "target_motion": np.lib.format.open_memmap(cache / "target_motion.npy", mode="w+", dtype=np.uint8, shape=(self.sample_count,)),
            "future_motion": np.lib.format.open_memmap(cache / "future_motion.npy", mode="w+", dtype=np.float32, shape=(self.sample_count, 1, 4)),
            "future_reliable": np.lib.format.open_memmap(cache / "future_reliable.npy", mode="w+", dtype=np.uint8, shape=(self.sample_count,)),
        }
        for index in range(self.sample_count):
            sample = self._sample_arrays(index)
            for name in names:
                arrays[name][index] = sample[name]
            done = index + 1
            if done == 1 or done % progress_interval == 0 or done == self.sample_count:
                print(json.dumps({"kind": "action_bank_cache_progress", "done": done, "total": self.sample_count, "cache": str(cache), "reused": False}), flush=True)
        for array in arrays.values():
            array.flush()
        manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        self.arrays = {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in names}
        self.items.clear()
        print(json.dumps({"kind": "action_bank_cache_ready", "done": self.sample_count, "total": self.sample_count, "cache": str(cache), "reused": False}), flush=True)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.arrays is None:
            sample = self._sample_arrays(index)
        else:
            sample = {name: self.arrays[name][index] for name in self.arrays}
        return {
            "short_tokens": torch.tensor(sample["short_tokens"], dtype=torch.float32),
            "short_mask": torch.tensor(sample["short_mask"], dtype=torch.float32),
            "long_tokens": torch.tensor(sample["long_tokens"], dtype=torch.float32),
            "long_mask": torch.tensor(sample["long_mask"], dtype=torch.float32),
            "target_motion": torch.tensor(float(sample["target_motion"]), dtype=torch.float32),
            "future_motion": torch.tensor(sample["future_motion"], dtype=torch.float32),
            "future_reliable": torch.tensor(float(sample["future_reliable"]), dtype=torch.float32),
        }
