from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.train_detection_row_score_head import MLP, iou_max

AUX_NAMES = [
    "action_bank_score",
    "action_bank_predicted_iou",
    "action_bank_velocity_similarity",
    "action_bank_direction_similarity",
    "action_bank_scale_similarity",
    "action_bank_learned_score",
    "action_bank_motion_probability",
    "action_bank_future_consistency",
    "action_bank_reliability",
    "samurai_cmc_score",
    "samurai_cmc_forward_iou",
    "samurai_cmc_residual_speed",
    "samurai_cmc_camera_validity",
    "camera_motion_validity",
    "online_action_bank_score",
    "online_action_bank_predicted_iou",
    "online_action_bank_center_similarity",
    "online_action_bank_direction_similarity",
    "online_action_bank_scale_similarity",
    "online_action_bank_track_quality",
    "online_action_bank_track_age_seconds",
    "online_action_bank_acceleration_similarity",
    "online_action_bank_motion_stability",
    "online_action_bank_hypotheses",
    "online_action_bank_future_consistency",
]
SHORT_TOKEN_COUNT = 8
LONG_TOKEN_COUNT = 16
TOKEN_FIELDS = (
    "valid",
    "residual_dx",
    "residual_dy",
    "velocity_error_x",
    "velocity_error_y",
    "acceleration_x",
    "acceleration_y",
    "scale_error_w",
    "scale_error_h",
    "motion_iou",
    "detector_score",
    "compatibility",
)
TOKEN_DIM = len(TOKEN_FIELDS)
TOKEN_FEATURE_NAMES = [
    *(f"online_action_bank_short_token_{index}_{field}" for index in range(SHORT_TOKEN_COUNT) for field in TOKEN_FIELDS),
    *(f"online_action_bank_long_token_{index}_{field}" for index in range(LONG_TOKEN_COUNT) for field in TOKEN_FIELDS),
]
NATIVE_NAMES = [
    "samurai_native_score",
    "samurai_native_iou",
    "samurai_native_predicted_iou",
    "samurai_native_object_score",
    "samurai_native_active",
    "samurai_native_reset_event",
    "samurai_native_low_object_frames",
    "samurai_native_disagreement_frames",
    "samurai_native_object_count",
    "samurai_native_best_object_id",
]

FEATURE_NAMES = [
    "raw_score",
    "raw_logit",
    "cx_norm",
    "cy_norm",
    "w_norm",
    "h_norm",
    "area_norm",
    "aspect_ratio",
    "candidate_count_log",
    "raw_rank_percentile",
    "raw_gap_to_max",
    "aux_present",
] + AUX_NAMES + TOKEN_FEATURE_NAMES + [
    "bank_rank_percentile",
    "bank_gap_to_max",
    "samurai_rank_percentile",
    "samurai_gap_to_max",
    "native_present",
] + NATIVE_NAMES + [
    "native_rank_percentile",
    "native_gap_to_max",
]


class ActionTokenRanker(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 192) -> None:
        super().__init__()
        token_start = FEATURE_NAMES.index(TOKEN_FEATURE_NAMES[0])
        token_stop = token_start + len(TOKEN_FEATURE_NAMES)
        static_indices = [index for index in range(in_dim) if not token_start <= index < token_stop]
        self.register_buffer("static_indices", torch.tensor(static_indices, dtype=torch.long), persistent=False)
        self.token_start = token_start
        self.short_width = TOKEN_DIM * SHORT_TOKEN_COUNT
        token_hidden = max(24, hidden // 4)
        self.static_net = torch.nn.Sequential(
            torch.nn.Linear(len(static_indices), hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        self.token_net = torch.nn.Sequential(
            torch.nn.Linear(TOKEN_DIM + 1, token_hidden),
            torch.nn.LayerNorm(token_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(token_hidden, token_hidden),
            torch.nn.SiLU(),
        )
        self.short_attention = torch.nn.Linear(token_hidden, 1)
        self.long_attention = torch.nn.Linear(token_hidden, 1)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden + 2 * token_hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(hidden, 1),
        )

    def _pool(self, tokens: torch.Tensor, attention: torch.nn.Linear) -> torch.Tensor:
        valid = tokens[..., 0]
        count = tokens.shape[1]
        age = torch.linspace(0.0, 1.0, count, device=tokens.device, dtype=tokens.dtype).view(1, count, 1)
        age = age.expand(tokens.shape[0], -1, -1)
        embedding = self.token_net(torch.cat((tokens, age), dim=-1))
        logits = attention(embedding).squeeze(-1).masked_fill(valid <= 0.5, -1e4)
        weights = torch.softmax(logits, dim=1)
        pooled = (weights.unsqueeze(-1) * embedding).sum(dim=1)
        return pooled * (valid > 0.5).any(dim=1, keepdim=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        static = self.static_net(features.index_select(1, self.static_indices))
        short_start = self.token_start
        short_stop = short_start + self.short_width
        long_stop = short_stop + TOKEN_DIM * LONG_TOKEN_COUNT
        short = features[:, short_start:short_stop].reshape(-1, SHORT_TOKEN_COUNT, TOKEN_DIM)
        long = features[:, short_stop:long_stop].reshape(-1, LONG_TOKEN_COUNT, TOKEN_DIM)
        short_pool = self._pool(short, self.short_attention)
        long_pool = self._pool(long, self.long_attention)
        return self.head(torch.cat((static, short_pool, long_pool), dim=1)).squeeze(-1)


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


def load_auxiliary(path: Path) -> tuple[dict[tuple[str, int, int], np.ndarray], dict[str, tuple[float, float]]]:
    values: dict[tuple[str, int, int], np.ndarray] = {}
    sequence_sizes: dict[str, tuple[float, float]] = {}
    payload_size = len(AUX_NAMES) + len(TOKEN_FEATURE_NAMES)
    bank_score_index = AUX_NAMES.index("action_bank_learned_score")
    with path.open("r", encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            meta = dict(item.get("meta") or {})
            for row in item.get("rows") or []:
                seq = str(row.get("seq") or meta.get("seq") or "")
                frame_id = row.get("frame_id")
                pred_index = row.get("prediction_index")
                if not seq or frame_id is None or pred_index is None:
                    continue
                width = max(1.0, finite(row.get("image_width")))
                height = max(1.0, finite(row.get("image_height")))
                sequence_sizes[seq] = (width, height)
                payload = np.zeros((payload_size,), dtype=np.float16)
                payload[: len(AUX_NAMES)] = np.asarray(
                    [finite(row.get(name, meta.get(name))) for name in AUX_NAMES], dtype=np.float16
                )
                short_tokens = np.asarray(
                    list(row.get("online_action_bank_short_tokens") or [])[: TOKEN_DIM * SHORT_TOKEN_COUNT],
                    dtype=np.float16,
                )
                long_tokens = np.asarray(
                    list(row.get("online_action_bank_long_tokens") or [])[: TOKEN_DIM * LONG_TOKEN_COUNT],
                    dtype=np.float16,
                )
                short_stop = len(AUX_NAMES) + min(len(short_tokens), TOKEN_DIM * SHORT_TOKEN_COUNT)
                payload[len(AUX_NAMES):short_stop] = short_tokens[: short_stop - len(AUX_NAMES)]
                long_start = len(AUX_NAMES) + TOKEN_DIM * SHORT_TOKEN_COUNT
                long_stop = long_start + min(len(long_tokens), TOKEN_DIM * LONG_TOKEN_COUNT)
                payload[long_start:long_stop] = long_tokens[: long_stop - long_start]
                key = (seq, int(float(frame_id)), int(float(pred_index)))
                previous = values.get(key)
                if previous is None or payload[bank_score_index] > previous[bank_score_index]:
                    values[key] = payload
    return values, sequence_sizes



def load_native(path: Path | None) -> dict[tuple[str, int, int], tuple[float, ...]]:
    values: dict[tuple[str, int, int], tuple[float, ...]] = {}
    if path is None:
        return values
    with path.open("r", encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            meta = dict(item.get("meta") or {})
            for row in item.get("rows") or []:
                seq = str(row.get("seq") or meta.get("seq") or "")
                frame_id = row.get("frame_id")
                pred_index = row.get("prediction_index")
                if not seq or frame_id is None or pred_index is None:
                    continue
                values[(seq, int(float(frame_id)), int(float(pred_index)))] = tuple(
                    finite(row.get(name, meta.get(name))) for name in NATIVE_NAMES
                )
    return values

def percentile(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.ones_like(values, dtype=np.float32)
    ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return ranks.astype(np.float32) / float(len(values) - 1)


def greedy_match_qualities(candidate_boxes: list[list[float]], gt_boxes: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    qualities = np.zeros((len(candidate_boxes),), dtype=np.float32)
    if not candidate_boxes or not len(gt_boxes):
        return qualities
    pairs: list[tuple[float, int, int]] = []
    for candidate_index, candidate_box in enumerate(candidate_boxes):
        for gt_index, gt_box in enumerate(gt_boxes):
            overlap = iou_max(candidate_box, np.asarray([gt_box], dtype=np.float32))
            if overlap >= threshold:
                pairs.append((float(overlap), candidate_index, gt_index))
    matched_candidates: set[int] = set()
    matched_gt: set[int] = set()
    for overlap, candidate_index, gt_index in sorted(pairs, reverse=True):
        if candidate_index in matched_candidates or gt_index in matched_gt:
            continue
        qualities[candidate_index] = overlap
        matched_candidates.add(candidate_index)
        matched_gt.add(gt_index)
    return qualities


def dataset_arrays(
    predictions: dict[str, Any],
    auxiliary: dict[tuple[str, int, int], np.ndarray],
    sequence_sizes: dict[str, tuple[float, float]],
    native: dict[tuple[str, int, int], tuple[float, ...]],
    with_labels: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], list[tuple[str, int]]]:
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    future_chunks: list[np.ndarray] = []
    groups: list[tuple[int, int]] = []
    locations: list[tuple[str, int]] = []
    bank_index = AUX_NAMES.index("action_bank_learned_score")
    online_bank_index = AUX_NAMES.index("online_action_bank_score")
    samurai_index = AUX_NAMES.index("samurai_cmc_forward_iou")
    future_index = AUX_NAMES.index("online_action_bank_future_consistency")
    native_score_index = NATIVE_NAMES.index("samurai_native_score")
    cursor = 0
    for image_id in sorted(predictions):
        item = predictions[image_id]
        detections = item.get("detections", [])
        if not detections:
            continue
        seq, frame_id, _ = image_key(str(image_id), 0)
        width, height = sequence_sizes.get(seq, (1920.0, 1280.0))
        raw = np.asarray([finite(row.get("score")) for row in detections], dtype=np.float32)
        aux_rows = [auxiliary.get((seq, frame_id, index)) for index in range(len(detections))]
        native_rows = [native.get((seq, frame_id, index)) for index in range(len(detections))]
        bank = np.asarray([max(float(payload[bank_index]), float(payload[online_bank_index])) if payload is not None else 0.0 for payload in aux_rows], dtype=np.float32)
        samurai = np.asarray([float(payload[samurai_index]) if payload is not None else 0.0 for payload in aux_rows], dtype=np.float32)
        native_score = np.asarray([payload[native_score_index] if payload is not None else 0.0 for payload in native_rows], dtype=np.float32)
        raw_rank = percentile(raw)
        bank_rank = percentile(bank)
        samurai_rank = percentile(samurai)
        native_rank = percentile(native_score)
        gt = np.asarray([row.get("bbox") for row in item.get("labels", []) if isinstance(row.get("bbox"), list) and len(row.get("bbox")) == 4], dtype=np.float32)
        if gt.size == 0:
            gt = np.zeros((0, 4), dtype=np.float32)
        candidate_boxes = [
            [finite(value) for value in detection["bbox"]]
            if isinstance(detection.get("bbox"), list) and len(detection["bbox"]) == 4
            else [0.0, 0.0, 0.0, 0.0]
            for detection in detections
        ]
        matched = greedy_match_qualities(candidate_boxes, gt) if with_labels else np.zeros((len(detections),), dtype=np.float32)
        frame_rows: list[np.ndarray] = []
        frame_labels: list[float] = []
        frame_future: list[float] = []
        for index, detection in enumerate(detections):
            bbox = detection.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = map(finite, bbox)
            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)
            score = raw[index]
            clipped = min(1.0 - 1e-6, max(1e-6, float(score)))
            payload = aux_rows[index]
            native_payload = native_rows[index]
            aux_values = np.asarray(payload, dtype=np.float32).copy() if payload is not None else np.zeros((len(AUX_NAMES) + len(TOKEN_FEATURE_NAMES),), dtype=np.float32)
            future_supervision = float(aux_values[future_index])
            aux_values[future_index] = 0.0
            native_values = np.asarray(native_payload, dtype=np.float32) if native_payload is not None else np.zeros((len(NATIVE_NAMES),), dtype=np.float32)
            prefix = np.asarray([
                score, math.log(clipped / (1.0 - clipped)),
                (x1 + x2) * 0.5 / width, (y1 + y2) * 0.5 / height,
                box_width / width, box_height / height,
                box_width * box_height / (width * height), box_width / max(1e-6, box_height),
                math.log1p(len(detections)), raw_rank[index], raw.max() - score, float(payload is not None),
            ], dtype=np.float32)
            suffix = np.asarray([
                bank_rank[index], bank.max() - bank[index], samurai_rank[index], samurai.max() - samurai[index],
                float(native_payload is not None),
            ], dtype=np.float32)
            tail = np.asarray([native_rank[index], native_score.max() - native_score[index]], dtype=np.float32)
            frame_rows.append(np.concatenate((prefix, aux_values, suffix, native_values, tail)))
            frame_labels.append(float(matched[index]) if with_labels else 0.0)
            frame_future.append(future_supervision if with_labels else 0.0)
            locations.append((str(image_id), index))
        if frame_rows:
            frame_array = np.stack(frame_rows).astype(np.float32, copy=False)
            feature_chunks.append(frame_array)
            label_chunks.append(np.asarray(frame_labels, dtype=np.float32))
            future_chunks.append(np.asarray(frame_future, dtype=np.float32))
            groups.append((cursor, cursor + len(frame_array)))
            cursor += len(frame_array)
    if not feature_chunks:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), groups, locations
    return np.concatenate(feature_chunks), np.concatenate(label_chunks), np.concatenate(future_chunks), groups, locations



def group_loss(logits: torch.Tensor, ious: torch.Tensor, future_consistency: torch.Tensor, groups: list[tuple[int, int]], device: torch.device) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for start, stop in groups:
        scores = logits[start:stop]
        overlaps = ious[start:stop]
        future = future_consistency[start:stop]
        positives = overlaps >= 0.5
        labels = positives.float()
        if positives.any():
            quality = 0.85 * overlaps + 0.15 * future
            target = torch.where(positives, torch.exp(12.0 * (quality - 0.5)), torch.zeros_like(overlaps))
            target /= target.sum().clamp_min(1e-6)
            listwise = -(target * torch.log_softmax(scores, dim=0)).sum()
            best_positive = scores[torch.argmax(quality)]
            hard_negative = scores[~positives].max() if (~positives).any() else best_positive.detach()
            pairwise = torch.nn.functional.softplus(1.0 - best_positive + hard_negative)
            weights = torch.where(positives, torch.full_like(scores, 4.0), torch.ones_like(scores))
            classification = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels, weight=weights)
            losses.append(0.55 * listwise + 1.50 * pairwise + 0.85 * classification)
        else:
            background = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
            hard_false_positive = torch.nn.functional.softplus(scores.max() + 1.0)
            losses.append(background + 0.50 * hard_false_positive)
    return torch.stack(losses).mean()



def write_score_jsonl(path: Path, scores: np.ndarray, locations: list[tuple[str, int]], score_field: str) -> None:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (image_id, pred_index), score in zip(locations, scores):
        grouped[image_id].append((pred_index, float(score)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for image_id in sorted(grouped):
            seq, frame_id, _ = image_key(image_id, 0)
            rows = [{"seq": seq, "frame_id": frame_id, "prediction_index": index, score_field: score} for index, score in sorted(grouped[image_id])]
            target.write(json.dumps({"meta": {"seq": seq, "image_id": image_id}, "rows": rows}, separators=(",", ":")) + "\n")


def predict(model: torch.nn.Module, features: np.ndarray, mean: np.ndarray, std: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    normalized = (features - mean) / std
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            logits = model(torch.from_numpy(normalized[start:start + batch_size]).to(device, non_blocking=True))
            outputs.append(torch.sigmoid(logits).cpu())
    return torch.cat(outputs).numpy() if outputs else np.zeros((0,), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train predictionsgt-native causal listwise Action Bank selector over every detector candidate.")
    parser.add_argument("--train-pkl", type=Path, required=True)
    parser.add_argument("--train-aux-tracklets", type=Path, required=True)
    parser.add_argument("--train-native-tracklets", type=Path)
    parser.add_argument("--val-pkl", type=Path, required=True)
    parser.add_argument("--val-aux-tracklets", type=Path, required=True)
    parser.add_argument("--val-native-tracklets", type=Path)
    parser.add_argument("--test-pkl", type=Path, required=True)
    parser.add_argument("--test-aux-tracklets", type=Path, required=True)
    parser.add_argument("--test-native-tracklets", type=Path)
    parser.add_argument("--out-val-scores", type=Path, required=True)
    parser.add_argument("--out-test-scores", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--score-field", default="action_bank_all_candidate_score")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--frame-batch-size", type=int, default=192)
    parser.add_argument("--inference-batch-size", type=int, default=16384)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--token-encoder", action="store_true")
    args = parser.parse_args()

    train_aux, train_sizes = load_auxiliary(args.train_aux_tracklets)
    train_native = load_native(args.train_native_tracklets)
    train_x, train_iou, train_future, train_groups, _ = dataset_arrays(load_predictionsgt(args.train_pkl), train_aux, train_sizes, train_native, True)
    val_aux, val_sizes = load_auxiliary(args.val_aux_tracklets)
    val_native = load_native(args.val_native_tracklets)
    val_x, _val_iou, _val_future, _val_groups, val_locations = dataset_arrays(load_predictionsgt(args.val_pkl), val_aux, val_sizes, val_native, False)
    test_aux, test_sizes = load_auxiliary(args.test_aux_tracklets)
    test_native = load_native(args.test_native_tracklets)
    test_x, _test_iou, _test_future, _test_groups, test_locations = dataset_arrays(load_predictionsgt(args.test_pkl), test_aux, test_sizes, test_native, False)
    del train_aux, val_aux, test_aux, train_native, val_native, test_native

    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    if args.token_encoder:
        token_valid_indices = [FEATURE_NAMES.index(name) for name in TOKEN_FEATURE_NAMES if name.endswith("_valid")]
        mean[token_valid_indices] = 0.0
        std[token_valid_indices] = 1.0
    normalized = (train_x - mean) / std
    positive_groups = np.asarray([index for index, (start, stop) in enumerate(train_groups) if (train_iou[start:stop] >= 0.5).any()], dtype=np.int64)
    background_groups = np.asarray([index for index, (start, stop) in enumerate(train_groups) if not (train_iou[start:stop] >= 0.5).any()], dtype=np.int64)
    if not len(positive_groups):
        raise RuntimeError("training candidate set contains no IoU>=0.5 positive frames")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (ActionTokenRanker(normalized.shape[1], hidden=args.hidden) if args.token_encoder else MLP(normalized.shape[1], hidden=args.hidden)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = np.random.default_rng(2026)
    history = []
    for epoch in range(1, args.epochs + 1):
        group_order = np.concatenate((positive_groups, background_groups))
        generator.shuffle(group_order)
        losses = []
        model.train()
        for batch_start in range(0, len(group_order), args.frame_batch_size):
            selected = group_order[batch_start:batch_start + args.frame_batch_size]
            chunks_x = []
            chunks_y = []
            chunks_future = []
            local_groups = []
            cursor = 0
            for group_index in selected:
                start, stop = train_groups[int(group_index)]
                chunks_x.append(normalized[start:stop])
                chunks_y.append(train_iou[start:stop])
                chunks_future.append(train_future[start:stop])
                local_groups.append((cursor, cursor + stop - start))
                cursor += stop - start
            batch_x = torch.from_numpy(np.concatenate(chunks_x)).to(device, non_blocking=True)
            batch_y = torch.from_numpy(np.concatenate(chunks_y)).to(device, non_blocking=True)
            batch_future = torch.from_numpy(np.concatenate(chunks_future)).to(device, non_blocking=True)
            logits = model(batch_x)
            loss = group_loss(logits, batch_y, batch_future, local_groups, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "epochs": args.epochs, "loss": float(np.mean(losses)), "device": str(device)}
        if device.type == "cuda":
            row["cuda_memory_allocated_mb"] = round(torch.cuda.memory_allocated(device) / 1048576, 3)
        history.append(row)
        print(json.dumps({"kind": "motion_token_ap_train_progress", **row}), flush=True)

    val_scores = predict(model, val_x, mean, std, args.inference_batch_size, device)
    test_scores = predict(model, test_x, mean, std, args.inference_batch_size, device)
    write_score_jsonl(args.out_val_scores, val_scores, val_locations, args.score_field)
    write_score_jsonl(args.out_test_scores, test_scores, test_locations, args.score_field)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "features": FEATURE_NAMES, "hidden": args.hidden, "score_field": args.score_field, "model_type": "action_token_ranker" if args.token_encoder else "mlp"}, args.out_model)
    summary = {
        "device": str(device),
        "model_type": "action_token_ranker" if args.token_encoder else "mlp",
        "action_tokens": {"short_seconds": 1.0, "short_count": SHORT_TOKEN_COUNT, "long_seconds": 3.0, "long_count": LONG_TOKEN_COUNT, "fields": TOKEN_FIELDS} if args.token_encoder else None,
        "train_rows": len(train_x),
        "train_frames": len(train_groups),
        "train_positive_rows": int((train_iou >= 0.5).sum()),
        "train_future_consistency_mean": float(train_future.mean()),
        "training_supervision": "greedy one-to-one candidate/GT matches with 20% future-1s consistency quality; unmatched duplicate detections are negatives; future field masked from all model inputs",
        "positive_frames": len(positive_groups),
        "background_frames": len(background_groups),
        "train_aux_coverage": float(train_x[:, FEATURE_NAMES.index("aux_present")].mean()),
        "train_native_coverage": float(train_x[:, FEATURE_NAMES.index("native_present")].mean()),
        "val_rows": len(val_x),
        "val_aux_coverage": float(val_x[:, FEATURE_NAMES.index("aux_present")].mean()),
        "val_native_coverage": float(val_x[:, FEATURE_NAMES.index("native_present")].mean()),
        "test_rows": len(test_x),
        "test_aux_coverage": float(test_x[:, FEATURE_NAMES.index("aux_present")].mean()),
        "test_native_coverage": float(test_x[:, FEATURE_NAMES.index("native_present")].mean()),
        "history": history,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"kind": "motion_token_ap_train_done", **summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

