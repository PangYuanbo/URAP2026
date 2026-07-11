from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import xgboost as xgb

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO, REPO / "tools"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.train_action_bank_motion_token_listwise import greedy_match_qualities, write_score_jsonl

WIDTH = 1280.0
HEIGHT = 960.0
CHAIN_FIELDS = (
    "action_chunk_bank_score",
    "action_chunk_bank_predicted_iou",
    "action_chunk_bank_center_similarity",
    "action_chunk_bank_direction_similarity",
    "action_chunk_bank_scale_similarity",
    "action_chunk_bank_track_quality",
    "action_chunk_bank_track_age_seconds",
    "action_chunk_bank_track_observations",
    "action_chunk_bank_chain_duration_seconds",
    "action_chunk_bank_acceleration_similarity",
    "action_chunk_bank_motion_stability",
    "action_chunk_bank_hypotheses",
    "action_chunk_bank_assigned",
)


def finite(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def frame_qualities(item) -> np.ndarray:
    detections = item.get("detections") or []
    boxes = [row.get("bbox") for row in detections]
    labels = [row.get("bbox") for row in item.get("labels") or []]
    valid_boxes = [box if isinstance(box, list) and len(box) == 4 else [0.0, 0.0, 0.0, 0.0] for box in boxes]
    valid_labels = [box for box in labels if isinstance(box, list) and len(box) == 4]
    gt = np.asarray(valid_labels, np.float32) if valid_labels else np.zeros((0, 4), np.float32)
    return greedy_match_qualities(valid_boxes, gt)


def history_features(history: deque, now: float) -> list[float]:
    while history and now - history[0][0] > 3.0:
        history.popleft()
    short = [event for event in history if now - event[0] <= 1.0]
    long = list(history)
    output = []
    for events, seconds in ((short, 1.0), (long, 3.0)):
        if not events:
            output.extend([0.0] * 7)
            continue
        values = np.asarray([event[1:] for event in events], np.float32)
        output.extend([
            min(1.0, len(events) / (8.0 if seconds == 1.0 else 16.0)),
            float(values[:, 0].mean()),
            float(values[:, 1].mean()),
            float(values[:, 2].mean()),
            float(values[:, 3].mean()),
            float(values[:, 1].std()),
            min(1.0, max(0.0, now - events[-1][0]) / seconds),
        ])
    return output


def build_arrays(predictions_path: Path, chain_path: Path, fps_path: Path, with_labels: bool):
    predictions = load_predictionsgt(predictions_path)
    fps_map = json.loads(fps_path.read_text(encoding="utf8"))
    total_candidates = sum(len(item.get("detections") or []) for item in predictions.values())
    features = np.empty((total_candidates, 38), np.float32)
    qualities = []
    groups = []
    locations = []
    sequences = []
    track_keys = []
    timestamps = []
    event_map = defaultdict(list)
    histories = defaultdict(deque)
    cursor = 0
    with chain_path.open("r", encoding="utf-8-sig") as source:
        for line in source:
            payload = json.loads(line)
            meta = payload.get("meta") or {}
            image_id = str(meta.get("image_id"))
            item = predictions[image_id]
            sequence, frame_id, _ = image_key(image_id, 0)
            fps = float(fps_map.get(sequence, meta.get("fps", 30.0)))
            now = frame_id / fps
            detections = item.get("detections") or []
            rows = payload.get("rows") or []
            current_quality = frame_qualities(item) if with_labels else np.zeros(len(detections), np.float32)
            raw_values = np.asarray([finite(row.get("score")) for row in detections], np.float32)
            order = np.argsort(np.argsort(raw_values)) if len(raw_values) else np.zeros(0, np.int64)
            rank = order / max(1, len(raw_values) - 1) if len(raw_values) else raw_values
            frame_features = []
            valid_indices = []
            pending_events = []
            for index, (detection, chain) in enumerate(zip(detections, rows)):
                bbox = detection.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [finite(value) for value in bbox]
                width = max(1e-3, x2 - x1)
                height = max(1e-3, y2 - y1)
                center_x = 0.5 * (x1 + x2)
                center_y = 0.5 * (y1 + y2)
                raw = finite(detection.get("score"))
                clipped = min(1.0 - 1e-6, max(1e-6, raw))
                track_id = int(chain.get("action_chunk_bank_track_id", -1))
                track_key = (sequence, track_id) if track_id >= 0 else None
                history = histories[track_key] if track_key is not None else deque()
                prefix = [
                    raw,
                    math.log(clipped / (1.0 - clipped)),
                    float(rank[index]),
                    float(raw_values.max() - raw) if len(raw_values) else 0.0,
                    center_x / WIDTH,
                    center_y / HEIGHT,
                    width / WIDTH,
                    height / HEIGHT,
                    width * height / (WIDTH * HEIGHT),
                    math.log(width / height),
                    math.log1p(len(detections)) / 6.0,
                ]
                chain_values = [finite(chain.get(field)) for field in CHAIN_FIELDS]
                frame_features.append(prefix + chain_values + history_features(history, now))
                valid_indices.append(index)
                locations.append((image_id, index))
                sequences.append(sequence)
                track_keys.append(track_key)
                timestamps.append(now)
                quality = float(current_quality[index]) if index < len(current_quality) else 0.0
                qualities.append(quality)
                assigned_track_id = int(chain.get("action_chunk_bank_assigned_track_id", -1))
                if int(chain.get("action_chunk_bank_assigned", 0)) and assigned_track_id >= 0:
                    assigned_key = (sequence, assigned_track_id)
                    pending_events.append((assigned_key, now, raw, finite(chain.get("action_chunk_bank_score")), finite(chain.get("action_chunk_bank_predicted_iou")), finite(chain.get("action_chunk_bank_track_quality")), quality))
            if frame_features:
                frame_count = len(frame_features)
                features[cursor:cursor + frame_count] = np.asarray(frame_features, np.float32)
                groups.append((cursor, cursor + frame_count))
                cursor += frame_count
            for assigned_key, event_time, raw, bank_score, predicted_iou, track_quality, quality in pending_events:
                histories[assigned_key].append((event_time, raw, bank_score, predicted_iou, track_quality))
                if with_labels:
                    event_map[assigned_key].append((event_time, quality))
    features = features[:cursor]
    quality_array = np.asarray(qualities, np.float32)
    future = np.zeros(len(features), np.float32)
    if with_labels:
        event_arrays = {}
        for key, events in event_map.items():
            event_arrays[key] = (np.asarray([event[0] for event in events], np.float64), np.asarray([event[1] for event in events], np.float32))
        for index, (key, now) in enumerate(zip(track_keys, timestamps)):
            if key is None or key not in event_arrays:
                continue
            event_times, event_quality = event_arrays[key]
            start = np.searchsorted(event_times, now + 1e-9, side="left")
            stop = np.searchsorted(event_times, now + 1.0 + 1e-9, side="right")
            if stop > start:
                future[index] = float(np.mean(event_quality[start:stop] >= 0.5))
    relevance = quality_array * (1.0 + future)
    return features, relevance.astype(np.float32), quality_array, groups, locations, np.asarray(sequences), future, np.asarray(timestamps, np.float64)


def selected_rows(features, quality, groups, margin: float = 0.35, max_negative: int = 28):
    keep = []
    qid = []
    group_id = 0
    for start, stop in groups:
        positives = np.flatnonzero(quality[start:stop] >= 0.5)
        negatives = np.flatnonzero(quality[start:stop] < 0.5)
        if not len(positives) or not len(negatives):
            continue
        raw = features[start:stop, 0]
        hard = negatives[raw[negatives] >= raw[positives].max() - margin]
        if len(hard):
            hard = hard[np.argsort(raw[hard])[::-1][:max_negative]]
        else:
            hard = negatives[np.argsort(raw[negatives])[::-1][:4]]
        local = np.sort(np.concatenate((positives, hard)))
        keep.extend((start + local).tolist())
        qid.extend([group_id] * len(local))
        group_id += 1
    return np.asarray(keep, np.int64), np.asarray(qid, np.int32)


def fit(features, relevance, quality, groups):
    keep, qid = selected_rows(features, quality, groups)
    model = xgb.XGBRanker(
        n_estimators=1100,
        max_depth=8,
        learning_rate=0.025,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=10,
        reg_alpha=0.1,
        gamma=0.02,
        objective="rank:pairwise",
        eval_metric="ndcg@5",
        tree_method="hist",
        device="cuda",
        max_bin=256,
        n_jobs=8,
        random_state=2026,
    )
    model.fit(features[keep], relevance[keep], qid=qid, verbose=False)
    return model, len(keep), int((quality[keep] >= 0.5).sum()), int(qid.max() + 1)


def normalize_by_group(values, groups):
    output = np.zeros(len(values), np.float32)
    for start, stop in groups:
        block = np.asarray(values[start:stop], np.float32)
        if not len(block):
            continue
        center = float(np.median(block))
        scale = max(1e-3, float(block.std()))
        z = np.clip((block - center) / scale, -12, 12)
        output[start:stop] = 1.0 / (1.0 + np.exp(-z))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Pure Action Chunk chain-consistency ranker with future-only training supervision.")
    for name in ("train-pkl", "train-chain", "val-pkl", "val-chain", "test-pkl", "test-chain", "fps-json", "out-val-scores", "out-test-scores", "out-model-dir", "out-summary"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--score-field", default="action_chunk_chain_consistency_score")
    args = parser.parse_args()
    train_x, train_rel, train_y, train_groups, _, _, train_future, _ = build_arrays(args.train_pkl, args.train_chain, args.fps_json, True)
    val_x, val_rel, val_y, val_groups, val_locations, val_sequences, val_future, _ = build_arrays(args.val_pkl, args.val_chain, args.fps_json, True)
    test_x, _, test_y, test_groups, test_locations, _, _, _ = build_arrays(args.test_pkl, args.test_chain, args.fps_json, False)
    oof_raw = np.zeros(len(val_x), np.float32)
    test_predictions = []
    models = []
    args.out_model_dir.mkdir(parents=True, exist_ok=True)
    for held_sequence in sorted(set(val_sequences)):
        parts = [train_x]
        relevance_parts = [train_rel]
        quality_parts = [train_y]
        groups = list(train_groups)
        cursor = len(train_x)
        for start, stop in val_groups:
            if val_sequences[start] == held_sequence:
                continue
            parts.append(val_x[start:stop])
            relevance_parts.append(val_rel[start:stop])
            quality_parts.append(val_y[start:stop])
            groups.append((cursor, cursor + stop - start))
            cursor += stop - start
        fit_x = np.concatenate(parts)
        fit_rel = np.concatenate(relevance_parts)
        fit_y = np.concatenate(quality_parts)
        model, rows, positives, group_count = fit(fit_x, fit_rel, fit_y, groups)
        mask = val_sequences == held_sequence
        oof_raw[mask] = model.predict(val_x[mask])
        test_predictions.append(model.predict(test_x))
        model_path = args.out_model_dir / f"action_chunk_chain_without_{held_sequence}.ubj"
        model.save_model(model_path)
        record = {"excluded_validation_video": held_sequence, "rank_rows": rows, "positive_rows": positives, "groups": group_count, "model": str(model_path)}
        models.append(record)
        print(json.dumps({"kind": "action_chunk_chain_model", **record}), flush=True)
        del fit_x, fit_rel, fit_y, model
        gc.collect()
    oof = normalize_by_group(oof_raw, val_groups)
    test = normalize_by_group(np.mean(np.stack(test_predictions), axis=0), test_groups)
    write_score_jsonl(args.out_val_scores, oof, val_locations, args.score_field)
    write_score_jsonl(args.out_test_scores, test, test_locations, args.score_field)
    summary = {
        "model": "pure Action Chunk causal chain ranker with future 1-second consistency supervision",
        "inference_boundary": "past-only 1s/3s chain features",
        "training_supervision": "future 1s assigned-chain correctness weights positive relevance",
        "features": int(train_x.shape[1]),
        "train_rows": len(train_x),
        "validation_rows": len(val_x),
        "test_rows": len(test_x),
        "train_future_mean": float(train_future.mean()),
        "validation_future_mean": float(val_future.mean()),
        "models": models,
    }
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
