from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from train_detection_row_score_head import EXTRA_FEATURES, MODEL_FEATURES, ROW_FEATURES, MLP, _float, feature_row, iou_max, load_gt

CONTEXT_FEATURES = [
    "frame_candidate_count_log",
    "raw_rank_percentile",
    "raw_gap_to_frame_max",
    "bank_rank_percentile",
    "bank_gap_to_frame_max",
    "samurai_rank_percentile",
    "samurai_gap_to_frame_max",
]
CAUSAL_BASE_FEATURES = [
    "objectness",
    "final_drone_score",
    "score",
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
    "cx_norm",
    "cy_norm",
    "w_norm",
    "h_norm",
    "area_norm",
    "aspect_ratio",
]
CAUSAL_BASE_INDICES = [MODEL_FEATURES.index(name) for name in CAUSAL_BASE_FEATURES]
ALL_FEATURES = CAUSAL_BASE_FEATURES + CONTEXT_FEATURES


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.ones_like(values, dtype=np.float32)
    order = np.argsort(np.argsort(values))
    return order.astype(np.float32) / float(len(values) - 1)


def load_records(paths: list[Path], gt: dict[tuple[str, int], np.ndarray] | None) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[tuple[Path, int, int]]]:
    records: list[dict[str, Any]] = []
    frame_members: dict[tuple[str, int], list[int]] = defaultdict(list)
    locations: list[tuple[Path, int, int]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as source:
            for line_index, line in enumerate(source):
                if not line.strip():
                    continue
                item = json.loads(line)
                meta = dict(item.get("meta") or {})
                rows = [dict(row) for row in item.get("rows") or []]
                for row_index, row in enumerate(rows):
                    seq = str(row.get("seq") or meta.get("seq") or "")
                    frame_id = int(float(row.get("frame_id", 0) or 0))
                    bbox = row.get("bbox")
                    if not seq or not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    raw = max(_float(row.get("score")), _float(row.get("objectness")), _float(row.get("final_drone_score")))
                    bank = _float(row.get("action_bank_learned_score"))
                    samurai = _float(row.get("samurai_cmc_forward_iou"))
                    target_iou = iou_max([_float(value) for value in bbox], (gt or {}).get((seq, frame_id), np.zeros((0, 4), dtype=np.float32))) if gt is not None else 0.0
                    index = len(records)
                    records.append({
                        "frame": (seq, frame_id),
                        "base": [feature_row(row, meta, row_index, len(rows))[index] for index in CAUSAL_BASE_INDICES],
                        "raw": raw,
                        "bank": bank,
                        "samurai": samurai,
                        "iou": target_iou,
                    })
                    frame_members[(seq, frame_id)].append(index)
                    locations.append((path, line_index, row_index))
    features = np.zeros((len(records), len(ALL_FEATURES)), dtype=np.float32)
    labels = np.zeros((len(records),), dtype=np.float32)
    groups: list[tuple[int, int]] = []
    ordered_indices: list[int] = []
    for key in sorted(frame_members):
        indices = frame_members[key]
        start = len(ordered_indices)
        ordered_indices.extend(indices)
        groups.append((start, len(ordered_indices)))
        raw = np.asarray([records[index]["raw"] for index in indices], dtype=np.float32)
        bank = np.asarray([records[index]["bank"] for index in indices], dtype=np.float32)
        samurai = np.asarray([records[index]["samurai"] for index in indices], dtype=np.float32)
        context = np.column_stack((
            np.full(len(indices), math.log1p(len(indices)), dtype=np.float32),
            percentile_ranks(raw),
            raw.max(initial=0.0) - raw,
            percentile_ranks(bank),
            bank.max(initial=0.0) - bank,
            percentile_ranks(samurai),
            samurai.max(initial=0.0) - samurai,
        ))
        for offset, index in enumerate(indices):
            features[index] = np.asarray(records[index]["base"] + context[offset].tolist(), dtype=np.float32)
            labels[index] = float(records[index]["iou"])
    permutation = np.asarray(ordered_indices, dtype=np.int64)
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    ordered_locations = [locations[index] for index in permutation]
    return features[permutation], labels[permutation], groups, ordered_locations


def frame_batch_loss(logits: torch.Tensor, ious: torch.Tensor, groups: list[tuple[int, int]], device: torch.device) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for start, stop in groups:
        frame_logits = logits[start:stop]
        frame_ious = ious[start:stop]
        positives = frame_ious >= 0.5
        if positives.any():
            target = torch.where(positives, torch.exp(8.0 * (frame_ious - 0.5)), torch.zeros_like(frame_ious))
            target = target / target.sum().clamp_min(1e-6)
            listwise = -(target * torch.log_softmax(frame_logits, dim=0)).sum()
            best_positive = frame_logits[positives].min()
            if (~positives).any():
                hardest_negative = frame_logits[~positives].max()
                pairwise = torch.nn.functional.softplus(-(best_positive - hardest_negative))
            else:
                pairwise = torch.zeros((), device=device)
            calibration = torch.nn.functional.binary_cross_entropy_with_logits(frame_logits, positives.float())
            losses.append(listwise + 0.5 * pairwise + 0.2 * calibration)
        else:
            losses.append(0.25 * torch.nn.functional.binary_cross_entropy_with_logits(frame_logits, torch.zeros_like(frame_logits)))
    return torch.stack(losses).mean()


def score_and_write(model: MLP, mean: np.ndarray, std: np.ndarray, input_path: Path, output_path: Path, score_field: str, batch_size: int, device: torch.device) -> dict[str, int]:
    features, _labels, _groups, locations = load_records([input_path], None)
    normalized = (features - mean) / std
    chunks: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            logits = model(torch.from_numpy(normalized[start:start + batch_size]).to(device, non_blocking=True))
            chunks.append(torch.sigmoid(logits).cpu())
    ordered_scores = torch.cat(chunks).numpy() if chunks else np.zeros((0,), dtype=np.float32)
    score_by_location = {location: float(score) for location, score in zip(locations, ordered_scores)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line_index = 0
    rows_scored = tracklets = 0
    with input_path.open("r", encoding="utf-8-sig") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            rows = [dict(row) for row in item.get("rows") or []]
            values: list[float] = []
            for row_index, row in enumerate(rows):
                score = score_by_location.get((input_path, line_index, row_index))
                if score is not None:
                    row[score_field] = score
                    values.append(score)
                    rows_scored += 1
            item["rows"] = rows
            meta = dict(item.get("meta") or {})
            meta[score_field] = float(np.mean(values)) if values else 0.0
            meta[f"{score_field}_max"] = float(np.max(values)) if values else 0.0
            item["meta"] = meta
            target.write(json.dumps(item, separators=(",", ":")) + "\n")
            line_index += 1
            tracklets += 1
    return {"tracklets": tracklets, "rows_scored": rows_scored}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a causal frame-listwise Action Bank candidate selector.")
    parser.add_argument("--train-tracklets", type=Path, required=True)
    parser.add_argument("--train-gt-csv", type=Path, required=True)
    parser.add_argument("--validation-tracklets", type=Path, required=True)
    parser.add_argument("--out-validation-tracklets", type=Path, required=True)
    parser.add_argument("--test-tracklets", type=Path, required=True)
    parser.add_argument("--out-test-tracklets", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--score-field", default="action_bank_listwise_score")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--frame-batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    gt = load_gt([args.train_gt_csv])
    features, ious, groups, _locations = load_records([args.train_tracklets], gt)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = (features - mean) / std
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(normalized.shape[1], hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = np.random.default_rng(2026)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        order = generator.permutation(len(groups))
        epoch_losses: list[float] = []
        model.train()
        for batch_start in range(0, len(order), args.frame_batch_size):
            selected = order[batch_start:batch_start + args.frame_batch_size]
            batch_features: list[np.ndarray] = []
            batch_ious: list[np.ndarray] = []
            batch_groups: list[tuple[int, int]] = []
            cursor = 0
            for group_index in selected:
                start, stop = groups[int(group_index)]
                batch_features.append(normalized[start:stop])
                batch_ious.append(ious[start:stop])
                batch_groups.append((cursor, cursor + stop - start))
                cursor += stop - start
            bx = torch.from_numpy(np.concatenate(batch_features)).to(device, non_blocking=True)
            by = torch.from_numpy(np.concatenate(batch_ious)).to(device, non_blocking=True)
            logits = model(bx)
            loss = frame_batch_loss(logits, by, batch_groups, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "epochs": args.epochs, "loss": float(np.mean(epoch_losses)), "device": str(device)}
        if device.type == "cuda":
            row["cuda_memory_allocated_mb"] = round(torch.cuda.memory_allocated(device) / 1048576, 3)
        history.append(row)
        print(json.dumps({"kind": "action_bank_listwise_train_progress", **row}), flush=True)

    validation = score_and_write(model, mean, std, args.validation_tracklets, args.out_validation_tracklets, args.score_field, args.inference_batch_size, device)
    test = score_and_write(model, mean, std, args.test_tracklets, args.out_test_tracklets, args.score_field, args.inference_batch_size, device)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "features": ALL_FEATURES, "hidden": args.hidden, "score_field": args.score_field}, args.out_model)
    summary = {
        "device": str(device),
        "train_rows": len(features),
        "train_frames": len(groups),
        "train_positive_rows": int((ious >= 0.5).sum()),
        "train_candidate_coverage_frames": int(sum(bool((ious[start:stop] >= 0.5).any()) for start, stop in groups)),
        "validation": validation,
        "test": test,
        "history": history,
        "model": str(args.out_model.resolve()),
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"kind": "action_bank_listwise_train_done", **summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


