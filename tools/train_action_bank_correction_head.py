from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
for entry in (str(REPO), str(TOOLS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.train_action_bank_all_candidate_listwise import AUX_NAMES, finite, load_auxiliary, percentile
from tools.train_detection_row_score_head import MLP, iou_max

SIGNALS = [
    "action_bank_score",
    "action_bank_motion_probability",
    "action_bank_learned_score",
    "action_bank_reliability",
    "samurai_cmc_forward_iou",
    "samurai_cmc_score",
]
FEATURE_NAMES = [
    "candidate_count_log", "top_raw", "challenger_raw", "raw_gap", "raw_ratio",
    "challenger_raw_rank", "pair_iou", "center_distance", "log_area_ratio", "aspect_delta",
] + [part + "_" + name for name in AUX_NAMES for part in ("top", "challenger", "delta")] + ["source_" + name for name in SIGNALS] + ["source_raw_runner_up"]


def safe_logit(value: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, value))
    return math.log(clipped / (1.0 - clipped))


def pair_geometry(top_box, challenger_box, width, height):
    tx1, ty1, tx2, ty2 = [float(value) for value in top_box]
    cx1, cy1, cx2, cy2 = [float(value) for value in challenger_box]
    tw, th = max(1e-6, tx2 - tx1), max(1e-6, ty2 - ty1)
    cw, ch = max(1e-6, cx2 - cx1), max(1e-6, cy2 - cy1)
    inter = max(0.0, min(tx2, cx2) - max(tx1, cx1)) * max(0.0, min(ty2, cy2) - max(ty1, cy1))
    union = tw * th + cw * ch - inter
    pair_iou = inter / max(1e-6, union)
    center_distance = math.hypot((tx1 + tx2 - cx1 - cx2) * 0.5 / width, (ty1 + ty2 - cy1 - cy2) * 0.5 / height)
    return pair_iou, center_distance, math.log((cw * ch) / (tw * th)), abs(math.log((cw / ch) / (tw / th)))


def frame_pairs(image_id, item, auxiliary, sequence_sizes, with_labels):
    detections = list(item.get("detections") or [])
    if len(detections) < 2:
        return []
    sequence, frame_id, _ = image_key(str(image_id), 0)
    width, height = sequence_sizes.get(sequence, (1920.0, 1280.0))
    raw = np.asarray([finite(row.get("score")) for row in detections], dtype=np.float32)
    raw_rank = percentile(raw)
    payloads = [auxiliary.get((sequence, frame_id, index)) for index in range(len(detections))]
    matrix = np.asarray([payload if payload is not None else (0.0,) * len(AUX_NAMES) for payload in payloads], dtype=np.float32)
    top = int(raw.argmax())
    challenger_sources: dict[int, set[str]] = {}
    for name in SIGNALS:
        values = matrix[:, AUX_NAMES.index(name)]
        index = int(values.argmax())
        if index != top and values[index] > 0:
            challenger_sources.setdefault(index, set()).add(name)
    for index in np.argsort(raw)[::-1][1:4]:
        challenger_sources.setdefault(int(index), set()).add("raw_runner_up")
    gt = np.asarray([row.get("bbox") for row in item.get("labels", []) if isinstance(row.get("bbox"), list) and len(row.get("bbox")) == 4], dtype=np.float32)
    if gt.size == 0:
        gt = np.zeros((0, 4), dtype=np.float32)
    top_iou = iou_max(np.asarray(detections[top]["bbox"], dtype=np.float32), gt) if with_labels else 0.0
    rows = []
    for challenger, sources in challenger_sources.items():
        pair_iou, center_distance, log_area_ratio, aspect_delta = pair_geometry(detections[top]["bbox"], detections[challenger]["bbox"], width, height)
        feature = [
            math.log1p(len(detections)), raw[top], raw[challenger], raw[top] - raw[challenger],
            raw[challenger] / max(1e-6, raw[top]), raw_rank[challenger], pair_iou,
            center_distance, log_area_ratio, aspect_delta,
        ]
        for aux_index in range(len(AUX_NAMES)):
            feature.extend((matrix[top, aux_index], matrix[challenger, aux_index], matrix[challenger, aux_index] - matrix[top, aux_index]))
        feature.extend(float(name in sources) for name in SIGNALS)
        feature.append(float("raw_runner_up" in sources))
        challenger_iou = iou_max(np.asarray(detections[challenger]["bbox"], dtype=np.float32), gt) if with_labels else 0.0
        target = float(top_iou < 0.5 and challenger_iou >= 0.5)
        rows.append((feature, target, sequence, frame_id, challenger, top_iou, challenger_iou))
    return rows


def build_dataset(predictions, auxiliary, sequence_sizes, with_labels):
    features, targets, locations = [], [], []
    for image_id in sorted(predictions):
        for feature, target, sequence, frame_id, challenger, top_iou, challenger_iou in frame_pairs(image_id, predictions[image_id], auxiliary, sequence_sizes, with_labels):
            features.append(feature)
            targets.append(target)
            locations.append((sequence, frame_id, challenger))
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), locations


def predict(model, features, mean, std, batch_size, device):
    normalized = (features - mean) / std
    output = np.zeros(len(features), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(normalized[start:start + batch_size]).to(device)
            output[start:start + batch_size] = torch.sigmoid(model(batch)).cpu().numpy()
    return output


def write_scores(path, scores, locations, field):
    grouped = {}
    for (sequence, frame_id, candidate), score in zip(locations, scores):
        grouped.setdefault((sequence, frame_id), []).append({"seq": sequence, "frame_id": frame_id, "prediction_index": candidate, field: float(score)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for (sequence, frame_id), rows in sorted(grouped.items()):
            target.write(json.dumps({"meta": {"seq": sequence, "frame_id": frame_id}, "rows": rows}, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a causal Action Bank head that only corrects detector conflicts.")
    parser.add_argument("--train-pkl", type=Path, required=True)
    parser.add_argument("--train-aux", type=Path, required=True)
    parser.add_argument("--val-pkl", type=Path, required=True)
    parser.add_argument("--val-aux", type=Path, required=True)
    parser.add_argument("--test-pkl", type=Path, required=True)
    parser.add_argument("--test-aux", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--out-val-scores", type=Path, required=True)
    parser.add_argument("--out-test-scores", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--score-field", default="action_bank_correction_score")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_aux, train_sizes = load_auxiliary(args.train_aux)
    val_aux, val_sizes = load_auxiliary(args.val_aux)
    test_aux, test_sizes = load_auxiliary(args.test_aux)
    train_x, train_y, _ = build_dataset(load_predictionsgt(args.train_pkl), train_aux, train_sizes, True)
    val_x, _, val_locations = build_dataset(load_predictionsgt(args.val_pkl), val_aux, val_sizes, False)
    test_x, _, test_locations = build_dataset(load_predictionsgt(args.test_pkl), test_aux, test_sizes, False)
    if not len(train_x) or not train_y.sum():
        raise RuntimeError("correction dataset has no positive examples")
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0).clip(min=1e-4)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = MLP(train_x.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    positive_weight = float((len(train_y) - train_y.sum()) / train_y.sum())
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    normalized = (train_x - mean) / std
    generator = np.random.default_rng(17)
    history = []
    for epoch in range(1, args.epochs + 1):
        order = generator.permutation(len(train_x))
        total = 0.0
        model.train()
        for start in range(0, len(order), args.batch_size):
            selected = order[start:start + args.batch_size]
            batch_x = torch.from_numpy(normalized[selected]).to(device)
            batch_y = torch.from_numpy(train_y[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(selected)
        row = {"epoch": epoch, "epochs": args.epochs, "loss": total / len(train_x), "device": str(device)}
        if device.type == "cuda":
            row["cuda_memory_allocated_mb"] = round(torch.cuda.memory_allocated(device) / 1048576, 3)
        history.append(row)
        print(json.dumps({"kind": "correction_train_progress", **row}), flush=True)
    val_scores = predict(model, val_x, mean, std, args.batch_size, device)
    test_scores = predict(model, test_x, mean, std, args.batch_size, device)
    write_scores(args.out_val_scores, val_scores, val_locations, args.score_field)
    write_scores(args.out_test_scores, test_scores, test_locations, args.score_field)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "features": FEATURE_NAMES, "hidden": args.hidden, "score_field": args.score_field}, args.out_model)
    summary = {
        "device": str(device), "train_pairs": len(train_x), "train_positive_pairs": int(train_y.sum()),
        "positive_weight": positive_weight, "val_pairs": len(val_x), "test_pairs": len(test_x),
        "val_score_mean": float(val_scores.mean()), "test_score_mean": float(test_scores.mean()), "history": history,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"kind": "correction_train_done", **summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
