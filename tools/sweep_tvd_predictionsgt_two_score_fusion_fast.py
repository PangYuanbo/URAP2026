from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval_tvd_predictionsgt_pkl import load_predictionsgt, process_batch, row_to_det, row_to_label
from rescore_li_tetc_diagnostics_from_tracklets import load_tracklet_scores
from sweep_tvd_predictionsgt_action_rescore import image_key, parse_csv_floats
from sweep_tvd_predictionsgt_score_fusion import load_row_scores
from sweep_tvd_predictionsgt_two_score_fusion import clone_with_scores


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def fuse(raw: np.ndarray, meta: np.ndarray, row: np.ndarray, valid: np.ndarray, mode: str, alpha: float, beta: float) -> np.ndarray:
    output = raw.copy()
    if not valid.any():
        return output
    r = raw[valid]
    m = meta[valid]
    w = row[valid]
    if mode == "logit-3mix":
        raw_weight = max(0.0, 1.0 - alpha - beta)
        values = sigmoid(raw_weight * logit(r) + alpha * logit(m) + beta * logit(w))
    elif mode == "meta-logit-row-geom":
        base = sigmoid((1.0 - alpha) * logit(r) + alpha * logit(m))
        values = np.exp((1.0 - beta) * np.log(np.maximum(base, 1e-9)) + beta * np.log(np.maximum(w, 1e-9)))
    elif mode == "meta-logit-row-suppress":
        base = sigmoid((1.0 - alpha) * logit(r) + alpha * logit(m))
        values = base * (1.0 - beta * (1.0 - w))
    elif mode == "meta-logit-row-boost":
        base = sigmoid((1.0 - alpha) * logit(r) + alpha * logit(m))
        values = base + beta * w * (1.0 - base)
    else:
        raise ValueError(mode)
    output[valid] = np.clip(values, 0.0, 1.0)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tvd-root", type=Path, required=True)
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--meta-tracklet-jsonl", type=Path, required=True)
    parser.add_argument("--meta-score-field", required=True)
    parser.add_argument("--row-tracklet-jsonl", type=Path, required=True)
    parser.add_argument("--row-score-field", required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--betas", nargs="+", type=float, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--write-best-pkl", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.tvd_root.resolve()))
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid
    from utils.metrics import ap_per_class

    data = load_predictionsgt(args.predictionsgt_pkl.resolve())
    meta_scores, meta_summary = load_tracklet_scores(args.meta_tracklet_jsonl.resolve(), args.meta_score_field, 1)
    row_scores, row_summary = load_row_scores(args.row_tracklet_jsonl.resolve(), args.row_score_field, 1)
    iouv = torch.linspace(0.5, 0.95, 10)
    correct_parts: list[np.ndarray] = []
    raw_values: list[float] = []
    meta_values: list[float] = []
    row_values: list[float] = []
    valid_values: list[bool] = []
    pred_classes: list[float] = []
    target_classes: list[float] = []
    labels_total = 0
    detections_total = 0
    for image_id in sorted(data):
        item = data[image_id]
        det_rows: list[list[float]] = []
        det_indices: list[int] = []
        for pred_index, record in enumerate(item.get("detections", [])):
            converted = row_to_det(record)
            if converted is not None:
                det_rows.append(converted)
                det_indices.append(pred_index)
        label_rows = [value for record in item.get("labels", []) if (value := row_to_label(record)) is not None]
        detections_total += len(det_rows)
        labels_total += len(label_rows)
        detections = torch.tensor(det_rows, dtype=torch.float32) if det_rows else torch.zeros((0, 6), dtype=torch.float32)
        labels = torch.tensor(label_rows, dtype=torch.float32) if label_rows else torch.zeros((0, 5), dtype=torch.float32)
        correct_parts.append(process_batch(detections, labels, iouv).numpy())
        target_classes.extend(labels[:, 0].tolist() if labels.numel() else [])
        for det_index, det in zip(det_indices, det_rows):
            key = image_key(str(image_id), det_index)
            meta_score = meta_scores.get(key)
            row_score = row_scores.get(key)
            raw_values.append(float(det[4]))
            meta_values.append(float(meta_score) if meta_score is not None else 0.0)
            row_values.append(float(row_score) if row_score is not None else 0.0)
            valid_values.append(meta_score is not None and row_score is not None)
            pred_classes.append(float(det[5]))
    correct = np.concatenate(correct_parts, axis=0)
    raw = np.asarray(raw_values, dtype=np.float64)
    meta = np.asarray(meta_values, dtype=np.float64)
    row = np.asarray(row_values, dtype=np.float64)
    valid = np.asarray(valid_values, dtype=bool)
    pred_cls = np.asarray(pred_classes, dtype=np.float32)
    target_cls = np.asarray(target_classes, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    total = len(args.modes) * len(args.alphas) * len(args.betas)
    completed = 0
    for mode in args.modes:
        for alpha in args.alphas:
            for beta in args.betas:
                confidence = fuse(raw, meta, row, valid, mode, alpha, beta)
                precision, recall, ap, f1, _ = ap_per_class(correct, confidence, pred_cls, target_cls, plot=False, save_dir=args.out_json.parent, names={0: "drone"})
                result = {"mode": mode, "alpha": alpha, "beta": beta, "missing_score_behavior": "keep", "images": len(data), "labels": labels_total, "detections": detections_total, "precision": float(precision.mean()), "recall": float(recall.mean()), "map50": float(ap[:, 0].mean()), "map5095": float(ap.mean(1).mean()), "f1": float(f1.mean())}
                rows.append(result)
                if best is None or result["map50"] > best["map50"]:
                    best = result
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(json.dumps({"kind": "fast_fusion_progress", "done": completed, "total": total, "best_map50": best["map50"] if best else None}), flush=True)
    summary = {"predictionsgt_pkl": str(args.predictionsgt_pkl.resolve()), "meta_summary": meta_summary, "row_summary": row_summary, "best": best, "top": sorted(rows, key=lambda value: (-value["map50"], -value["recall"]))[:20], "rows": rows, "fast_precomputed_iou": True}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.write_best_pkl and best:
        best_data = clone_with_scores(data, meta_scores, row_scores, best["mode"], best["alpha"], best["beta"], "keep")
        with args.write_best_pkl.open("wb") as file:
            pickle.dump(best_data, file)
    print(json.dumps({"kind": "fast_fusion_done", "best": best}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

