from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval_tvd_predictionsgt_pkl import load_predictionsgt, process_batch, row_to_det, row_to_label
from sweep_tvd_predictionsgt_action_rescore import image_key
from sweep_tvd_predictionsgt_score_fusion import load_tracklet_scores
from sweep_tvd_predictionsgt_two_score_fusion_fast import fuse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tvd-root", type=Path, required=True)
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--score-map-pkl", type=Path, required=True)
    parser.add_argument("--valid-tracklet-jsonl", type=Path)
    parser.add_argument("--valid-score-field")
    parser.add_argument("--fields", nargs="+")
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--betas", nargs="+", type=float, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tvd_root.resolve()))
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid
    from utils.metrics import ap_per_class

    data = load_predictionsgt(args.predictionsgt_pkl.resolve())
    with args.score_map_pkl.open("rb") as handle:
        score_payload = pickle.load(handle)
    all_fields = list(score_payload["fields"])
    selected_fields = args.fields or all_fields
    field_indices = [all_fields.index(field) for field in selected_fields]
    score_map = score_payload["scores"]
    valid_score_keys = None
    if args.valid_tracklet_jsonl:
        if not args.valid_score_field:
            parser.error("--valid-score-field is required with --valid-tracklet-jsonl")
        valid_scores, _valid_summary = load_tracklet_scores(args.valid_tracklet_jsonl.resolve(), args.valid_score_field, 1)
        valid_score_keys = set(valid_scores)
    iouv = torch.linspace(0.5, 0.95, 10)
    correct_parts: list[np.ndarray] = []
    raw_values: list[float] = []
    row_matrix: list[list[float]] = []
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
            values = score_map.get(key)
            raw_values.append(float(det[4]))
            row_matrix.append([float(values[index]) for index in field_indices] if values is not None else [0.0] * len(field_indices))
            valid_values.append(values is not None and (valid_score_keys is None or image_key(str(image_id), det_index) in valid_score_keys))
            pred_classes.append(float(det[5]))
    correct = np.concatenate(correct_parts, axis=0)
    raw = np.asarray(raw_values, dtype=np.float64)
    rows_np = np.asarray(row_matrix, dtype=np.float64)
    valid = np.asarray(valid_values, dtype=bool)
    pred_cls = np.asarray(pred_classes, dtype=np.float32)
    target_cls = np.asarray(target_classes, dtype=np.float32)
    zero_meta = np.full_like(raw, 0.5)
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    total = len(selected_fields) * len(args.modes) * len(args.alphas) * len(args.betas)
    completed = 0
    for field_index, field in enumerate(selected_fields):
        row = rows_np[:, field_index]
        for mode in args.modes:
            for alpha in args.alphas:
                for beta in args.betas:
                    confidence = fuse(raw, zero_meta, row, valid, mode, alpha, beta)
                    precision, recall, ap, f1, _ = ap_per_class(correct, confidence, pred_cls, target_cls, plot=False, save_dir=args.out_json.parent, names={0: "drone"})
                    result = {"field": field, "mode": mode, "alpha": alpha, "beta": beta, "images": len(data), "labels": labels_total, "detections": detections_total, "precision": float(precision.mean()), "recall": float(recall.mean()), "map50": float(ap[:, 0].mean()), "map5095": float(ap.mean(1).mean()), "f1": float(f1.mean())}
                    results.append(result)
                    if best is None or result["map50"] > best["map50"]:
                        best = result
                    completed += 1
                    if completed % 20 == 0 or completed == total:
                        print(json.dumps({"kind": "multi_score_progress", "done": completed, "total": total, "best": best}), flush=True)
    summary = {"score_map_pkl": str(args.score_map_pkl), "fields": selected_fields, "best": best, "top": sorted(results, key=lambda value: (-value["map50"], -value["recall"]))[:40], "rows": results, "fast_precomputed_iou": True}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"kind": "multi_score_done", "best": best}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
