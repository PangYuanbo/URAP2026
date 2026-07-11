from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_tvd_predictionsgt_pkl import process_batch, row_to_det, row_to_label


def _load_predictionsgt(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected dict, got {type(data)}")
    return data


def _filtered_predictionsgt(data: dict[str, Any], score_threshold: float, top_k: int) -> dict[str, Any]:
    out = deepcopy(data)
    for item in out.values():
        detections = item.get("detections", [])
        detections = [
            row for row in detections
            if isinstance(row, dict) and float(row.get("score", 0.0)) >= score_threshold
        ]
        detections.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        if top_k > 0:
            detections = detections[:top_k]
        item["detections"] = detections
    return out


def _evaluate_predictionsgt(data: dict[str, Any], tvd_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(tvd_root.resolve()))
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]
    from utils.metrics import ap_per_class  # type: ignore

    iouv = torch.linspace(0.5, 0.95, 10)
    stats = []
    images = 0
    labels_total = 0
    detections_total = 0
    for image_id in sorted(data):
        images += 1
        item = data[image_id]
        det_rows = [row_to_det(row) for row in item.get("detections", [])]
        label_rows = [row_to_label(row) for row in item.get("labels", [])]
        det_rows = [row for row in det_rows if row is not None]
        label_rows = [row for row in label_rows if row is not None]
        detections_total += len(det_rows)
        labels_total += len(label_rows)
        det = torch.tensor(det_rows, dtype=torch.float32) if det_rows else torch.zeros((0, 6), dtype=torch.float32)
        labels = torch.tensor(label_rows, dtype=torch.float32) if label_rows else torch.zeros((0, 5), dtype=torch.float32)
        correct = process_batch(det, labels, iouv)
        tcls = labels[:, 0].tolist() if labels.numel() else []
        stats.append((correct.numpy(), det[:, 4].numpy(), det[:, 5].numpy(), np.asarray(tcls)))

    arrays = [np.concatenate(x, 0) for x in zip(*stats)] if stats else []
    if len(arrays) and arrays[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*arrays, plot=False, save_dir=Path("."), names={0: "drone"})
        precision = float(p.mean())
        recall = float(r.mean())
        map50 = float(ap[:, 0].mean())
        map5095 = float(ap.mean(1).mean())
        f1_value = float(f1.mean())
    else:
        precision = recall = map50 = map5095 = f1_value = 0.0
        ap_class = np.asarray([], dtype=np.int64)

    return {
        "images": images,
        "labels": labels_total,
        "detections": detections_total,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map5095": map5095,
        "f1": f1_value,
        "ap_class": [int(v) for v in np.asarray(ap_class).tolist()],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep score/top-k filters for a TransVisDrone predictionsgt pkl.")
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--tvd-root", type=Path, default=Path("papers/TransVisDrone"))
    parser.add_argument("--score-thresholds", nargs="+", type=float, default=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--primary-metric", choices=["precision", "recall", "map50", "map5095", "f1"], default="map50")
    args = parser.parse_args()

    data = _load_predictionsgt(args.predictionsgt_pkl)
    rows = []
    for score_threshold in args.score_thresholds:
        for top_k in args.top_ks:
            filtered = _filtered_predictionsgt(data, score_threshold=score_threshold, top_k=top_k)
            metrics = _evaluate_predictionsgt(filtered, args.tvd_root)
            rows.append({
                "score_threshold": float(score_threshold),
                "top_k": int(top_k),
                **metrics,
            })

    best = max(rows, key=lambda row: (float(row[args.primary_metric]), float(row["recall"]), float(row["precision"]))) if rows else {}
    summary = {
        "predictionsgt_pkl": str(args.predictionsgt_pkl.resolve()),
        "primary_metric": args.primary_metric,
        "best": best,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["score_threshold", "top_k", "images", "labels", "detections", "precision", "recall", "map50", "map5095", "f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
