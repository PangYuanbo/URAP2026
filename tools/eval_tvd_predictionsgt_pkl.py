from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def add_tvd_to_path(tvd_root: Path) -> None:
    sys.path.insert(0, str(tvd_root.resolve()))


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = torch.maximum(ax1, bx1)
    iy1 = torch.maximum(ay1, by1)
    ix2 = torch.minimum(ax2, bx2)
    iy2 = torch.minimum(ay2, by2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    return inter / (area_a + area_b - inter).clamp(min=1e-9)


def process_batch(detections: torch.Tensor, labels: torch.Tensor, iouv: torch.Tensor) -> torch.Tensor:
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool)
    if detections.numel() == 0 or labels.numel() == 0:
        return correct
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches_t = torch.tensor(matches, dtype=torch.float32)
        correct[matches_t[:, 1].long()] = matches_t[:, 2:3] >= iouv.cpu()
    return correct


def load_predictionsgt(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected dict, got {type(data)}")
    return data


def row_to_det(row: dict[str, Any]) -> list[float] | None:
    bbox = row.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2, float(row.get("score", 0.0)), float(row.get("category_id", 0))]


def row_to_label(row: dict[str, Any]) -> list[float] | None:
    bbox = row.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None
    return [float(row.get("category_id", 0)), x1, y1, x2, y2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute YOLOv5-style metrics from TransVisDrone predictionsgt pkl.")
    parser.add_argument("--tvd-root", type=Path, default=Path("papers/TransVisDrone"))
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    add_tvd_to_path(args.tvd_root)
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]
    from utils.metrics import ap_per_class  # type: ignore

    data = load_predictionsgt(args.predictionsgt_pkl.resolve())
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

    arrays = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(arrays) and arrays[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*arrays, plot=bool(args.plot), save_dir=args.out_json.parent, names={0: "drone"})
        ap50 = ap[:, 0]
        ap5095 = ap.mean(1)
        mp = float(p.mean())
        mr = float(r.mean())
        map50 = float(ap50.mean())
        map5095 = float(ap5095.mean())
        mf1 = float(f1.mean())
    else:
        mp = mr = map50 = map5095 = mf1 = 0.0
        ap_class = np.asarray([], dtype=np.int64)

    summary = {
        "predictionsgt_pkl": str(args.predictionsgt_pkl.resolve()),
        "images": images,
        "labels": labels_total,
        "detections": detections_total,
        "precision": mp,
        "recall": mr,
        "map50": map50,
        "map5095": map5095,
        "f1": mf1,
        "ap_class": [int(v) for v in np.asarray(ap_class).tolist()],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
