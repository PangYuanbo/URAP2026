from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def _iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


def _cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Box:
    return Box(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _clamp_box(b: Box, w: int, h: int) -> Box:
    return Box(
        x1=max(0.0, min(float(w), b.x1)),
        y1=max(0.0, min(float(h), b.y1)),
        x2=max(0.0, min(float(w), b.x2)),
        y2=max(0.0, min(float(h), b.y2)),
    )


def _load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[Box]:
    if not label_path.is_file():
        return []
    boxes: list[Box] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            # class cx cy w h (normalized)
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            bw = float(parts[3]) * img_w
            bh = float(parts[4]) * img_h
            boxes.append(_cxcywh_to_xyxy(cx, cy, bw, bh))
    return boxes


def _iter_result_files(results_dir: Path, per_flight_filename: str = "result.json") -> list[Path]:
    files: list[Path] = []
    for child in sorted(results_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        cand = child / per_flight_filename
        if cand.is_file():
            files.append(cand)
    return files


def _load_predictions(results_dir: Path, min_score: float) -> dict[str, list[tuple[float, Box]]]:
    """
    Return mapping:
      img_name -> list[(score, pred_box_xyxy)]
    Input format is AOT-style JSON produced by AirbornePredictor.
    """
    pred_by_img: dict[str, list[tuple[float, Box]]] = {}

    per_flight = _iter_result_files(results_dir)
    if not per_flight:
        raise SystemExit(f"No per-flight result.json found under: {results_dir}")

    for p in per_flight:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise TypeError(f"{p}: expected list, got {type(data)}")
        for item in data:
            if not isinstance(item, dict) or "img_name" not in item or "detections" not in item:
                continue
            img_name = str(item["img_name"])
            dets = item.get("detections") or []
            if not isinstance(dets, list):
                continue
            for det in dets:
                if not isinstance(det, dict):
                    continue
                s = float(det.get("s", 0.0))
                if s < min_score:
                    continue
                cx = float(det.get("x", 0.0))
                cy = float(det.get("y", 0.0))
                w = float(det.get("w", 0.0))
                h = float(det.get("h", 0.0))
                b = _cxcywh_to_xyxy(cx, cy, w, h)
                pred_by_img.setdefault(img_name, []).append((s, b))

    return pred_by_img


def _compute_ap(tp: list[int], fp: list[int], n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    import numpy as np

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    rec = tp_cum / max(1, n_gt)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    # Precision envelope
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Integrate area under PR curve
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="NPS frames directory (e.g., .../AllFrames/val)")
    ap.add_argument("--labels-dir", required=True, help="NPS YOLO labels directory (e.g., .../NPSvisdroneStyle/val/labels)")
    ap.add_argument("--results-dir", required=True, help="Winner results folder containing per-clip subfolders")
    ap.add_argument("--img-w", type=int, default=1280, help="Image width (NPS val default: 1280)")
    ap.add_argument("--img-h", type=int, default=960, help="Image height (NPS val default: 960)")
    ap.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold (default: 0.5)")
    ap.add_argument("--min-score", type=float, default=0.0, help="Filter predictions by score >= min-score")
    ap.add_argument("--out-json", default=None, help="Write metrics summary JSON")
    args = ap.parse_args()

    images_dir = Path(args.images_dir).resolve()
    labels_dir = Path(args.labels_dir).resolve()
    results_dir = Path(args.results_dir).resolve()

    if not images_dir.is_dir():
        raise SystemExit(f"--images-dir not found: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"--labels-dir not found: {labels_dir}")
    if not results_dir.is_dir():
        raise SystemExit(f"--results-dir not found: {results_dir}")

    img_w = int(args.img_w)
    img_h = int(args.img_h)

    # Ground truth
    img_names = sorted([p.name for p in images_dir.iterdir() if p.is_file() and p.name.lower().endswith(".png")])
    gt_by_img: dict[str, list[Box]] = {}
    n_gt = 0
    for name in img_names:
        lab = labels_dir / (os.path.splitext(name)[0] + ".txt")
        gts = _load_yolo_labels(lab, img_w=img_w, img_h=img_h)
        gt_by_img[name] = gts
        n_gt += len(gts)

    # Predictions
    pred_by_img = _load_predictions(results_dir, min_score=float(args.min_score))
    preds_flat: list[tuple[str, float, Box]] = []
    for img_name, dets in pred_by_img.items():
        for s, b in dets:
            preds_flat.append((img_name, s, _clamp_box(b, w=img_w, h=img_h)))
    preds_flat.sort(key=lambda t: t[1], reverse=True)

    # Match predictions to GT (single class) for AP@IoU
    matched: dict[str, list[bool]] = {
        name: [False] * len(gts) for name, gts in gt_by_img.items()
    }

    tp: list[int] = []
    fp: list[int] = []

    for img_name, score, pred_box in preds_flat:
        gts = gt_by_img.get(img_name, [])
        if not gts:
            tp.append(0)
            fp.append(1)
            continue

        best_iou = -1.0
        best_idx = -1
        for i, gt_box in enumerate(gts):
            if matched[img_name][i]:
                continue
            v = _iou(pred_box, gt_box)
            if v > best_iou:
                best_iou = v
                best_idx = i

        if best_iou >= float(args.iou_thr) and best_idx >= 0:
            matched[img_name][best_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    ap50 = _compute_ap(tp, fp, n_gt=n_gt)

    # A simple operating point at the current min-score filter:
    tp_sum = sum(tp)
    fp_sum = sum(fp)
    fn_sum = n_gt - sum(sum(1 for m in ms if m) for ms in matched.values())
    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    recall = tp_sum / n_gt if n_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    summary: dict[str, Any] = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "results_dir": str(results_dir),
        "num_images": len(img_names),
        "num_gt_boxes": n_gt,
        "num_predictions": len(preds_flat),
        "iou_thr": float(args.iou_thr),
        "min_score": float(args.min_score),
        "AP@IoU": ap50,
        "precision_at_min_score": precision,
        "recall_at_min_score": recall,
        "f1_at_min_score": f1,
    }

    print(json.dumps(summary, indent=2))

    if args.out_json:
        outp = Path(args.out_json).resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote: {outp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

