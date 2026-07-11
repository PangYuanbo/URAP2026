from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _remap_path(path: Path) -> Path:
    if path.exists():
        return path
    text = str(path)
    for old, new in (("D:\\URAP_datasets\\", "U:\\URAP_datasets\\"), ("D:/URAP_datasets/", "U:/URAP_datasets/")):
        if text.startswith(old):
            candidate = Path(new + text[len(old) :])
            if candidate.exists():
                return candidate
    return path


def _read_images(path: Path, limit: int = 0) -> list[Path]:
    images = [_remap_path(Path(line.strip())) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return images[:limit] if limit else images


def _image_size(path: Path, width: int | None, height: int | None) -> tuple[int, int]:
    if width is not None and height is not None:
        return int(width), int(height)
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return int(img.shape[1]), int(img.shape[0])


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() in {"images", "images2"}:
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def _xywhn_to_xyxy(cx: float, cy: float, bw: float, bh: float, width: int, height: int) -> tuple[float, float, float, float]:
    x = cx * width
    y = cy * height
    w = bw * width
    h = bh * height
    return (
        max(0.0, x - w / 2.0),
        max(0.0, y - h / 2.0),
        min(float(width), x + w / 2.0),
        min(float(height), y + h / 2.0),
    )


def _read_yolo_labels(path: Path, width: int, height: int, conf_threshold: float | None) -> np.ndarray:
    rows: list[list[float]] = []
    if not path.exists():
        return np.zeros((0, 6), dtype=np.float32)
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        conf = float(parts[5]) if len(parts) >= 6 else 1.0
        if conf_threshold is not None and conf < conf_threshold:
            continue
        cls = float(parts[0])
        box = _xywhn_to_xyxy(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), width, height)
        rows.append([*box, conf, cls])
    rows.sort(key=lambda row: row[4], reverse=True)
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 6), dtype=np.float32)


def _box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _match_counts(pred: np.ndarray, gt: np.ndarray, iou_threshold: float) -> tuple[int, int, int]:
    if pred.shape[0] == 0:
        return 0, 0, int(gt.shape[0])
    if gt.shape[0] == 0:
        return 0, int(pred.shape[0]), 0
    iou = _box_iou_matrix(gt[:, :4], pred[:, :4])
    class_match = gt[:, 5:6] == pred[:, 5][None, :]
    candidates = np.argwhere((iou >= iou_threshold) & class_match)
    matches = [(int(label_i), int(pred_i), float(iou[label_i, pred_i])) for label_i, pred_i in candidates]
    matches.sort(key=lambda item: item[2], reverse=True)
    used_labels: set[int] = set()
    used_preds: set[int] = set()
    for label_i, pred_i, _ in matches:
        if label_i in used_labels or pred_i in used_preds:
            continue
        used_labels.add(label_i)
        used_preds.add(pred_i)
    tp = len(used_preds)
    fp = int(pred.shape[0]) - tp
    fn = int(gt.shape[0]) - len(used_labels)
    return tp, fp, fn


def _metrics(tp: int, fp: int, fn: int, frames: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fppi": fp / max(1, frames),
    }


def _parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--method must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("method name cannot be empty")
    return name, Path(raw_path.strip())


def _parse_thresholds(values: list[str] | None) -> list[float]:
    if not values:
        return [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    out: list[float] = []
    for value in values:
        out.extend(float(part) for part in value.replace(",", " ").split() if part.strip())
    return sorted(set(out))


def _select_baseline(rows: list[dict[str, Any]], baseline_method: str, baseline_threshold: float | None, primary_metric: str) -> dict[str, Any]:
    baseline_rows = [row for row in rows if row["method"] == baseline_method]
    if not baseline_rows:
        raise ValueError(f"baseline method not found in sweep rows: {baseline_method}")
    if baseline_threshold is not None:
        return min(baseline_rows, key=lambda row: abs(float(row["threshold"]) - float(baseline_threshold)))
    return max(baseline_rows, key=lambda row: (float(row[primary_metric]), float(row["recall"]), float(row["precision"]), -int(row["fp"])))


def _select_under_fp(rows: list[dict[str, Any]], fp_budget: int) -> dict[str, Any]:
    feasible = [row for row in rows if int(row["fp"]) <= fp_budget]
    pool = feasible or rows
    selected = max(pool, key=lambda row: (float(row["recall"]), float(row["f1"]), float(row["precision"]), -int(row["fp"])))
    selected = dict(selected)
    if not feasible:
        selected["exceeds_fp_budget"] = True
    return selected


def evaluate_methods(
    images: list[Path],
    methods: list[tuple[str, Path]],
    thresholds: list[float],
    iou_threshold: float,
    image_width: int | None = None,
    image_height: int | None = None,
    progress_every: int = 2000,
) -> list[dict[str, Any]]:
    gt_by_image: dict[Path, np.ndarray] = {}
    pred_by_method_image: dict[tuple[str, Path], np.ndarray] = {}
    size_by_image: dict[Path, tuple[int, int]] = {}
    for index, image in enumerate(images, start=1):
        width, height = _image_size(image, image_width, image_height)
        size_by_image[image] = (width, height)
        gt_by_image[image] = _read_yolo_labels(_label_path(image), width, height, conf_threshold=None)
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == len(images)):
            print(json.dumps({"kind": "matched_fp_progress", "stage": "cache_gt", "done": index, "total": len(images)}), flush=True)
    for method_name, label_dir in methods:
        for index, image in enumerate(images, start=1):
            width, height = size_by_image[image]
            pred_by_method_image[(method_name, image)] = _read_yolo_labels(label_dir / f"{image.stem}.txt", width, height, conf_threshold=None)
            if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == len(images)):
                print(json.dumps({"kind": "matched_fp_progress", "stage": "cache_pred", "method": method_name, "done": index, "total": len(images)}), flush=True)

    rows: list[dict[str, Any]] = []
    sweep_total = len(methods) * len(thresholds)
    sweep_done = 0
    for method_name, _ in methods:
        for threshold in thresholds:
            tp = fp = fn = detections = labels = 0
            for image in images:
                gt = gt_by_image[image]
                raw_pred = pred_by_method_image[(method_name, image)]
                pred = raw_pred[raw_pred[:, 4] >= threshold] if raw_pred.shape[0] else raw_pred
                frame_tp, frame_fp, frame_fn = _match_counts(pred, gt, iou_threshold)
                tp += frame_tp
                fp += frame_fp
                fn += frame_fn
                detections += int(pred.shape[0])
                labels += int(gt.shape[0])
            row = {
                "method": method_name,
                "threshold": float(threshold),
                "frames": len(images),
                "labels": labels,
                "detections": detections,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
            row.update(_metrics(tp, fp, fn, len(images)))
            rows.append(row)
            sweep_done += 1
            print(
                json.dumps(
                    {
                        "kind": "matched_fp_progress",
                        "stage": "sweep",
                        "method": method_name,
                        "threshold": threshold,
                        "done": sweep_done,
                        "total": sweep_total,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "recall": row["recall"],
                    }
                ),
                flush=True,
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare YOLO prediction-label methods at matched false positives.")
    parser.add_argument("--images-list", type=Path, required=True)
    parser.add_argument("--method", action="append", type=_parse_method, required=True, help="NAME=prediction_label_dir. Repeat for baseline and candidate methods.")
    parser.add_argument("--baseline-method", required=True)
    parser.add_argument("--baseline-threshold", type=float)
    parser.add_argument("--fp-budget", type=int, help="Override baseline-derived FP budget.")
    parser.add_argument("--thresholds", nargs="*", default=None)
    parser.add_argument("--primary-metric", choices=["precision", "recall", "f1"], default="f1")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=2000)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    if (args.image_width is None) != (args.image_height is None):
        raise ValueError("--image-width and --image-height must be provided together")
    thresholds = _parse_thresholds(args.thresholds)
    images = _read_images(args.images_list, args.max_frames)
    rows = evaluate_methods(
        images,
        args.method,
        thresholds,
        iou_threshold=args.match_iou,
        image_width=args.image_width,
        image_height=args.image_height,
        progress_every=args.progress_every,
    )
    baseline = _select_baseline(rows, args.baseline_method, args.baseline_threshold, args.primary_metric)
    fp_budget = int(args.fp_budget if args.fp_budget is not None else baseline["fp"])

    selected = []
    for method_name, _ in args.method:
        chosen = _select_under_fp([row for row in rows if row["method"] == method_name], fp_budget)
        chosen["delta_recall_vs_baseline"] = float(chosen["recall"]) - float(baseline["recall"])
        chosen["delta_precision_vs_baseline"] = float(chosen["precision"]) - float(baseline["precision"])
        chosen["delta_fp_vs_budget"] = int(chosen["fp"]) - fp_budget
        selected.append(chosen)

    summary = {
        "images_list": str(args.images_list),
        "methods": [{"name": name, "label_dir": str(path)} for name, path in args.method],
        "baseline_method": args.baseline_method,
        "baseline": baseline,
        "fp_budget": fp_budget,
        "thresholds": thresholds,
        "match_iou": args.match_iou,
        "selected_under_fp_budget": sorted(selected, key=lambda row: (float(row["recall"]), float(row["f1"]), float(row["precision"])), reverse=True),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["method", "threshold", "frames", "labels", "detections", "tp", "fp", "fn", "precision", "recall", "f1", "fppi"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
