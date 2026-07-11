from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def valid_box(row: dict[str, Any]) -> np.ndarray | None:
    bbox = row.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    values = np.asarray(bbox, dtype=np.float32)
    if not np.isfinite(values).all() or values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def best_iou(label: np.ndarray, detections: list[np.ndarray]) -> float:
    if not detections:
        return 0.0
    boxes = np.stack(detections)
    intersection_x1 = np.maximum(label[0], boxes[:, 0])
    intersection_y1 = np.maximum(label[1], boxes[:, 1])
    intersection_x2 = np.minimum(label[2], boxes[:, 2])
    intersection_y2 = np.minimum(label[3], boxes[:, 3])
    intersection = np.maximum(0.0, intersection_x2 - intersection_x1) * np.maximum(0.0, intersection_y2 - intersection_y1)
    label_area = (label[2] - label[0]) * (label[3] - label[1])
    detection_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = np.maximum(1e-9, label_area + detection_area - intersection)
    return float(np.max(intersection / union))


def summarize(name: str, source: Path) -> dict[str, Any]:
    with source.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{source}: expected dict, got {type(data)}")

    labels = 0
    detections = 0
    frames_with_labels = 0
    frames_without_detections = 0
    covered_iou50 = 0
    best_ious: list[float] = []
    scores: list[float] = []
    detection_max_x = 0.0
    detection_max_y = 0.0
    label_max_x = 0.0
    label_max_y = 0.0

    for item in data.values():
        detection_rows = item.get("detections", [])
        label_rows = item.get("labels", [])
        detection_boxes = [box for row in detection_rows if (box := valid_box(row)) is not None]
        label_boxes = [box for row in label_rows if (box := valid_box(row)) is not None]
        detections += len(detection_boxes)
        labels += len(label_boxes)
        frames_with_labels += int(bool(label_boxes))
        frames_without_detections += int(not detection_boxes)
        scores.extend(float(row.get("score", 0.0)) for row in detection_rows if valid_box(row) is not None)
        if detection_boxes:
            stacked = np.stack(detection_boxes)
            detection_max_x = max(detection_max_x, float(stacked[:, [0, 2]].max()))
            detection_max_y = max(detection_max_y, float(stacked[:, [1, 3]].max()))
        if label_boxes:
            stacked = np.stack(label_boxes)
            label_max_x = max(label_max_x, float(stacked[:, [0, 2]].max()))
            label_max_y = max(label_max_y, float(stacked[:, [1, 3]].max()))
        for label_box in label_boxes:
            overlap = best_iou(label_box, detection_boxes)
            best_ious.append(overlap)
            covered_iou50 += int(overlap >= 0.5)

    score_array = np.asarray(scores, dtype=np.float32)
    iou_array = np.asarray(best_ious, dtype=np.float32)
    images = len(data)
    result = {
        "name": name,
        "source": str(source.resolve()),
        "images": images,
        "labels": labels,
        "detections": detections,
        "detections_per_image": detections / max(1, images),
        "frames_with_labels": frames_with_labels,
        "frames_without_detections": frames_without_detections,
        "covered_labels_iou50": covered_iou50,
        "candidate_coverage_iou50": covered_iou50 / max(1, labels),
        "best_iou_mean": float(iou_array.mean()) if iou_array.size else None,
        "score_mean": float(score_array.mean()) if score_array.size else None,
        "score_p50": float(np.quantile(score_array, 0.5)) if score_array.size else None,
        "score_p90": float(np.quantile(score_array, 0.9)) if score_array.size else None,
        "score_p99": float(np.quantile(score_array, 0.99)) if score_array.size else None,
        "detection_max_xy": [detection_max_x, detection_max_y],
        "label_max_xy": [label_max_x, label_max_y],
    }
    print(json.dumps(result), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", nargs=2, metavar=("NAME", "PKL"), required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    rows = [summarize(name, Path(source)) for name, source in args.source]
    by_name = {row["name"]: row for row in rows}
    dense = by_name.get("dense_train")
    test = by_name.get("test")
    comparison = None
    if dense and test:
        comparison = {
            "dense_train_minus_test_detections_per_image": dense["detections_per_image"] - test["detections_per_image"],
            "density_ratio_dense_train_to_test": dense["detections_per_image"] / max(1e-9, test["detections_per_image"]),
            "coverage_delta_dense_train_vs_test": dense["candidate_coverage_iou50"] - test["candidate_coverage_iou50"],
        }
    payload = {"rows": rows, "comparison": comparison}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
