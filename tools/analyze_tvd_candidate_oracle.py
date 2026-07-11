from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def box_iou(left: list[float], right: list[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, left)
    bx1, by1, bx2, by2 = map(float, right)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_left = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_right = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(1e-9, area_left + area_right - intersection)


def valid_bbox(row: dict[str, Any]) -> list[float] | None:
    bbox = row.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    values = [float(value) for value in bbox]
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def greedy_matches(detections: list[dict[str, Any]], labels: list[dict[str, Any]], threshold: float) -> list[tuple[int, int, float]]:
    pairs: list[tuple[float, int, int]] = []
    for label_index, label in enumerate(labels):
        label_box = valid_bbox(label)
        if label_box is None:
            continue
        label_class = int(label.get("category_id", 0))
        for detection_index, detection in enumerate(detections):
            detection_box = valid_bbox(detection)
            if detection_box is None or int(detection.get("category_id", 0)) != label_class:
                continue
            overlap = box_iou(label_box, detection_box)
            if overlap >= threshold:
                pairs.append((overlap, label_index, detection_index))
    pairs.sort(reverse=True)
    used_labels: set[int] = set()
    used_detections: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for overlap, label_index, detection_index in pairs:
        if label_index in used_labels or detection_index in used_detections:
            continue
        used_labels.add(label_index)
        used_detections.add(detection_index)
        matches.append((label_index, detection_index, overlap))
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the ranking/localization oracle of a TransVisDrone predictionsgt candidate set.")
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-oracle-pkl", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    with args.predictionsgt_pkl.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"expected dict, got {type(data)}")

    thresholds = np.linspace(0.5, 0.95, 10)
    covered_by_threshold = np.zeros(len(thresholds), dtype=np.int64)
    labels_total = detections_total = images_with_labels = 0
    frames_all_covered = frames_partially_covered = frames_no_candidate = 0
    selected_detection_scores: list[float] = []
    selected_detection_ious: list[float] = []
    oracle = copy.deepcopy(data)

    for image_id, item in data.items():
        detections = [row for row in item.get("detections", []) if valid_bbox(row) is not None]
        labels = [row for row in item.get("labels", []) if valid_bbox(row) is not None]
        detections_total += len(detections)
        labels_total += len(labels)
        images_with_labels += int(bool(labels))
        per_label_best: list[float] = []
        for label in labels:
            label_box = valid_bbox(label)
            label_class = int(label.get("category_id", 0))
            best = max(
                (
                    box_iou(label_box, valid_bbox(detection))
                    for detection in detections
                    if int(detection.get("category_id", 0)) == label_class
                ),
                default=0.0,
            )
            per_label_best.append(best)
            covered_by_threshold += np.asarray(best >= thresholds, dtype=np.int64)
        covered_at_05 = sum(value >= args.iou_threshold for value in per_label_best)
        if labels:
            if covered_at_05 == len(labels):
                frames_all_covered += 1
            elif covered_at_05:
                frames_partially_covered += 1
            else:
                frames_no_candidate += 1

        matches = greedy_matches(detections, labels, args.iou_threshold)
        selected_indices = {detection_index for _, detection_index, _ in matches}
        output_detections: list[dict[str, Any]] = []
        for label_index, detection_index, overlap in matches:
            row = copy.deepcopy(detections[detection_index])
            row["score"] = float(1.0 - 1e-7 * label_index)
            row["oracle_iou"] = float(overlap)
            output_detections.append(row)
            selected_detection_scores.append(float(detections[detection_index].get("score", 0.0)))
            selected_detection_ious.append(float(overlap))
        oracle[image_id]["detections"] = output_detections

    coverage = [
        {"iou": float(threshold), "covered_labels": int(count), "coverage": float(count / max(1, labels_total))}
        for threshold, count in zip(thresholds, covered_by_threshold)
    ]
    summary = {
        "input": str(args.predictionsgt_pkl.resolve()),
        "images": len(data),
        "images_with_labels": images_with_labels,
        "labels": labels_total,
        "detections": detections_total,
        "coverage": coverage,
        "ranking_only_map50_upper_bound": float(covered_by_threshold[0] / max(1, labels_total)),
        "labels_missing_candidate_iou50": int(labels_total - covered_by_threshold[0]),
        "frames_all_labels_covered_iou50": frames_all_covered,
        "frames_partially_covered_iou50": frames_partially_covered,
        "frames_no_candidate_iou50": frames_no_candidate,
        "selected_candidate_original_score_mean": float(np.mean(selected_detection_scores)) if selected_detection_scores else None,
        "selected_candidate_original_score_p10": float(np.quantile(selected_detection_scores, 0.1)) if selected_detection_scores else None,
        "selected_candidate_iou_mean": float(np.mean(selected_detection_ious)) if selected_detection_ious else None,
        "oracle_pkl": str(args.out_oracle_pkl.resolve()),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_oracle_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_oracle_pkl.open("wb") as handle:
        pickle.dump(oracle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
