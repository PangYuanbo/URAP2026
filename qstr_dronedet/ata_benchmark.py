"""ATA dataset auditing and paper-style single-object tracking metrics."""

from __future__ import annotations

import json
import math
import os
import shutil
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


Box = tuple[float, float, float, float]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_SEQUENCES = {
    "train": [*(f"uav-m{i}" for i in range(1, 23)), *(f"uav-s{i}" for i in range(1, 19))],
    "test": [*(f"uav-m{i}" for i in range(23, 29)), *(f"uav-s{i}" for i in range(19, 23))],
}


@dataclass(frozen=True)
class SequenceMetrics:
    name: str
    frames: int
    auc: float
    op50: float
    precision_20: float
    normalized_precision_auc: float
    mean_iou: float


def read_boxes(path: str | Path) -> list[Box]:
    rows: list[Box] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        normalized = line.replace("\t", ",").replace(" ", ",")
        values = [float(value) for value in normalized.split(",") if value]
        if len(values) != 4:
            raise ValueError(f"Expected four xywh values in {path}:{line_number}, got {line!r}")
        rows.append((values[0], values[1], values[2], values[3]))
    return rows


def read_prediction_boxes(path: str | Path) -> list[Box]:
    path = Path(path)
    if path.suffix.lower() != ".csv":
        return read_boxes(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = ("pred_x", "pred_y", "pred_w", "pred_h")
        if not reader.fieldnames or any(field not in reader.fieldnames for field in required):
            raise ValueError(f"SAMURAI prediction CSV is missing {required}: {path}")
        return [tuple(float(row[field]) for field in required) for row in reader]


def list_images(path: str | Path) -> list[Path]:
    root = Path(path)
    if not root.is_dir():
        return []
    return sorted(item for item in root.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def audit_ata_release(root: str | Path, *, image_width: int = 1920, image_height: int = 1080) -> dict[str, object]:
    root = Path(root)
    result: dict[str, object] = {"dataset_root": str(root.resolve()), "splits": {}}
    all_boxes: list[Box] = []
    missing_sequences: list[str] = []
    unexpected_sequences: list[str] = []
    missing_image_sequences: list[str] = []
    frame_mismatches: list[dict[str, object]] = []
    out_of_bounds: list[dict[str, object]] = []

    for split, expected_names in EXPECTED_SEQUENCES.items():
        split_root = root / split
        actual_names = sorted(item.name for item in split_root.iterdir() if item.is_dir()) if split_root.is_dir() else []
        missing_sequences.extend(f"{split}/{name}" for name in expected_names if name not in actual_names)
        unexpected_sequences.extend(f"{split}/{name}" for name in actual_names if name not in expected_names)
        sequence_rows = []
        for name in expected_names:
            sequence_root = split_root / name
            gt_path = sequence_root / "groundtruth.txt"
            if not gt_path.is_file():
                continue
            boxes = read_boxes(gt_path)
            all_boxes.extend(boxes)
            images = list_images(sequence_root / "img")
            if not images:
                missing_image_sequences.append(f"{split}/{name}")
            elif len(images) != len(boxes):
                frame_mismatches.append(
                    {"sequence": f"{split}/{name}", "annotations": len(boxes), "images": len(images)}
                )
            for frame_index, (x, y, width, height) in enumerate(boxes, 1):
                if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
                    out_of_bounds.append(
                        {
                            "sequence": f"{split}/{name}",
                            "frame": frame_index,
                            "box": [x, y, width, height],
                        }
                    )
            sequence_rows.append(
                {
                    "name": name,
                    "annotations": len(boxes),
                    "images": len(images),
                    "language": (sequence_root / "language.txt").is_file(),
                }
            )
        result["splits"][split] = {
            "expected_sequences": len(expected_names),
            "available_sequences": len(sequence_rows),
            "annotation_frames": sum(row["annotations"] for row in sequence_rows),
            "image_frames": sum(row["images"] for row in sequence_rows),
            "sequences": sequence_rows,
        }

    widths = np.asarray([box[2] for box in all_boxes], dtype=np.float64)
    heights = np.asarray([box[3] for box in all_boxes], dtype=np.float64)
    areas = widths * heights
    result.update(
        {
            "annotation_frames": len(all_boxes),
            "paper_reported_frames": 38094,
            "paper_frame_delta": len(all_boxes) - 38094,
            "mean_width": float(widths.mean()) if len(widths) else 0.0,
            "mean_height": float(heights.mean()) if len(heights) else 0.0,
            "mean_area": float(areas.mean()) if len(areas) else 0.0,
            "missing_sequences": missing_sequences,
            "unexpected_sequences": unexpected_sequences,
            "missing_image_sequences": missing_image_sequences,
            "frame_mismatches": frame_mismatches,
            "out_of_bounds_boxes": out_of_bounds,
            "ready_for_tracking": not missing_sequences
            and not missing_image_sequences
            and not frame_mismatches,
        }
    )
    return result


def _box_iou(prediction: Box, target: Box) -> float:
    if prediction[2] <= 0 or prediction[3] <= 0 or target[2] <= 0 or target[3] <= 0:
        return 0.0
    px2, py2 = prediction[0] + prediction[2], prediction[1] + prediction[3]
    tx2, ty2 = target[0] + target[2], target[1] + target[3]
    intersection = max(0.0, min(px2, tx2) - max(prediction[0], target[0])) * max(
        0.0, min(py2, ty2) - max(prediction[1], target[1])
    )
    union = prediction[2] * prediction[3] + target[2] * target[3] - intersection
    return intersection / union if union > 0 else 0.0


def _center_errors(prediction: Box, target: Box) -> tuple[float, float]:
    if prediction[2] <= 0 or prediction[3] <= 0 or target[2] <= 0 or target[3] <= 0:
        return float("inf"), float("inf")
    dx = (prediction[0] + prediction[2] / 2) - (target[0] + target[2] / 2)
    dy = (prediction[1] + prediction[3] / 2) - (target[1] + target[3] / 2)
    absolute = math.hypot(dx, dy)
    normalized = math.hypot(dx / target[2], dy / target[3])
    return absolute, normalized


def evaluate_sequence(name: str, predictions: Sequence[Box], targets: Sequence[Box]) -> SequenceMetrics:
    if len(predictions) != len(targets):
        raise ValueError(f"Prediction length mismatch for {name}: {len(predictions)} != {len(targets)}")
    if not targets:
        raise ValueError(f"Sequence {name} has no frames")
    ious = np.asarray([_box_iou(pred, gt) for pred, gt in zip(predictions, targets)], dtype=np.float64)
    errors = np.asarray([_center_errors(pred, gt)[0] for pred, gt in zip(predictions, targets)], dtype=np.float64)
    normalized_errors = np.asarray(
        [_center_errors(pred, gt)[1] for pred, gt in zip(predictions, targets)], dtype=np.float64
    )
    success_thresholds = np.arange(0.0, 1.0001, 0.05)
    normalized_thresholds = np.arange(0.0, 0.5001, 0.01)
    return SequenceMetrics(
        name=name,
        frames=len(targets),
        auc=float(np.mean([(ious >= threshold).mean() for threshold in success_thresholds])),
        op50=float((ious >= 0.5).mean()),
        precision_20=float((errors <= 20.0).mean()),
        normalized_precision_auc=float(
            np.mean([(normalized_errors <= threshold).mean() for threshold in normalized_thresholds])
        ),
        mean_iou=float(ious.mean()),
    )


def evaluate_ata_predictions(
    dataset_root: str | Path,
    predictions_root: str | Path,
    *,
    split: str = "test",
) -> dict[str, object]:
    if split not in EXPECTED_SEQUENCES:
        raise ValueError(f"Unsupported ATA split: {split}")
    dataset_root = Path(dataset_root)
    predictions_root = Path(predictions_root)
    per_sequence = []
    for name in EXPECTED_SEQUENCES[split]:
        targets = read_boxes(dataset_root / split / name / "groundtruth.txt")
        prediction_path = predictions_root / f"{name}.txt"
        if not prediction_path.is_file():
            prediction_path = predictions_root / name / "predictions.txt"
        if not prediction_path.is_file():
            prediction_path = predictions_root / f"{name}.csv"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing predictions for {name} under {predictions_root}")
        metrics = evaluate_sequence(name, read_prediction_boxes(prediction_path), targets)
        per_sequence.append(metrics)
    return {
        "dataset": "ATA",
        "split": split,
        "aggregation": "macro average across sequences",
        "sequences": len(per_sequence),
        "frames": sum(item.frames for item in per_sequence),
        "auc": float(np.mean([item.auc for item in per_sequence])),
        "op50": float(np.mean([item.op50 for item in per_sequence])),
        "precision_20": float(np.mean([item.precision_20 for item in per_sequence])),
        "normalized_precision_auc": float(np.mean([item.normalized_precision_auc for item in per_sequence])),
        "mean_iou": float(np.mean([item.mean_iou for item in per_sequence])),
        "sequence_results": [item.__dict__ for item in per_sequence],
    }


def materialize_samurai_layout(
    source_root: str | Path,
    output_root: str | Path,
    *,
    split: str,
    image_mode: str = "hardlink",
) -> dict[str, object]:
    if split not in EXPECTED_SEQUENCES:
        raise ValueError(f"Unsupported ATA split: {split}")
    if image_mode not in {"hardlink", "copy"}:
        raise ValueError("image_mode must be 'hardlink' or 'copy'")
    source_root = Path(source_root)
    output_root = Path(output_root)
    names = EXPECTED_SEQUENCES[split]
    total_frames = 0
    for name in names:
        source_sequence = source_root / split / name
        boxes = read_boxes(source_sequence / "groundtruth.txt")
        images = list_images(source_sequence / "img")
        if len(images) != len(boxes):
            raise ValueError(
                f"ATA images are unavailable or incomplete for {split}/{name}: "
                f"{len(images)} images for {len(boxes)} annotations"
            )
        target_sequence = output_root / "lasot" / "uav" / name
        target_images = target_sequence / "img"
        target_images.mkdir(parents=True, exist_ok=True)
        for index, source_image in enumerate(images, 1):
            target_image = target_images / f"{index:08d}.jpg"
            if target_image.exists():
                continue
            if image_mode == "hardlink":
                os.link(source_image, target_image)
            else:
                shutil.copy2(source_image, target_image)
        shutil.copy2(source_sequence / "groundtruth.txt", target_sequence / "groundtruth.txt")
        language = source_sequence / "language.txt"
        if language.is_file():
            shutil.copy2(language, target_sequence / "language.txt")
        total_frames += len(boxes)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{split}_set.txt").write_text("\n".join(names) + "\n", encoding="ascii")
    manifest = {
        "dataset": "ATA",
        "split": split,
        "sequences": len(names),
        "frames": total_frames,
        "image_mode": image_mode,
        "first_frame_prompt": "groundtruth row 1, xywh",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
