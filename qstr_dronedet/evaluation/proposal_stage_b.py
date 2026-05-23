from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from qstr_dronedet.recognition.crop_recognizer import CropRecognizer
from qstr_dronedet.types import CLASSES


DRONE_INDEX = CLASSES.index("drone")
BACKGROUND_INDEX = CLASSES.index("background")


def _load_crop_model(weights: str | Path) -> tuple[CropRecognizer, str, torch.device]:
    ckpt = torch.load(weights, map_location="cpu")
    model = CropRecognizer()
    model.load_state_dict(ckpt["state_dict"])
    target_mode = str(ckpt.get("target_mode", "multiclass"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, target_mode, device


def _read_crop(path: str | Path, image_size: int = 128) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def _binary_drone_probability(logits: torch.Tensor, target_mode: str) -> float:
    if target_mode == "drone_binary":
        probs = torch.softmax(logits[:, [DRONE_INDEX, BACKGROUND_INDEX]], dim=1)
        return float(probs[0, 0].item())
    probs = torch.softmax(logits, dim=1)
    return float(probs[0, DRONE_INDEX].item())


def _empty_counter() -> dict[str, float]:
    return {
        "samples": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "mean_p_drone": 0.0,
    }


def _finalize_counter(counter: dict[str, float], prob_sum: float) -> dict[str, float]:
    samples = int(counter["samples"])
    tp = int(counter["tp"])
    fp = int(counter["fp"])
    tn = int(counter["tn"])
    fn = int(counter["fn"])
    counter["accuracy"] = (tp + tn) / max(1, samples)
    counter["precision"] = tp / max(1, tp + fp)
    counter["recall"] = tp / max(1, tp + fn)
    counter["mean_p_drone"] = prob_sum / max(1, samples)
    return counter


def evaluate_crop_recognizer_on_proposals(
    manifest_jsonl: str | Path,
    crop_weights: str | Path,
    out_dir: str | Path,
    threshold: float = 0.5,
    max_samples: int | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Evaluate a crop recognizer on detector-proposal Stage B manifest rows."""
    manifest_jsonl = Path(manifest_jsonl)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in manifest_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_samples is not None:
        rows = rows[: max(0, int(max_samples))]

    model, target_mode, device = _load_crop_model(crop_weights)
    predictions_path = out_dir / "proposal_stage_b_predictions.jsonl"
    summary_path = out_dir / "summary.json"

    overall = _empty_counter()
    by_bucket: dict[str, dict[str, float]] = defaultdict(_empty_counter)
    by_bucket_prob_sum: dict[str, float] = defaultdict(float)
    by_class: dict[str, dict[str, float]] = defaultdict(_empty_counter)
    by_class_prob_sum: dict[str, float] = defaultdict(float)
    overall_prob_sum = 0.0
    skipped_rows = 0
    missed_drone_rows = 0

    pending: list[tuple[dict[str, Any], torch.Tensor]] = []

    def consume(batch: list[tuple[dict[str, Any], torch.Tensor]], fh) -> None:
        nonlocal overall_prob_sum
        if not batch:
            return
        tensors = torch.stack([item[1] for item in batch], dim=0).to(device)
        with torch.no_grad():
            logits = model(tensors)
            if target_mode == "drone_binary":
                probs = torch.softmax(logits[:, [DRONE_INDEX, BACKGROUND_INDEX]], dim=1)[:, 0]
            else:
                probs = torch.softmax(logits, dim=1)[:, DRONE_INDEX]
        for (row, _), p in zip(batch, probs):
            p_drone = float(p.item())
            gt_class = str(row.get("class", "background"))
            bucket = str(row.get("diagnostic_bucket", gt_class))
            gt_is_drone = gt_class == "drone"
            pred_is_drone = p_drone >= threshold
            if gt_is_drone and pred_is_drone:
                key = "tp"
            elif (not gt_is_drone) and pred_is_drone:
                key = "fp"
            elif (not gt_is_drone) and (not pred_is_drone):
                key = "tn"
            else:
                key = "fn"

            for counter in (overall, by_bucket[bucket], by_class[gt_class]):
                counter["samples"] += 1
                counter[key] += 1
            overall_prob_sum += p_drone
            by_bucket_prob_sum[bucket] += p_drone
            by_class_prob_sum[gt_class] += p_drone

            out_row = {
                "crop_path": row.get("crop_path"),
                "source_video": row.get("source_video"),
                "frame_id": row.get("frame_id"),
                "class": gt_class,
                "diagnostic_bucket": bucket,
                "proposal_score": row.get("proposal_score"),
                "proposal_source": row.get("proposal_source"),
                "match_iou": row.get("match_iou"),
                "match_center_distance": row.get("match_center_distance"),
                "p_drone": p_drone,
                "predicted_is_drone": pred_is_drone,
                "correct": (gt_is_drone == pred_is_drone),
                "outcome": key,
            }
            fh.write(json.dumps(out_row) + "\n")

    with predictions_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if row.get("class") == "missed_drone":
                missed_drone_rows += 1
                continue
            crop_path = row.get("crop_path")
            if not crop_path:
                skipped_rows += 1
                continue
            try:
                pending.append((row, _read_crop(crop_path)))
            except FileNotFoundError:
                skipped_rows += 1
                continue
            if len(pending) >= max(1, int(batch_size)):
                consume(pending, fh)
                pending.clear()
        consume(pending, fh)

    summary = {
        "manifest_jsonl": str(manifest_jsonl),
        "crop_weights": str(crop_weights),
        "target_mode": target_mode,
        "threshold": threshold,
        "max_samples": max_samples,
        "evaluated_rows": int(overall["samples"]),
        "skipped_rows": skipped_rows,
        "missed_drone_rows": missed_drone_rows,
        "overall": _finalize_counter(overall, overall_prob_sum),
        "by_diagnostic_bucket": {
            bucket: _finalize_counter(counter, by_bucket_prob_sum[bucket])
            for bucket, counter in sorted(by_bucket.items())
        },
        "by_class": {
            cls: _finalize_counter(counter, by_class_prob_sum[cls])
            for cls, counter in sorted(by_class.items())
        },
        "predictions_jsonl": str(predictions_path),
        "summary_json": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
