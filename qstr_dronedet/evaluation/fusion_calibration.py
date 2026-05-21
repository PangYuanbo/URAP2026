from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

from qstr_dronedet.candidates.merge import bbox_iou, center_distance
from qstr_dronedet.types import CLASSES, normalize_probs


def _read_gt(csv_path: str | Path) -> dict[tuple[str, int], tuple[float, float, float, float]]:
    rows = list(csv.DictReader(Path(csv_path).open("r", encoding="utf-8")))
    out: dict[tuple[str, int], tuple[float, float, float, float]] = {}
    for row in rows:
        video = str(Path(row["video_path"]))
        frame_id = int(float(row["frame_id"]))
        out[(video, frame_id)] = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
    return out


def _nearest_gt(
    gt: dict[tuple[str, int], tuple[float, float, float, float]],
    video: str,
    frame_id: int,
    tolerance: int,
) -> tuple[float, float, float, float] | None:
    exact = gt.get((video, frame_id))
    if exact is not None:
        return exact
    best: tuple[int, tuple[float, float, float, float]] | None = None
    for (v, fid), box in gt.items():
        if v != video:
            continue
        d = abs(fid - frame_id)
        if d <= tolerance and (best is None or d < best[0]):
            best = (d, box)
    return best[1] if best else None


def _infer_video_key(row: dict[str, Any], gt: dict[tuple[str, int], tuple[float, float, float, float]], video_hint: str | None) -> str | None:
    if video_hint:
        return str(Path(video_hint))
    if row.get("video_path"):
        return str(Path(row["video_path"]))
    frame_id = int(row["frame_id"])
    matches = [video for (video, fid) in gt if fid == frame_id]
    if len(matches) == 1:
        return matches[0]
    return matches[0] if matches else None


def _score_candidate(row: dict[str, Any], weights: dict[str, float]) -> dict[str, float]:
    crop = normalize_probs(row.get("crop_probs", {}))
    feat = normalize_probs(row.get("feature_probs", {}))
    temp = normalize_probs(row.get("temporal_probs", {}))
    fused = {c: 0.0 for c in CLASSES}
    for c in CLASSES:
        fused[c] += weights["crop"] * crop[c]
        fused[c] += weights["feature"] * feat[c]
        fused[c] += weights["temporal"] * temp[c]
    fused["drone"] += weights["tracker"] * max(0.0, min(1.0, float(row.get("track_score") or 0.0)))
    fused["drone"] += weights["motion"] * max(0.0, min(1.0, float(row.get("motion_score") or 0.0)))
    return normalize_probs(fused)


def calibrate_fusion_from_diagnostics(
    diagnostics_jsonl: str | Path,
    gt_csv: str | Path,
    out: str | Path,
    video_hint: str | None = None,
    match_iou: float = 0.1,
    match_center_px: float = 24.0,
    threshold: float = 0.5,
    frame_tolerance: int = 0,
) -> dict[str, Any]:
    diagnostics_jsonl = Path(diagnostics_jsonl)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in diagnostics_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    gt = _read_gt(gt_csv)
    labeled: list[dict[str, Any]] = []
    for row in rows:
        video = _infer_video_key(row, gt, video_hint)
        if video is None:
            continue
        gt_box = _nearest_gt(gt, video, int(row["frame_id"]), frame_tolerance)
        if gt_box is None:
            continue
        iou = bbox_iou(tuple(float(x) for x in row["bbox"]), gt_box)
        dist = center_distance(tuple(float(x) for x in row["bbox"]), gt_box)
        item = dict(row)
        item["label_positive"] = bool(iou >= match_iou or dist <= match_center_px)
        item["match_iou"] = iou
        item["match_center_distance"] = dist
        labeled.append(item)

    crop_grid = [0.25, 0.35, 0.45, 0.55, 0.65]
    feature_grid = [0.0, 0.05, 0.10, 0.15, 0.25]
    temporal_grid = [0.20, 0.30, 0.40, 0.50]
    tracker_grid = [0.0, 0.05, 0.10]
    motion_grid = [0.0, 0.03]
    results = []
    positives = sum(1 for r in labeled if r["label_positive"])
    for crop_w, feat_w, temp_w, tracker_w, motion_w in product(crop_grid, feature_grid, temporal_grid, tracker_grid, motion_grid):
        s = crop_w + feat_w + temp_w + tracker_w + motion_w
        weights = {
            "crop": crop_w / s,
            "feature": feat_w / s,
            "temporal": temp_w / s,
            "tracker": tracker_w / s,
            "motion": motion_w / s,
        }
        tp = fp = fn = 0
        max_pos = 0.0
        max_neg = 0.0
        positive_frames: set[int] = set()
        hit_frames: set[int] = set()
        for row in labeled:
            score = float(row.get("objectness", 1.0)) * _score_candidate(row, weights)["drone"]
            pred = score >= threshold
            if row["label_positive"]:
                positive_frames.add(int(row["frame_id"]))
                max_pos = max(max_pos, score)
                if pred:
                    tp += 1
                    hit_frames.add(int(row["frame_id"]))
                else:
                    fn += 1
            else:
                max_neg = max(max_neg, score)
                if pred:
                    fp += 1
        precision = tp / max(1, tp + fp)
        recall = len(hit_frames) / max(1, len(positive_frames))
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        results.append(
            {
                "weights": weights,
                "tp_rows": tp,
                "fp_rows": fp,
                "fn_rows": fn,
                "precision_rows": precision,
                "recall_frames": recall,
                "f1": f1,
                "max_positive_score": max_pos,
                "max_negative_score": max_neg,
            }
        )
    results.sort(key=lambda r: (r["f1"], r["recall_frames"], -r["fp_rows"], r["max_positive_score"]), reverse=True)
    summary = {
        "diagnostics": str(diagnostics_jsonl),
        "gt_csv": str(gt_csv),
        "video_hint": video_hint,
        "num_candidates": len(rows),
        "num_labeled_candidates": len(labeled),
        "num_positive_candidates": positives,
        "threshold": threshold,
        "frame_tolerance": frame_tolerance,
        "best": results[:10],
    }
    (out / "fusion_calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
