from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qstr_dronedet.candidates.merge import bbox_iou


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_gt_csv(path: str | Path, max_frames: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            frame_id = int(float(row["frame_id"]))
            if max_frames is not None and frame_id >= max_frames:
                continue
            rows.append(
                {
                    "seq": Path(row["video_path"]).parent.name,
                    "frame_id": frame_id,
                    "bbox": [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
                    "class": row.get("class", ""),
                    "tag": row.get("tag", ""),
                }
            )
    return rows


def _iter_prediction_files(run_root: str | Path, profile: str, prediction_name: str) -> list[Path]:
    root = Path(run_root) / profile
    if not root.exists():
        return []
    return sorted(root.glob(f"*/{prediction_name}"))


def _load_predictions(run_root: str | Path, profile: str, prediction_name: str, max_frames: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _iter_prediction_files(run_root, profile, prediction_name):
        seq = path.parent.name
        for row in _load_jsonl(path):
            frame_id = int(row.get("frame_id", -1))
            if max_frames is not None and frame_id >= max_frames:
                continue
            item = dict(row)
            item["seq"] = seq
            item["frame_id"] = frame_id
            rows.append(item)
    return rows


def _match_frame(gt_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], iou_threshold: float) -> tuple[int, int, int, float, list[dict[str, Any]]]:
    preds = sorted(pred_rows, key=lambda r: float(r.get("final_drone_score", 0.0)), reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    hits: list[dict[str, Any]] = []
    for pred_idx, pred in enumerate(preds):
        best_idx = None
        best_iou = 0.0
        for gt_idx, gt in enumerate(gt_rows):
            if gt_idx in matched_gt:
                continue
            ov = bbox_iou(tuple(pred.get("bbox", [0, 0, 0, 0])), tuple(gt["bbox"]))
            if ov > best_iou:
                best_iou = ov
                best_idx = gt_idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            matched_pred.add(pred_idx)
            hits.append({"pred": pred, "gt": gt_rows[best_idx], "iou": best_iou})
    tp = len(matched_gt)
    fp = len(preds) - len(matched_pred)
    fn = len(gt_rows) - tp
    best_iou = 0.0
    for pred in preds:
        for gt in gt_rows:
            best_iou = max(best_iou, bbox_iou(tuple(pred.get("bbox", [0, 0, 0, 0])), tuple(gt["bbox"])))
    return tp, fp, fn, best_iou, hits


def _failure_type(gt_count: int, tp: int, fp: int, best_iou: float, best_score: float, pred_count: int) -> str:
    if gt_count == 0 and fp > 0:
        return "false_positive_no_gt"
    if gt_count == 0:
        return "true_negative"
    if tp >= gt_count:
        return "success"
    if pred_count == 0:
        return "no_drone_prediction"
    if best_iou < 0.1:
        return "proposal_or_localization_failure"
    if best_iou < 0.3:
        return "weak_localization"
    if best_score < 0.2:
        return "score_suppression"
    return "duplicate_or_matching_failure"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _color_for_row(row: dict[str, Any]) -> tuple[int, int, int]:
    if row["gt_count"] == 0:
        return (180, 180, 180) if row["fp"] == 0 else (0, 165, 255)
    if row["tp"] >= row["gt_count"]:
        return (0, 180, 0)
    if row["tp"] > 0:
        return (0, 220, 220)
    if row["pred_drone_count"] > 0:
        return (0, 0, 220)
    return (120, 0, 180)


def _draw_timeline(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    seqs = sorted({str(r["seq"]) for r in rows})
    max_frame = max((int(r["frame_id"]) for r in rows), default=0)
    cell_w, cell_h = 8, 18
    left, top = 230, 38
    width = left + (max_frame + 1) * cell_w + 20
    height = top + len(seqs) * cell_h + 55
    img = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    by_key = {(str(r["seq"]), int(r["frame_id"])): r for r in rows}
    for y_idx, seq in enumerate(seqs):
        y = top + y_idx * cell_h
        cv2.putText(img, seq, (8, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        for frame_id in range(max_frame + 1):
            row = by_key.get((seq, frame_id))
            color = (245, 245, 245) if row is None else _color_for_row(row)
            x = left + frame_id * cell_w
            cv2.rectangle(img, (x, y), (x + cell_w - 1, y + cell_h - 3), color, -1)
    legend = [
        ("success", (0, 180, 0)),
        ("partial", (0, 220, 220)),
        ("miss/localization", (0, 0, 220)),
        ("no prediction", (120, 0, 180)),
        ("FP no GT", (0, 165, 255)),
    ]
    x = 10
    y = height - 22
    for label, color in legend:
        cv2.rectangle(img, (x, y - 10), (x + 14, y + 4), color, -1)
        cv2.putText(img, label, (x + 19, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (30, 30, 30), 1, cv2.LINE_AA)
        x += 120
    cv2.imwrite(str(out_path), img)


def _summarize_frame_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt_frames = [r for r in rows if r["gt_count"] > 0]
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    gt = sum(int(r["gt_count"]) for r in rows)
    pred = sum(int(r["pred_drone_count"]) for r in rows)
    failures = defaultdict(int)
    for row in rows:
        failures[str(row["failure_type"])] += 1
    return {
        "frames": len(rows),
        "gt_objects": gt,
        "gt_frames": len(gt_frames),
        "pred_drone": pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "frame_success_rate": sum(1 for r in gt_frames if r["tp"] >= r["gt_count"]) / max(1, len(gt_frames)),
        "failure_counts": dict(sorted(failures.items())),
    }


def _linear_markdown(summary: dict[str, Any], out_dir: Path) -> str:
    raw = summary.get("raw", {})
    filtered = summary.get("filtered", {})
    return f"""# URAP-UAV: QSTR Frozen10 Per-Frame Failure Analysis

## Result

| output | TP | FP | FN | precision | recall | frame success |
|---|---:|---:|---:|---:|---:|---:|
| raw hard-recovery | {raw.get('tp', 0)} | {raw.get('fp', 0)} | {raw.get('fn', 0)} | {raw.get('precision', 0):.3f} | {raw.get('recall', 0):.3f} | {raw.get('frame_success_rate', 0):.3f} |
| tracklet filtered/promoted | {filtered.get('tp', 0)} | {filtered.get('fp', 0)} | {filtered.get('fn', 0)} | {filtered.get('precision', 0):.3f} | {filtered.get('recall', 0):.3f} | {filtered.get('frame_success_rate', 0):.3f} |

## Artifacts

- `{out_dir / 'raw_frame_timeline.png'}`
- `{out_dir / 'filtered_frame_timeline.png'}`
- `{out_dir / 'per_frame_raw.csv'}`
- `{out_dir / 'per_frame_filtered.csv'}`
- `{out_dir / 'summary.json'}`

## Failure Buckets

Raw:

```json
{json.dumps(raw.get('failure_counts', {}), indent=2)}
```

Filtered:

```json
{json.dumps(filtered.get('failure_counts', {}), indent=2)}
```

## Recommended Linear Issues

1. Tighten tracklet promotion on sequences where recall gain creates many FP.
2. Add per-frame UI overlays for `tracklet_promoted`, `tracklet_confirmed`, and `tracklet_rejected`.
3. Build train/adapt threshold sweep for reject-only vs promotion profiles without tuning on frozen10.
"""


def analyze_frame_failures(
    run_root: str | Path,
    gt_csv: str | Path,
    out: str | Path,
    profile: str = "hard_recovery",
    raw_prediction_name: str = "predictions_raw.jsonl",
    filtered_prediction_name: str = "predictions.jsonl",
    score_threshold: float = 0.2,
    iou_threshold: float = 0.3,
    max_frames: int | None = None,
) -> dict[str, Any]:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = _load_gt_csv(gt_csv, max_frames=max_frames)
    available_seqs = {p.parent.name for p in _iter_prediction_files(run_root, profile, filtered_prediction_name)}
    gt = [g for g in gt if g["seq"] in available_seqs]

    def build_rows(prediction_name: str) -> list[dict[str, Any]]:
        preds_all = _load_predictions(run_root, profile, prediction_name, max_frames=max_frames)
        preds = [
            p for p in preds_all
            if p.get("predicted_class") == "drone" and float(p.get("final_drone_score", 0.0)) >= score_threshold
        ]
        gt_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        pred_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for item in gt:
            gt_by_key[(item["seq"], int(item["frame_id"]))].append(item)
        for item in preds:
            pred_by_key[(str(item["seq"]), int(item["frame_id"]))].append(item)
        keys = sorted(set(gt_by_key) | set(pred_by_key))
        rows: list[dict[str, Any]] = []
        for seq, frame_id in keys:
            gt_rows = gt_by_key.get((seq, frame_id), [])
            pred_rows = pred_by_key.get((seq, frame_id), [])
            tp, fp, fn, best_iou, hits = _match_frame(gt_rows, pred_rows, iou_threshold)
            best_score = max((float(p.get("final_drone_score", 0.0)) for p in pred_rows), default=0.0)
            rows.append(
                {
                    "seq": seq,
                    "frame_id": frame_id,
                    "gt_count": len(gt_rows),
                    "pred_drone_count": len(pred_rows),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "best_iou": best_iou,
                    "best_score": best_score,
                    "hit_track_ids": ";".join(str(h["pred"].get("track_id")) for h in hits),
                    "failure_type": _failure_type(len(gt_rows), tp, fp, best_iou, best_score, len(pred_rows)),
                    "tags": ";".join(sorted({g.get("tag", "") for g in gt_rows if g.get("tag", "")})),
                }
            )
        return rows

    raw_rows = build_rows(raw_prediction_name)
    filtered_rows = build_rows(filtered_prediction_name)
    _write_csv(out_dir / "per_frame_raw.csv", raw_rows)
    _write_csv(out_dir / "per_frame_filtered.csv", filtered_rows)
    raw_summary = _summarize_frame_rows(raw_rows)
    filtered_summary = _summarize_frame_rows(filtered_rows)
    summary = {
        "run_root": str(run_root),
        "profile": profile,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "max_frames": max_frames,
        "raw": raw_summary,
        "filtered": filtered_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _draw_timeline(raw_rows, out_dir / "raw_frame_timeline.png", "Raw hard-recovery per-frame result")
    _draw_timeline(filtered_rows, out_dir / "filtered_frame_timeline.png", "Tracklet filtered/promoted per-frame result")
    linear_md = _linear_markdown(summary, out_dir)
    (out_dir / "URAP-UAV_linear_issue.md").write_text(linear_md, encoding="utf-8")
    return summary
