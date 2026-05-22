from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from qstr_dronedet.candidates.merge import bbox_iou
from qstr_dronedet.tracking.tracklet_classifier import filter_infer_rows_with_tracklet_classifier


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
                    "tag": row.get("tag", ""),
                }
            )
    return rows


def _load_run_rows(run_roots: list[str | Path], profile: str, max_frames: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    pred_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    seqs: set[str] = set()
    for run_root in run_roots:
        profile_root = Path(run_root) / profile
        for pred_path in sorted(profile_root.glob("*/predictions.jsonl")):
            seq = pred_path.parent.name
            diag_path = pred_path.parent / "diagnostics.jsonl"
            if not diag_path.exists():
                continue
            seqs.add(seq)
            for row in _load_jsonl(pred_path):
                frame_id = int(row.get("frame_id", -1))
                if max_frames is not None and frame_id >= max_frames:
                    continue
                item = dict(row)
                item["seq"] = seq
                pred_rows.append(item)
            for row in _load_jsonl(diag_path):
                frame_id = int(row.get("frame_id", -1))
                if max_frames is not None and frame_id >= max_frames:
                    continue
                item = dict(row)
                item["seq"] = seq
                diag_rows.append(item)
    return pred_rows, diag_rows, seqs


def _evaluate_rows(rows: list[dict[str, Any]], gt_rows: list[dict[str, Any]], score_threshold: float, iou_threshold: float) -> dict[str, Any]:
    preds = [
        r for r in rows
        if r.get("predicted_class") == "drone" and float(r.get("final_drone_score", 0.0)) >= score_threshold
    ]
    gt_by_key: dict[tuple[str, int], list[tuple[int, dict[str, Any]]]] = {}
    for idx, gt in enumerate(gt_rows):
        gt_by_key.setdefault((str(gt["seq"]), int(gt["frame_id"])), []).append((idx, gt))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for pred_idx, pred in enumerate(sorted(preds, key=lambda r: float(r.get("final_drone_score", 0.0)), reverse=True)):
        best_idx = None
        best_iou = 0.0
        for gt_idx, gt in gt_by_key.get((str(pred.get("seq")), int(pred.get("frame_id", -1))), []):
            if gt_idx in matched_gt:
                continue
            ov = bbox_iou(tuple(pred.get("bbox", [0, 0, 0, 0])), tuple(gt["bbox"]))
            if ov > best_iou:
                best_iou = ov
                best_idx = gt_idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            matched_pred.add(pred_idx)

    fp_frames = set()
    success_frames = set()
    gt_frames = {(str(g["seq"]), int(g["frame_id"])) for g in gt_rows}
    pred_frames = {(str(p.get("seq")), int(p.get("frame_id", -1))) for p in preds}
    for seq, frame_id in pred_frames - gt_frames:
        fp_frames.add((seq, frame_id))
    for gt_idx in matched_gt:
        gt = gt_rows[gt_idx]
        success_frames.add((str(gt["seq"]), int(gt["frame_id"])))

    tp = len(matched_pred)
    fp = len(preds) - tp
    fn = len(gt_rows) - len(matched_gt)
    return {
        "gt": len(gt_rows),
        "pred_drone": len(preds),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "frame_success_rate": len(success_frames) / max(1, len(gt_frames)),
        "false_positive_no_gt_frames": len(fp_frames),
    }


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


def run_tracklet_filter_sweep(
    run_roots: list[str | Path],
    gt_csv: str | Path,
    weights: str | Path,
    out: str | Path,
    profile: str = "hard_recovery",
    classifier_thresholds: list[float] | None = None,
    promotion_score_floors: list[float] | None = None,
    promotion_max_backgrounds: list[float] | None = None,
    promotion_min_branch_drone: float = 0.40,
    score_threshold: float = 0.20,
    iou_threshold: float = 0.30,
    max_frames: int | None = None,
) -> dict[str, Any]:
    classifier_thresholds = classifier_thresholds or [0.5, 0.7, 0.85, 0.95]
    promotion_score_floors = promotion_score_floors or [0.20, 0.22, 0.30]
    promotion_max_backgrounds = promotion_max_backgrounds or [0.55, 0.60, 0.68]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_rows, diag_rows, seqs = _load_run_rows(run_roots, profile, max_frames)
    gt_rows = [g for g in _load_gt_csv(gt_csv, max_frames=max_frames) if g["seq"] in seqs]
    raw_metrics = _evaluate_rows(pred_rows, gt_rows, score_threshold, iou_threshold)
    rows: list[dict[str, Any]] = []

    for threshold in classifier_thresholds:
        for promote in [False, True]:
            backgrounds = promotion_max_backgrounds if promote else [0.0]
            floors = promotion_score_floors if promote else [0.0]
            for floor in floors:
                for max_bg in backgrounds:
                    filtered, _, filter_summary = filter_infer_rows_with_tracklet_classifier(
                        pred_rows,
                        diag_rows,
                        weights,
                        threshold=threshold,
                        promote_positive_tracklets=promote,
                        promotion_score_floor=floor,
                        promotion_min_branch_drone=promotion_min_branch_drone,
                        promotion_max_background=max_bg,
                    )
                    metrics = _evaluate_rows(filtered, gt_rows, score_threshold, iou_threshold)
                    rows.append(
                        {
                            "classifier_threshold": threshold,
                            "promotion_enabled": int(promote),
                            "promotion_score_floor": floor,
                            "promotion_max_background": max_bg,
                            "promotion_min_branch_drone": promotion_min_branch_drone,
                            **metrics,
                            "delta_tp": metrics["tp"] - raw_metrics["tp"],
                            "delta_fp": metrics["fp"] - raw_metrics["fp"],
                            "delta_fn": metrics["fn"] - raw_metrics["fn"],
                            "delta_recall": metrics["recall"] - raw_metrics["recall"],
                            "delta_precision": metrics["precision"] - raw_metrics["precision"],
                            "raw_drone_predictions": filter_summary["raw_drone_predictions"],
                            "filtered_drone_predictions_pre_score": filter_summary["filtered_drone_predictions"],
                            "rejected_drone_predictions_pre_score": filter_summary["rejected_drone_predictions"],
                        }
                    )

    stable_target_recall = raw_metrics["recall"] - 0.02
    hard_target_recall = raw_metrics["recall"] + 0.05
    stable_candidates = [
        r for r in rows
        if r["recall"] >= stable_target_recall and r["fp"] <= raw_metrics["fp"]
    ]
    hard_candidates = [
        r for r in rows
        if r["recall"] >= hard_target_recall
    ]
    stable = sorted(stable_candidates or rows, key=lambda r: (-r["precision"], r["fp"], -r["recall"]))[0]
    hard = sorted(hard_candidates or rows, key=lambda r: (-r["recall"], r["fp"], -r["precision"]))[0]
    _write_csv(out_dir / "tracklet_filter_sweep.csv", rows)
    summary = {
        "run_roots": [str(p) for p in run_roots],
        "gt_csv": str(gt_csv),
        "weights": str(weights),
        "profile": profile,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "max_frames": max_frames,
        "raw": raw_metrics,
        "num_configs": len(rows),
        "stable_target_recall": stable_target_recall,
        "stable_met_target": bool(stable_candidates),
        "selected_stable": stable,
        "hard_recovery_target_recall": hard_target_recall,
        "hard_recovery_met_target": bool(hard_candidates),
        "selected_hard_recovery": hard,
    }
    (out_dir / "tracklet_filter_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
