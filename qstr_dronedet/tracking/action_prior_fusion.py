from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qstr_dronedet.evaluation.tracklet_filter_sweep import _evaluate_rows


@dataclass(frozen=True)
class ActionPriorFusionResult:
    out_path: Path
    summary: dict[str, Any]


def _iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _append_cause(cause: Any, value: str) -> str:
    if cause is None or cause == "":
        return value
    text = str(cause)
    if value in text.split("+"):
        return text
    return f"{text}+{value}"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


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
        for row in rows:
            writer.writerow(row)


def _metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {**metrics, "f1": f1}


def _load_run_prediction_rows(
    run_roots: list[str | Path],
    profile: str,
    prediction_name: str,
    max_frames: int | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    seqs: set[str] = set()
    for run_root in run_roots:
        profile_root = Path(run_root) / profile
        for pred_path in sorted(profile_root.glob(f"*/{prediction_name}")):
            seq = pred_path.parent.name
            seqs.add(seq)
            for row in _load_jsonl(pred_path):
                frame_id = int(row.get("frame_id", -1))
                if max_frames is not None and frame_id >= max_frames:
                    continue
                item = dict(row)
                item["seq"] = str(item.get("seq") or seq)
                rows.append(item)
    return rows, seqs


def _fuse_rows(
    rows: list[dict[str, Any]],
    prior_weight: float,
    min_prior_score: float,
    promote_threshold: float | None,
    min_base_score_for_promotion: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total = 0
    with_prior = 0
    fused_rows = 0
    promoted_rows = 0
    score_gain_sum = 0.0
    raw_drone_predictions = 0
    final_drone_predictions = 0
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        total += 1
        out_row = dict(row)
        raw_score = _clamp01(float(out_row.get("final_drone_score", 0.0) or 0.0))
        raw_pred = out_row.get("predicted_class")
        raw_drone_predictions += int(raw_pred == "drone")
        prior = _clamp01(float(out_row.get("action_frame_prior_score", 0.0) or 0.0))
        if prior > 0.0:
            with_prior += 1
        if prior >= min_prior_score:
            fused_score = raw_score + prior_weight * prior * (1.0 - raw_score)
            fused_score = _clamp01(fused_score)
            gain = max(0.0, fused_score - raw_score)
            out_row["raw_final_drone_score"] = out_row.get("raw_final_drone_score", out_row.get("final_drone_score"))
            out_row["action_prior_fused_score"] = fused_score
            out_row["action_prior_score_gain"] = gain
            out_row["final_drone_score"] = fused_score
            probs = dict(out_row.get("final_probs") or {})
            probs["drone"] = max(float(probs.get("drone", 0.0) or 0.0), fused_score)
            probs["background"] = min(float(probs.get("background", 1.0) or 1.0), 1.0 - fused_score)
            out_row["final_probs"] = probs
            out_row["diagnostic_cause"] = _append_cause(out_row.get("diagnostic_cause"), "action_prior_fused")
            fused_rows += 1
            score_gain_sum += gain
            if (
                promote_threshold is not None
                and raw_pred != "drone"
                and raw_score >= min_base_score_for_promotion
                and fused_score >= promote_threshold
            ):
                out_row["raw_predicted_class"] = out_row.get("raw_predicted_class", raw_pred)
                out_row["predicted_class"] = "drone"
                out_row["diagnostic_cause"] = _append_cause(out_row.get("diagnostic_cause"), "action_prior_promoted")
                promoted_rows += 1
        final_drone_predictions += int(out_row.get("predicted_class") == "drone")
        out_rows.append(out_row)
    summary = {
        "total_rows": total,
        "rows_with_prior": with_prior,
        "fused_rows": fused_rows,
        "promoted_rows": promoted_rows,
        "raw_drone_predictions": raw_drone_predictions,
        "final_drone_predictions": final_drone_predictions,
        "mean_score_gain": score_gain_sum / fused_rows if fused_rows else 0.0,
    }
    return out_rows, summary


def fuse_action_frame_prior_predictions(
    pred_jsonl: str | Path,
    out: str | Path,
    prior_weight: float = 0.35,
    min_prior_score: float = 0.25,
    promote_threshold: float | None = 0.20,
    min_base_score_for_promotion: float = 0.0,
) -> ActionPriorFusionResult:
    """Fuse action-prior heatmap support into flat frame-level prediction rows."""
    if prior_weight < 0.0 or prior_weight > 1.0:
        raise ValueError("prior_weight must be in [0, 1]")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(pred_jsonl)
    fused, stats = _fuse_rows(rows, prior_weight, min_prior_score, promote_threshold, min_base_score_for_promotion)
    with out_path.open("w", encoding="utf-8") as f:
        for row in fused:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "pred_jsonl": str(pred_jsonl),
        "jsonl": str(out_path),
        "prior_weight": prior_weight,
        "min_prior_score": min_prior_score,
        "promote_threshold": promote_threshold,
        "min_base_score_for_promotion": min_base_score_for_promotion,
        **stats,
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPriorFusionResult(out_path=out_path, summary=summary)


def sweep_action_frame_prior_fusion(
    pred_jsonl: str | Path,
    gt_csv: str | Path,
    out_dir: str | Path,
    prior_weights: list[float] | None = None,
    min_prior_scores: list[float] | None = None,
    promote_thresholds: list[float | None] | None = None,
    min_base_scores_for_promotion: list[float] | None = None,
    score_threshold: float = 0.20,
    iou_threshold: float = 0.30,
    max_frames: int | None = None,
) -> ActionPriorFusionResult:
    prior_weights = prior_weights or [0.2, 0.35, 0.5]
    min_prior_scores = min_prior_scores or [0.2, 0.35, 0.5]
    promote_thresholds = promote_thresholds or [None, 0.2, 0.3]
    min_base_scores_for_promotion = min_base_scores_for_promotion or [0.0]
    rows = _load_jsonl(pred_jsonl)
    if max_frames is not None:
        rows = [row for row in rows if int(row.get("frame_id", -1)) < max_frames]
    gt_rows = _load_gt_csv(gt_csv, max_frames=max_frames)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    raw_metrics = _metric_row(_evaluate_rows(rows, gt_rows, score_threshold=score_threshold, iou_threshold=iou_threshold))
    sweep_rows: list[dict[str, Any]] = [
        {
            "run_type": "raw",
            "prior_weight": 0.0,
            "min_prior_score": None,
            "promote_threshold": None,
            "min_base_score_for_promotion": None,
            "fused_rows": 0,
            "promoted_rows": 0,
            **raw_metrics,
        }
    ]
    best = sweep_rows[0]
    for prior_weight in prior_weights:
        if prior_weight < 0.0 or prior_weight > 1.0:
            raise ValueError("prior_weights must be in [0, 1]")
        for min_prior_score in min_prior_scores:
            for promote_threshold in promote_thresholds:
                for min_base in min_base_scores_for_promotion:
                    fused, stats = _fuse_rows(rows, prior_weight, min_prior_score, promote_threshold, min_base)
                    metrics = _metric_row(_evaluate_rows(fused, gt_rows, score_threshold=score_threshold, iou_threshold=iou_threshold))
                    row = {
                        "run_type": "action_prior_fused",
                        "prior_weight": prior_weight,
                        "min_prior_score": min_prior_score,
                        "promote_threshold": promote_threshold,
                        "min_base_score_for_promotion": min_base,
                        **stats,
                        **metrics,
                    }
                    sweep_rows.append(row)
                    if (float(row["f1"]), float(row["recall"]), -float(row["fp"])) > (
                        float(best["f1"]),
                        float(best["recall"]),
                        -float(best["fp"]),
                    ):
                        best = row
    csv_path = out_root / "action_frame_prior_fusion_sweep.csv"
    summary_path = out_root / "action_frame_prior_fusion_sweep_summary.json"
    _write_csv(csv_path, sweep_rows)
    summary = {
        "pred_jsonl": str(pred_jsonl),
        "gt_csv": str(gt_csv),
        "out_dir": str(out_root),
        "csv": str(csv_path),
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "max_frames": max_frames,
        "num_configs": len(sweep_rows) - 1,
        "raw": sweep_rows[0],
        "best": best,
        "rows": sweep_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPriorFusionResult(out_path=csv_path, summary=summary)


def _sweep_rows(
    rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    prior_weights: list[float] | None,
    min_prior_scores: list[float] | None,
    promote_thresholds: list[float | None] | None,
    min_base_scores_for_promotion: list[float] | None,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    prior_weights = prior_weights or [0.2, 0.35, 0.5]
    min_prior_scores = min_prior_scores or [0.2, 0.35, 0.5]
    promote_thresholds = promote_thresholds or [None, 0.2, 0.3]
    min_base_scores_for_promotion = min_base_scores_for_promotion or [0.0]
    raw_metrics = _metric_row(_evaluate_rows(rows, gt_rows, score_threshold=score_threshold, iou_threshold=iou_threshold))
    sweep_rows: list[dict[str, Any]] = [
        {
            "run_type": "raw",
            "prior_weight": 0.0,
            "min_prior_score": None,
            "promote_threshold": None,
            "min_base_score_for_promotion": None,
            "fused_rows": 0,
            "promoted_rows": 0,
            **raw_metrics,
        }
    ]
    best = sweep_rows[0]
    for prior_weight in prior_weights:
        if prior_weight < 0.0 or prior_weight > 1.0:
            raise ValueError("prior_weights must be in [0, 1]")
        for min_prior_score in min_prior_scores:
            for promote_threshold in promote_thresholds:
                for min_base in min_base_scores_for_promotion:
                    fused, stats = _fuse_rows(rows, prior_weight, min_prior_score, promote_threshold, min_base)
                    metrics = _metric_row(_evaluate_rows(fused, gt_rows, score_threshold=score_threshold, iou_threshold=iou_threshold))
                    row = {
                        "run_type": "action_prior_fused",
                        "prior_weight": prior_weight,
                        "min_prior_score": min_prior_score,
                        "promote_threshold": promote_threshold,
                        "min_base_score_for_promotion": min_base,
                        **stats,
                        **metrics,
                    }
                    sweep_rows.append(row)
                    if (float(row["f1"]), float(row["recall"]), -float(row["fp"])) > (
                        float(best["f1"]),
                        float(best["recall"]),
                        -float(best["fp"]),
                    ):
                        best = row
    return sweep_rows, sweep_rows[0], best


def sweep_action_frame_prior_fusion_run_root(
    run_roots: list[str | Path],
    gt_csv: str | Path,
    out_dir: str | Path,
    profile: str = "hard_recovery",
    prediction_name: str = "predictions.jsonl",
    prior_weights: list[float] | None = None,
    min_prior_scores: list[float] | None = None,
    promote_thresholds: list[float | None] | None = None,
    min_base_scores_for_promotion: list[float] | None = None,
    score_threshold: float = 0.20,
    iou_threshold: float = 0.30,
    max_frames: int | None = None,
) -> ActionPriorFusionResult:
    rows, seqs = _load_run_prediction_rows(run_roots, profile, prediction_name, max_frames=max_frames)
    gt_rows = [g for g in _load_gt_csv(gt_csv, max_frames=max_frames) if str(g["seq"]) in seqs]
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    sweep_rows, raw, best = _sweep_rows(
        rows,
        gt_rows,
        prior_weights=prior_weights,
        min_prior_scores=min_prior_scores,
        promote_thresholds=promote_thresholds,
        min_base_scores_for_promotion=min_base_scores_for_promotion,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
    csv_path = out_root / "action_frame_prior_fusion_run_sweep.csv"
    summary_path = out_root / "action_frame_prior_fusion_run_sweep_summary.json"
    _write_csv(csv_path, sweep_rows)
    summary = {
        "run_roots": [str(path) for path in run_roots],
        "gt_csv": str(gt_csv),
        "out_dir": str(out_root),
        "csv": str(csv_path),
        "profile": profile,
        "prediction_name": prediction_name,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "max_frames": max_frames,
        "sequences": sorted(seqs),
        "num_sequences": len(seqs),
        "num_prediction_rows": len(rows),
        "num_gt_rows": len(gt_rows),
        "num_configs": len(sweep_rows) - 1,
        "raw": raw,
        "best": best,
        "rows": sweep_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPriorFusionResult(out_path=csv_path, summary=summary)
