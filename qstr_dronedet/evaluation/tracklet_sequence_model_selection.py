from __future__ import annotations

import csv
import fnmatch
import json
import shutil
from pathlib import Path
from typing import Any

from qstr_dronedet.evaluation.tracklet_filter_sweep import _evaluate_rows, _load_gt_csv, _load_run_rows
from qstr_dronedet.tracking.tracklet_sequence_classifier import (
    SequenceSample,
    filter_infer_rows_with_tracklet_sequence_classifier,
    train_tracklet_sequence_classifier,
)


def _load_samples(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sample_seq(item: dict[str, Any]) -> str:
    return str((item.get("meta") or {}).get("seq", ""))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _split_samples(
    samples: list[dict[str, Any]],
    calib_seqs: list[str] | None,
    calib_seq_patterns: list[str] | None,
    calib_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seqs = sorted({_sample_seq(item) for item in samples if _sample_seq(item)})
    requested = set(calib_seqs or [])
    patterns = calib_seq_patterns or []
    if requested or patterns:
        calib_set = {seq for seq in seqs if seq in requested or any(fnmatch.fnmatch(seq, p) or p in seq for p in patterns)}
    else:
        n_calib = min(max(1, int(round(len(seqs) * calib_fraction))), max(1, len(seqs) - 1))
        calib_set = set(seqs[-n_calib:])
    if not calib_set:
        raise ValueError("Calibration split is empty")
    if len(calib_set) == len(seqs):
        raise ValueError("Calibration split consumed all sequences")
    train = [item for item in samples if _sample_seq(item) not in calib_set]
    calib = [item for item in samples if _sample_seq(item) in calib_set]
    return train, calib, {
        "train_sequences": sorted({_sample_seq(item) for item in train}),
        "calibration_sequences": sorted(calib_set),
        "num_train_tracklets": len(train),
        "num_calibration_tracklets": len(calib),
    }


def _evaluate_sequence_checkpoint(
    run_roots: list[str | Path],
    gt_csv: str | Path,
    weights: str | Path,
    profile: str,
    allowed_sequences: set[str],
    classifier_threshold: float,
    promotion_enabled: bool,
    promotion_score_floor: float,
    promotion_max_background: float,
    selective_promotion: bool,
    selective_max_promoted_tracklets_per_sequence: int,
    score_threshold: float,
    iou_threshold: float,
    max_frames: int | None,
) -> dict[str, Any]:
    pred_rows, diag_rows, seqs = _load_run_rows(run_roots, profile, max_frames)
    active = set(seqs) & allowed_sequences
    pred_rows = [row for row in pred_rows if str(row.get("seq")) in active]
    diag_rows = [row for row in diag_rows if str(row.get("seq")) in active]
    gt_rows = [row for row in _load_gt_csv(gt_csv, max_frames=max_frames) if str(row.get("seq")) in active]
    raw_metrics = _evaluate_rows(pred_rows, gt_rows, score_threshold, iou_threshold)
    filtered, _, filter_summary = filter_infer_rows_with_tracklet_sequence_classifier(
        pred_rows,
        diag_rows,
        weights,
        threshold=classifier_threshold,
        promote_positive_tracklets=promotion_enabled,
        promotion_score_floor=promotion_score_floor,
        promotion_min_branch_drone=0.40,
        promotion_max_background=promotion_max_background,
        selective_promotion=selective_promotion and promotion_enabled,
        selective_max_promoted_tracklets_per_sequence=selective_max_promoted_tracklets_per_sequence,
    )
    metrics = _evaluate_rows(filtered, gt_rows, score_threshold, iou_threshold)
    return {
        "raw": raw_metrics,
        "filtered": metrics,
        "filter_summary": filter_summary,
        "delta_tp": metrics["tp"] - raw_metrics["tp"],
        "delta_fp": metrics["fp"] - raw_metrics["fp"],
        "delta_recall": metrics["recall"] - raw_metrics["recall"],
        "delta_precision": metrics["precision"] - raw_metrics["precision"],
    }


def _selection_key(row: dict[str, Any], raw_recall: float, raw_precision: float, raw_fp: int, max_recall_drop: float) -> tuple[float, float, float, float]:
    recall = float(row["recall"])
    precision = float(row["precision"])
    fp = int(row["fp"])
    recall_ok = float(recall >= raw_recall - max_recall_drop)
    fp_ok = float(fp <= raw_fp)
    precision_ok = float(precision >= raw_precision)
    return (recall_ok + fp_ok + precision_ok, recall_ok, precision, -float(fp))


def run_tracklet_sequence_model_selection(
    tracklet_jsonl: str | Path,
    run_roots: list[str | Path],
    gt_csv: str | Path,
    out: str | Path,
    profile: str = "hard_recovery",
    calib_seqs: list[str] | None = None,
    calib_seq_patterns: list[str] | None = None,
    calib_fraction: float = 0.4,
    epochs_values: list[int] | None = None,
    hidden_values: list[int] | None = None,
    max_len_values: list[int] | None = None,
    hard_negative_augments_values: list[int] | None = None,
    classifier_thresholds: list[float] | None = None,
    promotion_enabled_values: list[bool] | None = None,
    promotion_score_floors: list[float] | None = None,
    promotion_max_backgrounds: list[float] | None = None,
    selective_promotion: bool = True,
    selective_max_promoted_tracklets_per_sequence_values: list[int] | None = None,
    score_threshold: float = 0.20,
    iou_threshold: float = 0.30,
    max_frames: int | None = None,
    max_recall_drop: float = 0.02,
) -> dict[str, Any]:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = _load_samples(tracklet_jsonl)
    train_samples, calib_samples, split_summary = _split_samples(samples, calib_seqs, calib_seq_patterns, calib_fraction)
    train_jsonl = out_dir / "train_tracklets_sequence.jsonl"
    calib_jsonl = out_dir / "calibration_tracklets_sequence.jsonl"
    _write_jsonl(train_jsonl, train_samples)
    _write_jsonl(calib_jsonl, calib_samples)

    epochs_values = epochs_values or [30]
    hidden_values = hidden_values or [32]
    max_len_values = max_len_values or [12, 24]
    hard_negative_augments_values = hard_negative_augments_values or [0, 2]
    classifier_thresholds = classifier_thresholds or [0.5, 0.7, 0.85]
    promotion_enabled_values = promotion_enabled_values if promotion_enabled_values is not None else [False, True]
    promotion_score_floors = promotion_score_floors or [0.22, 0.30]
    promotion_max_backgrounds = promotion_max_backgrounds or [0.55, 0.60]
    selective_max_promoted_tracklets_per_sequence_values = selective_max_promoted_tracklets_per_sequence_values or [1, 2]

    rows: list[dict[str, Any]] = []
    calib_sequences = set(split_summary["calibration_sequences"])
    candidate_id = 0
    for epochs in epochs_values:
        for hidden in hidden_values:
            for max_len in max_len_values:
                for hard_neg_aug in hard_negative_augments_values:
                    candidate_id += 1
                    ckpt = out_dir / "checkpoints" / f"tracklet_sequence_candidate_{candidate_id:03d}.pt"
                    train_tracklet_sequence_classifier(
                        train_jsonl,
                        ckpt,
                        epochs=epochs,
                        hidden=hidden,
                        max_len=max_len,
                        hard_negative_augments=hard_neg_aug,
                    )
                    for threshold in classifier_thresholds:
                        for promote in promotion_enabled_values:
                            floors = promotion_score_floors if promote else [0.0]
                            backgrounds = promotion_max_backgrounds if promote else [0.0]
                            budgets = selective_max_promoted_tracklets_per_sequence_values if (promote and selective_promotion) else [0]
                            for floor in floors:
                                for max_bg in backgrounds:
                                    for budget in budgets:
                                        ev = _evaluate_sequence_checkpoint(
                                            run_roots,
                                            gt_csv,
                                            ckpt,
                                            profile=profile,
                                            allowed_sequences=calib_sequences,
                                            classifier_threshold=threshold,
                                            promotion_enabled=promote,
                                            promotion_score_floor=floor,
                                            promotion_max_background=max_bg,
                                            selective_promotion=selective_promotion,
                                            selective_max_promoted_tracklets_per_sequence=budget,
                                            score_threshold=score_threshold,
                                            iou_threshold=iou_threshold,
                                            max_frames=max_frames,
                                        )
                                        metrics = ev["filtered"]
                                        raw = ev["raw"]
                                        rows.append(
                                            {
                                                "candidate_id": candidate_id,
                                                "weights": str(ckpt),
                                                "epochs": epochs,
                                                "hidden": hidden,
                                                "max_len": max_len,
                                                "hard_negative_augments": hard_neg_aug,
                                                "classifier_threshold": threshold,
                                                "promotion_enabled": int(promote),
                                                "promotion_score_floor": floor,
                                                "promotion_max_background": max_bg,
                                                "selective_promotion": int(bool(promote and selective_promotion)),
                                                "selective_max_promoted_tracklets_per_sequence": budget,
                                                **metrics,
                                                "raw_tp": raw["tp"],
                                                "raw_fp": raw["fp"],
                                                "raw_fn": raw["fn"],
                                                "raw_precision": raw["precision"],
                                                "raw_recall": raw["recall"],
                                                "delta_tp": ev["delta_tp"],
                                                "delta_fp": ev["delta_fp"],
                                                "delta_recall": ev["delta_recall"],
                                                "delta_precision": ev["delta_precision"],
                                                "filtered_drone_predictions_pre_score": ev["filter_summary"]["filtered_drone_predictions"],
                                                "rejected_drone_predictions_pre_score": ev["filter_summary"]["rejected_drone_predictions"],
                                                "promoted_drone_predictions_pre_score": ev["filter_summary"]["promoted_drone_predictions"],
                                            }
                                        )

    raw_recall = float(rows[0]["raw_recall"])
    raw_precision = float(rows[0]["raw_precision"])
    raw_fp = int(rows[0]["raw_fp"])
    deployable = [
        row for row in rows
        if float(row["recall"]) >= raw_recall - max_recall_drop
        and int(row["fp"]) <= raw_fp
        and float(row["precision"]) >= raw_precision
    ]
    selected = sorted(deployable or rows, key=lambda r: _selection_key(r, raw_recall, raw_precision, raw_fp, max_recall_drop), reverse=True)[0]
    selected["passes_calibration"] = bool(deployable)
    if not deployable:
        selected["not_recommended_reason"] = "No sequence candidate preserved recall while keeping FP and precision no worse than raw."
    selected_weights = out_dir / "selected_tracklet_sequence_classifier.pt"
    shutil.copy2(selected["weights"], selected_weights)
    selected["selected_weights"] = str(selected_weights)

    _write_csv(out_dir / "sequence_model_selection.csv", rows)
    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "run_roots": [str(path) for path in run_roots],
        "gt_csv": str(gt_csv),
        "profile": profile,
        "split": split_summary,
        "num_candidates": len(rows),
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "max_frames": max_frames,
        "max_recall_drop": max_recall_drop,
        "raw_calibration": {
            "tp": int(selected["raw_tp"]),
            "fp": int(selected["raw_fp"]),
            "fn": int(selected["raw_fn"]),
            "precision": float(selected["raw_precision"]),
            "recall": float(selected["raw_recall"]),
        },
        "selected": selected,
    }
    (out_dir / "sequence_model_selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
