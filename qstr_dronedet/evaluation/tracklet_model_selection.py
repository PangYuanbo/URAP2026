from __future__ import annotations

import csv
import fnmatch
import json
import shutil
from pathlib import Path
from typing import Any

from qstr_dronedet.evaluation.tracklet_filter_sweep import _evaluate_rows, _load_gt_csv, _load_run_rows
from qstr_dronedet.tracking.tracklet_classifier import (
    TRACKLET_FEATURES,
    filter_infer_rows_with_tracklet_classifier,
    train_tracklet_classifier,
)


def _read_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _matches_any(value: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(value, pattern) or pattern in value for pattern in patterns)


def _split_by_sequence(
    rows: list[dict[str, str]],
    calib_seqs: list[str] | None,
    calib_seq_patterns: list[str] | None,
    calib_fraction: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    seqs = sorted({str(row.get("seq", "")) for row in rows if row.get("seq", "")})
    if not seqs:
        raise ValueError("Tracklet CSV must include a non-empty seq column for calibration splitting")

    requested = set(calib_seqs or [])
    patterns = calib_seq_patterns or []
    if requested or patterns:
        calib_set = {seq for seq in seqs if seq in requested or _matches_any(seq, patterns)}
    else:
        n_calib = max(1, int(round(len(seqs) * calib_fraction)))
        n_calib = min(max(1, n_calib), max(1, len(seqs) - 1))
        calib_set = set(seqs[-n_calib:])

    if not calib_set:
        raise ValueError("Calibration split is empty; adjust --calib-seqs or --calib-seq-patterns")
    if len(calib_set) == len(seqs):
        raise ValueError("Calibration split consumed all sequences; leave at least one train sequence")

    train_rows = [row for row in rows if str(row.get("seq", "")) not in calib_set]
    calib_rows = [row for row in rows if str(row.get("seq", "")) in calib_set]
    return train_rows, calib_rows, {
        "train_sequences": sorted({str(row.get("seq", "")) for row in train_rows}),
        "calibration_sequences": sorted(calib_set),
        "num_train_rows": len(train_rows),
        "num_calibration_rows": len(calib_rows),
    }


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def _is_hard_negative(row: dict[str, str]) -> bool:
    if int(float(row.get("label", "0") or 0)) != 0:
        return False
    branch_like = max(_float(row, "mean_crop_drone"), _float(row, "mean_temporal_drone"), _float(row, "mean_final_drone"))
    background = _float(row, "mean_background")
    final_score = _float(row, "max_final_score")
    fallback = _float(row, "fallback_rate")
    weak_temporal = _float(row, "weak_detector_temporal_signal")
    return (
        branch_like >= 0.30
        or final_score >= 0.12
        or (fallback > 0.0 and background < 0.75)
        or weak_temporal >= 0.05
    )


def _augment_hard_negatives(rows: list[dict[str, str]], repeats: int) -> tuple[list[dict[str, str]], int]:
    if repeats <= 0:
        return rows, 0
    augmented: list[dict[str, str]] = []
    for row in rows:
        if not _is_hard_negative(row):
            continue
        for i in range(repeats):
            dup = dict(row)
            dup["track_id"] = f"{row.get('track_id', '')}__hardneg_aug{i + 1}"
            augmented.append(dup)
    return rows + augmented, len(augmented)


def _evaluate_checkpoint_on_frames(
    run_roots: list[str | Path],
    gt_csv: str | Path,
    weights: str | Path,
    profile: str,
    allowed_sequences: set[str],
    classifier_threshold: float,
    promotion_enabled: bool,
    promotion_score_floor: float,
    promotion_max_background: float,
    promotion_min_branch_drone: float,
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
    filtered, _, filter_summary = filter_infer_rows_with_tracklet_classifier(
        pred_rows,
        diag_rows,
        weights,
        threshold=classifier_threshold,
        promote_positive_tracklets=promotion_enabled,
        promotion_score_floor=promotion_score_floor,
        promotion_min_branch_drone=promotion_min_branch_drone,
        promotion_max_background=promotion_max_background,
        selective_promotion=selective_promotion,
        selective_max_promoted_tracklets_per_sequence=selective_max_promoted_tracklets_per_sequence,
    )
    metrics = _evaluate_rows(filtered, gt_rows, score_threshold, iou_threshold)
    return {
        "sequences": sorted(active),
        "raw": raw_metrics,
        "filtered": metrics,
        "filter_summary": filter_summary,
        "delta_tp": metrics["tp"] - raw_metrics["tp"],
        "delta_fp": metrics["fp"] - raw_metrics["fp"],
        "delta_fn": metrics["fn"] - raw_metrics["fn"],
        "delta_recall": metrics["recall"] - raw_metrics["recall"],
        "delta_precision": metrics["precision"] - raw_metrics["precision"],
    }


def _selection_key(row: dict[str, Any], raw_recall: float, max_recall_drop: float) -> tuple[float, float, float, float]:
    recall = float(row["recall"])
    precision = float(row["precision"])
    fp = float(row["fp"])
    tp = float(row["tp"])
    recall_ok = 1.0 if recall >= raw_recall - max_recall_drop else 0.0
    return (recall_ok, precision, -fp, tp)


def run_tracklet_model_selection(
    tracklet_csv: str | Path,
    run_roots: list[str | Path],
    gt_csv: str | Path,
    out: str | Path,
    profile: str = "hard_recovery",
    calib_seqs: list[str] | None = None,
    calib_seq_patterns: list[str] | None = None,
    calib_fraction: float = 0.4,
    epochs_values: list[int] | None = None,
    hidden_values: list[int] | None = None,
    lr_values: list[float] | None = None,
    hard_tiny_positive_augments_values: list[int] | None = None,
    hard_negative_augments_values: list[int] | None = None,
    classifier_thresholds: list[float] | None = None,
    promotion_enabled_values: list[bool] | None = None,
    promotion_score_floors: list[float] | None = None,
    promotion_max_backgrounds: list[float] | None = None,
    promotion_min_branch_drone: float = 0.40,
    selective_promotion: bool = True,
    selective_max_promoted_tracklets_per_sequence_values: list[int] | None = None,
    score_threshold: float = 0.20,
    iou_threshold: float = 0.30,
    max_frames: int | None = None,
    max_recall_drop: float = 0.02,
) -> dict[str, Any]:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, fields = _read_csv_rows(tracklet_csv)
    missing = [feature for feature in TRACKLET_FEATURES if feature not in fields]
    if missing:
        raise ValueError(f"Tracklet CSV is missing features required by current classifier: {missing[:5]}")

    train_rows_base, calib_rows, split_summary = _split_by_sequence(
        rows,
        calib_seqs=calib_seqs,
        calib_seq_patterns=calib_seq_patterns,
        calib_fraction=calib_fraction,
    )
    _write_csv_rows(out_dir / "calibration_tracklets.csv", calib_rows, fields)

    epochs_values = epochs_values or [30, 60]
    hidden_values = hidden_values or [32]
    lr_values = lr_values or [1e-3]
    hard_tiny_positive_augments_values = hard_tiny_positive_augments_values or [0, 2]
    hard_negative_augments_values = hard_negative_augments_values or [0, 2]
    classifier_thresholds = classifier_thresholds or [0.50, 0.70, 0.85]
    promotion_enabled_values = promotion_enabled_values if promotion_enabled_values is not None else [False, True]
    promotion_score_floors = promotion_score_floors or [0.22, 0.30]
    promotion_max_backgrounds = promotion_max_backgrounds or [0.55, 0.60]
    selective_max_promoted_tracklets_per_sequence_values = selective_max_promoted_tracklets_per_sequence_values or [1, 2]

    calib_sequences = set(split_summary["calibration_sequences"])
    candidate_rows: list[dict[str, Any]] = []
    candidate_idx = 0

    for hard_neg_aug in hard_negative_augments_values:
        train_rows, num_hard_neg_augmented = _augment_hard_negatives(train_rows_base, hard_neg_aug)
        train_csv = out_dir / f"train_tracklets_hn{hard_neg_aug}.csv"
        _write_csv_rows(train_csv, train_rows, fields)
        for epochs in epochs_values:
            for hidden in hidden_values:
                for lr in lr_values:
                    for hard_pos_aug in hard_tiny_positive_augments_values:
                        candidate_idx += 1
                        ckpt = out_dir / "checkpoints" / f"tracklet_v3_candidate_{candidate_idx:03d}.pt"
                        train_tracklet_classifier(
                            train_csv,
                            ckpt,
                            epochs=epochs,
                            lr=lr,
                            hidden=hidden,
                            hard_tiny_positive_augments=hard_pos_aug,
                        )
                        for threshold in classifier_thresholds:
                            for promote in promotion_enabled_values:
                                floors = promotion_score_floors if promote else [0.0]
                                backgrounds = promotion_max_backgrounds if promote else [0.0]
                                budgets = selective_max_promoted_tracklets_per_sequence_values if (promote and selective_promotion) else [0]
                                for floor in floors:
                                    for max_bg in backgrounds:
                                        for budget in budgets:
                                            evaluation = _evaluate_checkpoint_on_frames(
                                                run_roots,
                                                gt_csv,
                                                ckpt,
                                                profile=profile,
                                                allowed_sequences=calib_sequences,
                                                classifier_threshold=threshold,
                                                promotion_enabled=promote,
                                                promotion_score_floor=floor,
                                                promotion_max_background=max_bg,
                                                promotion_min_branch_drone=promotion_min_branch_drone,
                                                selective_promotion=bool(promote and selective_promotion),
                                                selective_max_promoted_tracklets_per_sequence=budget,
                                                score_threshold=score_threshold,
                                                iou_threshold=iou_threshold,
                                                max_frames=max_frames,
                                            )
                                            metrics = evaluation["filtered"]
                                            raw = evaluation["raw"]
                                            candidate_rows.append(
                                                {
                                                    "candidate_id": candidate_idx,
                                                    "weights": str(ckpt),
                                                    "epochs": epochs,
                                                    "hidden": hidden,
                                                    "lr": lr,
                                                    "hard_tiny_positive_augments": hard_pos_aug,
                                                    "hard_negative_augments": hard_neg_aug,
                                                    "num_hard_negative_augmented": num_hard_neg_augmented,
                                                    "classifier_threshold": threshold,
                                                    "promotion_enabled": int(promote),
                                                    "promotion_score_floor": floor,
                                                    "promotion_max_background": max_bg,
                                                    "promotion_min_branch_drone": promotion_min_branch_drone,
                                                    "selective_promotion": int(bool(promote and selective_promotion)),
                                                    "selective_max_promoted_tracklets_per_sequence": budget,
                                                    **metrics,
                                                    "raw_tp": raw["tp"],
                                                    "raw_fp": raw["fp"],
                                                    "raw_fn": raw["fn"],
                                                    "raw_precision": raw["precision"],
                                                    "raw_recall": raw["recall"],
                                                    "delta_tp": evaluation["delta_tp"],
                                                    "delta_fp": evaluation["delta_fp"],
                                                    "delta_fn": evaluation["delta_fn"],
                                                    "delta_recall": evaluation["delta_recall"],
                                                    "delta_precision": evaluation["delta_precision"],
                                                    "filtered_drone_predictions_pre_score": evaluation["filter_summary"]["filtered_drone_predictions"],
                                                    "rejected_drone_predictions_pre_score": evaluation["filter_summary"]["rejected_drone_predictions"],
                                                    "promoted_drone_predictions_pre_score": evaluation["filter_summary"]["promoted_drone_predictions"],
                                                }
                                            )

    if not candidate_rows:
        raise ValueError("No model-selection candidates were evaluated")

    raw_recall = float(candidate_rows[0]["raw_recall"])
    raw_precision = float(candidate_rows[0]["raw_precision"])
    raw_fp = int(candidate_rows[0]["raw_fp"])
    deployable = [
        row for row in candidate_rows
        if float(row["recall"]) >= raw_recall - max_recall_drop
        and int(row["fp"]) <= raw_fp
        and float(row["precision"]) >= raw_precision
    ]
    selected = sorted(deployable or candidate_rows, key=lambda row: _selection_key(row, raw_recall, max_recall_drop), reverse=True)[0]
    selected["passes_calibration"] = bool(deployable)
    if not deployable:
        selected["not_recommended_reason"] = (
            "No candidate matched raw recall within max_recall_drop while also keeping FP and precision no worse than raw."
        )
    fp_control_candidates = [row for row in candidate_rows if int(row["fp"]) <= raw_fp]
    selected_fp_control = sorted(fp_control_candidates or candidate_rows, key=lambda row: (float(row["precision"]), -int(row["fp"]), float(row["recall"])), reverse=True)[0]
    selected_high_recall = sorted(candidate_rows, key=lambda row: (float(row["recall"]), -int(row["fp"]), float(row["precision"])), reverse=True)[0]
    selected_weights = out_dir / "selected_tracklet_classifier.pt"
    shutil.copy2(selected["weights"], selected_weights)
    selected["selected_weights"] = str(selected_weights)

    result_fields: list[str] = []
    for row in candidate_rows:
        for key in row:
            if key not in result_fields:
                result_fields.append(key)
    _write_csv_rows(out_dir / "model_selection.csv", candidate_rows, result_fields)
    summary = {
        "tracklet_csv": str(tracklet_csv),
        "run_roots": [str(path) for path in run_roots],
        "gt_csv": str(gt_csv),
        "profile": profile,
        "split": split_summary,
        "num_candidates": len(candidate_rows),
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
        "selected_fp_control": selected_fp_control,
        "selected_high_recall": selected_high_recall,
    }
    (out_dir / "model_selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
