from __future__ import annotations

import csv
import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from qstr_dronedet.candidates.merge import bbox_iou, center_distance

TRACKLET_FEATURES = [
    "num_rows",
    "mean_objectness",
    "max_objectness",
    "mean_final_score",
    "max_final_score",
    "mean_crop_drone",
    "mean_temporal_drone",
    "mean_final_drone",
    "mean_background",
    "temporal_minus_crop_mean",
    "temporal_minus_background_mean",
    "final_minus_background_mean",
    "temporal_gain_rate",
    "detector_update_rate",
    "fallback_rate",
    "validated_rate",
    "mean_track_drift",
    "max_track_drift",
    "mean_track_speed",
    "mean_box_side",
    "std_box_side",
    "mean_center_step",
    "max_center_step",
    "std_center_step",
    "track_span_frames",
    "frame_density",
    "weak_detector_temporal_signal",
    "score_above_02_rate",
    "score_slope",
    "objectness_slope",
    "temporal_drone_slope",
    "background_slope",
    "final_margin_mean",
    "final_margin_min",
    "final_margin_slope",
    "background_dominance_rate",
    "background_dominance_longest_streak",
    "temporal_over_background_rate",
    "temporal_over_background_longest_streak",
    "score_above_02_longest_streak",
    "max_frame_gap",
    "mean_frame_gap",
    "gap_rate",
    "first_final_score",
    "last_final_score",
    "first_background",
    "last_background",
    "mean_action_dynamics_score",
    "min_action_dynamics_score",
    "mean_action_error_improvement_vs_cv",
    "mean_action_learned_center_error",
    "mean_action_frame_prior_score",
    "max_action_frame_prior_score",
    "action_frame_prior_coverage_rate",
    "mean_action_frame_prior_tracklet_support",
]


@dataclass
class TrackletDatasetResult:
    csv_path: Path
    json_path: Path
    summary: dict[str, Any]


@dataclass
class TrackletClassifierThresholdSweepResult:
    csv_path: Path
    summary_path: Path
    summary: dict[str, Any]


@dataclass
class TrackletClassifierMergeResult:
    csv_path: Path
    manifest_path: Path
    summary: dict[str, Any]


@dataclass
class TrackletClassifierBenchmarkResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class TrackletClassifierFrameBenchmarkResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class TrackletClassifierFramePreflightResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class TrackletClassifierOfficialEvalBundleResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class TrackletClassifierOfficialPredictionExportResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class TrackletClassifierAotPredictionExportResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class TrackletClassifierAotEvalPreflightResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class TrackletClassifierAotTrackletExportResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class TrackletClassifierAotTrackletFilterResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class TrackletClassifierAotTrackletRescoreResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class TrackletClassifierMixturePreflightResult:
    out_path: Path
    summary: dict[str, Any]


class TrackletMLP(torch.nn.Module):
    def __init__(self, in_dim: int = len(TRACKLET_FEATURES), hidden: int = 32) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_checkpoint(weights: str | Path) -> tuple[TrackletMLP, list[str], torch.Tensor, torch.Tensor]:
    ckpt = torch.load(weights, map_location="cpu")
    features = list(ckpt.get("features", TRACKLET_FEATURES))
    hidden = int(ckpt.get("hidden", ckpt["state_dict"]["net.0.weight"].shape[0]))
    model = TrackletMLP(in_dim=len(features), hidden=hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, features, ckpt["mean"], ckpt["std"].clamp_min(1e-6)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_gt_csv(path: str | Path, max_frames: int | None = None) -> dict[tuple[str, int], list[tuple[float, float, float, float]]]:
    out: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            frame_id = int(float(row["frame_id"]))
            if max_frames is not None and frame_id >= max_frames:
                continue
            seq = Path(row["video_path"]).parent.name
            box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            out.setdefault((seq, frame_id), []).append(box)
    return out


def _row_track_key(row: dict[str, Any], scope_by_seq: bool = True) -> str | None:
    track_id = row.get("track_id")
    if track_id is None or track_id == "":
        return None
    if scope_by_seq:
        seq = row.get("seq")
        if seq is not None and seq != "":
            return f"{seq}:{track_id}"
    return str(track_id)


def _box_side(row: dict[str, Any]) -> float:
    x1, y1, x2, y2 = [float(v) for v in row.get("bbox", [0, 0, 0, 0])]
    return max(0.0, max(x2 - x1, y2 - y1))


def _box_center(row: dict[str, Any]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in row.get("bbox", [0, 0, 0, 0])]
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _prob(row: dict[str, Any], branch: str, key: str) -> float:
    return float((row.get(branch) or {}).get(key, 0.0))


def _safe_mean(vals: list[float], default: float = 0.0) -> float:
    return float(np.mean(vals)) if vals else default


def _safe_max(vals: list[float], default: float = 0.0) -> float:
    return float(np.max(vals)) if vals else default


def _safe_std(vals: list[float], default: float = 0.0) -> float:
    return float(np.std(vals)) if vals else default


def _safe_slope(xs: list[int], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    x = x - float(x.mean())
    denom = float(np.sum(x * x))
    if denom <= 1e-6:
        return 0.0
    return float(np.sum(x * (y - float(y.mean()))) / denom)


def _longest_true_streak(flags: list[bool]) -> float:
    best = 0
    cur = 0
    for flag in flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best)


def _tracklet_label(seq: str, rows: list[dict[str, Any]], gt_by_key: dict[tuple[str, int], list[tuple[float, float, float, float]]], iou_threshold: float, center_threshold: float) -> tuple[int, float, int]:
    best = 0.0
    matched = 0
    for row in rows:
        frame_id = int(row.get("frame_id", -1))
        box = tuple(float(v) for v in row.get("bbox", [0, 0, 0, 0]))
        frame_best = 0.0
        frame_match = False
        for gt in gt_by_key.get((seq, frame_id), []):
            ov = bbox_iou(box, gt)
            dist = center_distance(box, gt)
            frame_best = max(frame_best, ov)
            if ov >= iou_threshold or dist <= center_threshold:
                frame_match = True
        best = max(best, frame_best)
        matched += int(frame_match)
    return int(matched > 0), best, matched


def _features(rows: list[dict[str, Any]]) -> dict[str, float]:
    rows = sorted(rows, key=lambda r: int(r.get("frame_id", 0)))
    crop_drone = [_prob(r, "crop_probs", "drone") for r in rows]
    temp_drone = [_prob(r, "temporal_probs", "drone") for r in rows]
    final_drone = [_prob(r, "final_probs", "drone") for r in rows]
    bg = [max(_prob(r, "crop_probs", "background"), _prob(r, "temporal_probs", "background"), _prob(r, "final_probs", "background")) for r in rows]
    objectness = [float(r.get("objectness", 0.0)) for r in rows]
    final_scores = [float(r.get("final_drone_score", 0.0)) for r in rows]
    action_scores = [float(r.get("action_dynamics_score", 0.0) or 0.0) for r in rows]
    action_improvements = [float(r.get("action_error_improvement_vs_cv", 0.0) or 0.0) for r in rows]
    action_errors = [float(r.get("action_mean_learned_center_error", 0.0) or 0.0) for r in rows]
    action_prior_scores = [float(r.get("action_frame_prior_score", 0.0) or 0.0) for r in rows]
    action_prior_support = [float(r.get("action_frame_prior_num_tracklet_priors", 0.0) or 0.0) for r in rows]
    drifts = [float(r.get("track_drift", 0.0) or 0.0) for r in rows]
    speeds = [float(r.get("track_speed", 0.0) or 0.0) for r in rows]
    sides = [_box_side(r) for r in rows]
    sources = [str(r.get("source", "")) for r in rows]
    centers = [_box_center(r) for r in rows]
    center_steps = [
        float(np.hypot(centers[i][0] - centers[i - 1][0], centers[i][1] - centers[i - 1][1]))
        for i in range(1, len(centers))
    ]
    frame_ids = [int(r.get("frame_id", 0)) for r in rows]
    track_span = float(max(frame_ids) - min(frame_ids) + 1) if frame_ids else 0.0
    frame_gaps = [max(0, frame_ids[i] - frame_ids[i - 1] - 1) for i in range(1, len(frame_ids))]
    detector_update_rate = _safe_mean([float("yolo" in s or "fallback" in s or "motion" in s or "seed" in s) for s in sources])
    validated_rate = _safe_mean([float(bool(r.get("track_validated", False))) for r in rows])
    temporal_gain_rate = _safe_mean([float(t > c + 0.05) for c, t in zip(crop_drone, temp_drone)])
    final_margin = [d - b for d, b in zip(final_drone, bg)]
    temporal_over_background = [t > b for t, b in zip(temp_drone, bg)]
    background_dominance = [b >= max(c, t, d) for b, c, t, d in zip(bg, crop_drone, temp_drone, final_drone)]
    score_above_02 = [s >= 0.2 for s in final_scores]
    out = {
        "num_rows": float(len(rows)),
        "mean_objectness": _safe_mean(objectness),
        "max_objectness": _safe_max(objectness),
        "mean_final_score": _safe_mean(final_scores),
        "max_final_score": _safe_max(final_scores),
        "mean_crop_drone": _safe_mean(crop_drone),
        "mean_temporal_drone": _safe_mean(temp_drone),
        "mean_final_drone": _safe_mean(final_drone),
        "mean_background": _safe_mean(bg),
        "temporal_minus_crop_mean": _safe_mean([t - c for c, t in zip(crop_drone, temp_drone)]),
        "temporal_minus_background_mean": _safe_mean([t - b for t, b in zip(temp_drone, bg)]),
        "final_minus_background_mean": _safe_mean([d - b for d, b in zip(final_drone, bg)]),
        "temporal_gain_rate": temporal_gain_rate,
        "detector_update_rate": detector_update_rate,
        "fallback_rate": _safe_mean([float("fallback" in s) for s in sources]),
        "validated_rate": validated_rate,
        "mean_track_drift": _safe_mean(drifts),
        "max_track_drift": _safe_max(drifts),
        "mean_track_speed": _safe_mean(speeds),
        "mean_box_side": _safe_mean(sides),
        "std_box_side": _safe_std(sides),
        "mean_center_step": _safe_mean(center_steps),
        "max_center_step": _safe_max(center_steps),
        "std_center_step": _safe_std(center_steps),
        "track_span_frames": track_span,
        "frame_density": float(len(rows)) / max(1.0, track_span),
        "weak_detector_temporal_signal": temporal_gain_rate * (1.0 - detector_update_rate) * (1.0 - validated_rate),
        "score_above_02_rate": _safe_mean([float(s >= 0.2) for s in final_scores]),
        "score_slope": _safe_slope(frame_ids, final_scores),
        "objectness_slope": _safe_slope(frame_ids, objectness),
        "temporal_drone_slope": _safe_slope(frame_ids, temp_drone),
        "background_slope": _safe_slope(frame_ids, bg),
        "final_margin_mean": _safe_mean(final_margin),
        "final_margin_min": float(min(final_margin)) if final_margin else 0.0,
        "final_margin_slope": _safe_slope(frame_ids, final_margin),
        "background_dominance_rate": _safe_mean([float(v) for v in background_dominance]),
        "background_dominance_longest_streak": _longest_true_streak(background_dominance),
        "temporal_over_background_rate": _safe_mean([float(v) for v in temporal_over_background]),
        "temporal_over_background_longest_streak": _longest_true_streak(temporal_over_background),
        "score_above_02_longest_streak": _longest_true_streak(score_above_02),
        "max_frame_gap": float(max(frame_gaps)) if frame_gaps else 0.0,
        "mean_frame_gap": _safe_mean([float(g) for g in frame_gaps]),
        "gap_rate": _safe_mean([float(g > 0) for g in frame_gaps]),
        "first_final_score": float(final_scores[0]) if final_scores else 0.0,
        "last_final_score": float(final_scores[-1]) if final_scores else 0.0,
        "first_background": float(bg[0]) if bg else 0.0,
        "last_background": float(bg[-1]) if bg else 0.0,
        "mean_action_dynamics_score": _safe_mean(action_scores),
        "min_action_dynamics_score": float(np.min(action_scores)) if action_scores else 0.0,
        "mean_action_error_improvement_vs_cv": _safe_mean(action_improvements),
        "mean_action_learned_center_error": _safe_mean(action_errors),
        "mean_action_frame_prior_score": _safe_mean(action_prior_scores),
        "max_action_frame_prior_score": _safe_max(action_prior_scores),
        "action_frame_prior_coverage_rate": _safe_mean([float(score > 0.0) for score in action_prior_scores]),
        "mean_action_frame_prior_tracklet_support": _safe_mean(action_prior_support),
    }
    return out


def build_tracklet_dataset(
    diagnostics: list[str | Path],
    gt_csv: str | Path,
    out: str | Path,
    max_frames: int | None = None,
    iou_threshold: float = 0.3,
    center_threshold: float = 24.0,
) -> TrackletDatasetResult:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_by_key = _load_gt_csv(gt_csv, max_frames=max_frames)
    tracklets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for diag in diagnostics:
        diag_path = Path(diag)
        seq = diag_path.parent.name
        for row in _load_jsonl(diag_path):
            if max_frames is not None and int(row.get("frame_id", -1)) >= max_frames:
                continue
            key = _row_track_key(row, scope_by_seq=False)
            if key is None:
                continue
            tracklets.setdefault((seq, key), []).append(row)
    csv_path = out_dir / "tracklets.csv"
    json_path = out_dir / "tracklets.jsonl"
    fields = ["seq", "track_id", "label", "best_iou", "matched_frames"] + TRACKLET_FEATURES
    positives = 0
    rows_out = []
    with csv_path.open("w", encoding="utf-8", newline="") as f_csv, json_path.open("w", encoding="utf-8") as f_json:
        writer = csv.DictWriter(f_csv, fieldnames=fields)
        writer.writeheader()
        for (seq, track_id), rows in sorted(tracklets.items()):
            rows = sorted(rows, key=lambda r: int(r.get("frame_id", 0)))
            label, best_iou, matched_frames = _tracklet_label(seq, rows, gt_by_key, iou_threshold, center_threshold)
            positives += int(label)
            feats = _features(rows)
            out_row = {"seq": seq, "track_id": track_id, "label": label, "best_iou": best_iou, "matched_frames": matched_frames, **feats}
            writer.writerow(out_row)
            f_json.write(json.dumps({"meta": out_row, "rows": rows}, ensure_ascii=False) + "\n")
            rows_out.append(out_row)
    summary = {"num_tracklets": len(rows_out), "positives": positives, "negatives": len(rows_out) - positives, "features": TRACKLET_FEATURES}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletDatasetResult(csv_path, json_path, summary)


def export_tracklet_jsonl_classifier_dataset(
    tracklet_jsonl: str | Path,
    out: str | Path,
    dataset_source: str | None = None,
) -> TrackletDatasetResult:
    """Convert nested proposal/tracklet JSONL into the classifier CSV schema."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "tracklets.csv"
    json_path = out_dir / "tracklets.jsonl"
    fields = ["seq", "track_id", "label", "bucket", "dataset_source", "best_iou", "matched_frames"] + TRACKLET_FEATURES
    rows_out: list[dict[str, Any]] = []
    positives = 0
    attached_dynamics = 0
    source_counts: dict[str, int] = {}
    with csv_path.open("w", encoding="utf-8", newline="") as f_csv, json_path.open("w", encoding="utf-8") as f_json:
        writer = csv.DictWriter(f_csv, fieldnames=fields)
        writer.writeheader()
        for item in _load_jsonl(tracklet_jsonl):
            meta = dict(item.get("meta") or {})
            rows = [dict(row) for row in (item.get("rows") or [])]
            if not rows:
                continue
            feats = _features(rows)
            label = int(float(meta.get("label", 0)))
            positives += int(label)
            if any(float(row.get("action_num_windows", 0) or 0) > 0 for row in rows):
                attached_dynamics += 1
            out_row = {
                "seq": str(meta.get("seq", rows[0].get("seq", ""))),
                "track_id": str(meta.get("track_id", rows[0].get("track_id", ""))),
                "label": label,
                "bucket": str(meta.get("bucket", "")),
                "dataset_source": str(dataset_source or meta.get("dataset_source", rows[0].get("dataset_source", ""))),
                "best_iou": float(meta.get("best_iou", 0.0) or 0.0),
                "matched_frames": int(float(meta.get("matched_frames", 0) or 0)),
                **feats,
            }
            source_counts[out_row["dataset_source"]] = source_counts.get(out_row["dataset_source"], 0) + 1
            writer.writerow(out_row)
            f_json.write(json.dumps({"meta": out_row, "rows": rows}, ensure_ascii=False) + "\n")
            rows_out.append(out_row)
    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "dataset_source": dataset_source,
        "num_tracklets": len(rows_out),
        "positives": positives,
        "negatives": len(rows_out) - positives,
        "tracklets_with_action_dynamics": attached_dynamics,
        "dataset_source_counts": dict(sorted(source_counts.items())),
        "features": TRACKLET_FEATURES,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletDatasetResult(csv_path, json_path, summary)


def score_tracklets_from_rows(rows: list[dict[str, Any]], weights: str | Path, threshold: float = 0.5) -> dict[str, dict[str, Any]]:
    tracklets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = _row_track_key(row)
        if key is None:
            continue
        tracklets.setdefault(key, []).append(row)
    if not tracklets:
        return {}

    model, features, mean, std = _load_checkpoint(weights)
    ordered = []
    matrix = []
    for track_id, track_rows in sorted(tracklets.items(), key=lambda kv: kv[0]):
        feats = _features(track_rows)
        ordered.append((track_id, feats, len(track_rows)))
        matrix.append([float(feats.get(k, 0.0)) for k in features])
    x = torch.tensor(matrix, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(model((x - mean) / std), dim=1)[:, 1].tolist()
    scores: dict[str, dict[str, Any]] = {}
    for (track_id, feats, n), prob in zip(ordered, probs):
        scores[track_id] = {
            "prob_tracklet_drone": float(prob),
            "tracklet_is_drone": bool(prob >= threshold),
            "num_rows": n,
            "features": feats,
        }
    return scores


def _append_cause(cause: Any, value: str) -> str:
    if cause is None or cause == "":
        return value
    text = str(cause)
    if value in text.split("+"):
        return text
    return f"{text}+{value}"


def _row_seq(row: dict[str, Any]) -> str:
    seq = row.get("seq")
    return str(seq) if seq is not None and seq != "" else "__single_sequence__"


def _promotion_evidence(row: dict[str, Any], score: dict[str, Any]) -> dict[str, float]:
    feats = score.get("features", {})
    crop_drone = _prob(row, "crop_probs", "drone")
    temporal_drone = _prob(row, "temporal_probs", "drone")
    final_drone = _prob(row, "final_probs", "drone")
    background = max(_prob(row, "crop_probs", "background"), _prob(row, "temporal_probs", "background"), _prob(row, "final_probs", "background"))
    tracklet_crop = float(feats.get("mean_crop_drone", 0.0))
    tracklet_temporal = float(feats.get("mean_temporal_drone", 0.0))
    tracklet_final = float(feats.get("mean_final_drone", 0.0))
    tracklet_background = float(feats.get("mean_background", 1.0))
    branch_drone = max(crop_drone, temporal_drone, final_drone, tracklet_crop, tracklet_temporal, tracklet_final)
    effective_background = min(background if background > 0 else 1.0, tracklet_background)
    effective_temporal = max(temporal_drone, tracklet_temporal)
    effective_crop = max(crop_drone, tracklet_crop)
    source = str(row.get("source", ""))
    has_recovery_source = float("fallback" in source or "tracker" in source or float(feats.get("fallback_rate", 0.0)) > 0.0)
    return {
        "branch_drone": branch_drone,
        "effective_background": effective_background,
        "temporal_crop_delta": effective_temporal - effective_crop,
        "temporal_background_margin": effective_temporal - effective_background,
        "tracklet_background": tracklet_background,
        "tracklet_rows": float(feats.get("num_rows", 0.0)),
        "tracklet_prob": float(score.get("prob_tracklet_drone", 0.0)),
        "fallback_rate": float(feats.get("fallback_rate", 0.0)),
        "detector_update_rate": float(feats.get("detector_update_rate", 0.0)),
        "temporal_gain_rate": float(feats.get("temporal_gain_rate", 0.0)),
        "weak_detector_temporal_signal": float(feats.get("weak_detector_temporal_signal", 0.0)),
        "max_objectness": float(feats.get("max_objectness", 0.0)),
        "has_recovery_source": has_recovery_source,
    }


def _selective_tracklet_key_allowlist(
    rows: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    promotion_min_branch_drone: float,
    promotion_max_background: float,
    min_temporal_crop_delta: float,
    min_temporal_background_margin: float,
    max_tracklet_background: float,
    max_tracklet_objectness: float,
    min_tracklet_rows: int,
    min_temporal_gain_rate: float,
    min_weak_detector_temporal_signal: float,
    require_recovery_source: bool,
    max_promoted_tracklets_per_sequence: int,
) -> set[str]:
    candidates_by_seq: dict[str, dict[str, float]] = {}
    for row in rows:
        key = _row_track_key(row)
        if key is None or row.get("predicted_class") == "drone":
            continue
        score = scores.get(key)
        if not score or not bool(score.get("tracklet_is_drone", False)):
            continue
        evidence = _promotion_evidence(row, score)
        if evidence["branch_drone"] < promotion_min_branch_drone:
            continue
        if evidence["effective_background"] > promotion_max_background:
            continue
        if evidence["temporal_crop_delta"] < min_temporal_crop_delta:
            continue
        if evidence["temporal_background_margin"] < min_temporal_background_margin:
            continue
        if evidence["tracklet_background"] > max_tracklet_background:
            continue
        if evidence["max_objectness"] > max_tracklet_objectness and evidence["fallback_rate"] <= 0.0:
            continue
        if evidence["tracklet_rows"] < float(min_tracklet_rows):
            continue
        if evidence["temporal_gain_rate"] < min_temporal_gain_rate and evidence["weak_detector_temporal_signal"] < min_weak_detector_temporal_signal:
            continue
        if require_recovery_source and evidence["has_recovery_source"] <= 0.0:
            continue
        rank_score = (
            evidence["tracklet_prob"]
            + 0.5 * evidence["temporal_background_margin"]
            + 0.25 * evidence["temporal_crop_delta"]
            + 0.20 * evidence["temporal_gain_rate"]
            - 0.30 * evidence["effective_background"]
        )
        seq = _row_seq(row)
        current = candidates_by_seq.setdefault(seq, {})
        current[key] = max(current.get(key, -1e9), rank_score)

    allowed: set[str] = set()
    for seq_candidates in candidates_by_seq.values():
        ordered = sorted(seq_candidates.items(), key=lambda item: item[1], reverse=True)
        if max_promoted_tracklets_per_sequence > 0:
            ordered = ordered[:max_promoted_tracklets_per_sequence]
        allowed.update(key for key, _ in ordered)
    return allowed


def apply_tracklet_filter_to_infer_outputs(
    predictions_path: str | Path,
    diagnostics_path: str | Path,
    weights: str | Path,
    threshold: float = 0.5,
    untracked_policy: str = "keep",
    promote_positive_tracklets: bool = True,
    promotion_score_floor: float = 0.22,
    promotion_min_branch_drone: float = 0.40,
    promotion_max_background: float = 0.68,
    selective_promotion: bool = False,
    selective_min_temporal_crop_delta: float = 0.05,
    selective_min_temporal_background_margin: float = -0.05,
    selective_max_tracklet_background: float = 0.60,
    selective_max_tracklet_objectness: float = 0.50,
    selective_min_tracklet_rows: int = 2,
    selective_min_temporal_gain_rate: float = 0.40,
    selective_min_weak_detector_temporal_signal: float = 0.05,
    selective_require_recovery_source: bool = True,
    selective_max_promoted_tracklets_per_sequence: int = 2,
) -> dict[str, Any]:
    pred_path = Path(predictions_path)
    diag_path = Path(diagnostics_path)
    pred_rows = _load_jsonl(pred_path)
    diag_rows = _load_jsonl(diag_path)
    filtered_pred_rows, filtered_diag_rows, summary = filter_infer_rows_with_tracklet_classifier(
        pred_rows,
        diag_rows,
        weights,
        threshold=threshold,
        untracked_policy=untracked_policy,
        promote_positive_tracklets=promote_positive_tracklets,
        promotion_score_floor=promotion_score_floor,
        promotion_min_branch_drone=promotion_min_branch_drone,
        promotion_max_background=promotion_max_background,
        selective_promotion=selective_promotion,
        selective_min_temporal_crop_delta=selective_min_temporal_crop_delta,
        selective_min_temporal_background_margin=selective_min_temporal_background_margin,
        selective_max_tracklet_background=selective_max_tracklet_background,
        selective_max_tracklet_objectness=selective_max_tracklet_objectness,
        selective_min_tracklet_rows=selective_min_tracklet_rows,
        selective_min_temporal_gain_rate=selective_min_temporal_gain_rate,
        selective_min_weak_detector_temporal_signal=selective_min_weak_detector_temporal_signal,
        selective_require_recovery_source=selective_require_recovery_source,
        selective_max_promoted_tracklets_per_sequence=selective_max_promoted_tracklets_per_sequence,
    )

    raw_pred_path = pred_path.with_name("predictions_raw.jsonl")
    raw_diag_path = diag_path.with_name("diagnostics_raw.jsonl")
    pred_path.replace(raw_pred_path)
    diag_path.replace(raw_diag_path)
    with pred_path.open("w", encoding="utf-8") as f:
        for row in filtered_pred_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with diag_path.open("w", encoding="utf-8") as f:
        for row in filtered_diag_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary.update(
        {
            "raw_predictions_path": str(raw_pred_path),
            "raw_diagnostics_path": str(raw_diag_path),
            "predictions_path": str(pred_path),
            "diagnostics_path": str(diag_path),
        }
    )
    (pred_path.parent / "tracklet_filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _discover_infer_pairs(
    run_root: str | Path,
    prediction_name: str = "predictions.jsonl",
    diagnostics_name: str = "diagnostics.jsonl",
) -> list[dict[str, Any]]:
    root = Path(run_root)
    direct_pred = root / prediction_name
    direct_diag = root / diagnostics_name
    if direct_pred.exists() and direct_diag.exists():
        return [{"seq": root.name, "predictions": direct_pred, "diagnostics": direct_diag}]
    pairs = []
    for pred in sorted(root.rglob(prediction_name)):
        diag = pred.parent / diagnostics_name
        if diag.exists():
            pairs.append({"seq": pred.parent.name, "predictions": pred, "diagnostics": diag})
    return pairs


def _summarize_frame_jsonl(path: Path, default_seq: str, max_frames: int | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "rows": 0,
        "drone_rows": 0,
        "track_id_rows": 0,
        "missing_frame_id": 0,
        "missing_bbox": 0,
        "missing_predicted_class": 0,
        "sequences": {},
        "frame_min": None,
        "frame_max": None,
        "errors": [],
    }
    if not path.exists():
        summary["errors"].append(f"missing file: {path}")
        return summary

    try:
        with path.open("r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    summary["errors"].append(f"line {line_no}: invalid json: {exc.msg}")
                    continue
                frame_value = row.get("frame_id")
                try:
                    frame_id = int(frame_value)
                except (TypeError, ValueError):
                    frame_id = None
                    summary["missing_frame_id"] += 1
                if frame_id is not None and max_frames is not None and frame_id >= max_frames:
                    continue
                summary["rows"] += 1
                if row.get("predicted_class") == "drone":
                    summary["drone_rows"] += 1
                if row.get("track_id") not in (None, ""):
                    summary["track_id_rows"] += 1
                if "predicted_class" not in row:
                    summary["missing_predicted_class"] += 1
                bbox = row.get("bbox")
                if not isinstance(bbox, list | tuple) or len(bbox) != 4:
                    summary["missing_bbox"] += 1
                seq = str(row.get("seq") or default_seq)
                seq_counts = summary["sequences"]
                seq_counts[seq] = int(seq_counts.get(seq, 0)) + 1
                if frame_id is not None:
                    summary["frame_min"] = frame_id if summary["frame_min"] is None else min(int(summary["frame_min"]), frame_id)
                    summary["frame_max"] = frame_id if summary["frame_max"] is None else max(int(summary["frame_max"]), frame_id)
    except OSError as exc:
        summary["errors"].append(f"could not read file: {exc}")
    summary["sequences"] = dict(sorted(summary["sequences"].items()))
    return summary


def validate_tracklet_classifier_frame_benchmark_inputs(
    run_roots: list[str | Path],
    gt_csvs: list[str | Path],
    weights: str | Path,
    out: str | Path,
    dataset_names: list[str] | None = None,
    prediction_name: str = "predictions.jsonl",
    diagnostics_name: str = "diagnostics.jsonl",
    thresholds: list[float] | None = None,
    baseline_csv: str | Path | None = None,
    baseline_metric: str = "frame_best_f1",
    max_frames: int | None = None,
    min_prediction_rows: int = 1,
    min_gt_boxes: int = 1,
    require_diagnostics: bool = True,
) -> TrackletClassifierFramePreflightResult:
    errors: list[str] = []
    warnings: list[str] = []
    datasets: list[dict[str, Any]] = []

    if not run_roots:
        errors.append("run_roots must contain at least one inference output root")
    if len(run_roots) != len(gt_csvs):
        errors.append("run_roots and gt_csvs must have the same length")
    if dataset_names is not None and len(dataset_names) != len(run_roots):
        errors.append("dataset_names must be empty or have the same length as run_roots")

    weights_path = Path(weights)
    if not weights_path.exists():
        errors.append(f"weights not found: {weights_path}")

    threshold_values = [float(v) for v in thresholds] if thresholds else []
    nonfinite_thresholds = [value for value in threshold_values if not np.isfinite(value)]
    if nonfinite_thresholds:
        errors.append(f"thresholds contain non-finite values: {nonfinite_thresholds}")

    total_pairs = 0
    total_prediction_rows = 0
    total_drone_rows = 0
    total_track_id_rows = 0
    total_gt_boxes = 0
    combined_prediction_sequences: set[str] = set()
    combined_gt_sequences: set[str] = set()

    pair_count = min(len(run_roots), len(gt_csvs))
    for index in range(pair_count):
        run_root = Path(run_roots[index])
        gt_csv = Path(gt_csvs[index])
        dataset = str(dataset_names[index]) if dataset_names is not None and index < len(dataset_names) else run_root.name
        dataset_summary: dict[str, Any] = {
            "dataset": dataset,
            "run_root": str(run_root),
            "gt_csv": str(gt_csv),
            "exists": run_root.exists(),
            "pairs": [],
            "gt_rows": 0,
            "gt_sequences": {},
            "prediction_sequences": {},
            "sequence_overlap": [],
            "errors": [],
            "warnings": [],
        }

        if not run_root.exists():
            dataset_summary["errors"].append(f"run root not found: {run_root}")
        elif not run_root.is_dir():
            dataset_summary["errors"].append(f"run root is not a directory: {run_root}")
        else:
            pairs = _discover_infer_pairs(run_root, prediction_name=prediction_name, diagnostics_name=diagnostics_name)
            if not pairs:
                loose_predictions = sorted(run_root.rglob(prediction_name))
                if loose_predictions:
                    message = f"found {len(loose_predictions)} prediction files but no matching {diagnostics_name}"
                    target = dataset_summary["errors"] if require_diagnostics else dataset_summary["warnings"]
                    target.append(message)
                else:
                    dataset_summary["errors"].append(f"no {prediction_name}/{diagnostics_name} pairs found")
            for pair in pairs:
                seq = str(pair["seq"])
                pred_summary = _summarize_frame_jsonl(Path(pair["predictions"]), seq, max_frames=max_frames)
                diag_summary = _summarize_frame_jsonl(Path(pair["diagnostics"]), seq, max_frames=max_frames)
                pair_summary = {
                    "seq": seq,
                    "predictions": pred_summary,
                    "diagnostics": diag_summary,
                }
                dataset_summary["pairs"].append(pair_summary)
                total_pairs += 1
                total_prediction_rows += int(pred_summary["rows"])
                total_drone_rows += int(pred_summary["drone_rows"])
                total_track_id_rows += int(pred_summary["track_id_rows"])
                for seq_name in pred_summary.get("sequences", {}):
                    combined_prediction_sequences.add(str(seq_name))
                for error in pred_summary.get("errors", []):
                    dataset_summary["errors"].append(f"{seq} predictions: {error}")
                for error in diag_summary.get("errors", []):
                    target = dataset_summary["errors"] if require_diagnostics else dataset_summary["warnings"]
                    target.append(f"{seq} diagnostics: {error}")
                if int(pred_summary["missing_frame_id"]) > 0:
                    dataset_summary["errors"].append(f"{seq} predictions have missing/invalid frame_id rows")
                if int(pred_summary["missing_bbox"]) > 0:
                    dataset_summary["errors"].append(f"{seq} predictions have missing/invalid bbox rows")
                if int(pred_summary["missing_predicted_class"]) > 0:
                    dataset_summary["warnings"].append(f"{seq} predictions have rows without predicted_class")
                if int(diag_summary["rows"]) == 0 and require_diagnostics:
                    dataset_summary["errors"].append(f"{seq} diagnostics has no rows")

        if not gt_csv.exists():
            dataset_summary["errors"].append(f"GT CSV not found: {gt_csv}")
        else:
            try:
                gt_rows = _load_gt_csv_flat(gt_csv, max_frames=max_frames)
                dataset_summary["gt_rows"] = len(gt_rows)
                total_gt_boxes += len(gt_rows)
                gt_seq_counts: dict[str, int] = {}
                for row in gt_rows:
                    seq_name = str(row["seq"])
                    gt_seq_counts[seq_name] = gt_seq_counts.get(seq_name, 0) + 1
                    combined_gt_sequences.add(seq_name)
                dataset_summary["gt_sequences"] = dict(sorted(gt_seq_counts.items()))
            except Exception as exc:
                dataset_summary["errors"].append(f"could not read GT CSV: {exc}")

        pred_seq_counts: dict[str, int] = {}
        for pair_summary in dataset_summary["pairs"]:
            for seq_name, count in pair_summary["predictions"].get("sequences", {}).items():
                pred_seq_counts[str(seq_name)] = pred_seq_counts.get(str(seq_name), 0) + int(count)
        dataset_summary["prediction_sequences"] = dict(sorted(pred_seq_counts.items()))
        overlap = sorted(set(pred_seq_counts) & set(dataset_summary["gt_sequences"]))
        dataset_summary["sequence_overlap"] = overlap
        if dataset_summary["pairs"] and dataset_summary["gt_rows"] > 0 and not overlap:
            dataset_summary["errors"].append("prediction sequences do not overlap GT sequences")
        if total_prediction_rows < min_prediction_rows and index == pair_count - 1:
            pass

        for item in dataset_summary["errors"]:
            errors.append(f"{dataset}: {item}")
        for item in dataset_summary["warnings"]:
            warnings.append(f"{dataset}: {item}")
        datasets.append(dataset_summary)

    if total_prediction_rows < min_prediction_rows:
        errors.append(f"combined prediction rows {total_prediction_rows} < required {min_prediction_rows}")
    if total_gt_boxes < min_gt_boxes:
        errors.append(f"combined GT boxes {total_gt_boxes} < required {min_gt_boxes}")
    if total_pairs <= 0:
        errors.append("no prediction/diagnostics pairs found across all run roots")

    baseline_validation = None
    if baseline_csv:
        baseline_path = Path(baseline_csv)
        if not baseline_path.exists():
            errors.append(f"baseline CSV not found: {baseline_path}")
        else:
            try:
                from qstr_dronedet.tracking.action_policy import validate_route_b_baseline_csv

                baseline_out = Path(out).with_suffix(Path(out).suffix + ".baseline_validation.json")
                baseline_result = validate_route_b_baseline_csv(
                    baseline_path,
                    baseline_out,
                    metric=baseline_metric,
                    require_metric_values=True,
                )
                baseline_validation = baseline_result.summary
                if not baseline_result.summary.get("valid", False):
                    errors.append(f"baseline CSV invalid for metric {baseline_metric}")
            except Exception as exc:
                errors.append(f"could not validate baseline CSV: {exc}")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "out": str(out_path),
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "requirements": {
            "min_prediction_rows": min_prediction_rows,
            "min_gt_boxes": min_gt_boxes,
            "require_diagnostics": require_diagnostics,
            "max_frames": max_frames,
        },
        "inputs": {
            "run_roots": [str(path) for path in run_roots],
            "gt_csvs": [str(path) for path in gt_csvs],
            "weights": str(weights_path),
            "dataset_names": dataset_names,
            "prediction_name": prediction_name,
            "diagnostics_name": diagnostics_name,
            "thresholds": threshold_values,
            "baseline_csv": str(baseline_csv) if baseline_csv else None,
            "baseline_metric": baseline_metric,
        },
        "combined": {
            "datasets": len(datasets),
            "pairs": total_pairs,
            "prediction_rows": total_prediction_rows,
            "drone_prediction_rows": total_drone_rows,
            "track_id_prediction_rows": total_track_id_rows,
            "gt_boxes": total_gt_boxes,
            "prediction_sequences": sorted(combined_prediction_sequences),
            "gt_sequences": sorted(combined_gt_sequences),
            "sequence_overlap": sorted(combined_prediction_sequences & combined_gt_sequences),
        },
        "baseline_validation": baseline_validation,
        "datasets": datasets,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletClassifierFramePreflightResult(out_path=out_path, summary=summary)


def _copy_and_enrich_jsonl(src: Path, dst: Path, default_seq: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8-sig") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("seq") in (None, ""):
                row["seq"] = default_seq
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_gt_csv_flat(path: str | Path, max_frames: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            frame_id = int(float(row["frame_id"]))
            if max_frames is not None and frame_id >= max_frames:
                continue
            seq = str(row.get("seq") or Path(row.get("video_path", "")).parent.name)
            rows.append(
                {
                    "seq": seq,
                    "frame_id": frame_id,
                    "bbox": (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
                }
            )
    return rows


def evaluate_frame_predictions_against_gt_csv(
    pred_rows: list[dict[str, Any]],
    gt_csv: str | Path,
    seqs: set[str] | None = None,
    iou_threshold: float = 0.3,
    score_threshold: float = 0.0,
    max_frames: int | None = None,
) -> dict[str, Any]:
    gt_rows = _load_gt_csv_flat(gt_csv, max_frames=max_frames)
    if seqs is not None:
        gt_rows = [row for row in gt_rows if str(row["seq"]) in seqs]
    gt_by_key: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
    for row in gt_rows:
        gt_by_key.setdefault((str(row["seq"]), int(row["frame_id"])), []).append(tuple(row["bbox"]))

    pred_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in pred_rows:
        if row.get("predicted_class") != "drone":
            continue
        if float(row.get("final_drone_score", 0.0) or 0.0) < score_threshold:
            continue
        frame_id = int(row.get("frame_id", -1))
        if max_frames is not None and frame_id >= max_frames:
            continue
        seq = str(row.get("seq") or "")
        if seqs is not None and seq not in seqs:
            continue
        pred_by_key.setdefault((seq, frame_id), []).append(row)

    tp = 0
    fp = 0
    fn = 0
    matched_gt = 0
    keys = sorted(set(gt_by_key) | set(pred_by_key))
    for key in keys:
        gt_boxes = list(gt_by_key.get(key, []))
        preds = sorted(pred_by_key.get(key, []), key=lambda row: float(row.get("final_drone_score", 0.0) or 0.0), reverse=True)
        used = [False] * len(gt_boxes)
        for pred in preds:
            box = tuple(float(v) for v in pred.get("bbox", [0, 0, 0, 0]))
            best_i = -1
            best_iou = 0.0
            for index, gt in enumerate(gt_boxes):
                if used[index]:
                    continue
                ov = bbox_iou(box, gt)
                if ov > best_iou:
                    best_i = index
                    best_iou = ov
            if best_i >= 0 and best_iou >= iou_threshold:
                used[best_i] = True
                tp += 1
                matched_gt += 1
            else:
                fp += 1
        fn += sum(1 for flag in used if not flag)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "gt_csv": str(gt_csv),
        "seqs": sorted(seqs) if seqs is not None else None,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "max_frames": max_frames,
        "num_gt_boxes": len(gt_rows),
        "num_prediction_boxes": sum(len(v) for v in pred_by_key.values()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_gt": matched_gt,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_tracklet_classifier_frame_benchmark(
    run_roots: list[str | Path],
    gt_csvs: list[str | Path],
    weights: str | Path,
    out_dir: str | Path,
    dataset_names: list[str] | None = None,
    prediction_name: str = "predictions.jsonl",
    diagnostics_name: str = "diagnostics.jsonl",
    threshold: float = 0.5,
    thresholds: list[float] | None = None,
    untracked_policy: str = "keep",
    promote_positive_tracklets: bool = True,
    promotion_score_floor: float = 0.22,
    promotion_min_branch_drone: float = 0.40,
    promotion_max_background: float = 0.68,
    selective_promotion: bool = False,
    selective_min_temporal_crop_delta: float = 0.05,
    selective_min_temporal_background_margin: float = -0.05,
    selective_max_tracklet_background: float = 0.60,
    selective_max_tracklet_objectness: float = 0.50,
    selective_min_tracklet_rows: int = 2,
    selective_min_temporal_gain_rate: float = 0.40,
    selective_min_weak_detector_temporal_signal: float = 0.05,
    selective_require_recovery_source: bool = True,
    selective_max_promoted_tracklets_per_sequence: int = 2,
    iou_threshold: float = 0.3,
    score_threshold: float = 0.0,
    max_frames: int | None = None,
    baseline_csv: str | Path | None = None,
    baseline_metric: str = "frame_best_f1",
    baseline_lower_is_better: bool = False,
    baseline_digits: int = 3,
    allow_invalid_baselines: bool = False,
) -> TrackletClassifierFrameBenchmarkResult:
    if not run_roots:
        raise ValueError("run_roots must contain at least one inference output root")
    if len(run_roots) != len(gt_csvs):
        raise ValueError("run_roots and gt_csvs must have the same length")
    if dataset_names is not None and len(dataset_names) != len(run_roots):
        raise ValueError("dataset_names must have the same length as run_roots")
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Tracklet classifier weights not found: {weights_path}")
    threshold_values = [float(v) for v in (thresholds if thresholds is not None and len(thresholds) > 0 else [threshold])]

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rows_csv = []
    dataset_summaries = []
    for index, (run_root, gt_csv) in enumerate(zip(run_roots, gt_csvs)):
        dataset = str(dataset_names[index]) if dataset_names is not None else Path(run_root).name
        safe_dataset = dataset.replace("/", "_").replace("\\", "_") or f"dataset_{index}"
        dataset_dir = out_root / safe_dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        pairs = _discover_infer_pairs(run_root, prediction_name=prediction_name, diagnostics_name=diagnostics_name)
        if not pairs:
            raise FileNotFoundError(f"No prediction/diagnostics pairs found under {run_root}")

        raw_rows_all: list[dict[str, Any]] = []
        seqs: set[str] = set()
        for pair in pairs:
            seq = str(pair["seq"])
            seqs.add(seq)
            raw_rows = []
            for line in Path(pair["predictions"]).read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get("seq") in (None, ""):
                        row["seq"] = seq
                    raw_rows.append(row)
            raw_rows_all.extend(raw_rows)

        raw_metrics = evaluate_frame_predictions_against_gt_csv(
            raw_rows_all,
            gt_csv,
            seqs=seqs,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            max_frames=max_frames,
        )
        threshold_summaries = []
        best_threshold_summary: dict[str, Any] | None = None
        for threshold_value in threshold_values:
            suffix = str(threshold_value).replace(".", "p").replace("-", "m")
            threshold_dir = dataset_dir / f"threshold_{suffix}"
            copied_pairs = []
            filtered_rows_all: list[dict[str, Any]] = []
            filter_summaries = []
            for pair in pairs:
                seq = str(pair["seq"])
                seq_dir = threshold_dir / seq
                pred_dst = seq_dir / prediction_name
                diag_dst = seq_dir / diagnostics_name
                _copy_and_enrich_jsonl(Path(pair["predictions"]), pred_dst, seq)
                _copy_and_enrich_jsonl(Path(pair["diagnostics"]), diag_dst, seq)
                filter_summary = apply_tracklet_filter_to_infer_outputs(
                    pred_dst,
                    diag_dst,
                    weights_path,
                    threshold=threshold_value,
                    untracked_policy=untracked_policy,
                    promote_positive_tracklets=promote_positive_tracklets,
                    promotion_score_floor=promotion_score_floor,
                    promotion_min_branch_drone=promotion_min_branch_drone,
                    promotion_max_background=promotion_max_background,
                    selective_promotion=selective_promotion,
                    selective_min_temporal_crop_delta=selective_min_temporal_crop_delta,
                    selective_min_temporal_background_margin=selective_min_temporal_background_margin,
                    selective_max_tracklet_background=selective_max_tracklet_background,
                    selective_max_tracklet_objectness=selective_max_tracklet_objectness,
                    selective_min_tracklet_rows=selective_min_tracklet_rows,
                    selective_min_temporal_gain_rate=selective_min_temporal_gain_rate,
                    selective_min_weak_detector_temporal_signal=selective_min_weak_detector_temporal_signal,
                    selective_require_recovery_source=selective_require_recovery_source,
                    selective_max_promoted_tracklets_per_sequence=selective_max_promoted_tracklets_per_sequence,
                )
                filtered_rows_all.extend(_load_jsonl(pred_dst))
                filter_summaries.append({"seq": seq, **filter_summary})
                copied_pairs.append(
                    {
                        "seq": seq,
                        "source_predictions": str(pair["predictions"]),
                        "source_diagnostics": str(pair["diagnostics"]),
                        "filtered_predictions": str(pred_dst),
                        "filtered_diagnostics": str(diag_dst),
                    }
                )
            filtered_metrics = evaluate_frame_predictions_against_gt_csv(
                filtered_rows_all,
                gt_csv,
                seqs=seqs,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                max_frames=max_frames,
            )
            delta = {
                "precision": filtered_metrics["precision"] - raw_metrics["precision"],
                "recall": filtered_metrics["recall"] - raw_metrics["recall"],
                "f1": filtered_metrics["f1"] - raw_metrics["f1"],
                "tp": filtered_metrics["tp"] - raw_metrics["tp"],
                "fp": filtered_metrics["fp"] - raw_metrics["fp"],
                "fn": filtered_metrics["fn"] - raw_metrics["fn"],
            }
            threshold_summary = {
                "threshold": threshold_value,
                "out_dir": str(threshold_dir),
                "pairs": copied_pairs,
                "filter": filter_summaries,
                "filtered_metrics": filtered_metrics,
                "delta_filtered_minus_raw": delta,
            }
            (threshold_dir / "tracklet_classifier_frame_threshold_summary.json").write_text(json.dumps(threshold_summary, indent=2), encoding="utf-8")
            threshold_summaries.append(threshold_summary)
            if best_threshold_summary is None or (
                filtered_metrics["f1"],
                filtered_metrics["recall"],
                filtered_metrics["precision"],
                -filtered_metrics["fp"],
            ) > (
                best_threshold_summary["filtered_metrics"]["f1"],
                best_threshold_summary["filtered_metrics"]["recall"],
                best_threshold_summary["filtered_metrics"]["precision"],
                -best_threshold_summary["filtered_metrics"]["fp"],
            ):
                best_threshold_summary = threshold_summary

        assert best_threshold_summary is not None
        dataset_summary = {
            "dataset": dataset,
            "run_root": str(run_root),
            "gt_csv": str(gt_csv),
            "out_dir": str(dataset_dir),
            "seqs": sorted(seqs),
            "raw_metrics": raw_metrics,
            "thresholds": threshold_summaries,
            "best": best_threshold_summary,
            "filtered_metrics": best_threshold_summary["filtered_metrics"],
            "delta_filtered_minus_raw": best_threshold_summary["delta_filtered_minus_raw"],
        }
        (dataset_dir / "tracklet_classifier_frame_benchmark_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
        dataset_summaries.append(dataset_summary)
        rows_csv.append(
            {
                "dataset": dataset,
                "variant": "raw",
                "method": "route_b_raw",
                "threshold": "",
                "precision": raw_metrics["precision"],
                "recall": raw_metrics["recall"],
                "f1": raw_metrics["f1"],
                "tp": raw_metrics["tp"],
                "fp": raw_metrics["fp"],
                "fn": raw_metrics["fn"],
                "num_prediction_boxes": raw_metrics["num_prediction_boxes"],
                "num_gt_boxes": raw_metrics["num_gt_boxes"],
            }
        )
        for threshold_summary in threshold_summaries:
            metrics = threshold_summary["filtered_metrics"]
            rows_csv.append(
                {
                    "dataset": dataset,
                    "variant": "filtered",
                    "method": "route_b_tracklet_classifier_filtered",
                    "threshold": threshold_summary["threshold"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "num_prediction_boxes": metrics["num_prediction_boxes"],
                    "num_gt_boxes": metrics["num_gt_boxes"],
                }
            )

    csv_path = out_root / "tracklet_classifier_frame_benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "variant",
                "method",
                "threshold",
                "precision",
                "recall",
                "f1",
                "tp",
                "fp",
                "fn",
                "num_prediction_boxes",
                "num_gt_boxes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_csv)
    route_b_results_csv = _write_tracklet_classifier_frame_route_b_results_csv(
        dataset_summaries,
        out_root / "tracklet_classifier_frame_route_b_results.csv",
    )
    baseline_report = None
    if baseline_csv is not None:
        from qstr_dronedet.tracking.action_policy import (
            compare_route_b_results_to_baselines,
            export_route_b_baseline_markdown_table,
            validate_route_b_baseline_csv,
        )

        baseline_root = out_root / "baseline_report"
        baseline_root.mkdir(parents=True, exist_ok=True)
        validation = validate_route_b_baseline_csv(
            baseline_csv,
            baseline_root / "baseline_validation.json",
            metric=baseline_metric,
            require_metric_values=True,
        )
        if allow_invalid_baselines or validation.summary.get("valid", False):
            comparison = compare_route_b_results_to_baselines(
                route_b_results_csv,
                baseline_csv,
                baseline_root / "comparison",
                metric=baseline_metric,
                higher_is_better=not baseline_lower_is_better,
            )
            markdown = export_route_b_baseline_markdown_table(
                comparison.summary["json"],
                baseline_root / "route_b_tracklet_classifier_frame_baseline_report.md",
                digits=baseline_digits,
            )
            baseline_report = {
                "valid": bool(validation.summary.get("valid", False)),
                "baseline_csv": str(baseline_csv),
                "metric": baseline_metric,
                "higher_is_better": not baseline_lower_is_better,
                "validation_json": str(validation.out_path),
                "comparison_csv": str(comparison.out_path),
                "comparison_json": str(comparison.summary["json"]),
                "ranking_csv": str(comparison.summary["ranking_csv"]),
                "markdown": str(markdown.out_path),
                "route_b_wins": int(comparison.summary["route_b_wins"]),
                "num_comparisons": int(comparison.summary["num_comparisons"]),
                "comparison_rows": comparison.summary["comparison_rows"],
            }
        else:
            baseline_report = {
                "valid": False,
                "baseline_csv": str(baseline_csv),
                "metric": baseline_metric,
                "validation_json": str(validation.out_path),
                "baseline_validation": validation.summary,
            }
            raise ValueError(f"Frame benchmark baseline CSV failed validation: {validation.summary.get('issues', [])}")
    summary = {
        "out_dir": str(out_root),
        "csv": str(csv_path),
        "weights": str(weights_path),
        "run_roots": [str(path) for path in run_roots],
        "gt_csvs": [str(path) for path in gt_csvs],
        "dataset_names": dataset_names,
        "prediction_name": prediction_name,
        "diagnostics_name": diagnostics_name,
        "threshold": threshold,
        "thresholds": threshold_values,
        "untracked_policy": untracked_policy,
        "promote_positive_tracklets": promote_positive_tracklets,
        "selective_promotion": selective_promotion,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "max_frames": max_frames,
        "datasets": dataset_summaries,
        "route_b_results_csv": str(route_b_results_csv),
        "baseline_csv": str(baseline_csv) if baseline_csv is not None else None,
        "baseline_metric": baseline_metric,
        "baseline_report": baseline_report,
    }
    summary_path = out_root / "tracklet_classifier_frame_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletClassifierFrameBenchmarkResult(out_path=csv_path, summary=summary)


def build_tracklet_classifier_official_eval_bundle(
    frame_summary: str | Path,
    out_dir: str | Path,
    preflight_json: str | Path | None = None,
    baseline_comparison_json: str | Path | None = None,
    copy_predictions: bool = True,
    require_valid_preflight: bool = True,
    require_baseline_comparison: bool = False,
) -> TrackletClassifierOfficialEvalBundleResult:
    summary_path = Path(frame_summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"frame benchmark summary not found: {summary_path}")
    frame = json.loads(summary_path.read_text(encoding="utf-8"))
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    warnings: list[str] = []
    dataset_entries: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []

    preflight_summary = None
    if preflight_json:
        preflight_path = Path(preflight_json)
        if not preflight_path.exists():
            errors.append(f"preflight JSON not found: {preflight_path}")
        else:
            preflight_summary = json.loads(preflight_path.read_text(encoding="utf-8"))
            if require_valid_preflight and not preflight_summary.get("valid", False):
                errors.append(f"preflight JSON is invalid: {preflight_path}")
    elif require_valid_preflight:
        warnings.append("no preflight JSON was provided; official evaluator inputs are less auditable")

    baseline_summary = None
    if baseline_comparison_json:
        baseline_path = Path(baseline_comparison_json)
        if not baseline_path.exists():
            errors.append(f"baseline comparison JSON not found: {baseline_path}")
        else:
            baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
    elif require_baseline_comparison:
        errors.append("baseline comparison JSON is required")
    else:
        warnings.append("no baseline comparison JSON was provided")

    for dataset in frame.get("datasets", []):
        name = str(dataset.get("dataset") or "")
        if not name:
            errors.append("frame summary contains a dataset with no name")
            continue
        safe_name = name.replace("/", "_").replace("\\", "_")
        best = dict(dataset.get("best") or {})
        if not best:
            errors.append(f"{name}: missing best threshold summary")
            continue
        pairs = list(best.get("pairs") or [])
        if not pairs:
            errors.append(f"{name}: best threshold has no filtered prediction pairs")

        copied_pairs = []
        for pair in pairs:
            seq = str(pair.get("seq") or "")
            src_pred = Path(str(pair.get("filtered_predictions") or ""))
            src_diag = Path(str(pair.get("filtered_diagnostics") or ""))
            if not src_pred.exists():
                errors.append(f"{name}/{seq}: filtered predictions not found: {src_pred}")
                continue
            if not src_diag.exists():
                warnings.append(f"{name}/{seq}: filtered diagnostics not found: {src_diag}")
            pred_summary = _summarize_frame_jsonl(src_pred, seq)
            if copy_predictions:
                seq_dir = out_root / "best_filtered" / safe_name / seq
                seq_dir.mkdir(parents=True, exist_ok=True)
                prediction_path = seq_dir / src_pred.name
                diagnostics_path = seq_dir / src_diag.name
                shutil.copy2(src_pred, prediction_path)
                if src_diag.exists():
                    shutil.copy2(src_diag, diagnostics_path)
                else:
                    diagnostics_path = None
            else:
                prediction_path = src_pred
                diagnostics_path = src_diag if src_diag.exists() else None

            copied = {
                "dataset": name,
                "seq": seq,
                "source_filtered_predictions": str(src_pred),
                "source_filtered_diagnostics": str(src_diag),
                "official_eval_predictions": str(prediction_path),
                "official_eval_diagnostics": str(diagnostics_path) if diagnostics_path is not None else None,
                "prediction_rows": int(pred_summary["rows"]),
                "drone_prediction_rows": int(pred_summary["drone_rows"]),
                "track_id_prediction_rows": int(pred_summary["track_id_rows"]),
            }
            copied_pairs.append(copied)
            index_rows.append(
                {
                    "dataset": name,
                    "seq": seq,
                    "threshold": best.get("threshold"),
                    "predictions": str(prediction_path),
                    "diagnostics": str(diagnostics_path) if diagnostics_path is not None else "",
                    "gt_csv": str(dataset.get("gt_csv") or ""),
                    "prediction_rows": int(pred_summary["rows"]),
                    "drone_prediction_rows": int(pred_summary["drone_rows"]),
                    "track_id_prediction_rows": int(pred_summary["track_id_rows"]),
                }
            )

        dataset_entries.append(
            {
                "dataset": name,
                "run_root": dataset.get("run_root"),
                "gt_csv": dataset.get("gt_csv"),
                "seqs": dataset.get("seqs"),
                "best_threshold": best.get("threshold"),
                "best_threshold_out_dir": best.get("out_dir"),
                "raw_metrics": dataset.get("raw_metrics"),
                "filtered_metrics": best.get("filtered_metrics") or dataset.get("filtered_metrics"),
                "delta_filtered_minus_raw": best.get("delta_filtered_minus_raw"),
                "pairs": copied_pairs,
                "official_eval_status": "ready_for_external_evaluator" if copied_pairs else "missing_predictions",
            }
        )

    index_csv = out_root / "official_eval_prediction_index.csv"
    fieldnames = [
        "dataset",
        "seq",
        "threshold",
        "predictions",
        "diagnostics",
        "gt_csv",
        "prediction_rows",
        "drone_prediction_rows",
        "track_id_prediction_rows",
    ]
    with index_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    manifest_path = out_root / "official_eval_bundle_manifest.json"
    summary = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "out_dir": str(out_root),
        "manifest": str(manifest_path),
        "prediction_index_csv": str(index_csv),
        "frame_summary": str(summary_path),
        "preflight_json": str(preflight_json) if preflight_json else None,
        "baseline_comparison_json": str(baseline_comparison_json) if baseline_comparison_json else None,
        "copy_predictions": copy_predictions,
        "requirements": {
            "require_valid_preflight": require_valid_preflight,
            "require_baseline_comparison": require_baseline_comparison,
        },
        "combined": {
            "datasets": len(dataset_entries),
            "pairs": len(index_rows),
            "prediction_rows": sum(int(row["prediction_rows"]) for row in index_rows),
            "drone_prediction_rows": sum(int(row["drone_prediction_rows"]) for row in index_rows),
        },
        "preflight": preflight_summary,
        "baseline_comparison": baseline_summary,
        "datasets": dataset_entries,
        "next_step": "Run the dataset-specific official evaluator on official_eval_predictions paths; do not report paper-level wins from this proxy manifest alone.",
    }
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletClassifierOfficialEvalBundleResult(out_path=manifest_path, summary=summary)


def _official_export_image_stem(row: dict[str, Any], seq: str, frame_id: int) -> str:
    image_path = str(row.get("image_path") or "").strip()
    if image_path:
        return Path(image_path).stem
    return f"{seq}_{frame_id:06d}"


def _official_export_image_size(
    row: dict[str, Any],
    default_image_size: tuple[int, int] | None,
) -> tuple[float, float] | None:
    width = row.get("image_width")
    height = row.get("image_height")
    if width is not None and height is not None:
        try:
            width_f = float(width)
            height_f = float(height)
            if width_f > 0 and height_f > 0:
                return width_f, height_f
        except (TypeError, ValueError):
            pass
    if default_image_size is not None:
        return float(default_image_size[0]), float(default_image_size[1])
    return None


def export_tracklet_classifier_official_predictions(
    bundle_manifest: str | Path,
    out_dir: str | Path,
    formats: list[str] | None = None,
    default_image_size: tuple[int, int] | None = None,
    score_field: str = "final_drone_score",
    min_score: float = 0.0,
    class_id: int = 0,
    include_background: bool = False,
) -> TrackletClassifierOfficialPredictionExportResult:
    manifest_path = Path(bundle_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"official eval bundle manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    requested_formats = formats or ["flat_csv", "yolo_txt"]
    allowed = {"flat_csv", "yolo_txt"}
    unknown = sorted(set(requested_formats) - allowed)
    if unknown:
        raise ValueError(f"unsupported export formats: {unknown}")

    errors: list[str] = []
    warnings: list[str] = []
    flat_rows: list[dict[str, Any]] = []
    yolo_rows_by_file: dict[Path, list[str]] = {}
    image_index_rows: list[dict[str, Any]] = []

    for dataset in manifest.get("datasets", []):
        dataset_name = str(dataset.get("dataset") or "")
        safe_dataset = dataset_name.replace("/", "_").replace("\\", "_")
        for pair in dataset.get("pairs", []):
            seq = str(pair.get("seq") or "")
            pred_path = Path(str(pair.get("official_eval_predictions") or ""))
            if not pred_path.exists():
                errors.append(f"{dataset_name}/{seq}: predictions not found: {pred_path}")
                continue
            try:
                rows = _load_jsonl(pred_path)
            except Exception as exc:
                errors.append(f"{dataset_name}/{seq}: could not read predictions: {exc}")
                continue
            for row_index, row in enumerate(rows):
                predicted_class = str(row.get("predicted_class") or "")
                if predicted_class != "drone" and not include_background:
                    continue
                try:
                    score = float(row.get(score_field, row.get("final_drone_score", row.get("objectness", 0.0))) or 0.0)
                except (TypeError, ValueError):
                    warnings.append(f"{dataset_name}/{seq}: row {row_index} has invalid score")
                    score = 0.0
                if score < min_score:
                    continue
                try:
                    frame_id = int(row.get("frame_id"))
                    x1, y1, x2, y2 = [float(v) for v in row.get("bbox", [])]
                except (TypeError, ValueError):
                    errors.append(f"{dataset_name}/{seq}: row {row_index} missing frame_id or bbox")
                    continue
                image_stem = _official_export_image_stem(row, seq, frame_id)
                image_size = _official_export_image_size(row, default_image_size)
                flat_rows.append(
                    {
                        "dataset": dataset_name,
                        "seq": seq,
                        "frame_id": frame_id,
                        "image_stem": image_stem,
                        "class_id": class_id,
                        "score": score,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "track_id": row.get("track_id", ""),
                        "source_predictions": str(pred_path),
                    }
                )
                image_index_rows.append(
                    {
                        "dataset": dataset_name,
                        "seq": seq,
                        "frame_id": frame_id,
                        "image_stem": image_stem,
                        "image_path": row.get("image_path", ""),
                        "image_width": image_size[0] if image_size is not None else "",
                        "image_height": image_size[1] if image_size is not None else "",
                    }
                )
                if "yolo_txt" in requested_formats:
                    if image_size is None:
                        warnings.append(f"{dataset_name}/{seq}/{frame_id}: missing image size; skipped YOLO txt row")
                        continue
                    width, height = image_size
                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)
                    cx = x1 + bw / 2.0
                    cy = y1 + bh / 2.0
                    label_path = out_root / "yolo_txt" / safe_dataset / "labels" / f"{image_stem}.txt"
                    yolo_rows_by_file.setdefault(label_path, []).append(
                        f"{class_id} {cx / width:.8f} {cy / height:.8f} {bw / width:.8f} {bh / height:.8f} {score:.8f}"
                    )

    flat_csv = out_root / "flat_xyxy_predictions.csv"
    if "flat_csv" in requested_formats:
        with flat_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["dataset", "seq", "frame_id", "image_stem", "class_id", "score", "x1", "y1", "x2", "y2", "track_id", "source_predictions"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)
    else:
        flat_csv = Path("")

    if "yolo_txt" in requested_formats:
        for label_path, lines in yolo_rows_by_file.items():
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    image_index_csv = out_root / "image_index.csv"
    deduped_index: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in image_index_rows:
        key = (str(row["dataset"]), str(row["seq"]), int(row["frame_id"]), str(row["image_stem"]))
        deduped_index[key] = row
    with image_index_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["dataset", "seq", "frame_id", "image_stem", "image_path", "image_width", "image_height"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped_index.values())

    summary_path = out_root / "official_prediction_export_summary.json"
    summary = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "bundle_manifest": str(manifest_path),
        "out_dir": str(out_root),
        "formats": requested_formats,
        "flat_csv": str(flat_csv) if "flat_csv" in requested_formats else None,
        "image_index_csv": str(image_index_csv),
        "yolo_txt_dir": str(out_root / "yolo_txt") if "yolo_txt" in requested_formats else None,
        "default_image_size": list(default_image_size) if default_image_size is not None else None,
        "score_field": score_field,
        "min_score": min_score,
        "class_id": class_id,
        "include_background": include_background,
        "combined": {
            "prediction_rows": len(flat_rows),
            "image_rows": len(deduped_index),
            "yolo_label_files": len(yolo_rows_by_file),
        },
        "next_step": "Use flat_xyxy_predictions.csv or yolo_txt/<dataset>/labels as the input adapter for the dataset-specific official evaluator.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletClassifierOfficialPredictionExportResult(out_path=summary_path, summary=summary)


def _format_aot_image_name(template: str, row: dict[str, str]) -> str:
    values = dict(row)
    values.setdefault("image_stem", "")
    values.setdefault("dataset", "")
    values.setdefault("seq", "")
    values.setdefault("frame_id", "0")
    try:
        frame_id_int = int(float(values["frame_id"]))
    except (TypeError, ValueError):
        frame_id_int = 0
    values["frame_id_int"] = frame_id_int
    values["frame_id_05d"] = f"{frame_id_int:05d}"
    values["frame_id_06d"] = f"{frame_id_int:06d}"
    name = template.format(**values)
    return name if Path(name).suffix else f"{name}.png"


def _parse_clip_id_from_row(row: dict[str, str]) -> int | None:
    for key in ["seq", "image_stem"]:
        text = str(row.get(key) or "")
        parts = Path(text).stem.split("_")
        if len(parts) >= 2 and parts[0] == "Clip":
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def _format_aot_clip_frame_image_name(row: dict[str, str], frame_id_offset: int = 0) -> str:
    clip_id = _parse_clip_id_from_row(row)
    if clip_id is None:
        raise ValueError(f"could not infer AOT clip id from seq/image_stem: seq={row.get('seq')} image_stem={row.get('image_stem')}")
    try:
        frame_id = int(float(row.get("frame_id", "0") or 0)) + int(frame_id_offset)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid frame_id for AOT clip-frame name: {row.get('frame_id')}") from exc
    return f"Clip_{clip_id}_{frame_id:05d}.png"


def export_tracklet_classifier_aot_prediction_parts(
    flat_csv: str | Path,
    out_dir: str | Path,
    image_name_template: str = "{image_stem}.png",
    image_name_mode: str = "template",
    frame_id_offset: int = 0,
    part_name: str = "predictions_split_0.pkl",
    min_score: float = 0.0,
    score_field: str = "score",
    class_name: str = "airborne",
    group_by_image: bool = True,
) -> TrackletClassifierAotPredictionExportResult:
    csv_path = Path(flat_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"flat prediction CSV not found: {csv_path}")
    out_root = Path(out_dir)
    pred_dir = out_root / "aotpredictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    part_path = pred_dir / part_name
    if image_name_mode not in {"template", "aot_clip_frame"}:
        raise ValueError(f"unsupported image_name_mode: {image_name_mode}")

    errors: list[str] = []
    warnings: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    ungrouped: list[dict[str, Any]] = []
    rows_seen = 0
    rows_exported = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"dataset", "seq", "frame_id", "image_stem", "x1", "y1", "x2", "y2", score_field}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"flat prediction CSV is missing required columns: {missing}")
        for row_index, row in enumerate(reader, start=2):
            rows_seen += 1
            try:
                score = float(row.get(score_field, "") or 0.0)
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
            except ValueError as exc:
                errors.append(f"row {row_index}: invalid numeric value: {exc}")
                continue
            if score < min_score:
                continue
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            if width <= 0.0 or height <= 0.0:
                warnings.append(f"row {row_index}: skipped non-positive bbox")
                continue
            try:
                track_id = int(float(row.get("track_id", "") or rows_exported))
            except ValueError:
                track_id = rows_exported
            detection = {
                "track_id": track_id,
                "x": x1 + width / 2.0,
                "y": y1 + height / 2.0,
                "w": width,
                "h": height,
                "n": class_name,
                "s": score,
            }
            try:
                img_name = (
                    _format_aot_clip_frame_image_name(row, frame_id_offset=frame_id_offset)
                    if image_name_mode == "aot_clip_frame"
                    else _format_aot_image_name(image_name_template, row)
                )
            except ValueError as exc:
                errors.append(f"row {row_index}: {exc}")
                continue
            rows_exported += 1
            if group_by_image:
                grouped.setdefault(img_name, []).append(detection)
            else:
                ungrouped.append({"img_name": img_name, "detections": [detection]})

    if group_by_image:
        result_rows = [{"img_name": img_name, "detections": detections} for img_name, detections in sorted(grouped.items())]
    else:
        result_rows = ungrouped
    with part_path.open("wb") as f:
        pickle.dump(result_rows, f)

    result_json = out_root / "result_preview.json"
    result_json.write_text(json.dumps(result_rows, indent=2), encoding="utf-8")
    summary_path = out_root / "aot_prediction_export_summary.json"
    summary = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "flat_csv": str(csv_path),
        "out_dir": str(out_root),
        "aotpredictions_dir": str(pred_dir),
        "part_path": str(part_path),
        "result_preview_json": str(result_json),
        "image_name_template": image_name_template,
        "image_name_mode": image_name_mode,
        "frame_id_offset": frame_id_offset,
        "part_name": part_name,
        "min_score": min_score,
        "score_field": score_field,
        "class_name": class_name,
        "group_by_image": group_by_image,
        "combined": {
            "rows_seen": rows_seen,
            "detections_exported": rows_exported,
            "result_records": len(result_rows),
            "image_records": len(grouped) if group_by_image else len({row["img_name"] for row in ungrouped}),
        },
        "next_step": "Run papers/TransVisDrone/evaluate_aot.py with --results_folder pointing at aotpredictions and --dataset-path set to the matching AOT root.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletClassifierAotPredictionExportResult(out_path=summary_path, summary=summary)


def _parse_aot_clip_frame_name(img_name: str) -> tuple[str, int] | None:
    stem = Path(str(img_name)).stem
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0] == "Clip" and tokens[1].isdigit() and tokens[2].isdigit():
        return f"Clip_{int(tokens[1])}", int(tokens[2])
    return None


def export_aot_prediction_parts_to_tracklets(
    results_folder: str | Path,
    out: str | Path,
    min_score: float = 0.0,
    dataset_source: str = "aot",
    image_width: int | None = None,
    image_height: int | None = None,
    min_tracklet_rows: int = 1,
    max_frame_gap: int | None = None,
    clip_id_to_flight_id_path: str | Path | None = None,
    aot_groundtruth_json: str | Path | None = None,
) -> TrackletClassifierAotTrackletExportResult:
    folder = Path(results_folder)
    if not folder.exists():
        raise FileNotFoundError(f"AOT results folder not found: {folder}")
    parts = sorted(folder.glob("*.pkl"))
    if not parts:
        raise FileNotFoundError(f"no .pkl prediction parts found: {folder}")
    if min_tracklet_rows < 1:
        raise ValueError("min_tracklet_rows must be >= 1")
    if max_frame_gap is not None and max_frame_gap < 1:
        raise ValueError("max_frame_gap must be >= 1 when provided")
    clip_to_flight: dict[int, str] = {}
    frame_image_lookup: dict[tuple[str, int], str] = {}
    if clip_id_to_flight_id_path is not None:
        clip_map_path = Path(clip_id_to_flight_id_path)
        if not clip_map_path.exists():
            raise FileNotFoundError(f"clip id map not found: {clip_map_path}")
        with clip_map_path.open("rb") as f:
            raw_map = pickle.load(f)
        clip_to_flight = {int(k): str(v) for k, v in dict(raw_map).items()}
    if aot_groundtruth_json is not None:
        gt_path = Path(aot_groundtruth_json)
        if not gt_path.exists():
            raise FileNotFoundError(f"AOT groundtruth JSON not found: {gt_path}")
        with gt_path.open("r", encoding="utf-8-sig") as f:
            gt = json.load(f)
        samples = gt.get("samples", {})
        if not isinstance(samples, dict):
            raise ValueError("AOT groundtruth JSON must contain object field samples")
        for flight_id, sample in samples.items():
            if not isinstance(sample, dict):
                continue
            ordered_frame_images: list[tuple[int, str]] = []
            seen_frames: set[int] = set()
            for entity in sample.get("entities", []) or []:
                if not isinstance(entity, dict):
                    continue
                blob = entity.get("blob") or {}
                try:
                    frame_id = int(blob.get("frame"))
                except (TypeError, ValueError):
                    continue
                img_name = entity.get("img_name")
                if img_name:
                    frame_image_lookup.setdefault((str(flight_id), frame_id), str(img_name))
                    if frame_id not in seen_frames:
                        ordered_frame_images.append((frame_id, str(img_name)))
                        seen_frames.add(frame_id)
            for clip_frame_index, (_frame_id, img_name) in enumerate(ordered_frame_images):
                frame_image_lookup[(str(flight_id), clip_frame_index)] = img_name

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    errors: list[str] = []
    warnings: list[str] = []
    records_seen = 0
    detections_seen = 0
    detections_used = 0
    skipped_low_score = 0
    skipped_bad_name = 0
    skipped_bad_detection = 0
    clip_ids: set[str] = set()

    for part in parts:
        try:
            with part.open("rb") as f:
                records = pickle.load(f)
        except Exception as exc:
            errors.append(f"{part}: could not read pkl: {exc}")
            continue
        if not isinstance(records, list):
            errors.append(f"{part}: pkl root is not a list")
            continue
        for record_index, record in enumerate(records):
            records_seen += 1
            if not isinstance(record, dict):
                errors.append(f"{part}: record {record_index} is not a dict")
                continue
            parsed = _parse_aot_clip_frame_name(str(record.get("img_name") or ""))
            if parsed is None:
                skipped_bad_name += 1
                if skipped_bad_name <= 20:
                    warnings.append(f"{part}: record {record_index} has non-AOT img_name: {record.get('img_name')}")
                continue
            seq, frame_id = parsed
            clip_ids.add(seq)
            clip_id = int(seq.split("_", 1)[1])
            flight_id = clip_to_flight.get(clip_id)
            frame_img_name = frame_image_lookup.get((flight_id, frame_id)) if flight_id is not None else None
            detections = record.get("detections")
            if not isinstance(detections, list):
                errors.append(f"{part}: record {record_index} detections is not a list")
                continue
            for det_index, det in enumerate(detections):
                detections_seen += 1
                if not isinstance(det, dict):
                    skipped_bad_detection += 1
                    continue
                try:
                    score = float(det.get("s", 0.0) or 0.0)
                    cx = float(det["x"])
                    cy = float(det["y"])
                    width = float(det["w"])
                    height = float(det["h"])
                except (TypeError, ValueError, KeyError):
                    skipped_bad_detection += 1
                    continue
                if score < min_score:
                    skipped_low_score += 1
                    continue
                if width <= 0.0 or height <= 0.0:
                    skipped_bad_detection += 1
                    continue
                track_id_raw = det.get("track_id", f"det_{record_index}_{det_index}")
                track_id = str(track_id_raw)
                row = {
                    "seq": seq,
                    "track_id": track_id,
                    "frame_id": frame_id,
                    "bbox": [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0],
                    "objectness": score,
                    "final_drone_score": score,
                    "predicted_class": str(det.get("n", "airborne")),
                    "source": "aot_prediction_pkl",
                    "visible": True,
                }
                if flight_id is not None:
                    row["flight_id"] = flight_id
                if frame_img_name is not None:
                    row["image_name"] = frame_img_name
                    row["image_path"] = str(Path("Images") / flight_id / frame_img_name) if flight_id is not None else frame_img_name
                if image_width is not None:
                    row["image_width"] = int(image_width)
                if image_height is not None:
                    row["image_height"] = int(image_height)
                grouped.setdefault((seq, track_id), []).append(row)
                detections_used += 1

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    dropped_short = 0
    segments_seen = 0
    with out_path.open("w", encoding="utf-8") as f:
        for (seq, track_id), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
            ordered = sorted(rows, key=lambda row: int(row["frame_id"]))
            segments: list[list[dict[str, Any]]] = []
            current: list[dict[str, Any]] = []
            previous_frame: int | None = None
            for row in ordered:
                frame_id = int(row["frame_id"])
                if current and max_frame_gap is not None and previous_frame is not None and frame_id - previous_frame > max_frame_gap:
                    segments.append(current)
                    current = []
                current.append(row)
                previous_frame = frame_id
            if current:
                segments.append(current)
            segments_seen += len(segments)
            for segment_index, segment_rows in enumerate(segments):
                if len(segment_rows) < min_tracklet_rows:
                    dropped_short += 1
                    continue
                segment_track_id = track_id if len(segments) == 1 else f"{track_id}:seg{segment_index}"
                item = {
                    "meta": {
                        "seq": seq,
                        "track_id": segment_track_id,
                        "raw_track_id": track_id,
                        "segment_index": segment_index,
                        "label": 0,
                        "bucket": "unlabeled_aot_prediction",
                        "dataset_source": dataset_source,
                        "num_rows": len(segment_rows),
                    },
                    "rows": [{**row, "track_id": segment_track_id, "raw_track_id": track_id, "segment_index": segment_index} for row in segment_rows],
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1

    summary = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "results_folder": str(folder),
        "out": str(out_path),
        "parts": [str(part) for part in parts],
        "min_score": min_score,
        "dataset_source": dataset_source,
        "image_width": image_width,
        "image_height": image_height,
        "min_tracklet_rows": min_tracklet_rows,
        "max_frame_gap": max_frame_gap,
        "clip_id_to_flight_id_path": str(clip_id_to_flight_id_path) if clip_id_to_flight_id_path is not None else None,
        "aot_groundtruth_json": str(aot_groundtruth_json) if aot_groundtruth_json is not None else None,
        "combined": {
            "records_seen": records_seen,
            "detections_seen": detections_seen,
            "detections_used": detections_used,
            "skipped_low_score": skipped_low_score,
            "skipped_bad_name": skipped_bad_name,
            "skipped_bad_detection": skipped_bad_detection,
            "clip_count": len(clip_ids),
            "raw_tracklets": len(grouped),
            "segments_seen": segments_seen,
            "tracklets_written": written,
            "dropped_short_tracklets": dropped_short,
            "clip_id_mappings": len(clip_to_flight),
            "frame_image_mappings": len(frame_image_lookup),
        },
        "next_step": "Use this tracklet JSONL with score-tracklets-with-action-policy or run-action-dynamics-tracklet-pipeline.",
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletClassifierAotTrackletExportResult(out_path=out_path, summary=summary)


def filter_aot_prediction_parts_by_tracklets(
    results_folder: str | Path,
    tracklet_jsonl: str | Path,
    out_dir: str | Path,
    part_name: str = "predictions_split_0.pkl",
    score_field: str | None = None,
    min_tracklet_score: float | None = None,
    min_tracklet_rows: int = 1,
) -> TrackletClassifierAotTrackletFilterResult:
    folder = Path(results_folder)
    if not folder.exists():
        raise FileNotFoundError(f"AOT results folder not found: {folder}")
    parts = sorted(folder.glob("*.pkl"))
    if not parts:
        raise FileNotFoundError(f"no .pkl prediction parts found: {folder}")
    tracklet_path = Path(tracklet_jsonl)
    if not tracklet_path.exists():
        raise FileNotFoundError(f"tracklet JSONL not found: {tracklet_path}")
    if min_tracklet_rows < 1:
        raise ValueError("min_tracklet_rows must be >= 1")

    allowed: set[tuple[str, int, str]] = set()
    kept_tracklets = 0
    skipped_tracklets = 0
    total_tracklets = 0
    score_values: list[float] = []
    with tracklet_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            total_tracklets += 1
            item = json.loads(line)
            meta = dict(item.get("meta") or {})
            rows = [dict(row) for row in (item.get("rows") or [])]
            if len(rows) < min_tracklet_rows:
                skipped_tracklets += 1
                continue
            if score_field:
                score_raw = meta.get(score_field)
                if score_raw is None and rows:
                    score_raw = rows[0].get(score_field)
                try:
                    score_value = float(score_raw)
                except (TypeError, ValueError):
                    skipped_tracklets += 1
                    continue
                if min_tracklet_score is not None and score_value < min_tracklet_score:
                    skipped_tracklets += 1
                    continue
                score_values.append(score_value)
            kept_tracklets += 1
            for row in rows:
                seq = str(row.get("seq") or meta.get("seq") or "")
                try:
                    frame_id = int(float(row.get("frame_id", 0) or 0))
                except (TypeError, ValueError):
                    continue
                raw_track_id = str(row.get("raw_track_id") or meta.get("raw_track_id") or row.get("track_id") or meta.get("track_id") or "")
                allowed.add((seq, frame_id, raw_track_id))

    out_root = Path(out_dir)
    pred_dir = out_root / "aotpredictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_part = pred_dir / part_name
    errors: list[str] = []
    warnings: list[str] = []
    records_seen = 0
    records_written = 0
    detections_seen = 0
    detections_written = 0
    missing_name = 0
    result_rows: list[dict[str, Any]] = []
    for part in parts:
        try:
            with part.open("rb") as f:
                records = pickle.load(f)
        except Exception as exc:
            errors.append(f"{part}: could not read pkl: {exc}")
            continue
        if not isinstance(records, list):
            errors.append(f"{part}: pkl root is not a list")
            continue
        for record_index, record in enumerate(records):
            records_seen += 1
            if not isinstance(record, dict):
                errors.append(f"{part}: record {record_index} is not a dict")
                continue
            img_name = str(record.get("img_name") or "")
            parsed = _parse_aot_clip_frame_name(img_name)
            if parsed is None:
                missing_name += 1
                if missing_name <= 20:
                    warnings.append(f"{part}: record {record_index} has non-AOT img_name: {img_name}")
                continue
            seq, frame_id = parsed
            detections = record.get("detections")
            if not isinstance(detections, list):
                errors.append(f"{part}: record {record_index} detections is not a list")
                continue
            kept_dets = []
            for det in detections:
                detections_seen += 1
                if not isinstance(det, dict):
                    continue
                raw_track_id = str(det.get("track_id", ""))
                if (seq, frame_id, raw_track_id) in allowed:
                    kept_dets.append(det)
                    detections_written += 1
            if kept_dets:
                result_rows.append({"img_name": img_name, "detections": kept_dets})
                records_written += 1

    with out_part.open("wb") as f:
        pickle.dump(result_rows, f)

    summary = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "results_folder": str(folder),
        "tracklet_jsonl": str(tracklet_path),
        "out_dir": str(out_root),
        "aotpredictions_dir": str(pred_dir),
        "part_path": str(out_part),
        "part_name": part_name,
        "score_field": score_field,
        "min_tracklet_score": min_tracklet_score,
        "min_tracklet_rows": min_tracklet_rows,
        "combined": {
            "source_parts": len(parts),
            "tracklets_seen": total_tracklets,
            "tracklets_kept": kept_tracklets,
            "tracklets_skipped": skipped_tracklets,
            "allowed_detection_keys": len(allowed),
            "records_seen": records_seen,
            "records_written": records_written,
            "detections_seen": detections_seen,
            "detections_written": detections_written,
            "non_aot_image_names": missing_name,
            "mean_tracklet_score": float(np.mean(score_values)) if score_values else None,
        },
        "next_step": "Run AOT preflight/evaluation with --results-folder pointing at aotpredictions.",
    }
    summary_path = out_root / "aot_tracklet_filter_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletClassifierAotTrackletFilterResult(out_path=summary_path, summary=summary)


def rescore_aot_prediction_parts_by_tracklets(
    results_folder: str | Path,
    tracklet_jsonl: str | Path,
    out_dir: str | Path,
    part_name: str = "predictions_split_0.pkl",
    score_field: str = "video_action_model_fusion_score",
    center: float = 0.486,
    beta: float = 0.4,
    mode: str = "suppress-only",
    min_tracklet_rows: int = 1,
    missing_score_behavior: str = "keep",
    protect_raw_score_at: float | None = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> TrackletClassifierAotTrackletRescoreResult:
    if mode not in {"additive", "suppress-only", "boost-only"}:
        raise ValueError("mode must be one of: additive, suppress-only, boost-only")
    if missing_score_behavior not in {"keep", "error"}:
        raise ValueError("missing_score_behavior must be 'keep' or 'error'")
    if min_tracklet_rows < 1:
        raise ValueError("min_tracklet_rows must be >= 1")
    if clip_min > clip_max:
        raise ValueError("clip_min must be <= clip_max")

    folder = Path(results_folder)
    if not folder.exists():
        raise FileNotFoundError(f"AOT results folder not found: {folder}")
    parts = sorted(folder.glob("*.pkl"))
    if not parts:
        raise FileNotFoundError(f"no .pkl prediction parts found: {folder}")
    tracklet_path = Path(tracklet_jsonl)
    if not tracklet_path.exists():
        raise FileNotFoundError(f"tracklet JSONL not found: {tracklet_path}")

    score_by_detection: dict[tuple[str, int, str], float] = {}
    total_tracklets = 0
    scored_tracklets = 0
    skipped_short_tracklets = 0
    missing_score_tracklets = 0
    score_values: list[float] = []
    with tracklet_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            total_tracklets += 1
            item = json.loads(line)
            meta = dict(item.get("meta") or {})
            rows = [dict(row) for row in (item.get("rows") or [])]
            if len(rows) < min_tracklet_rows:
                skipped_short_tracklets += 1
                continue
            score_raw = meta.get(score_field)
            if score_raw is None and rows:
                score_raw = rows[0].get(score_field)
            try:
                score_value = float(score_raw)
            except (TypeError, ValueError):
                missing_score_tracklets += 1
                if missing_score_behavior == "error":
                    raise ValueError(f"missing or non-numeric {score_field} for tracklet {meta}")
                continue
            scored_tracklets += 1
            score_values.append(score_value)
            for row in rows:
                seq = str(row.get("seq") or meta.get("seq") or "")
                try:
                    frame_id = int(float(row.get("frame_id", 0) or 0))
                except (TypeError, ValueError):
                    continue
                raw_track_id = str(row.get("raw_track_id") or meta.get("raw_track_id") or row.get("track_id") or meta.get("track_id") or "")
                if seq and raw_track_id:
                    score_by_detection[(seq, frame_id, raw_track_id)] = score_value

    out_root = Path(out_dir)
    pred_dir = out_root / "aotpredictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_part = pred_dir / part_name
    errors: list[str] = []
    warnings: list[str] = []
    records_seen = 0
    records_written = 0
    detections_seen = 0
    detections_written = 0
    detections_scored = 0
    detections_changed = 0
    detections_protected = 0
    missing_detection_scores = 0
    non_aot_image_names = 0
    raw_scores: list[float] = []
    rescored_scores: list[float] = []
    result_rows: list[dict[str, Any]] = []

    def adjusted_score(raw_score: float, action_score: float) -> float:
        delta = action_score - center
        if mode == "suppress-only":
            value = raw_score - beta * max(0.0, -delta)
        elif mode == "boost-only":
            value = raw_score + beta * max(0.0, delta)
        else:
            value = raw_score + beta * delta
        return min(float(clip_max), max(float(clip_min), float(value)))

    for part in parts:
        try:
            with part.open("rb") as f:
                records = pickle.load(f)
        except Exception as exc:
            errors.append(f"{part}: could not read pkl: {exc}")
            continue
        if not isinstance(records, list):
            errors.append(f"{part}: pkl root is not a list")
            continue
        for record_index, record in enumerate(records):
            records_seen += 1
            if not isinstance(record, dict):
                errors.append(f"{part}: record {record_index} is not a dict")
                continue
            out_record = dict(record)
            img_name = str(record.get("img_name") or "")
            parsed = _parse_aot_clip_frame_name(img_name)
            if parsed is None:
                non_aot_image_names += 1
                if non_aot_image_names <= 20:
                    warnings.append(f"{part}: record {record_index} has non-AOT img_name: {img_name}")
                result_rows.append(out_record)
                records_written += 1
                continue
            seq, frame_id = parsed
            detections = record.get("detections")
            if not isinstance(detections, list):
                errors.append(f"{part}: record {record_index} detections is not a list")
                continue
            out_dets = []
            for det in detections:
                detections_seen += 1
                if not isinstance(det, dict):
                    continue
                out_det = dict(det)
                raw_track_id = str(det.get("track_id", ""))
                try:
                    raw_score = float(det.get("s", 0.0) or 0.0)
                except (TypeError, ValueError):
                    out_dets.append(out_det)
                    detections_written += 1
                    continue
                raw_scores.append(raw_score)
                action_score = score_by_detection.get((seq, frame_id, raw_track_id))
                if action_score is None:
                    missing_detection_scores += 1
                    new_score = raw_score
                elif protect_raw_score_at is not None and raw_score >= protect_raw_score_at:
                    detections_scored += 1
                    detections_protected += 1
                    new_score = raw_score
                else:
                    detections_scored += 1
                    new_score = adjusted_score(raw_score, action_score)
                    if abs(new_score - raw_score) > 1e-12:
                        detections_changed += 1
                    out_det["s"] = float(new_score)
                    out_det[f"{score_field}_tracklet_score"] = float(action_score)
                    out_det["tracklet_rescore_raw_s"] = float(raw_score)
                    out_det["tracklet_rescore_mode"] = mode
                rescored_scores.append(float(new_score))
                out_dets.append(out_det)
                detections_written += 1
            out_record["detections"] = out_dets
            result_rows.append(out_record)
            records_written += 1

    with out_part.open("wb") as f:
        pickle.dump(result_rows, f)

    summary = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "results_folder": str(folder),
        "tracklet_jsonl": str(tracklet_path),
        "out_dir": str(out_root),
        "aotpredictions_dir": str(pred_dir),
        "part_path": str(out_part),
        "part_name": part_name,
        "score_field": score_field,
        "center": center,
        "beta": beta,
        "mode": mode,
        "min_tracklet_rows": min_tracklet_rows,
        "missing_score_behavior": missing_score_behavior,
        "protect_raw_score_at": protect_raw_score_at,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "combined": {
            "source_parts": len(parts),
            "tracklets_seen": total_tracklets,
            "tracklets_scored": scored_tracklets,
            "tracklets_skipped_short": skipped_short_tracklets,
            "tracklets_missing_score": missing_score_tracklets,
            "scored_detection_keys": len(score_by_detection),
            "records_seen": records_seen,
            "records_written": records_written,
            "detections_seen": detections_seen,
            "detections_written": detections_written,
            "detections_scored": detections_scored,
            "detections_changed": detections_changed,
            "detections_protected": detections_protected,
            "detections_missing_tracklet_score": missing_detection_scores,
            "non_aot_image_names": non_aot_image_names,
            "mean_tracklet_score": float(np.mean(score_values)) if score_values else None,
            "mean_raw_score": float(np.mean(raw_scores)) if raw_scores else None,
            "mean_rescored_score": float(np.mean(rescored_scores)) if rescored_scores else None,
        },
        "next_step": "Run AOT preflight/evaluation with --results-folder pointing at aotpredictions.",
    }
    summary_path = out_root / "aot_tracklet_rescore_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletClassifierAotTrackletRescoreResult(out_path=summary_path, summary=summary)


def validate_tracklet_classifier_aot_eval_inputs(
    results_folder: str | Path,
    out: str | Path,
    clip_id_to_flight_id_path: str | Path | None = None,
    require_clip_pattern: bool = True,
    require_known_clip_ids: bool = True,
    max_records: int | None = None,
) -> TrackletClassifierAotEvalPreflightResult:
    folder = Path(results_folder)
    errors: list[str] = []
    warnings: list[str] = []
    part_summaries: list[dict[str, Any]] = []
    clip_map = None
    if clip_id_to_flight_id_path is not None:
        clip_map_path = Path(clip_id_to_flight_id_path)
        if not clip_map_path.exists():
            errors.append(f"clip id map not found: {clip_map_path}")
        else:
            try:
                with clip_map_path.open("rb") as f:
                    clip_map = pickle.load(f)
            except Exception as exc:
                errors.append(f"could not read clip id map: {exc}")

    if not folder.exists():
        errors.append(f"results folder not found: {folder}")
        parts: list[Path] = []
    else:
        parts = sorted(folder.glob("*.pkl"))
        if not parts:
            errors.append(f"no .pkl prediction parts found: {folder}")

    total_records = 0
    total_detections = 0
    image_names: set[str] = set()
    clip_ids: set[int] = set()
    pattern_errors = 0
    unknown_clip_ids: set[int] = set()
    numeric_errors = 0
    schema_errors = 0
    for part in parts:
        part_errors: list[str] = []
        part_warnings: list[str] = []
        try:
            with part.open("rb") as f:
                rows = pickle.load(f)
        except Exception as exc:
            part_errors.append(f"could not read pkl: {exc}")
            rows = []
        if not isinstance(rows, list):
            part_errors.append("pkl root is not a list")
            rows = []
        checked_rows = rows if max_records is None else rows[:max_records]
        for index, record in enumerate(checked_rows):
            total_records += 1
            if not isinstance(record, dict):
                schema_errors += 1
                part_errors.append(f"record {index}: not a dict")
                continue
            img_name = str(record.get("img_name") or "")
            if not img_name:
                schema_errors += 1
                part_errors.append(f"record {index}: missing img_name")
                continue
            image_names.add(img_name)
            stem = Path(img_name).stem
            tokens = stem.split("_")
            if len(tokens) >= 3 and tokens[0] == "Clip" and tokens[1].isdigit() and tokens[2].isdigit():
                clip_id = int(tokens[1])
                clip_ids.add(clip_id)
                if clip_map is not None and clip_id not in clip_map:
                    unknown_clip_ids.add(clip_id)
            else:
                pattern_errors += 1
                if require_clip_pattern and pattern_errors <= 20:
                    part_errors.append(f"record {index}: img_name does not match Clip_<id>_<frame>: {img_name}")
            detections = record.get("detections")
            if not isinstance(detections, list):
                schema_errors += 1
                part_errors.append(f"record {index}: detections is not a list")
                continue
            for det_index, det in enumerate(detections):
                total_detections += 1
                if not isinstance(det, dict):
                    schema_errors += 1
                    part_errors.append(f"record {index} detection {det_index}: not a dict")
                    continue
                for field in ["x", "y", "w", "h", "s"]:
                    try:
                        value = float(det[field])
                        if not np.isfinite(value):
                            raise ValueError("non-finite")
                    except Exception:
                        numeric_errors += 1
                        if numeric_errors <= 20:
                            part_errors.append(f"record {index} detection {det_index}: invalid {field}")
                if str(det.get("n") or "") != "airborne":
                    part_warnings.append(f"record {index} detection {det_index}: n is not airborne")
        if max_records is not None and isinstance(rows, list) and len(rows) > max_records:
            part_warnings.append(f"checked first {max_records} of {len(rows)} records")
        part_summaries.append(
            {
                "part": str(part),
                "records": len(rows) if isinstance(rows, list) else 0,
                "checked_records": len(checked_rows),
                "errors": part_errors,
                "warnings": part_warnings[:20],
            }
        )
        errors.extend(f"{part.name}: {err}" for err in part_errors)
        warnings.extend(f"{part.name}: {warn}" for warn in part_warnings[:20])

    if require_known_clip_ids and clip_map is not None and unknown_clip_ids:
        errors.append(f"unknown AOT clip ids: {sorted(unknown_clip_ids)[:20]}")
    if require_clip_pattern and pattern_errors:
        errors.append(f"{pattern_errors} image names do not match Clip_<id>_<frame>")
    if total_detections <= 0 and parts:
        warnings.append("no detections found in AOT prediction parts")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "results_folder": str(folder),
        "clip_id_to_flight_id_path": str(clip_id_to_flight_id_path) if clip_id_to_flight_id_path else None,
        "requirements": {
            "require_clip_pattern": require_clip_pattern,
            "require_known_clip_ids": require_known_clip_ids,
            "max_records": max_records,
        },
        "combined": {
            "parts": len(parts),
            "records_checked": total_records,
            "detections_checked": total_detections,
            "unique_image_names": len(image_names),
            "clip_ids": sorted(clip_ids),
            "pattern_errors": pattern_errors,
            "unknown_clip_ids": sorted(unknown_clip_ids),
            "schema_errors": schema_errors,
            "numeric_errors": numeric_errors,
        },
        "parts": part_summaries,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletClassifierAotEvalPreflightResult(out_path=out_path, summary=summary)


def filter_infer_rows_with_tracklet_classifier(
    pred_rows: list[dict[str, Any]],
    diag_rows: list[dict[str, Any]],
    weights: str | Path,
    threshold: float = 0.5,
    untracked_policy: str = "keep",
    promote_positive_tracklets: bool = True,
    promotion_score_floor: float = 0.22,
    promotion_min_branch_drone: float = 0.40,
    promotion_max_background: float = 0.68,
    selective_promotion: bool = False,
    selective_min_temporal_crop_delta: float = 0.05,
    selective_min_temporal_background_margin: float = -0.05,
    selective_max_tracklet_background: float = 0.60,
    selective_max_tracklet_objectness: float = 0.50,
    selective_min_tracklet_rows: int = 2,
    selective_min_temporal_gain_rate: float = 0.40,
    selective_min_weak_detector_temporal_signal: float = 0.05,
    selective_require_recovery_source: bool = True,
    selective_max_promoted_tracklets_per_sequence: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scores = score_tracklets_from_rows(diag_rows, weights, threshold=threshold)
    selective_allowlist = (
        _selective_tracklet_key_allowlist(
            diag_rows,
            scores,
            promotion_min_branch_drone=promotion_min_branch_drone,
            promotion_max_background=promotion_max_background,
            min_temporal_crop_delta=selective_min_temporal_crop_delta,
            min_temporal_background_margin=selective_min_temporal_background_margin,
            max_tracklet_background=selective_max_tracklet_background,
            max_tracklet_objectness=selective_max_tracklet_objectness,
            min_tracklet_rows=selective_min_tracklet_rows,
            min_temporal_gain_rate=selective_min_temporal_gain_rate,
            min_weak_detector_temporal_signal=selective_min_weak_detector_temporal_signal,
            require_recovery_source=selective_require_recovery_source,
            max_promoted_tracklets_per_sequence=selective_max_promoted_tracklets_per_sequence,
        )
        if selective_promotion
        else None
    )

    def update_row(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        track_id = _row_track_key(out)
        score = scores.get(track_id) if track_id is not None else None
        if score is None:
            out["tracklet_classifier_prob"] = None
            out["tracklet_is_drone"] = None
            out["tracklet_filter_applied"] = bool(untracked_policy == "suppress" and out.get("predicted_class") == "drone")
            if untracked_policy == "suppress" and out.get("predicted_class") == "drone":
                out["raw_predicted_class"] = out.get("predicted_class")
                out["raw_final_drone_score"] = out.get("final_drone_score")
                out["predicted_class"] = "background"
                out["final_drone_score"] = 0.0
                out["diagnostic_cause"] = _append_cause(out.get("diagnostic_cause"), "tracklet_untracked_rejected")
            return out

        is_drone = bool(score["tracklet_is_drone"])
        out["tracklet_classifier_prob"] = score["prob_tracklet_drone"]
        out["tracklet_is_drone"] = is_drone
        out["tracklet_filter_applied"] = bool(out.get("predicted_class") == "drone")
        if out.get("predicted_class") == "drone" and not is_drone:
            out["raw_predicted_class"] = out.get("predicted_class")
            out["raw_final_drone_score"] = out.get("final_drone_score")
            out["predicted_class"] = "background"
            out["final_drone_score"] = 0.0
            probs = dict(out.get("final_probs") or {})
            probs["drone"] = min(float(probs.get("drone", 0.0)), float(score["prob_tracklet_drone"]))
            probs["background"] = max(float(probs.get("background", 0.0)), 1.0 - float(score["prob_tracklet_drone"]))
            out["final_probs"] = probs
            out["diagnostic_cause"] = _append_cause(out.get("diagnostic_cause"), "tracklet_rejected")
        elif out.get("predicted_class") == "drone" and is_drone:
            out["raw_final_drone_score"] = out.get("final_drone_score")
            out["final_drone_score"] = max(float(out.get("final_drone_score", 0.0)), promotion_score_floor * float(score["prob_tracklet_drone"]))
            out["diagnostic_cause"] = _append_cause(out.get("diagnostic_cause"), "tracklet_confirmed")
        elif promote_positive_tracklets and is_drone:
            evidence = _promotion_evidence(out, score)
            branch_drone = evidence["branch_drone"]
            effective_background = evidence["effective_background"]
            selective_allowed = selective_allowlist is None or (track_id is not None and track_id in selective_allowlist)
            if branch_drone >= promotion_min_branch_drone and effective_background <= promotion_max_background:
                if not selective_allowed:
                    out["tracklet_promotion_suppressed"] = True
                    out["diagnostic_cause"] = _append_cause(out.get("diagnostic_cause"), "tracklet_promotion_budget_rejected")
                    return out
                out["raw_predicted_class"] = out.get("predicted_class")
                out["raw_final_drone_score"] = out.get("final_drone_score")
                out["predicted_class"] = "drone"
                out["final_drone_score"] = max(float(out.get("final_drone_score", 0.0)), promotion_score_floor * float(score["prob_tracklet_drone"]))
                out["diagnostic_cause"] = _append_cause(out.get("diagnostic_cause"), "tracklet_selective_promoted" if selective_promotion else "tracklet_promoted")
        return out

    filtered_pred_rows = [update_row(row) for row in pred_rows]
    filtered_diag_rows = [update_row(row) for row in diag_rows]

    raw_drone = sum(1 for row in pred_rows if row.get("predicted_class") == "drone")
    filtered_drone = sum(1 for row in filtered_pred_rows if row.get("predicted_class") == "drone")
    rejected = raw_drone - filtered_drone
    promoted = sum(1 for row in filtered_pred_rows if row.get("predicted_class") == "drone" and row.get("raw_predicted_class") not in (None, "drone"))
    summary = {
        "weights": str(weights),
        "threshold": threshold,
        "untracked_policy": untracked_policy,
        "promote_positive_tracklets": promote_positive_tracklets,
        "promotion_score_floor": promotion_score_floor,
        "promotion_min_branch_drone": promotion_min_branch_drone,
        "promotion_max_background": promotion_max_background,
        "selective_promotion": selective_promotion,
        "selective_min_temporal_crop_delta": selective_min_temporal_crop_delta,
        "selective_min_temporal_background_margin": selective_min_temporal_background_margin,
        "selective_max_tracklet_background": selective_max_tracklet_background,
        "selective_max_tracklet_objectness": selective_max_tracklet_objectness,
        "selective_min_tracklet_rows": selective_min_tracklet_rows,
        "selective_min_temporal_gain_rate": selective_min_temporal_gain_rate,
        "selective_min_weak_detector_temporal_signal": selective_min_weak_detector_temporal_signal,
        "selective_require_recovery_source": selective_require_recovery_source,
        "selective_max_promoted_tracklets_per_sequence": selective_max_promoted_tracklets_per_sequence,
        "selective_allowed_tracklets": len(selective_allowlist) if selective_allowlist is not None else None,
        "num_tracklets": len(scores),
        "raw_drone_predictions": raw_drone,
        "filtered_drone_predictions": filtered_drone,
        "rejected_drone_predictions": rejected,
        "promoted_drone_predictions": promoted,
    }
    return filtered_pred_rows, filtered_diag_rows, summary


def _coerce_feature(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return 0.0
    return float(value)


def _load_tracklet_csv(path: str | Path, features: list[str] | None = None) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, str]]]:
    feature_names = features or TRACKLET_FEATURES
    xs, ys, meta = [], [], []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            xs.append([_coerce_feature(row, k) for k in feature_names])
            ys.append(int(row["label"]))
            meta.append(row)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long), meta


def _summarize_classifier_csv(path: str | Path, dataset_name: str | None = None) -> dict[str, Any]:
    csv_path = Path(path)
    fieldnames = ["seq", "track_id", "label", "bucket", "dataset_source", "best_iou", "matched_frames"] + TRACKLET_FEATURES
    rows = 0
    positives = 0
    negatives = 0
    bucket_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    seq_counts: dict[str, int] = {}
    missing_feature_values: dict[str, int] = {}
    nonfinite_feature_values: dict[str, int] = {}
    errors: list[str] = []

    if not csv_path.exists():
        return {
            "path": str(csv_path),
            "dataset": dataset_name,
            "exists": False,
            "rows": 0,
            "positives": 0,
            "negatives": 0,
            "bucket_counts": {},
            "dataset_source_counts": {},
            "seq_counts": {},
            "errors": [f"missing file: {csv_path}"],
        }

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        reader_fields = reader.fieldnames or []
        missing_fields = [field for field in fieldnames if field not in reader_fields]
        if missing_fields:
            errors.append(f"missing classifier fields: {missing_fields}")
        for row in reader:
            rows += 1
            try:
                label = int(float(row.get("label") or 0))
            except ValueError:
                label = 0
                errors.append(f"row {rows} has invalid label: {row.get('label')}")
            positives += int(label > 0)
            negatives += int(label <= 0)
            bucket = str(row.get("bucket") or "__missing__")
            source = str(row.get("dataset_source") or "__missing__")
            seq = str(row.get("seq") or "__missing__")
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
            seq_counts[seq] = seq_counts.get(seq, 0) + 1
            for feature in TRACKLET_FEATURES:
                value = row.get(feature, "")
                if value == "":
                    missing_feature_values[feature] = missing_feature_values.get(feature, 0) + 1
                    continue
                try:
                    if not np.isfinite(float(value)):
                        nonfinite_feature_values[feature] = nonfinite_feature_values.get(feature, 0) + 1
                except ValueError:
                    nonfinite_feature_values[feature] = nonfinite_feature_values.get(feature, 0) + 1

    return {
        "path": str(csv_path),
        "dataset": dataset_name,
        "exists": True,
        "rows": rows,
        "positives": positives,
        "negatives": negatives,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "dataset_source_counts": dict(sorted(source_counts.items())),
        "seq_counts": dict(sorted(seq_counts.items())),
        "num_sequences": len(seq_counts),
        "missing_feature_values": dict(sorted(missing_feature_values.items())),
        "nonfinite_feature_values": dict(sorted(nonfinite_feature_values.items())),
        "errors": errors,
    }


def validate_tracklet_classifier_mixture_inputs(
    train_csvs: list[str | Path],
    eval_csvs: list[str | Path],
    out: str | Path,
    train_source_names: list[str] | None = None,
    eval_dataset_names: list[str] | None = None,
    min_train_rows: int = 1,
    min_eval_rows: int = 1,
    min_train_positives: int = 1,
    min_eval_positives: int = 1,
    require_train_negatives: bool = True,
    require_eval_negatives: bool = True,
    fail_on_train_eval_sequence_overlap: bool = True,
) -> TrackletClassifierMixturePreflightResult:
    if not train_csvs:
        raise ValueError("train_csvs must contain at least one classifier CSV")
    if not eval_csvs:
        raise ValueError("eval_csvs must contain at least one classifier CSV")
    if train_source_names is not None and len(train_source_names) != len(train_csvs):
        raise ValueError("train_source_names must have the same length as train_csvs")
    if eval_dataset_names is not None and len(eval_dataset_names) != len(eval_csvs):
        raise ValueError("eval_dataset_names must have the same length as eval_csvs")

    train = [
        _summarize_classifier_csv(path, train_source_names[index] if train_source_names is not None else None)
        for index, path in enumerate(train_csvs)
    ]
    evals = [
        _summarize_classifier_csv(path, eval_dataset_names[index] if eval_dataset_names is not None else None)
        for index, path in enumerate(eval_csvs)
    ]
    errors: list[str] = []
    warnings: list[str] = []

    train_rows = sum(int(row["rows"]) for row in train)
    train_pos = sum(int(row["positives"]) for row in train)
    train_neg = sum(int(row["negatives"]) for row in train)
    eval_rows = sum(int(row["rows"]) for row in evals)
    eval_pos = sum(int(row["positives"]) for row in evals)
    eval_neg = sum(int(row["negatives"]) for row in evals)

    for split, summaries, min_rows, min_pos, require_neg in [
        ("train", train, min_train_rows, min_train_positives, require_train_negatives),
        ("eval", evals, min_eval_rows, min_eval_positives, require_eval_negatives),
    ]:
        for summary in summaries:
            label = summary.get("dataset") or summary["path"]
            for error in summary.get("errors", []):
                errors.append(f"{split} {label}: {error}")
            if int(summary["rows"]) < min_rows:
                errors.append(f"{split} {label}: rows {summary['rows']} < required {min_rows}")
            if int(summary["positives"]) < min_pos:
                errors.append(f"{split} {label}: positives {summary['positives']} < required {min_pos}")
            if require_neg and int(summary["negatives"]) <= 0:
                errors.append(f"{split} {label}: no negative tracklets")
            if "__missing__" in summary.get("dataset_source_counts", {}):
                errors.append(f"{split} {label}: missing dataset_source values")
            if summary.get("missing_feature_values"):
                warnings.append(f"{split} {label}: missing feature values present")
            if summary.get("nonfinite_feature_values"):
                errors.append(f"{split} {label}: non-finite or invalid feature values present")

    if train_rows < min_train_rows:
        errors.append(f"combined train rows {train_rows} < required {min_train_rows}")
    if train_pos < min_train_positives:
        errors.append(f"combined train positives {train_pos} < required {min_train_positives}")
    if require_train_negatives and train_neg <= 0:
        errors.append("combined train has no negative tracklets")
    if eval_rows < min_eval_rows:
        errors.append(f"combined eval rows {eval_rows} < required {min_eval_rows}")
    if eval_pos < min_eval_positives:
        errors.append(f"combined eval positives {eval_pos} < required {min_eval_positives}")
    if require_eval_negatives and eval_neg <= 0:
        errors.append("combined eval has no negative tracklets")

    eval_names = [str(name) for name in eval_dataset_names] if eval_dataset_names is not None else [Path(path).parent.name or Path(path).stem for path in eval_csvs]
    duplicate_eval_names = sorted({name for name in eval_names if eval_names.count(name) > 1})
    if duplicate_eval_names:
        errors.append(f"duplicate eval dataset names: {duplicate_eval_names}")

    train_seqs = set()
    for summary in train:
        train_seqs.update(seq for seq in summary.get("seq_counts", {}) if seq != "__missing__")
    eval_seqs = set()
    for summary in evals:
        eval_seqs.update(seq for seq in summary.get("seq_counts", {}) if seq != "__missing__")
    overlaps = sorted(train_seqs & eval_seqs)
    if overlaps:
        message = f"train/eval sequence overlap: {overlaps[:20]}"
        if fail_on_train_eval_sequence_overlap:
            errors.append(message)
        else:
            warnings.append(message)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "out": str(out_path),
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "train_csvs": [str(path) for path in train_csvs],
        "eval_csvs": [str(path) for path in eval_csvs],
        "train_source_names": train_source_names,
        "eval_dataset_names": eval_dataset_names,
        "requirements": {
            "min_train_rows": min_train_rows,
            "min_eval_rows": min_eval_rows,
            "min_train_positives": min_train_positives,
            "min_eval_positives": min_eval_positives,
            "require_train_negatives": require_train_negatives,
            "require_eval_negatives": require_eval_negatives,
            "fail_on_train_eval_sequence_overlap": fail_on_train_eval_sequence_overlap,
        },
        "combined": {
            "train_rows": train_rows,
            "train_positives": train_pos,
            "train_negatives": train_neg,
            "eval_rows": eval_rows,
            "eval_positives": eval_pos,
            "eval_negatives": eval_neg,
            "train_sequences": len(train_seqs),
            "eval_sequences": len(eval_seqs),
            "train_eval_sequence_overlap": overlaps,
        },
        "train": train,
        "eval": evals,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletClassifierMixturePreflightResult(out_path=out_path, summary=summary)


def merge_tracklet_classifier_datasets(
    inputs: list[str | Path],
    out: str | Path,
    source_names: list[str] | None = None,
    manifest_out: str | Path | None = None,
) -> TrackletClassifierMergeResult:
    if not inputs:
        raise ValueError("inputs must contain at least one classifier CSV")
    if source_names is not None and len(source_names) != len(inputs):
        raise ValueError("source_names must have the same length as inputs")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_out) if manifest_out is not None else out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seq", "track_id", "label", "bucket", "dataset_source", "best_iou", "matched_frames"] + TRACKLET_FEATURES

    total_rows = 0
    positives = 0
    negatives = 0
    bucket_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    dataset_summaries: list[dict[str, Any]] = []

    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for index, input_path in enumerate(inputs):
            path = Path(input_path)
            source_override = source_names[index] if source_names is not None else None
            per_rows = 0
            per_positive = 0
            per_negative = 0
            per_buckets: dict[str, int] = {}
            per_sources: dict[str, int] = {}
            with path.open("r", encoding="utf-8-sig", newline="") as f_in:
                reader = csv.DictReader(f_in)
                missing = [field for field in fieldnames if field not in (reader.fieldnames or [])]
                if missing:
                    raise ValueError(f"{path} is missing classifier fields: {missing}")
                for row in reader:
                    out_row = {field: row.get(field, "") for field in fieldnames}
                    source = source_override or out_row.get("dataset_source") or path.parent.name or path.stem
                    out_row["dataset_source"] = str(source)
                    label = int(float(out_row.get("label") or 0))
                    bucket = str(out_row.get("bucket") or "")
                    writer.writerow(out_row)
                    total_rows += 1
                    per_rows += 1
                    positives += int(label > 0)
                    negatives += int(label <= 0)
                    per_positive += int(label > 0)
                    per_negative += int(label <= 0)
                    bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
                    source_counts[str(source)] = source_counts.get(str(source), 0) + 1
                    per_buckets[bucket] = per_buckets.get(bucket, 0) + 1
                    per_sources[str(source)] = per_sources.get(str(source), 0) + 1
            dataset_summaries.append(
                {
                    "input": str(path),
                    "source_override": source_override,
                    "rows": per_rows,
                    "positives": per_positive,
                    "negatives": per_negative,
                    "bucket_counts": dict(sorted(per_buckets.items())),
                    "dataset_source_counts": dict(sorted(per_sources.items())),
                }
            )

    summary = {
        "out": str(out_path),
        "manifest": str(manifest_path),
        "inputs": [str(path) for path in inputs],
        "source_names": source_names,
        "rows": total_rows,
        "positives": positives,
        "negatives": negatives,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "dataset_source_counts": dict(sorted(source_counts.items())),
        "features": TRACKLET_FEATURES,
        "datasets": dataset_summaries,
    }
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletClassifierMergeResult(csv_path=out_path, manifest_path=manifest_path, summary=summary)


def _augment_hard_tiny_positives(
    x: torch.Tensor,
    y: torch.Tensor,
    meta: list[dict[str, str]],
    repeats: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, str]], int]:
    if repeats <= 0 or len(y) == 0:
        return x, y, meta, 0
    pos_idx = torch.nonzero(y == 1, as_tuple=False).flatten()
    if pos_idx.numel() == 0:
        return x, y, meta, 0
    feature_index = {name: idx for idx, name in enumerate(TRACKLET_FEATURES)}
    augmented = []
    augmented_meta: list[dict[str, str]] = []
    for idx in pos_idx.tolist():
        base = x[idx].clone()
        for rep in range(repeats):
            row = base.clone()
            severity = 0.75 + 0.10 * (rep % 3)
            row[feature_index["num_rows"]] = min(float(row[feature_index["num_rows"]]), 6.0 + rep)
            row[feature_index["mean_objectness"]] = min(float(row[feature_index["mean_objectness"]]), 0.12 + 0.03 * rep)
            row[feature_index["max_objectness"]] = min(float(row[feature_index["max_objectness"]]), 0.20 + 0.04 * rep)
            row[feature_index["mean_final_score"]] = min(float(row[feature_index["mean_final_score"]]), 0.13 + 0.02 * rep)
            row[feature_index["max_final_score"]] = min(float(row[feature_index["max_final_score"]]), 0.22 + 0.03 * rep)
            crop = min(float(row[feature_index["mean_crop_drone"]]), 0.44 + 0.03 * rep)
            temporal = max(float(row[feature_index["mean_temporal_drone"]]), crop + 0.11, 0.56)
            background = max(float(row[feature_index["mean_background"]]), 0.55)
            final_drone = min(float(row[feature_index["mean_final_drone"]]), 0.34 + 0.04 * rep)
            row[feature_index["mean_crop_drone"]] = crop
            row[feature_index["mean_temporal_drone"]] = min(0.72, temporal)
            row[feature_index["mean_final_drone"]] = final_drone
            row[feature_index["mean_background"]] = min(0.68, background)
            row[feature_index["temporal_minus_crop_mean"]] = row[feature_index["mean_temporal_drone"]] - row[feature_index["mean_crop_drone"]]
            row[feature_index["temporal_minus_background_mean"]] = row[feature_index["mean_temporal_drone"]] - row[feature_index["mean_background"]]
            row[feature_index["final_minus_background_mean"]] = row[feature_index["mean_final_drone"]] - row[feature_index["mean_background"]]
            row[feature_index["temporal_gain_rate"]] = max(float(row[feature_index["temporal_gain_rate"]]), severity)
            row[feature_index["detector_update_rate"]] = 0.0
            row[feature_index["fallback_rate"]] = 0.0
            row[feature_index["validated_rate"]] = 0.0
            row[feature_index["mean_track_drift"]] = min(float(row[feature_index["mean_track_drift"]]), 1.0)
            row[feature_index["max_track_drift"]] = min(float(row[feature_index["max_track_drift"]]), 2.0)
            row[feature_index["mean_track_speed"]] = min(float(row[feature_index["mean_track_speed"]]), 2.0)
            row[feature_index["mean_box_side"]] = min(max(float(row[feature_index["mean_box_side"]]), 90.0), 120.0)
            row[feature_index["std_box_side"]] = min(float(row[feature_index["std_box_side"]]), 8.0)
            row[feature_index["mean_center_step"]] = min(float(row[feature_index["mean_center_step"]]), 5.0)
            row[feature_index["max_center_step"]] = min(float(row[feature_index["max_center_step"]]), 12.0)
            row[feature_index["std_center_step"]] = min(float(row[feature_index["std_center_step"]]), 4.0)
            row[feature_index["track_span_frames"]] = min(float(row[feature_index["track_span_frames"]]), 8.0)
            row[feature_index["frame_density"]] = max(float(row[feature_index["frame_density"]]), 0.8)
            row[feature_index["weak_detector_temporal_signal"]] = row[feature_index["temporal_gain_rate"]]
            row[feature_index["score_above_02_rate"]] = min(float(row[feature_index["score_above_02_rate"]]), 0.30)
            augmented.append(row)
            meta_row = dict(meta[idx]) if idx < len(meta) else {"label": "1"}
            meta_row["label"] = "1"
            meta_row["augmented"] = "hard_tiny_positive"
            augmented_meta.append(meta_row)
    if not augmented:
        return x, y, meta, 0
    aug_x = torch.stack(augmented)
    aug_y = torch.ones(aug_x.shape[0], dtype=y.dtype)
    return torch.cat([x, aug_x], dim=0), torch.cat([y, aug_y], dim=0), meta + augmented_meta, int(aug_x.shape[0])


def _balance_sample_weights(meta: list[dict[str, str]], y: torch.Tensor, balance_by: list[str] | None) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if not balance_by:
        return None, {"enabled": False, "balance_by": []}
    groups: list[str] = []
    counts: dict[str, int] = {}
    for index, row in enumerate(meta):
        parts = []
        for key in balance_by:
            if key == "label":
                value = str(int(y[index].item())) if index < len(y) else str(row.get("label", ""))
            else:
                value = str(row.get(key, "") or "__missing__")
            parts.append(value)
        group = "|".join(parts)
        groups.append(group)
        counts[group] = counts.get(group, 0) + 1
    if len(groups) != len(y):
        raise ValueError("metadata length must match labels for balanced tracklet classifier training")
    weights = torch.tensor([1.0 / max(1, counts[group]) for group in groups], dtype=torch.float32)
    return weights, {
        "enabled": True,
        "balance_by": list(balance_by),
        "group_counts": dict(sorted(counts.items())),
        "min_weight": float(weights.min().item()) if len(weights) else 0.0,
        "max_weight": float(weights.max().item()) if len(weights) else 0.0,
    }


def train_tracklet_classifier(
    csv_path: str | Path,
    out: str | Path,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 32,
    hard_tiny_positive_augments: int = 0,
    balance_by: list[str] | None = None,
) -> Path:
    x, y, meta = _load_tracklet_csv(csv_path)
    if len(y) == 0:
        raise ValueError("Tracklet dataset is empty")
    x, y, meta, num_augmented = _augment_hard_tiny_positives(x, y, meta, hard_tiny_positive_augments)
    sample_weights, balance_summary = _balance_sample_weights(meta, y, balance_by)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    x_norm = (x - mean) / std
    dataset = TensorDataset(x_norm, y)
    sampler = (
        WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        if sample_weights is not None
        else None
    )
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=sampler is None, sampler=sampler)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrackletMLP(in_dim=x.shape[1], hidden=hidden).to(device)
    counts = torch.bincount(y, minlength=2).float()
    weights = counts.sum() / counts.clamp_min(1.0) / 2.0
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(by.numel())
            total += int(by.numel())
        history.append({"epoch": epoch + 1, "loss": total_loss / max(1, total)})
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "features": TRACKLET_FEATURES,
            "mean": mean,
            "std": std,
            "history": history,
            "hidden": hidden,
            "num_training_rows": int(len(y)),
            "num_hard_tiny_positive_augmented": num_augmented,
            "balance": balance_summary,
        },
        out_path,
    )
    return out_path


def evaluate_tracklet_classifier(csv_path: str | Path, weights: str | Path, out: str | Path | None = None, threshold: float = 0.5) -> dict[str, Any]:
    model, features, mean, std = _load_checkpoint(weights)
    x, y, meta = _load_tracklet_csv(csv_path, features=features)
    x_norm = (x - mean) / std
    with torch.no_grad():
        probs = torch.softmax(model(x_norm), dim=1)[:, 1]
    pred = probs >= threshold
    y_bool = y.bool()
    tp = int((pred & y_bool).sum().item())
    fp = int((pred & ~y_bool).sum().item())
    fn = int((~pred & y_bool).sum().item())
    tn = int((~pred & ~y_bool).sum().item())
    metrics = {
        "num_tracklets": int(len(y)),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "accuracy": (tp + tn) / max(1, len(y)),
    }
    if out is not None:
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with (out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
            fields = ["seq", "track_id", "label", "prob_tracklet_drone", "pred_tracklet_drone"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row, prob, p in zip(meta, probs.tolist(), pred.tolist()):
                writer.writerow({"seq": row["seq"], "track_id": row["track_id"], "label": row["label"], "prob_tracklet_drone": prob, "pred_tracklet_drone": int(p)})
    return metrics


def evaluate_tracklet_classifier_thresholds(
    csv_path: str | Path,
    weights: str | Path,
    out: str | Path,
    thresholds: list[float] | None = None,
) -> TrackletClassifierThresholdSweepResult:
    thresholds = thresholds or [round(i / 20.0, 3) for i in range(1, 20)]
    model, features, mean, std = _load_checkpoint(weights)
    x, y, _ = _load_tracklet_csv(csv_path, features=features)
    x_norm = (x - mean) / std
    with torch.no_grad():
        probs = torch.softmax(model(x_norm), dim=1)[:, 1]
    y_bool = y.bool()
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        pred = probs >= float(threshold)
        tp = int((pred & y_bool).sum().item())
        fp = int((pred & ~y_bool).sum().item())
        fn = int((~pred & y_bool).sum().item())
        tn = int((~pred & ~y_bool).sum().item())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
        row = {
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": (tp + tn) / max(1, len(y)),
        }
        rows.append(row)
        if best is None or (row["f1"], row["recall"], -row["fp"]) > (best["f1"], best["recall"], -best["fp"]):
            best = row
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path_out = out_dir / "tracklet_classifier_threshold_sweep.csv"
    json_path = out_dir / "tracklet_classifier_threshold_summary.json"
    with csv_path_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "csv": str(csv_path_out),
        "csv_path": str(csv_path),
        "weights": str(weights),
        "num_tracklets": int(len(y)),
        "positives": int(y_bool.sum().item()),
        "negatives": int((~y_bool).sum().item()),
        "thresholds": [float(v) for v in thresholds],
        "best": best or {},
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletClassifierThresholdSweepResult(csv_path=csv_path_out, summary_path=json_path, summary=summary)


def _write_tracklet_classifier_route_b_results_csv(
    eval_summaries: list[dict[str, Any]],
    out: str | Path,
) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "model_type",
        "tracklet_best_f1",
        "best_f1",
        "best_precision",
        "best_recall",
        "best_accuracy",
        "best_threshold",
        "best_tp",
        "best_fp",
        "best_fn",
        "best_tn",
        "summary_json",
        "threshold_sweep",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in eval_summaries:
            best = dict(row.get("best") or {})
            writer.writerow(
                {
                    "dataset": row.get("dataset", ""),
                    "method": "route_b:tracklet_classifier_mixture",
                    "model_type": "tracklet_classifier_mixture",
                    "tracklet_best_f1": float(best.get("f1", 0.0)),
                    "best_f1": float(best.get("f1", 0.0)),
                    "best_precision": float(best.get("precision", 0.0)),
                    "best_recall": float(best.get("recall", 0.0)),
                    "best_accuracy": float(best.get("accuracy", 0.0)),
                    "best_threshold": float(best.get("threshold", 0.0)),
                    "best_tp": int(best.get("tp", 0)),
                    "best_fp": int(best.get("fp", 0)),
                    "best_fn": int(best.get("fn", 0)),
                    "best_tn": int(best.get("tn", 0)),
                    "summary_json": row.get("threshold_summary", ""),
                    "threshold_sweep": row.get("threshold_sweep", ""),
                }
            )
    return out_path


def _write_tracklet_classifier_frame_route_b_results_csv(
    dataset_summaries: list[dict[str, Any]],
    out: str | Path,
) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "model_type",
        "frame_best_f1",
        "best_f1",
        "best_precision",
        "best_recall",
        "best_accuracy",
        "best_threshold",
        "best_tp",
        "best_fp",
        "best_fn",
        "num_prediction_boxes",
        "num_gt_boxes",
        "summary_json",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset_summaries:
            best = dict(row.get("best") or {})
            metrics = dict(best.get("filtered_metrics") or row.get("filtered_metrics") or {})
            tp = int(metrics.get("tp", 0))
            fp = int(metrics.get("fp", 0))
            fn = int(metrics.get("fn", 0))
            tn = 0
            accuracy = tp / max(1, tp + fp + fn + tn)
            writer.writerow(
                {
                    "dataset": row.get("dataset", ""),
                    "method": "route_b:tracklet_classifier_frame_filtered",
                    "model_type": "tracklet_classifier_frame_filtered",
                    "frame_best_f1": float(metrics.get("f1", 0.0)),
                    "best_f1": float(metrics.get("f1", 0.0)),
                    "best_precision": float(metrics.get("precision", 0.0)),
                    "best_recall": float(metrics.get("recall", 0.0)),
                    "best_accuracy": accuracy,
                    "best_threshold": float(best.get("threshold", 0.0)),
                    "best_tp": tp,
                    "best_fp": fp,
                    "best_fn": fn,
                    "num_prediction_boxes": int(metrics.get("num_prediction_boxes", 0)),
                    "num_gt_boxes": int(metrics.get("num_gt_boxes", 0)),
                    "summary_json": str(Path(row.get("out_dir", "")) / "tracklet_classifier_frame_benchmark_summary.json") if row.get("out_dir") else "",
                }
            )
    return out_path


def run_tracklet_classifier_mixture_benchmark(
    train_csvs: list[str | Path],
    eval_csvs: list[str | Path],
    out_dir: str | Path,
    train_source_names: list[str] | None = None,
    eval_dataset_names: list[str] | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 32,
    hard_tiny_positive_augments: int = 0,
    balance_by: list[str] | None = None,
    thresholds: list[float] | None = None,
    preflight: bool = True,
    strict_preflight: bool = True,
    fail_on_train_eval_sequence_overlap: bool = True,
    baseline_csv: str | Path | None = None,
    baseline_metric: str = "tracklet_best_f1",
    baseline_lower_is_better: bool = False,
    baseline_digits: int = 3,
    allow_invalid_baselines: bool = False,
) -> TrackletClassifierBenchmarkResult:
    if not train_csvs:
        raise ValueError("train_csvs must contain at least one classifier CSV")
    if not eval_csvs:
        raise ValueError("eval_csvs must contain at least one classifier CSV")
    if train_source_names is not None and len(train_source_names) != len(train_csvs):
        raise ValueError("train_source_names must have the same length as train_csvs")
    if eval_dataset_names is not None and len(eval_dataset_names) != len(eval_csvs):
        raise ValueError("eval_dataset_names must have the same length as eval_csvs")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    preflight_result: TrackletClassifierMixturePreflightResult | None = None
    if preflight:
        preflight_result = validate_tracklet_classifier_mixture_inputs(
            train_csvs,
            eval_csvs,
            out_root / "tracklet_classifier_mixture_preflight.json",
            train_source_names=train_source_names,
            eval_dataset_names=eval_dataset_names,
            fail_on_train_eval_sequence_overlap=fail_on_train_eval_sequence_overlap,
        )
        if strict_preflight and not preflight_result.summary.get("valid", False):
            raise ValueError(f"Tracklet classifier mixture preflight failed: {preflight_result.summary.get('errors', [])}")

    train_dir = out_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    merge_result = merge_tracklet_classifier_datasets(
        train_csvs,
        train_dir / "mixed_tracklets.csv",
        source_names=train_source_names,
        manifest_out=train_dir / "mixed_tracklets.manifest.json",
    )
    weights = train_tracklet_classifier(
        merge_result.csv_path,
        train_dir / "joint_tracklet_classifier.pt",
        epochs=epochs,
        lr=lr,
        hidden=hidden,
        hard_tiny_positive_augments=hard_tiny_positive_augments,
        balance_by=balance_by,
    )
    ckpt = torch.load(weights, map_location="cpu")
    eval_summaries = []
    eval_root = out_root / "eval"
    for index, eval_csv in enumerate(eval_csvs):
        dataset_name = eval_dataset_names[index] if eval_dataset_names is not None else Path(eval_csv).parent.name or Path(eval_csv).stem
        safe_name = str(dataset_name).replace("/", "_").replace("\\", "_") or f"dataset_{index}"
        dataset_dir = eval_root / safe_name
        sweep = evaluate_tracklet_classifier_thresholds(eval_csv, weights, dataset_dir, thresholds=thresholds)
        eval_summaries.append(
            {
                "dataset": str(dataset_name),
                "csv": str(eval_csv),
                "out_dir": str(dataset_dir),
                "threshold_sweep": str(sweep.csv_path),
                "threshold_summary": str(sweep.summary_path),
                "best": sweep.summary.get("best", {}),
                "summary": sweep.summary,
            }
        )

    summary_path = out_root / "tracklet_classifier_mixture_benchmark_summary.json"
    route_b_results_csv = _write_tracklet_classifier_route_b_results_csv(
        eval_summaries,
        out_root / "tracklet_classifier_mixture_route_b_results.csv",
    )
    baseline_report = None
    if baseline_csv is not None:
        from qstr_dronedet.tracking.action_policy import (
            compare_route_b_results_to_baselines,
            export_route_b_baseline_markdown_table,
            validate_route_b_baseline_csv,
        )

        baseline_root = out_root / "baseline_report"
        baseline_root.mkdir(parents=True, exist_ok=True)
        validation = validate_route_b_baseline_csv(
            baseline_csv,
            baseline_root / "baseline_validation.json",
            metric=baseline_metric,
            require_metric_values=True,
        )
        if allow_invalid_baselines or validation.summary.get("valid", False):
            comparison = compare_route_b_results_to_baselines(
                route_b_results_csv,
                baseline_csv,
                baseline_root / "comparison",
                metric=baseline_metric,
                higher_is_better=not baseline_lower_is_better,
            )
            markdown = export_route_b_baseline_markdown_table(
                comparison.summary["json"],
                baseline_root / "route_b_tracklet_classifier_baseline_report.md",
                digits=baseline_digits,
            )
            baseline_report = {
                "valid": bool(validation.summary.get("valid", False)),
                "baseline_csv": str(baseline_csv),
                "metric": baseline_metric,
                "higher_is_better": not baseline_lower_is_better,
                "validation_json": str(validation.out_path),
                "comparison_csv": str(comparison.out_path),
                "comparison_json": str(comparison.summary["json"]),
                "ranking_csv": str(comparison.summary["ranking_csv"]),
                "markdown": str(markdown.out_path),
                "route_b_wins": int(comparison.summary["route_b_wins"]),
                "num_comparisons": int(comparison.summary["num_comparisons"]),
                "comparison_rows": comparison.summary["comparison_rows"],
            }
        else:
            baseline_report = {
                "valid": False,
                "baseline_csv": str(baseline_csv),
                "metric": baseline_metric,
                "validation_json": str(validation.out_path),
                "baseline_validation": validation.summary,
            }
            raise ValueError(f"Tracklet classifier baseline CSV failed validation: {validation.summary.get('issues', [])}")

    best_by_dataset = {row["dataset"]: row["best"] for row in eval_summaries}
    summary = {
        "out_dir": str(out_root),
        "summary_json": str(summary_path),
        "train_csvs": [str(path) for path in train_csvs],
        "eval_csvs": [str(path) for path in eval_csvs],
        "train_source_names": train_source_names,
        "eval_dataset_names": [row["dataset"] for row in eval_summaries],
        "epochs": epochs,
        "lr": lr,
        "hidden": hidden,
        "hard_tiny_positive_augments": hard_tiny_positive_augments,
        "balance_by": balance_by,
        "thresholds": thresholds,
        "preflight": preflight_result.summary if preflight_result is not None else None,
        "mixed_train_csv": str(merge_result.csv_path),
        "mixed_train_manifest": str(merge_result.manifest_path),
        "mixed_train": merge_result.summary,
        "weights": str(weights),
        "checkpoint_balance": ckpt.get("balance", {}),
        "eval": eval_summaries,
        "best_by_dataset": best_by_dataset,
        "route_b_results_csv": str(route_b_results_csv),
        "baseline_csv": str(baseline_csv) if baseline_csv is not None else None,
        "baseline_metric": baseline_metric,
        "baseline_report": baseline_report,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletClassifierBenchmarkResult(out_path=summary_path, summary=summary)
