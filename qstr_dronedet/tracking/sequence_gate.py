from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qstr_dronedet.candidates.merge import bbox_iou, center_distance


@dataclass
class SequenceGateConfig:
    score_threshold: float = 0.20
    candidate_min_score: float = 0.10
    max_gap: int = 2
    link_radius: float = 20.0
    link_radius_per_side: float = 1.5
    link_min_iou: float = 0.02
    min_tracklet_rows: int = 2
    min_streak: int = 2
    min_mean_drone: float = 0.34
    min_mean_margin: float = -0.05
    max_mean_background: float = 0.62
    max_center_step: float = 90.0
    max_frame_gap_for_confirm: int = 2
    strong_single_crop: float = 0.72
    strong_single_temporal: float = 0.72
    strong_single_max_background: float = 0.42
    hard_tiny_max_side: float = 40.0
    hard_tiny_min_rows: int = 2
    hard_tiny_min_temporal: float = 0.55
    hard_tiny_min_temporal_crop_delta: float = 0.10
    hard_tiny_max_background: float = 1.01
    suppress_unlinked: bool = False


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    vals = row.get("bbox", [0.0, 0.0, 0.0, 0.0])
    return tuple(float(v) for v in vals)


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    box = _bbox(row)
    return (
        row.get("seq", "__single_sequence__"),
        int(row.get("frame_id", -1)),
        tuple(round(v, 2) for v in box),
        str(row.get("source", "")),
    )


def _center(row: dict[str, Any]) -> tuple[float, float]:
    x1, y1, x2, y2 = _bbox(row)
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _side(row: dict[str, Any]) -> float:
    x1, y1, x2, y2 = _bbox(row)
    return max(0.0, max(x2 - x1, y2 - y1))


def _prob(row: dict[str, Any], branch: str, cls: str) -> float:
    try:
        return float((row.get(branch) or {}).get(cls, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _max_background(row: dict[str, Any]) -> float:
    return max(
        _prob(row, "crop_probs", "background"),
        _prob(row, "temporal_probs", "background"),
        _prob(row, "feature_probs", "background"),
        _prob(row, "final_probs", "background"),
        _prob(row, "crop_probs", "alignment_artifact"),
        _prob(row, "temporal_probs", "alignment_artifact"),
        _prob(row, "feature_probs", "alignment_artifact"),
        _prob(row, "final_probs", "alignment_artifact"),
    )


def _max_drone(row: dict[str, Any]) -> float:
    return max(
        _prob(row, "crop_probs", "drone"),
        _prob(row, "temporal_probs", "drone"),
        _prob(row, "final_probs", "drone"),
    )


def _row_score(row: dict[str, Any]) -> float:
    return max(float(row.get("final_drone_score", 0.0) or 0.0), float(row.get("objectness", 0.0) or 0.0) * _max_drone(row))


def _source_has_detector(row: dict[str, Any]) -> bool:
    source = str(row.get("source", ""))
    return any(token in source for token in ("yolo", "fallback", "motion", "seed", "oracle", "detector"))


def _track_key(row: dict[str, Any]) -> str | None:
    track_id = row.get("track_id")
    if track_id is None or track_id == "":
        return None
    seq = row.get("seq")
    return f"{seq}:{track_id}" if seq not in (None, "") else str(track_id)


def _longest_true_streak(flags: list[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _candidate_rows(rows: list[dict[str, Any]], config: SequenceGateConfig) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if "bbox" not in row:
            continue
        if row.get("predicted_class") == "drone" or _row_score(row) >= config.candidate_min_score:
            candidates.append(row)
    return candidates


def _link_untracked_sequence(rows: list[dict[str, Any]], config: SequenceGateConfig, start_id: int = 1) -> tuple[list[list[dict[str, Any]]], int]:
    rows = sorted(rows, key=lambda r: (int(r.get("frame_id", 0)), -_row_score(r)))
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_frame.setdefault(int(row.get("frame_id", 0)), []).append(row)
    active: list[dict[str, Any]] = []
    tracklets: list[list[dict[str, Any]]] = []
    next_id = start_id

    for frame_id in sorted(by_frame):
        used: set[int] = set()
        for row in by_frame[frame_id]:
            best_idx = None
            best_score = -1e9
            box = _bbox(row)
            side = _side(row)
            for idx, tr in enumerate(active):
                if idx in used:
                    continue
                gap = frame_id - int(tr["last_frame"])
                if gap <= 0 or gap > config.max_gap:
                    continue
                last_box = tr["last_bbox"]
                last_side = max(last_box[2] - last_box[0], last_box[3] - last_box[1])
                radius = config.link_radius + config.link_radius_per_side * max(side, last_side) + 4.0 * max(0, gap - 1)
                dist = center_distance(last_box, box)
                ov = bbox_iou(last_box, box)
                if dist > radius and ov < config.link_min_iou:
                    continue
                score = 2.0 * ov - dist / max(radius, 1e-6) + 0.05 * _row_score(row)
                if score > best_score:
                    best_idx = idx
                    best_score = score
            item = dict(row)
            if best_idx is None:
                item["sequence_track_id"] = f"seqgate_{next_id}"
                next_id += 1
                tracklets.append([item])
                active.append({"rows": tracklets[-1], "last_bbox": box, "last_frame": frame_id})
            else:
                tr = active[best_idx]
                item["sequence_track_id"] = tr["rows"][0]["sequence_track_id"]
                tr["rows"].append(item)
                tr["last_bbox"] = box
                tr["last_frame"] = frame_id
                used.add(best_idx)
        active = [tr for tr in active if frame_id - int(tr["last_frame"]) <= config.max_gap]
    return tracklets, next_id


def build_sequence_tracklets(rows: list[dict[str, Any]], config: SequenceGateConfig) -> list[list[dict[str, Any]]]:
    candidates = _candidate_rows(rows, config)
    existing: dict[str, list[dict[str, Any]]] = {}
    untracked_by_seq: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        key = _track_key(row)
        if key is not None:
            item = dict(row)
            item["sequence_track_id"] = key
            existing.setdefault(key, []).append(item)
        else:
            seq = str(row.get("seq", "__single_sequence__"))
            untracked_by_seq.setdefault(seq, []).append(row)

    tracklets = [sorted(v, key=lambda r: int(r.get("frame_id", 0))) for v in existing.values()]
    next_id = 1
    for seq, seq_rows in sorted(untracked_by_seq.items()):
        linked, next_id = _link_untracked_sequence(seq_rows, config, start_id=next_id)
        for tr in linked:
            for row in tr:
                row["seq"] = seq
            tracklets.append(tr)
    return tracklets


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _max(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _std_over_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = float(np.mean(values))
    if abs(mean) < 1e-6:
        return 0.0
    return float(np.std(values) / abs(mean))


def _float_field(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def sequence_tracklet_features(rows: list[dict[str, Any]], config: SequenceGateConfig) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: int(r.get("frame_id", 0)))
    frame_ids = [int(r.get("frame_id", 0)) for r in rows]
    centers = [_center(r) for r in rows]
    center_steps = [
        float(np.hypot(centers[i][0] - centers[i - 1][0], centers[i][1] - centers[i - 1][1]))
        for i in range(1, len(centers))
    ]
    gaps = [max(0, frame_ids[i] - frame_ids[i - 1] - 1) for i in range(1, len(frame_ids))]
    crop = [_prob(r, "crop_probs", "drone") for r in rows]
    temporal = [_prob(r, "temporal_probs", "drone") for r in rows]
    final = [_prob(r, "final_probs", "drone") for r in rows]
    background = [_max_background(r) for r in rows]
    scores = [float(r.get("final_drone_score", 0.0) or 0.0) for r in rows]
    objectness = [_float_field(r, "objectness", 0.0) for r in rows]
    sides = [_side(r) for r in rows]
    detector_support = [float(_source_has_detector(r) or bool(r.get("track_validated", False))) for r in rows]
    detector_flags = [v > 0.0 for v in detector_support]
    span = max(frame_ids) - min(frame_ids) + 1 if frame_ids else 0
    score_flags = [r.get("predicted_class") == "drone" and float(r.get("final_drone_score", 0.0) or 0.0) >= config.score_threshold for r in rows]
    objectness_flags = [v >= max(0.10, config.candidate_min_score) for v in objectness]
    mean_side = _mean(sides)
    center_step_per_side = [step / max(mean_side, 1.0) for step in center_steps]
    high_background_flags = [b >= 0.60 for b in background]
    max_drone_per_row = [max(c, t, f) for c, t, f in zip(crop, temporal, final)]
    high_drone_flags = [d >= 0.55 for d in max_drone_per_row]
    detector_high_background = [d and b for d, b in zip(detector_flags, high_background_flags)]
    detector_high_background_drone = [
        d and b and h for d, b, h in zip(detector_flags, high_background_flags, high_drone_flags)
    ]
    detector_high_background_flags = [bool(v) for v in detector_high_background]
    detector_high_background_drone_flags = [bool(v) for v in detector_high_background_drone]
    detector_objectness = [o for o, d in zip(objectness, detector_flags) if d]
    detector_background = [b for b, d in zip(background, detector_flags) if d]
    detector_drone = [dmax for dmax, d in zip(max_drone_per_row, detector_flags) if d]
    track_history = [_float_field(r, "track_history_len", 0.0) for r in rows]
    track_detector_updates = [_float_field(r, "track_detector_updates", 0.0) for r in rows]
    frames_since_detector = [_float_field(r, "track_frames_since_detector_update", 999.0) for r in rows]
    finite_frames_since_detector = [v for v in frames_since_detector if v < 999.0]
    return {
        "num_rows": len(rows),
        "span_frames": span,
        "frame_density": len(rows) / max(1, span),
        "longest_score_streak": _longest_true_streak(score_flags),
        "score_persistence": _longest_true_streak(score_flags) / max(1, span),
        "mean_crop_drone": _mean(crop),
        "mean_temporal_drone": _mean(temporal),
        "mean_final_drone": _mean(final),
        "mean_background": _mean(background),
        "max_background": _max(background),
        "mean_score": _mean(scores),
        "max_score": _max(scores),
        "mean_objectness": _mean(objectness),
        "max_objectness": _max(objectness),
        "longest_objectness_streak": _longest_true_streak(objectness_flags),
        "objectness_persistence": _longest_true_streak(objectness_flags) / max(1, span),
        "mean_box_side": _mean(sides),
        "max_box_side": _max(sides),
        "box_side_cv": _std_over_mean(sides),
        "mean_center_step": _mean(center_steps),
        "max_center_step": _max(center_steps),
        "mean_center_step_per_side": _mean(center_step_per_side),
        "max_center_step_per_side": _max(center_step_per_side),
        "max_frame_gap": max(gaps) if gaps else 0,
        "detector_support_count": int(sum(detector_flags)),
        "detector_support_rate": _mean(detector_support),
        "longest_detector_streak": _longest_true_streak(detector_flags),
        "detector_persistence": _longest_true_streak(detector_flags) / max(1, span),
        "mean_drone": _mean(max_drone_per_row),
        "temporal_minus_crop_mean": _mean([t - c for c, t in zip(crop, temporal)]),
        "drone_background_margin": _mean([max(c, t, f) - b for c, t, f, b in zip(crop, temporal, final, background)]),
        "high_background_rate": _mean([float(v) for v in high_background_flags]),
        "detector_high_background_rate": _mean([float(v) for v in detector_high_background]),
        "detector_high_background_drone_rate": _mean([float(v) for v in detector_high_background_drone]),
        "longest_detector_high_background_streak": _longest_true_streak(detector_high_background_flags),
        "detector_high_background_persistence": _longest_true_streak(detector_high_background_flags) / max(1, span),
        "longest_detector_high_background_drone_streak": _longest_true_streak(detector_high_background_drone_flags),
        "detector_high_background_drone_persistence": _longest_true_streak(detector_high_background_drone_flags) / max(1, span),
        "mean_detector_objectness": _mean(detector_objectness),
        "mean_detector_background": _mean(detector_background),
        "mean_detector_drone": _mean(detector_drone),
        "background_detector_contradiction": _mean(
            [
                max(0.0, b - d) * float(s)
                for b, d, s in zip(background, max_drone_per_row, detector_support)
            ]
        ),
        "drone_detector_contradiction": _mean(
            [
                max(0.0, d - b) * float(s)
                for b, d, s in zip(background, max_drone_per_row, detector_support)
            ]
        ),
        "max_track_history_len": _max(track_history),
        "max_track_detector_updates": _max(track_detector_updates),
        "min_frames_since_detector_update": float(np.min(finite_frames_since_detector)) if finite_frames_since_detector else 999.0,
    }


def _tracklet_confirmed(features: dict[str, Any], config: SequenceGateConfig) -> tuple[bool, str]:
    strong_single = (
        int(features["num_rows"]) == 1
        and float(features["mean_crop_drone"]) >= config.strong_single_crop
        and float(features["mean_temporal_drone"]) >= config.strong_single_temporal
        and float(features["mean_background"]) <= config.strong_single_max_background
    )
    if strong_single:
        return True, "strong_single_frame"

    hard_tiny = (
        int(features["num_rows"]) >= config.hard_tiny_min_rows
        and int(features["longest_score_streak"]) >= config.min_streak
        and float(features["mean_box_side"]) <= config.hard_tiny_max_side
        and float(features["mean_temporal_drone"]) >= config.hard_tiny_min_temporal
        and float(features["temporal_minus_crop_mean"]) >= config.hard_tiny_min_temporal_crop_delta
        and float(features["mean_background"]) <= config.hard_tiny_max_background
        and float(features["max_center_step"]) <= config.max_center_step
        and int(features["max_frame_gap"]) <= max(config.max_frame_gap_for_confirm, config.max_gap + 2)
        and float(features["detector_support_rate"]) > 0.0
    )
    if hard_tiny:
        return True, "hard_tiny_sequence"

    confirmed = (
        int(features["num_rows"]) >= config.min_tracklet_rows
        and int(features["longest_score_streak"]) >= config.min_streak
        and float(features["mean_drone"]) >= config.min_mean_drone
        and float(features["drone_background_margin"]) >= config.min_mean_margin
        and float(features["mean_background"]) <= config.max_mean_background
        and float(features["max_center_step"]) <= config.max_center_step
        and int(features["max_frame_gap"]) <= config.max_frame_gap_for_confirm
    )
    return confirmed, "sequence_consistent" if confirmed else "sequence_inconsistent"


def _append_cause(cause: Any, value: str) -> str:
    if cause is None or cause == "":
        return value
    text = str(cause)
    if value in text.split("+"):
        return text
    return f"{text}+{value}"


def sequence_consistency_gate_rows(
    pred_rows: list[dict[str, Any]],
    diag_rows: list[dict[str, Any]],
    config: SequenceGateConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = config or SequenceGateConfig()
    tracklets = build_sequence_tracklets(diag_rows, config)
    decisions: dict[tuple[Any, ...], dict[str, Any]] = {}
    confirmed_tracklets = 0
    rejected_tracklets = 0
    reasons: dict[str, int] = {}
    for tracklet in tracklets:
        features = sequence_tracklet_features(tracklet, config)
        confirmed, reason = _tracklet_confirmed(features, config)
        confirmed_tracklets += int(confirmed)
        rejected_tracklets += int(not confirmed)
        reasons[reason] = reasons.get(reason, 0) + 1
        track_id = str(tracklet[0].get("sequence_track_id", ""))
        for row in tracklet:
            decisions[_row_key(row)] = {
                "sequence_track_id": track_id,
                "sequence_gate_confirmed": confirmed,
                "sequence_gate_reason": reason,
                "sequence_gate_features": features,
            }

    def update(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        decision = decisions.get(_row_key(out))
        if decision is None:
            out["sequence_gate_confirmed"] = None
            out["sequence_gate_reason"] = "unlinked"
            if config.suppress_unlinked and out.get("predicted_class") == "drone" and float(out.get("final_drone_score", 0.0) or 0.0) >= config.score_threshold:
                out["raw_predicted_class"] = out.get("predicted_class")
                out["raw_final_drone_score"] = out.get("final_drone_score")
                out["predicted_class"] = "background"
                out["final_drone_score"] = 0.0
                out["diagnostic_cause"] = _append_cause(out.get("diagnostic_cause"), "sequence_unlinked_rejected")
            return out
        out.update(decision)
        should_filter = (
            out.get("predicted_class") == "drone"
            and float(out.get("final_drone_score", 0.0) or 0.0) >= config.score_threshold
            and not bool(decision["sequence_gate_confirmed"])
        )
        if should_filter:
            out["raw_predicted_class"] = out.get("predicted_class")
            out["raw_final_drone_score"] = out.get("final_drone_score")
            out["predicted_class"] = "background"
            out["final_drone_score"] = 0.0
            probs = dict(out.get("final_probs") or {})
            probs["drone"] = min(float(probs.get("drone", 0.0)), 0.05)
            probs["background"] = max(float(probs.get("background", 0.0)), 0.85)
            out["final_probs"] = probs
            out["diagnostic_cause"] = _append_cause(out.get("diagnostic_cause"), "sequence_gate_rejected")
        elif out.get("predicted_class") == "drone" and bool(decision["sequence_gate_confirmed"]):
            out["diagnostic_cause"] = _append_cause(out.get("diagnostic_cause"), "sequence_gate_confirmed")
        return out

    filtered_pred = [update(r) for r in pred_rows]
    filtered_diag = [update(r) for r in diag_rows]
    raw_drone = sum(1 for r in pred_rows if r.get("predicted_class") == "drone" and float(r.get("final_drone_score", 0.0) or 0.0) >= config.score_threshold)
    filtered_drone = sum(1 for r in filtered_pred if r.get("predicted_class") == "drone" and float(r.get("final_drone_score", 0.0) or 0.0) >= config.score_threshold)
    summary = {
        "score_threshold": config.score_threshold,
        "candidate_min_score": config.candidate_min_score,
        "num_sequence_tracklets": len(tracklets),
        "confirmed_tracklets": confirmed_tracklets,
        "rejected_tracklets": rejected_tracklets,
        "decision_reasons": reasons,
        "raw_drone_predictions": raw_drone,
        "filtered_drone_predictions": filtered_drone,
        "rejected_drone_predictions": raw_drone - filtered_drone,
    }
    return filtered_pred, filtered_diag, summary


def apply_sequence_consistency_gate_to_infer_outputs(
    predictions_path: str | Path,
    diagnostics_path: str | Path,
    out_dir: str | Path | None = None,
    config: SequenceGateConfig | None = None,
) -> dict[str, Any]:
    pred_path = Path(predictions_path)
    diag_path = Path(diagnostics_path)
    pred_rows = _load_jsonl(pred_path)
    diag_rows = _load_jsonl(diag_path)
    filtered_pred, filtered_diag, summary = sequence_consistency_gate_rows(pred_rows, diag_rows, config=config)

    if out_dir is None:
        pred_before = pred_path.with_name("predictions_before_sequence_gate.jsonl")
        diag_before = diag_path.with_name("diagnostics_before_sequence_gate.jsonl")
        pred_path.replace(pred_before)
        diag_path.replace(diag_before)
        pred_out = pred_path
        diag_out = diag_path
    else:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pred_before = pred_path
        diag_before = diag_path
        pred_out = out / pred_path.name
        diag_out = out / diag_path.name

    _write_jsonl(pred_out, filtered_pred)
    _write_jsonl(diag_out, filtered_diag)
    summary.update(
        {
            "raw_predictions_path": str(pred_before),
            "raw_diagnostics_path": str(diag_before),
            "predictions_path": str(pred_out),
            "diagnostics_path": str(diag_out),
        }
    )
    (Path(out_dir) if out_dir else pred_out.parent).joinpath("sequence_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
