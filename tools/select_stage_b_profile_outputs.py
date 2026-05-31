from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_dji_new2_profile_compare import evaluate as evaluate_profile_compare
from qstr_dronedet.tracking.sequence_gate import SequenceGateConfig, build_sequence_tracklets, sequence_tracklet_features


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _bbox(row: dict[str, Any]) -> list[float]:
    value = row.get("bbox") or row.get("bbox_xyxy") or row.get("proposal_bbox_xyxy") or [0, 0, 0, 0]
    return [float(v) for v in value[:4]]


def _key(row: dict[str, Any]) -> tuple[int, tuple[float, float, float, float]]:
    bbox = _bbox(row)
    return (
        int(row.get("frame_id", -1)),
        (round(bbox[0], 3), round(bbox[1], 3), round(bbox[2], 3), round(bbox[3], 3)),
    )


def _max_side(row: dict[str, Any]) -> float:
    x1, y1, x2, y2 = _bbox(row)
    return max(0.0, x2 - x1, y2 - y1)


def _prob(row: dict[str, Any], key: str) -> float:
    probs = row.get("final_probs")
    if not isinstance(probs, dict):
        return 0.0
    try:
        return float(probs.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _score(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    try:
        return float(row.get("final_drone_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _is_drone(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("predicted_class") == "drone")


def _track_support(row: dict[str, Any]) -> bool:
    source = str(row.get("source") or "")
    cause = str(row.get("diagnostic_cause") or "")
    if "tracker" in source:
        return True
    if cause in {"tracklet_confirmed", "tracklet_temporal_only_protected", "hard_tiny_recovery"}:
        return True
    if row.get("track_recognition_confirmed") is True:
        return True
    try:
        return float(row.get("tracklet_classifier_prob", 0.0)) >= 0.75 and row.get("tracklet_is_drone") is True
    except (TypeError, ValueError):
        return False


def _scene_support(row: dict[str, Any]) -> bool:
    return str(row.get("mode") or "") in {
        "static_or_hovering",
        "bad_alignment_fast_egomotion",
        "fast_target",
    }


def _float_field(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _scene_recovery_support(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if not _scene_support(row):
        return False
    if getattr(args, "scene_recovery_allow_untracked", False):
        return True
    history_len = _float_field(row, "track_history_len", 0.0)
    detector_updates = _float_field(row, "track_detector_updates", 0.0)
    frames_since_detector = _float_field(row, "track_frames_since_detector_update", 999.0)
    track_score = _float_field(row, "track_score", 0.0)
    evidence_len = _float_field(row, "track_evidence_len", 0.0)
    return (
        history_len >= getattr(args, "scene_min_track_history_len", 2)
        and detector_updates >= getattr(args, "scene_min_track_detector_updates", 2)
        and frames_since_detector <= getattr(args, "scene_max_frames_since_detector_update", 1)
        and track_score >= getattr(args, "scene_min_track_score", 0.10)
        and evidence_len >= getattr(args, "scene_min_track_evidence_len", 0)
    )


def _load_scene_tracklet_gate(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        gate = json.load(f)
    for key in ("feature_names", "mean", "std", "weights", "bias", "threshold"):
        if key not in gate:
            raise ValueError(f"Scene tracklet gate is missing key: {key}")
    return gate


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _score_scene_gate_features(features: dict[str, Any], gate: dict[str, Any]) -> float:
    names = list(gate["feature_names"])
    mean = [float(v) for v in gate["mean"]]
    std = [max(float(v), 1e-6) for v in gate["std"]]
    weights = [float(v) for v in gate["weights"]]
    value = float(gate["bias"])
    for idx, name in enumerate(names):
        x = (float(features.get(name, 0.0)) - mean[idx]) / std[idx]
        value += weights[idx] * x
    return _sigmoid(value)


def _scene_gate_row_scores(
    rows: list[dict[str, Any]],
    gate: dict[str, Any] | None,
    sequence: str,
) -> dict[tuple[int, tuple[float, float, float, float]], dict[str, Any]]:
    if gate is None:
        return {}
    keyed: dict[tuple[int, tuple[float, float, float, float]], dict[str, Any]] = {}
    enriched = []
    for row in rows:
        item = dict(row)
        item["seq"] = sequence
        enriched.append(item)
    config_data = gate.get("sequence_gate_config") or {}
    config = SequenceGateConfig(
        candidate_min_score=float(config_data.get("candidate_min_score", 0.0)),
        max_gap=int(config_data.get("max_gap", 4)),
        link_radius=float(config_data.get("link_radius", 24.0)),
        link_radius_per_side=float(config_data.get("link_radius_per_side", 1.0)),
    )
    for tracklet in build_sequence_tracklets(enriched, config):
        if not tracklet:
            continue
        features = sequence_tracklet_features(tracklet, config)
        prob = _score_scene_gate_features(features, gate)
        passed = prob >= float(gate.get("threshold", 1.0))
        track_id = str(tracklet[0].get("sequence_track_id", ""))
        for row in tracklet:
            keyed[_key(row)] = {
                "scene_tracklet_gate_prob": float(prob),
                "scene_tracklet_gate_pass": bool(passed),
                "scene_tracklet_gate_id": track_id,
                "scene_tracklet_gate_num_rows": int(features.get("num_rows", 0)),
            }
    return keyed


def _enrich_with_diagnostics(pred_rows: list[dict[str, Any]], diag_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diag_by_key = {_key(row): row for row in diag_rows}
    enriched: list[dict[str, Any]] = []
    for row in pred_rows:
        diag = diag_by_key.get(_key(row))
        if diag is None:
            enriched.append(row)
            continue
        item = dict(row)
        for key in (
            "crop_probs",
            "feature_probs",
            "temporal_probs",
            "track_history_len",
            "track_detector_updates",
            "track_frames_since_detector_update",
            "track_score",
            "track_evidence_len",
            "track_id",
            "track_validated",
        ):
            if key in diag and key not in item:
                item[key] = diag[key]
        enriched.append(item)
    return enriched


def _recovery_reason(
    recall_row: dict[str, Any],
    strict_row: dict[str, Any] | None,
    args: argparse.Namespace,
    scene_gate_score: dict[str, Any] | None = None,
) -> str | None:
    if not _is_drone(recall_row):
        return None
    if _is_drone(strict_row):
        return None
    if _score(recall_row) < args.recall_min_score:
        return None
    drone_prob = _prob(recall_row, "drone")
    background = _prob(recall_row, "background")
    hard_tiny = _max_side(recall_row) <= args.hard_tiny_max_side
    gate_background_override = (
        hard_tiny
        and getattr(args, "scene_tracklet_gate_required", False)
        and getattr(args, "scene_tracklet_gate_override_background", False)
        and scene_gate_score
        and scene_gate_score.get("scene_tracklet_gate_pass") is True
    )
    if drone_prob < args.recall_min_prob:
        return None
    if background > args.recall_max_background and not gate_background_override:
        return None

    if _track_support(recall_row):
        return "recall_track_supported_recovery"
    if hard_tiny and getattr(args, "scene_tracklet_gate_required", False):
        if scene_gate_score and scene_gate_score.get("scene_tracklet_gate_pass") is True:
            return "recall_scene_hard_tiny_recovery"
        return None
    if hard_tiny and _scene_recovery_support(recall_row, args):
        return "recall_scene_hard_tiny_recovery"
    return None


def _row_sort_key(row: dict[str, Any]) -> tuple[int, float, float, float, float]:
    x1, y1, x2, y2 = _bbox(row)
    return (int(row.get("frame_id", -1)), x1, y1, x2, y2)


def select_rows(
    recall_rows: list[dict[str, Any]],
    strict_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    scene_gate_scores: dict[tuple[int, tuple[float, float, float, float]], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    recall_by_key = {_key(row): row for row in recall_rows}
    strict_by_key = {_key(row): row for row in strict_rows}
    keys = set(recall_by_key) | set(strict_by_key)
    selected: list[dict[str, Any]] = []
    counts = {
        "strict_default": 0,
        "recall_track_supported_recovery": 0,
        "recall_scene_hard_tiny_recovery": 0,
        "recall_only_recovery": 0,
    }
    for key in sorted(keys):
        recall_row = recall_by_key.get(key)
        strict_row = strict_by_key.get(key)
        chosen = dict(strict_row or recall_row or {})
        reason = None
        if recall_row is not None:
            scene_gate_score = (scene_gate_scores or {}).get(key)
            reason = _recovery_reason(recall_row, strict_row, args, scene_gate_score=scene_gate_score)
        if reason and recall_row is not None:
            chosen = dict(recall_row)
            counts[reason] += 1
            if strict_row is None:
                counts["recall_only_recovery"] += 1
        else:
            counts["strict_default"] += 1
            reason = "strict_default"
        chosen["stage_b_profile_selected"] = "recall_oriented" if reason != "strict_default" else "strict_fp_control"
        chosen["stage_b_profile_selection_reason"] = reason
        chosen["recall_profile_score"] = _score(recall_row)
        chosen["strict_profile_score"] = _score(strict_row)
        if recall_row is not None and scene_gate_scores:
            chosen.update(scene_gate_scores.get(_key(recall_row), {}))
        selected.append(chosen)
    return sorted(selected, key=_row_sort_key), counts


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def select_profile(args: argparse.Namespace) -> dict[str, Any]:
    recall_root = Path(args.recall_root)
    strict_root = Path(args.strict_root)
    out_root = Path(args.out) / args.profile_name
    scene_tracklet_gate = _load_scene_tracklet_gate(args.scene_tracklet_gate)
    args.scene_tracklet_gate_required = scene_tracklet_gate is not None
    sequences = args.sequences
    if not sequences:
        recall_sequences = {p.name for p in recall_root.iterdir() if p.is_dir()}
        strict_sequences = {p.name for p in strict_root.iterdir() if p.is_dir()}
        sequences = sorted(recall_sequences & strict_sequences)
    if not sequences:
        raise ValueError("No common sequence directories found")

    total_counts: dict[str, int] = {}
    sequence_summaries: list[dict[str, Any]] = []
    for sequence in sequences:
        recall_dir = recall_root / sequence
        strict_dir = strict_root / sequence
        out_dir = out_root / sequence
        if not recall_dir.exists() or not strict_dir.exists():
            raise FileNotFoundError(f"Missing sequence in recall or strict root: {sequence}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("frame_annotations.csv", "run_meta.json"):
            _copy_if_exists(strict_dir / name, out_dir / name)
        for name in ("predictions_raw.jsonl", "diagnostics_raw.jsonl"):
            _copy_if_exists(strict_dir / name, out_dir / name)

        recall_pred_rows = _load_jsonl(recall_dir / "predictions.jsonl")
        recall_diag_rows = _load_jsonl(recall_dir / "diagnostics.jsonl")
        pred_gate_rows = _enrich_with_diagnostics(recall_pred_rows, recall_diag_rows)
        pred_gate_scores = _scene_gate_row_scores(pred_gate_rows, scene_tracklet_gate, sequence)
        diag_gate_scores = _scene_gate_row_scores(recall_diag_rows, scene_tracklet_gate, sequence)

        pred_rows, pred_counts = select_rows(
            recall_pred_rows,
            _load_jsonl(strict_dir / "predictions.jsonl"),
            args,
            scene_gate_scores=pred_gate_scores,
        )
        diag_rows, diag_counts = select_rows(
            recall_diag_rows,
            _load_jsonl(strict_dir / "diagnostics.jsonl"),
            args,
            scene_gate_scores=diag_gate_scores,
        )
        _write_jsonl(out_dir / "predictions.jsonl", pred_rows)
        _write_jsonl(out_dir / "diagnostics.jsonl", diag_rows)
        for key, value in pred_counts.items():
            total_counts[key] = total_counts.get(key, 0) + value
        sequence_summaries.append(
            {
                "sequence": sequence,
                "predictions": len(pred_rows),
                "diagnostics": len(diag_rows),
                **{f"pred_{key}": value for key, value in pred_counts.items()},
                **{f"diag_{key}": value for key, value in diag_counts.items()},
            }
        )

    result = {
        "profile_name": args.profile_name,
        "recall_root": str(recall_root),
        "strict_root": str(strict_root),
        "out_root": str(out_root),
        "sequences": sequences,
        "selection_counts": total_counts,
        "sequence_summaries": sequence_summaries,
        "rules": {
            "hard_tiny_max_side": args.hard_tiny_max_side,
            "recall_min_score": args.recall_min_score,
            "recall_min_prob": args.recall_min_prob,
            "recall_max_background": args.recall_max_background,
            "scene_recovery_allow_untracked": args.scene_recovery_allow_untracked,
            "scene_min_track_history_len": args.scene_min_track_history_len,
            "scene_min_track_detector_updates": args.scene_min_track_detector_updates,
            "scene_max_frames_since_detector_update": args.scene_max_frames_since_detector_update,
            "scene_min_track_score": args.scene_min_track_score,
            "scene_min_track_evidence_len": args.scene_min_track_evidence_len,
            "scene_tracklet_gate": args.scene_tracklet_gate,
            "scene_tracklet_gate_required": args.scene_tracklet_gate_required,
            "scene_tracklet_gate_override_background": args.scene_tracklet_gate_override_background,
        },
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    _write_csv(Path(args.out) / "selection_summary.csv", sequence_summaries)
    (Path(args.out) / "selection_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.evaluate:
        eval_result = evaluate_profile_compare(
            Path(args.out),
            sequences,
            [args.profile_name],
            iou_threshold=args.iou_threshold,
            center_threshold=args.center_threshold,
        )
        _write_csv(Path(args.out) / "summary.csv", eval_result["aggregate_rows"] + eval_result["summary_rows"])
        _write_csv(Path(args.out) / "frame_timeline.csv", eval_result["frame_rows"])
        (Path(args.out) / "summary.json").write_text(json.dumps(eval_result, indent=2), encoding="utf-8")
        result["evaluation"] = eval_result["aggregate_rows"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recall-root", required=True, help="yolo_only directory from recall-oriented Stage B run")
    parser.add_argument("--strict-root", required=True, help="yolo_only directory from strict FP-control Stage B run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile-name", default="source_scene_stageb_select")
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--hard-tiny-max-side", type=float, default=32.0)
    parser.add_argument("--recall-min-score", type=float, default=0.18)
    parser.add_argument("--recall-min-prob", type=float, default=0.55)
    parser.add_argument("--recall-max-background", type=float, default=0.60)
    parser.add_argument("--scene-recovery-allow-untracked", action="store_true")
    parser.add_argument("--scene-min-track-history-len", type=int, default=2)
    parser.add_argument("--scene-min-track-detector-updates", type=int, default=2)
    parser.add_argument("--scene-max-frames-since-detector-update", type=int, default=1)
    parser.add_argument("--scene-min-track-score", type=float, default=0.10)
    parser.add_argument("--scene-min-track-evidence-len", type=int, default=0)
    parser.add_argument("--scene-tracklet-gate", default="", help="Optional JSON logistic gate for scene hard-tiny recovery")
    parser.add_argument(
        "--scene-tracklet-gate-override-background",
        action="store_true",
        help="Allow a passing scene tracklet gate to recover hard-tiny rows even when final background exceeds recall-max-background.",
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--center-threshold", type=float, default=16.0)
    args = parser.parse_args()
    result = select_profile(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
