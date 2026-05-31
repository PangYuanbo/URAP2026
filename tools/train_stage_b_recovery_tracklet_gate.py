from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qstr_dronedet.tracking.sequence_gate import SequenceGateConfig, build_sequence_tracklets, sequence_tracklet_features
from tools.evaluate_dji_new2_profile_compare import _center_distance, _iou, _load_gt
from tools.select_stage_b_profile_outputs import _is_drone, _key, _load_jsonl, _recovery_reason, _score


FEATURE_NAMES = [
    "num_rows",
    "span_frames",
    "frame_density",
    "longest_score_streak",
    "score_persistence",
    "mean_crop_drone",
    "mean_temporal_drone",
    "mean_final_drone",
    "mean_background",
    "max_background",
    "mean_score",
    "max_score",
    "mean_objectness",
    "max_objectness",
    "longest_objectness_streak",
    "objectness_persistence",
    "mean_box_side",
    "max_box_side",
    "box_side_cv",
    "mean_center_step",
    "max_center_step",
    "mean_center_step_per_side",
    "max_center_step_per_side",
    "max_frame_gap",
    "detector_support_count",
    "detector_support_rate",
    "longest_detector_streak",
    "detector_persistence",
    "mean_drone",
    "temporal_minus_crop_mean",
    "drone_background_margin",
    "high_background_rate",
    "detector_high_background_rate",
    "detector_high_background_drone_rate",
    "background_detector_contradiction",
    "drone_detector_contradiction",
    "max_track_history_len",
    "max_track_detector_updates",
    "min_frames_since_detector_update",
]


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


def _old_scene_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "scene_recovery_allow_untracked": True,
            "scene_min_track_history_len": 2,
            "scene_min_track_detector_updates": 2,
            "scene_max_frames_since_detector_update": 1,
            "scene_min_track_score": 0.10,
            "scene_min_track_evidence_len": 0,
        }
    )
    return argparse.Namespace(**values)


def _bbox(row: dict[str, Any]) -> list[float]:
    value = row.get("bbox") or row.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
    return [float(v) for v in value[:4]]


def _max_side(row: dict[str, Any]) -> float:
    x1, y1, x2, y2 = _bbox(row)
    return max(0.0, x2 - x1, y2 - y1)


def _match_tracklet(tracklet: list[dict[str, Any]], gt_by_frame: dict[int, list[dict[str, Any]]], iou_threshold: float, center_threshold: float) -> tuple[int, int]:
    hit_rows = 0
    for row in tracklet:
        bbox = _bbox(row)
        frame_id = int(row.get("frame_id", -1))
        if any(
            _iou(bbox, gt["bbox"]) >= iou_threshold or _center_distance(bbox, gt["bbox"]) <= center_threshold
            for gt in gt_by_frame.get(frame_id, [])
        ):
            hit_rows += 1
    return int(hit_rows > 0), hit_rows


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


def _sample_row(
    row: dict[str, Any],
    strict_row: dict[str, Any] | None,
    old_args: argparse.Namespace,
    args: argparse.Namespace,
) -> bool:
    if args.sample_mode == "old_scene_rule":
        return _recovery_reason(row, strict_row, old_args) == "recall_scene_hard_tiny_recovery"
    if args.sample_mode == "suppressed_recall_drone":
        if not _is_drone(row) or _is_drone(strict_row):
            return False
        if _score(row) < args.recall_min_score:
            return False
        if _max_side(row) > args.hard_tiny_max_side:
            return False
        return True
    raise ValueError(f"Unknown sample mode: {args.sample_mode}")


def _collect_recall_root_samples(
    recall_root: Path,
    strict_root: Path | None,
    args: argparse.Namespace,
    source_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old_args = _old_scene_args(args)
    config = SequenceGateConfig(candidate_min_score=0.0, max_gap=args.max_gap, link_radius=args.link_radius, link_radius_per_side=args.link_radius_per_side)
    samples: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for recall_dir in sorted(p for p in recall_root.iterdir() if p.is_dir()):
        strict_dir = strict_root / recall_dir.name if strict_root is not None else None
        if strict_root is not None and (strict_dir is None or not strict_dir.exists()):
            continue
        strict_by_key = {_key(row): row for row in _load_jsonl(strict_dir / "predictions.jsonl")} if strict_dir is not None else {}
        gt_by_frame: dict[int, list[dict[str, Any]]] = {}
        gt_path = (strict_dir / "frame_annotations.csv") if strict_dir is not None else (recall_dir / "frame_annotations.csv")
        if not gt_path.exists():
            continue
        for gt in _load_gt(gt_path):
            gt_by_frame.setdefault(int(gt["frame_id"]), []).append(gt)
        recall_pred_rows = _load_jsonl(recall_dir / "predictions.jsonl")
        recall_diag_rows = _load_jsonl(recall_dir / "diagnostics.jsonl")
        enriched_rows = _enrich_with_diagnostics(recall_pred_rows, recall_diag_rows)
        enriched_by_key = {_key(row): row for row in enriched_rows}
        rows = []
        for row in recall_pred_rows:
            if not _sample_row(row, strict_by_key.get(_key(row)), old_args, args):
                continue
            item = dict(enriched_by_key.get(_key(row), row))
            item["seq"] = recall_dir.name
            rows.append(item)
        for idx, tracklet in enumerate(build_sequence_tracklets(rows, config)):
            if not tracklet:
                continue
            label, hit_rows = _match_tracklet(tracklet, gt_by_frame, args.iou_threshold, args.center_threshold)
            features = sequence_tracklet_features(tracklet, config)
            feature_vector = [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
            track_id = str(tracklet[0].get("sequence_track_id") or f"{recall_dir.name}:{idx}")
            samples.append(
                {
                    "seq": recall_dir.name,
                    "track_id": track_id,
                    "source_name": source_name,
                    "label": label,
                    "hit_rows": hit_rows,
                    "features": feature_vector,
                    "feature_dict": features,
                    "num_rows": len(tracklet),
                }
            )
            csv_rows.append(
                {
                    "source_name": source_name,
                    "seq": recall_dir.name,
                    "track_id": track_id,
                    "label": label,
                    "hit_rows": hit_rows,
                    "sample_mode": args.sample_mode,
                    **{name: feature_vector[i] for i, name in enumerate(FEATURE_NAMES)},
                }
            )
    return samples, csv_rows


def _load_scene_tracklet_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_samples: list[dict[str, Any]] = []
    all_csv_rows: list[dict[str, Any]] = []
    paired_samples, paired_rows = _collect_recall_root_samples(
        Path(args.recall_root),
        Path(args.strict_root),
        args,
        source_name="paired_recall_strict",
    )
    all_samples.extend(paired_samples)
    all_csv_rows.extend(paired_rows)
    for idx, root in enumerate(args.extra_recall_roots or []):
        samples, rows = _collect_recall_root_samples(
            Path(root),
            None,
            args,
            source_name=f"extra_recall_only_{idx + 1}",
        )
        all_samples.extend(samples)
        all_csv_rows.extend(rows)
    return all_samples, all_csv_rows


def _train_logistic(x: np.ndarray, y: np.ndarray, epochs: int, lr: float, seed: int) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    torch.manual_seed(seed)
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    model = torch.nn.Linear(x.shape[1], 1)
    pos = float(y.sum())
    neg = float(len(y) - y.sum())
    pos_weight = torch.tensor([max(1.0, neg / max(1.0, pos))], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = model(x_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == epochs:
            history.append({"epoch": float(epoch + 1), "loss": float(loss.item())})
    weights = model.weight.detach().cpu().numpy().reshape(-1)
    bias = float(model.bias.detach().cpu().numpy()[0])
    return weights, bias, history


def _sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(v, -50.0, 50.0)))


def _select_threshold(probs: np.ndarray, y: np.ndarray, min_precision: float, min_tp: int, tp_weight: float, fp_weight: float) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    thresholds = sorted({float(v) for v in probs.tolist()} | {0.0, 0.5, 0.95, 1.01}, reverse=True)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        pred = probs >= threshold
        tp = int(np.logical_and(pred, y == 1).sum())
        fp = int(np.logical_and(pred, y == 0).sum())
        fn = int(np.logical_and(~pred, y == 1).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, int((y == 1).sum()))
        passes = tp >= min_tp and precision >= min_precision
        utility = tp_weight * tp - fp_weight * fp
        row = {
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "passes": passes,
            "utility": utility,
        }
        rows.append(row)
        if passes:
            if best is None or (utility, precision, recall, -fp) > (best["utility"], best["precision"], best["recall"], -best["fp"]):
                best = row
    if best is None:
        best = {
            "threshold": 1.01,
            "tp": 0,
            "fp": 0,
            "fn": int((y == 1).sum()),
            "precision": 0.0,
            "recall": 0.0,
            "passes": False,
            "utility": 0.0,
            "not_recommended_reason": "No threshold met min precision and min TP; gate disables scene recovery.",
        }
    return float(best["threshold"]), rows, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recall-root", required=True)
    parser.add_argument("--strict-root", required=True)
    parser.add_argument("--extra-recall-roots", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-mode", choices=["old_scene_rule", "suppressed_recall_drone"], default="old_scene_rule")
    parser.add_argument("--hard-tiny-max-side", type=float, default=32.0)
    parser.add_argument("--recall-min-score", type=float, default=0.18)
    parser.add_argument("--recall-min-prob", type=float, default=0.55)
    parser.add_argument("--recall-max-background", type=float, default=0.60)
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--link-radius", type=float, default=24.0)
    parser.add_argument("--link-radius-per-side", type=float, default=1.0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--center-threshold", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=530)
    parser.add_argument("--min-precision", type=float, default=0.20)
    parser.add_argument("--min-tp", type=int, default=1)
    parser.add_argument("--tp-weight", type=float, default=100.0)
    parser.add_argument("--fp-weight", type=float, default=1.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples, csv_rows = _load_scene_tracklet_samples(args)
    _write_csv(out / "scene_tracklet_gate_training_samples.csv", csv_rows)
    if not samples:
        raise ValueError("No scene recovery tracklet samples found")
    y = np.asarray([sample["label"] for sample in samples], dtype=np.float32)
    x_raw = np.asarray([sample["features"] for sample in samples], dtype=np.float32)
    mean = x_raw.mean(axis=0)
    std = x_raw.std(axis=0)
    std[std < 1e-6] = 1.0
    x = (x_raw - mean) / std
    weights, bias, history = _train_logistic(x, y, epochs=args.epochs, lr=args.lr, seed=args.seed)
    probs = _sigmoid(x @ weights + bias)
    threshold, threshold_rows, selected = _select_threshold(
        probs,
        y,
        min_precision=args.min_precision,
        min_tp=args.min_tp,
        tp_weight=args.tp_weight,
        fp_weight=args.fp_weight,
    )
    for row, prob in zip(csv_rows, probs.tolist()):
        row["gate_prob"] = float(prob)
    _write_csv(out / "scene_tracklet_gate_training_samples_scored.csv", csv_rows)
    _write_csv(out / "scene_tracklet_gate_threshold_sweep.csv", threshold_rows)
    gate = {
        "kind": "qstr_scene_recovery_tracklet_logistic_v2",
        "feature_names": FEATURE_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
        "threshold": threshold,
        "sequence_gate_config": {
            "candidate_min_score": 0.0,
            "max_gap": args.max_gap,
            "link_radius": args.link_radius,
            "link_radius_per_side": args.link_radius_per_side,
        },
        "training": {
            "recall_root": args.recall_root,
            "strict_root": args.strict_root,
            "extra_recall_roots": args.extra_recall_roots,
            "num_tracklets": len(samples),
            "positive_tracklets": int(y.sum()),
            "negative_tracklets": int(len(y) - y.sum()),
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "history": history,
            "selected_threshold_metrics": selected,
        },
    }
    gate_path = out / "scene_tracklet_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    summary = {
        "gate": str(gate_path),
        "num_tracklets": len(samples),
        "positive_tracklets": int(y.sum()),
        "negative_tracklets": int(len(y) - y.sum()),
        "threshold": threshold,
        "selected": selected,
    }
    (out / "scene_tracklet_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
