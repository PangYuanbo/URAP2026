from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

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
]


@dataclass
class TrackletDatasetResult:
    csv_path: Path
    json_path: Path
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


def _row_track_key(row: dict[str, Any]) -> str | None:
    track_id = row.get("track_id")
    if track_id is None or track_id == "":
        return None
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
    detector_update_rate = _safe_mean([float("yolo" in s or "fallback" in s or "motion" in s or "seed" in s) for s in sources])
    validated_rate = _safe_mean([float(bool(r.get("track_validated", False))) for r in rows])
    temporal_gain_rate = _safe_mean([float(t > c + 0.05) for c, t in zip(crop_drone, temp_drone)])
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
            key = _row_track_key(row)
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
) -> dict[str, Any]:
    pred_path = Path(predictions_path)
    diag_path = Path(diagnostics_path)
    pred_rows = _load_jsonl(pred_path)
    diag_rows = _load_jsonl(diag_path)
    scores = score_tracklets_from_rows(diag_rows, weights, threshold=threshold)

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
            crop_drone = _prob(out, "crop_probs", "drone")
            temporal_drone = _prob(out, "temporal_probs", "drone")
            final_drone = _prob(out, "final_probs", "drone")
            background = max(_prob(out, "crop_probs", "background"), _prob(out, "temporal_probs", "background"), _prob(out, "final_probs", "background"))
            feats = score.get("features", {})
            tracklet_branch_drone = max(
                float(feats.get("mean_crop_drone", 0.0)),
                float(feats.get("mean_temporal_drone", 0.0)),
                float(feats.get("mean_final_drone", 0.0)),
            )
            tracklet_background = float(feats.get("mean_background", 1.0))
            branch_drone = max(crop_drone, temporal_drone, final_drone, tracklet_branch_drone)
            effective_background = min(background if background > 0 else 1.0, tracklet_background)
            if branch_drone >= promotion_min_branch_drone and effective_background <= promotion_max_background:
                out["raw_predicted_class"] = out.get("predicted_class")
                out["raw_final_drone_score"] = out.get("final_drone_score")
                out["predicted_class"] = "drone"
                out["final_drone_score"] = max(float(out.get("final_drone_score", 0.0)), promotion_score_floor * float(score["prob_tracklet_drone"]))
                out["diagnostic_cause"] = _append_cause(out.get("diagnostic_cause"), "tracklet_promoted")
        return out

    filtered_pred_rows = [update_row(row) for row in pred_rows]
    filtered_diag_rows = [update_row(row) for row in diag_rows]

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

    raw_drone = sum(1 for row in pred_rows if row.get("predicted_class") == "drone")
    filtered_drone = sum(1 for row in filtered_pred_rows if row.get("predicted_class") == "drone")
    rejected = raw_drone - filtered_drone
    summary = {
        "weights": str(weights),
        "threshold": threshold,
        "untracked_policy": untracked_policy,
        "promote_positive_tracklets": promote_positive_tracklets,
        "promotion_score_floor": promotion_score_floor,
        "promotion_min_branch_drone": promotion_min_branch_drone,
        "promotion_max_background": promotion_max_background,
        "num_tracklets": len(scores),
        "raw_drone_predictions": raw_drone,
        "filtered_drone_predictions": filtered_drone,
        "rejected_drone_predictions": rejected,
        "raw_predictions_path": str(raw_pred_path),
        "raw_diagnostics_path": str(raw_diag_path),
        "predictions_path": str(pred_path),
        "diagnostics_path": str(diag_path),
    }
    (pred_path.parent / "tracklet_filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


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


def _augment_hard_tiny_positives(x: torch.Tensor, y: torch.Tensor, repeats: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    if repeats <= 0 or len(y) == 0:
        return x, y, 0
    pos_idx = torch.nonzero(y == 1, as_tuple=False).flatten()
    if pos_idx.numel() == 0:
        return x, y, 0
    feature_index = {name: idx for idx, name in enumerate(TRACKLET_FEATURES)}
    augmented = []
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
    if not augmented:
        return x, y, 0
    aug_x = torch.stack(augmented)
    aug_y = torch.ones(aug_x.shape[0], dtype=y.dtype)
    return torch.cat([x, aug_x], dim=0), torch.cat([y, aug_y], dim=0), int(aug_x.shape[0])


def train_tracklet_classifier(
    csv_path: str | Path,
    out: str | Path,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 32,
    hard_tiny_positive_augments: int = 0,
) -> Path:
    x, y, _ = _load_tracklet_csv(csv_path)
    if len(y) == 0:
        raise ValueError("Tracklet dataset is empty")
    x, y, num_augmented = _augment_hard_tiny_positives(x, y, hard_tiny_positive_augments)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    x_norm = (x - mean) / std
    dataset = TensorDataset(x_norm, y)
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
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
