from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from qstr_dronedet.tracking.tracklet_classifier import (
    _append_cause,
    _box_center,
    _box_side,
    _features,
    _prob,
    _promotion_evidence,
    _row_track_key,
    _selective_tracklet_key_allowlist,
)


TRACKLET_FRAME_FEATURES = [
    "objectness",
    "final_drone_score",
    "crop_drone",
    "feature_drone",
    "temporal_drone",
    "final_drone",
    "crop_background",
    "temporal_background",
    "final_background",
    "max_background",
    "alignment_artifact",
    "motion_score",
    "alignment_quality",
    "track_score",
    "track_drift",
    "track_speed",
    "box_side",
    "center_step",
    "frame_gap",
    "source_yolo",
    "source_fallback",
    "source_tracker",
    "source_motion",
    "track_validated",
    "predicted_drone",
    "temporal_minus_crop",
    "temporal_minus_background",
    "final_minus_background",
]


@dataclass
class SequenceSample:
    seq: str
    track_id: str
    label: int
    rows: list[dict[str, Any]]


class TrackletSequenceGRU(torch.nn.Module):
    def __init__(self, in_dim: int = len(TRACKLET_FRAME_FEATURES), hidden: int = 32) -> None:
        super().__init__()
        self.gru = torch.nn.GRU(in_dim, hidden, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(h[-1])


def _load_sequence_checkpoint(weights: str | Path) -> tuple[TrackletSequenceGRU, list[str], torch.Tensor, torch.Tensor, int]:
    ckpt = torch.load(weights, map_location="cpu")
    features = list(ckpt.get("frame_features", TRACKLET_FRAME_FEATURES))
    hidden = int(ckpt.get("hidden", ckpt["state_dict"]["gru.weight_hh_l0"].shape[1]))
    max_len = int(ckpt.get("max_len", 16))
    model = TrackletSequenceGRU(in_dim=len(features), hidden=hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, features, ckpt["mean"], ckpt["std"].clamp_min(1e-6), max_len


def _load_tracklet_jsonl(path: str | Path) -> list[SequenceSample]:
    samples: list[SequenceSample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            meta = item.get("meta") or {}
            rows = item.get("rows") or []
            samples.append(
                SequenceSample(
                    seq=str(meta.get("seq", "")),
                    track_id=str(meta.get("track_id", "")),
                    label=int(float(meta.get("label", 0))),
                    rows=rows,
                )
            )
    return samples


def _row_frame_features(row: dict[str, Any], prev_row: dict[str, Any] | None) -> list[float]:
    source = str(row.get("source", ""))
    center = _box_center(row)
    if prev_row is None:
        center_step = 0.0
        frame_gap = 0.0
    else:
        prev_center = _box_center(prev_row)
        center_step = float(np.hypot(center[0] - prev_center[0], center[1] - prev_center[1]))
        frame_gap = float(max(0, int(row.get("frame_id", 0)) - int(prev_row.get("frame_id", 0)) - 1))
    crop_drone = _prob(row, "crop_probs", "drone")
    feature_drone = _prob(row, "feature_probs", "drone")
    temporal_drone = _prob(row, "temporal_probs", "drone")
    final_drone = _prob(row, "final_probs", "drone")
    crop_bg = _prob(row, "crop_probs", "background")
    temporal_bg = _prob(row, "temporal_probs", "background")
    final_bg = _prob(row, "final_probs", "background")
    max_bg = max(crop_bg, temporal_bg, final_bg)
    artifact = max(
        _prob(row, "crop_probs", "alignment_artifact"),
        _prob(row, "feature_probs", "alignment_artifact"),
        _prob(row, "temporal_probs", "alignment_artifact"),
        _prob(row, "final_probs", "alignment_artifact"),
    )
    return [
        float(row.get("objectness", 0.0)),
        float(row.get("final_drone_score", 0.0)),
        crop_drone,
        feature_drone,
        temporal_drone,
        final_drone,
        crop_bg,
        temporal_bg,
        final_bg,
        max_bg,
        artifact,
        float(row.get("motion_score", 0.0)),
        float(row.get("alignment_quality", 0.0)),
        float(row.get("track_score", 0.0) or 0.0),
        float(row.get("track_drift", 0.0) or 0.0),
        float(row.get("track_speed", 0.0) or 0.0),
        _box_side(row),
        center_step,
        frame_gap,
        float("yolo" in source),
        float("fallback" in source),
        float("tracker" in source),
        float("motion" in source),
        float(bool(row.get("track_validated", False))),
        float(row.get("predicted_class") == "drone"),
        temporal_drone - crop_drone,
        temporal_drone - max_bg,
        final_drone - max_bg,
    ]


def _rows_to_sequence(rows: list[dict[str, Any]], max_len: int) -> tuple[torch.Tensor, int]:
    ordered = sorted(rows, key=lambda r: int(r.get("frame_id", 0)))
    if len(ordered) > max_len:
        ordered = ordered[-max_len:]
    feats = []
    prev = None
    for row in ordered:
        feats.append(_row_frame_features(row, prev))
        prev = row
    length = max(1, len(feats))
    if not feats:
        feats = [[0.0] * len(TRACKLET_FRAME_FEATURES)]
    while len(feats) < max_len:
        feats.append([0.0] * len(TRACKLET_FRAME_FEATURES))
    return torch.tensor(feats, dtype=torch.float32), length


def _samples_to_tensors(samples: list[SequenceSample], max_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, lengths, ys = [], [], []
    for sample in samples:
        x, n = _rows_to_sequence(sample.rows, max_len=max_len)
        xs.append(x)
        lengths.append(n)
        ys.append(sample.label)
    return torch.stack(xs), torch.tensor(lengths, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def _augment_hard_negative_samples(samples: list[SequenceSample], repeats: int) -> list[SequenceSample]:
    if repeats <= 0:
        return samples
    out = list(samples)
    for sample in samples:
        if sample.label != 0:
            continue
        feats = _features(sample.rows)
        branch_like = max(float(feats.get("mean_crop_drone", 0.0)), float(feats.get("mean_temporal_drone", 0.0)), float(feats.get("mean_final_drone", 0.0)))
        if branch_like < 0.30 and float(feats.get("max_final_score", 0.0)) < 0.12 and float(feats.get("fallback_rate", 0.0)) <= 0.0:
            continue
        for idx in range(repeats):
            out.append(SequenceSample(sample.seq, f"{sample.track_id}__seq_hn{idx + 1}", sample.label, sample.rows))
    return out


def train_tracklet_sequence_classifier(
    jsonl_path: str | Path,
    out: str | Path,
    epochs: int = 40,
    lr: float = 1e-3,
    hidden: int = 32,
    max_len: int = 16,
    hard_negative_augments: int = 0,
) -> Path:
    samples = _augment_hard_negative_samples(_load_tracklet_jsonl(jsonl_path), hard_negative_augments)
    if not samples:
        raise ValueError("Tracklet sequence dataset is empty")
    x, lengths, y = _samples_to_tensors(samples, max_len=max_len)
    valid = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    valid_x = x[valid]
    mean = valid_x.mean(dim=0)
    std = valid_x.std(dim=0).clamp_min(1e-6)
    x_norm = (x - mean) / std
    loader = DataLoader(TensorDataset(x_norm, lengths, y), batch_size=min(32, len(y)), shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrackletSequenceGRU(in_dim=x.shape[-1], hidden=hidden).to(device)
    counts = torch.bincount(y, minlength=2).float()
    weights = counts.sum() / counts.clamp_min(1.0) / 2.0
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for bx, bl, by in loader:
            bx, bl, by = bx.to(device), bl.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx, bl), by)
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
            "frame_features": TRACKLET_FRAME_FEATURES,
            "mean": mean,
            "std": std,
            "hidden": hidden,
            "max_len": max_len,
            "history": history,
            "num_training_tracklets": int(len(y)),
            "num_positive_tracklets": int((y == 1).sum().item()),
            "num_negative_tracklets": int((y == 0).sum().item()),
            "hard_negative_augments": hard_negative_augments,
        },
        out_path,
    )
    return out_path


def score_tracklets_from_rows_sequence(rows: list[dict[str, Any]], weights: str | Path, threshold: float = 0.5) -> dict[str, dict[str, Any]]:
    tracklets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = _row_track_key(row)
        if key is None:
            continue
        tracklets.setdefault(key, []).append(row)
    if not tracklets:
        return {}
    model, _, mean, std, max_len = _load_sequence_checkpoint(weights)
    ordered = []
    xs, lengths = [], []
    for track_id, track_rows in sorted(tracklets.items(), key=lambda kv: kv[0]):
        x, n = _rows_to_sequence(track_rows, max_len=max_len)
        ordered.append((track_id, _features(track_rows), len(track_rows)))
        xs.append(x)
        lengths.append(n)
    x_tensor = (torch.stack(xs) - mean) / std
    length_tensor = torch.tensor(lengths, dtype=torch.long)
    with torch.no_grad():
        probs = torch.softmax(model(x_tensor, length_tensor), dim=1)[:, 1].tolist()
    scores: dict[str, dict[str, Any]] = {}
    for (track_id, feats, n), prob in zip(ordered, probs):
        scores[track_id] = {
            "prob_tracklet_drone": float(prob),
            "tracklet_is_drone": bool(prob >= threshold),
            "num_rows": n,
            "features": feats,
        }
    return scores


def filter_infer_rows_with_tracklet_sequence_classifier(
    pred_rows: list[dict[str, Any]],
    diag_rows: list[dict[str, Any]],
    weights: str | Path,
    threshold: float = 0.5,
    promote_positive_tracklets: bool = True,
    promotion_score_floor: float = 0.22,
    promotion_min_branch_drone: float = 0.40,
    promotion_max_background: float = 0.68,
    selective_promotion: bool = False,
    selective_max_promoted_tracklets_per_sequence: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scores = score_tracklets_from_rows_sequence(diag_rows, weights, threshold=threshold)
    selective_allowlist = (
        _selective_tracklet_key_allowlist(
            diag_rows,
            scores,
            promotion_min_branch_drone=promotion_min_branch_drone,
            promotion_max_background=promotion_max_background,
            min_temporal_crop_delta=0.05,
            min_temporal_background_margin=-0.05,
            max_tracklet_background=0.60,
            max_tracklet_objectness=0.50,
            min_tracklet_rows=2,
            min_temporal_gain_rate=0.40,
            min_weak_detector_temporal_signal=0.05,
            require_recovery_source=True,
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
            out["tracklet_sequence_prob"] = None
            out["tracklet_sequence_is_drone"] = None
            return out
        is_drone = bool(score["tracklet_is_drone"])
        out["tracklet_sequence_prob"] = score["prob_tracklet_drone"]
        out["tracklet_sequence_is_drone"] = is_drone
        if out.get("predicted_class") == "drone" and not is_drone:
            out["raw_predicted_class"] = out.get("predicted_class")
            out["raw_final_drone_score"] = out.get("final_drone_score")
            out["predicted_class"] = "background"
            out["final_drone_score"] = 0.0
            probs = dict(out.get("final_probs") or {})
            probs["drone"] = min(float(probs.get("drone", 0.0)), float(score["prob_tracklet_drone"]))
            probs["background"] = max(float(probs.get("background", 0.0)), 1.0 - float(score["prob_tracklet_drone"]))
            out["final_probs"] = probs
            out["diagnostic_cause"] = _append_cause(out.get("diagnostic_cause"), "tracklet_sequence_rejected")
        elif out.get("predicted_class") == "drone" and is_drone:
            out["raw_final_drone_score"] = out.get("final_drone_score")
            out["final_drone_score"] = max(float(out.get("final_drone_score", 0.0)), promotion_score_floor * float(score["prob_tracklet_drone"]))
            out["diagnostic_cause"] = _append_cause(out.get("diagnostic_cause"), "tracklet_sequence_confirmed")
        elif promote_positive_tracklets and is_drone:
            evidence = _promotion_evidence(out, score)
            selective_allowed = selective_allowlist is None or (track_id is not None and track_id in selective_allowlist)
            if evidence["branch_drone"] >= promotion_min_branch_drone and evidence["effective_background"] <= promotion_max_background and selective_allowed:
                out["raw_predicted_class"] = out.get("predicted_class")
                out["raw_final_drone_score"] = out.get("final_drone_score")
                out["predicted_class"] = "drone"
                out["final_drone_score"] = max(float(out.get("final_drone_score", 0.0)), promotion_score_floor * float(score["prob_tracklet_drone"]))
                out["diagnostic_cause"] = _append_cause(out.get("diagnostic_cause"), "tracklet_sequence_promoted")
        return out

    filtered_pred_rows = [update_row(row) for row in pred_rows]
    filtered_diag_rows = [update_row(row) for row in diag_rows]
    raw_drone = sum(1 for row in pred_rows if row.get("predicted_class") == "drone")
    filtered_drone = sum(1 for row in filtered_pred_rows if row.get("predicted_class") == "drone")
    summary = {
        "weights": str(weights),
        "threshold": threshold,
        "promote_positive_tracklets": promote_positive_tracklets,
        "promotion_score_floor": promotion_score_floor,
        "promotion_min_branch_drone": promotion_min_branch_drone,
        "promotion_max_background": promotion_max_background,
        "selective_promotion": selective_promotion,
        "selective_max_promoted_tracklets_per_sequence": selective_max_promoted_tracklets_per_sequence,
        "num_tracklets": len(scores),
        "raw_drone_predictions": raw_drone,
        "filtered_drone_predictions": filtered_drone,
        "rejected_drone_predictions": raw_drone - filtered_drone,
        "promoted_drone_predictions": sum(1 for row in filtered_pred_rows if row.get("predicted_class") == "drone" and row.get("raw_predicted_class") not in (None, "drone")),
    }
    return filtered_pred_rows, filtered_diag_rows, summary


def evaluate_tracklet_sequence_classifier(jsonl_path: str | Path, weights: str | Path, out: str | Path | None = None, threshold: float = 0.5) -> dict[str, Any]:
    samples = _load_tracklet_jsonl(jsonl_path)
    model, _, mean, std, max_len = _load_sequence_checkpoint(weights)
    x, lengths, y = _samples_to_tensors(samples, max_len=max_len)
    with torch.no_grad():
        probs = torch.softmax(model((x - mean) / std, lengths), dim=1)[:, 1]
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
    return metrics
