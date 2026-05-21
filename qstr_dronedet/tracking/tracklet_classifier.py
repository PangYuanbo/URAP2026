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
    "temporal_gain_rate",
    "detector_update_rate",
    "fallback_rate",
    "validated_rate",
    "mean_track_drift",
    "max_track_drift",
    "mean_track_speed",
    "mean_box_side",
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


def _prob(row: dict[str, Any], branch: str, key: str) -> float:
    return float((row.get(branch) or {}).get(key, 0.0))


def _safe_mean(vals: list[float], default: float = 0.0) -> float:
    return float(np.mean(vals)) if vals else default


def _safe_max(vals: list[float], default: float = 0.0) -> float:
    return float(np.max(vals)) if vals else default


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
        "temporal_gain_rate": _safe_mean([float(t > c + 0.05) for c, t in zip(crop_drone, temp_drone)]),
        "detector_update_rate": _safe_mean([float("yolo" in s or "fallback" in s or "motion" in s or "seed" in s) for s in sources]),
        "fallback_rate": _safe_mean([float("fallback" in s) for s in sources]),
        "validated_rate": _safe_mean([float(bool(r.get("track_validated", False))) for r in rows]),
        "mean_track_drift": _safe_mean(drifts),
        "max_track_drift": _safe_max(drifts),
        "mean_track_speed": _safe_mean(speeds),
        "mean_box_side": _safe_mean(sides),
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


def _load_tracklet_csv(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, str]]]:
    xs, ys, meta = [], [], []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            xs.append([float(row[k]) for k in TRACKLET_FEATURES])
            ys.append(int(row["label"]))
            meta.append(row)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long), meta


def train_tracklet_classifier(
    csv_path: str | Path,
    out: str | Path,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 32,
) -> Path:
    x, y, _ = _load_tracklet_csv(csv_path)
    if len(y) == 0:
        raise ValueError("Tracklet dataset is empty")
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
    torch.save({"state_dict": model.cpu().state_dict(), "features": TRACKLET_FEATURES, "mean": mean, "std": std, "history": history}, out_path)
    return out_path


def evaluate_tracklet_classifier(csv_path: str | Path, weights: str | Path, out: str | Path | None = None, threshold: float = 0.5) -> dict[str, Any]:
    x, y, meta = _load_tracklet_csv(csv_path)
    ckpt = torch.load(weights, map_location="cpu")
    model = TrackletMLP(in_dim=len(ckpt["features"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    x_norm = (x - ckpt["mean"]) / ckpt["std"].clamp_min(1e-6)
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
