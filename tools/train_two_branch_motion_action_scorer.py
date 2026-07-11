from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


EPS = 1e-6


EVIDENCE_FEATURES = [
    "objectness",
    "final_score",
    "raw_score",
    "background",
    "score_margin",
    "box_side",
    "box_area",
    "visible",
    "dt",
    "valid",
]

MOTION_FEATURES = [
    "cx",
    "cy",
    "w",
    "h",
    "dx",
    "dy",
    "speed",
    "dw",
    "dh",
    "dt",
    "valid",
]

SUMMARY_FEATURES = [
    "num_rows",
    "mean_objectness",
    "max_objectness",
    "mean_final_score",
    "max_final_score",
    "score_above_02_rate",
    "score_slope",
    "objectness_slope",
    "mean_box_side",
    "frame_density",
    "max_frame_gap",
    "gap_rate",
    "mean_background",
    "background_slope",
    "final_margin_mean",
]

SUMMARY_FEATURE_GROUPS = {
    "full": SUMMARY_FEATURES,
    "detector_confidence": [
        "num_rows",
        "mean_objectness",
        "max_objectness",
        "mean_final_score",
        "max_final_score",
        "score_above_02_rate",
        "score_slope",
        "objectness_slope",
    ],
}


@dataclass(frozen=True)
class Sample:
    item_index: int
    start: int
    evidence: np.ndarray
    motion: np.ndarray
    summary: np.ndarray
    future_actions: np.ndarray
    action_mask: float
    label: float


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _row_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    value = row.get("bbox") or row.get("bbox_xyxy")
    if value is None:
        value = [row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")]
    if value is None or len(value) != 4:
        raise ValueError("row must contain bbox/bbox_xyxy or x1/y1/x2/y2")
    x1, y1, x2, y2 = [_safe_float(v) for v in value]
    return x1, y1, x2, y2


def _row_image_size(row: dict[str, Any]) -> tuple[float, float]:
    width = row.get("image_width")
    height = row.get("image_height")
    if width is None or height is None:
        size = row.get("image_size")
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            width, height = size[0], size[1]
    width_f = max(1.0, _safe_float(width, 1.0))
    height_f = max(1.0, _safe_float(height, 1.0))
    return width_f, height_f


def _norm_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = _row_box(row)
    width, height = _row_image_size(row)
    x1n = min(1.0, max(0.0, x1 / width))
    y1n = min(1.0, max(0.0, y1 / height))
    x2n = min(1.0, max(0.0, x2 / width))
    y2n = min(1.0, max(0.0, y2 / height))
    return x1n, y1n, x2n, y2n


def _cxcywh(box: Iterable[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(EPS, x2 - x1)
    h = max(EPS, y2 - y1)
    return x1 + w * 0.5, y1 + h * 0.5, w, h


def _box_action(prev_box: Iterable[float], next_box: Iterable[float]) -> tuple[float, float, float, float]:
    pcx, pcy, pw, ph = _cxcywh(prev_box)
    ncx, ncy, nw, nh = _cxcywh(next_box)
    return ncx - pcx, ncy - pcy, math.log(max(EPS, nw) / max(EPS, pw)), math.log(max(EPS, nh) / max(EPS, ph))


def _apply_action(prev_box: Iterable[float], action: Iterable[float]) -> tuple[float, float, float, float]:
    cx, cy, w, h = _cxcywh(prev_box)
    dx, dy, dlogw, dlogh = [float(v) for v in action]
    w2 = max(EPS, w * math.exp(min(6.0, max(-6.0, dlogw))))
    h2 = max(EPS, h * math.exp(min(6.0, max(-6.0, dlogh))))
    cx2 = cx + dx
    cy2 = cy + dy
    return cx2 - w2 * 0.5, cy2 - h2 * 0.5, cx2 + w2 * 0.5, cy2 + h2 * 0.5


def _reconstruct(start_box: Iterable[float], actions: np.ndarray) -> np.ndarray:
    current = tuple(float(v) for v in start_box)
    boxes = []
    for action in actions:
        current = _apply_action(current, action)
        boxes.append(current)
    return np.asarray(boxes, dtype=np.float32)


def _row_scores(row: dict[str, Any]) -> tuple[float, float, float, float]:
    objectness = _safe_float(row.get("objectness"), 0.0)
    final_score = _safe_float(row.get("final_drone_score", row.get("final_score", row.get("score"))), 0.0)
    raw_score = _safe_float(row.get("score"), final_score)
    background = _safe_float(row.get("background"), np.nan)
    probs = row.get("final_probs")
    if not math.isfinite(background) and isinstance(probs, dict):
        background = _safe_float(probs.get("background"), 1.0 - final_score)
    if not math.isfinite(background):
        background = 1.0 - final_score
    return objectness, final_score, raw_score, background


def _summary_row(meta: dict[str, Any], summary_features: list[str]) -> np.ndarray:
    return np.asarray([_safe_float(meta.get(name), 0.0) for name in summary_features], dtype=np.float32)


def _make_window(
    rows: list[dict[str, Any]],
    meta: dict[str, Any],
    start: int,
    past_len: int,
    future_len: int,
    item_index: int,
    label: float,
    summary_features: list[str],
) -> Sample:
    ordered = sorted(rows, key=lambda row: int(_safe_float(row.get("frame_id"), 0.0)))
    total = past_len + future_len
    if len(ordered) >= total:
        window = ordered[start : start + total]
        action_mask = 1.0
    else:
        start = 0
        window = ordered[:]
        action_mask = 0.0
    past_rows = window[:past_len]
    future_rows = window[past_len : past_len + future_len]
    if not past_rows:
        raise ValueError("tracklet has no rows")
    while len(past_rows) < past_len:
        past_rows.insert(0, past_rows[0])
    boxes = [_norm_box(row) for row in past_rows]
    frame_ids = [int(_safe_float(row.get("frame_id"), 0.0)) for row in past_rows]
    evidence = np.zeros((past_len, len(EVIDENCE_FEATURES)), dtype=np.float32)
    motion = np.zeros((past_len, len(MOTION_FEATURES)), dtype=np.float32)
    prev_cx = prev_cy = prev_w = prev_h = 0.0
    prev_frame = frame_ids[0]
    for idx, (row, box, frame_id) in enumerate(zip(past_rows, boxes, frame_ids)):
        cx, cy, bw, bh = _cxcywh(box)
        objectness, final_score, raw_score, background = _row_scores(row)
        dt = 0.0 if idx == 0 else float(max(0, frame_id - prev_frame))
        dx = 0.0 if idx == 0 else cx - prev_cx
        dy = 0.0 if idx == 0 else cy - prev_cy
        dw = 0.0 if idx == 0 else bw - prev_w
        dh = 0.0 if idx == 0 else bh - prev_h
        visible = 1.0 if row.get("visible", True) else 0.0
        evidence[idx] = np.asarray(
            [
                objectness,
                final_score,
                raw_score,
                background,
                final_score - background,
                max(bw, bh),
                bw * bh,
                visible,
                min(10.0, dt) / 10.0,
                1.0,
            ],
            dtype=np.float32,
        )
        motion[idx] = np.asarray(
            [
                cx,
                cy,
                bw,
                bh,
                dx,
                dy,
                math.sqrt(dx * dx + dy * dy),
                dw,
                dh,
                min(10.0, dt) / 10.0,
                1.0,
            ],
            dtype=np.float32,
        )
        prev_cx, prev_cy, prev_w, prev_h, prev_frame = cx, cy, bw, bh, frame_id
    actions = np.zeros((future_len, 4), dtype=np.float32)
    if action_mask > 0.0 and len(future_rows) == future_len:
        future_boxes = [_norm_box(row) for row in future_rows]
        action_boxes = boxes[-1:] + future_boxes
        actions = np.asarray([_box_action(action_boxes[i], action_boxes[i + 1]) for i in range(future_len)], dtype=np.float32)
    return Sample(
        item_index=item_index,
        start=start,
        evidence=np.nan_to_num(evidence, nan=0.0, posinf=1.0, neginf=-1.0),
        motion=np.nan_to_num(motion, nan=0.0, posinf=1.0, neginf=-1.0),
        summary=np.nan_to_num(_summary_row(meta, summary_features), nan=0.0, posinf=1.0, neginf=-1.0),
        future_actions=np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0),
        action_mask=action_mask,
        label=label,
    )


def build_samples(
    items: list[dict[str, Any]],
    past_len: int,
    future_len: int,
    window_stride: int,
    max_windows_per_tracklet: int | None = None,
    summary_features: list[str] | None = None,
) -> list[Sample]:
    samples: list[Sample] = []
    if summary_features is None:
        summary_features = SUMMARY_FEATURES
    total = past_len + future_len
    stride = max(1, int(window_stride))
    for item_index, item in enumerate(items):
        rows = sorted(list(item.get("rows") or []), key=lambda row: int(_safe_float(row.get("frame_id"), 0.0)))
        if not rows:
            continue
        meta = dict(item.get("meta") or {})
        label = 1.0 if int(_safe_float(meta.get("label"), 0.0)) > 0 else 0.0
        if len(rows) >= total:
            starts = list(range(0, len(rows) - total + 1, stride))
            if starts[-1] != len(rows) - total:
                starts.append(len(rows) - total)
            if max_windows_per_tracklet is not None and len(starts) > max_windows_per_tracklet:
                indices = np.linspace(0, len(starts) - 1, max_windows_per_tracklet).round().astype(int)
                starts = [starts[int(index)] for index in indices]
            for start in starts:
                samples.append(_make_window(rows, meta, start, past_len, future_len, item_index, label, summary_features))
        else:
            samples.append(_make_window(rows, meta, 0, past_len, future_len, item_index, label, summary_features))
    return samples


class TwoBranchMotionActionNet(torch.nn.Module):
    def __init__(
        self,
        evidence_dim: int,
        motion_dim: int,
        summary_dim: int,
        past_len: int,
        future_len: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        self.evidence_in = torch.nn.Sequential(torch.nn.Linear(evidence_dim, d_model), torch.nn.SiLU(), torch.nn.Linear(d_model, d_model))
        self.motion_in = torch.nn.Sequential(torch.nn.Linear(motion_dim, d_model), torch.nn.SiLU(), torch.nn.Linear(d_model, d_model))
        self.summary_in = torch.nn.Sequential(torch.nn.LayerNorm(summary_dim), torch.nn.Linear(summary_dim, d_model), torch.nn.SiLU(), torch.nn.Linear(d_model, d_model))
        self.evidence_cls = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        self.motion_cls = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        self.evidence_pos = torch.nn.Parameter(torch.zeros(1, past_len + 1, d_model))
        self.motion_pos = torch.nn.Parameter(torch.zeros(1, past_len + 1, d_model))
        evidence_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        motion_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.evidence_encoder = torch.nn.TransformerEncoder(evidence_layer, num_layers=num_layers)
        self.motion_encoder = torch.nn.TransformerEncoder(motion_layer, num_layers=num_layers)
        self.fusion = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model * 3),
            torch.nn.Linear(d_model * 3, d_model),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(),
        )
        self.cls_head = torch.nn.Linear(d_model, 1)
        self.summary_direct_head = torch.nn.Sequential(
            torch.nn.LayerNorm(summary_dim),
            torch.nn.Linear(summary_dim, d_model),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model, 1),
        )
        self.action_head = torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, future_len * 4))

    def forward(self, evidence: torch.Tensor, motion: torch.Tensor, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = evidence.shape[0]
        evidence_tokens = self.evidence_in(evidence)
        motion_tokens = self.motion_in(motion)
        evidence_tokens = torch.cat([self.evidence_cls.expand(bsz, -1, -1), evidence_tokens], dim=1)
        motion_tokens = torch.cat([self.motion_cls.expand(bsz, -1, -1), motion_tokens], dim=1)
        evidence_encoded = self.evidence_encoder(evidence_tokens + self.evidence_pos[:, : evidence_tokens.shape[1], :])
        motion_encoded = self.motion_encoder(motion_tokens + self.motion_pos[:, : motion_tokens.shape[1], :])
        fused = self.fusion(torch.cat([evidence_encoded[:, 0, :], motion_encoded[:, 0, :], self.summary_in(summary)], dim=1))
        logits = (self.cls_head(fused) + self.summary_direct_head(summary)).squeeze(1)
        actions = self.action_head(fused).reshape(bsz, self.future_len, 4)
        return logits, actions


def _stack(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    evidence = np.stack([sample.evidence for sample in samples]).astype(np.float32)
    motion = np.stack([sample.motion for sample in samples]).astype(np.float32)
    summary = np.stack([sample.summary for sample in samples]).astype(np.float32)
    actions = np.stack([sample.future_actions for sample in samples]).astype(np.float32)
    action_mask = np.asarray([sample.action_mask for sample in samples], dtype=np.float32)
    labels = np.asarray([sample.label for sample in samples], dtype=np.float32)
    return evidence, motion, summary, actions, action_mask, labels


def _normalize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = train.reshape(-1, train.shape[-1])
    valid = flat[:, -1] > 0.5 if train.shape[-1] and np.any(flat[:, -1] > 0.5) else np.ones((flat.shape[0],), dtype=bool)
    mean = flat[valid].mean(axis=0)
    std = flat[valid].std(axis=0)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (test - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def _infer_scores(
    model: TwoBranchMotionActionNet,
    evidence: np.ndarray,
    motion: np.ndarray,
    summary: np.ndarray,
    actions: np.ndarray,
    action_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores = []
    consistency = []
    with torch.no_grad():
        for start in range(0, len(evidence), batch_size):
            ev = torch.from_numpy(evidence[start : start + batch_size]).to(device)
            mo = torch.from_numpy(motion[start : start + batch_size]).to(device)
            su = torch.from_numpy(summary[start : start + batch_size]).to(device)
            logits, pred_actions = model(ev, mo, su)
            scores.append(torch.sigmoid(logits).cpu().numpy())
            pred_np = pred_actions.cpu().numpy()
            target_np = actions[start : start + batch_size]
            mask_np = action_mask[start : start + batch_size]
            err = np.mean(np.abs(pred_np - target_np), axis=(1, 2))
            cons = np.where(mask_np > 0.0, np.exp(-err / 0.01), 0.0)
            consistency.append(cons.astype(np.float32))
    return np.concatenate(scores).astype(np.float32), np.concatenate(consistency).astype(np.float32)


def write_scored(
    items: list[dict[str, Any]],
    samples: list[Sample],
    scores: np.ndarray,
    consistency: np.ndarray,
    out: Path,
    score_field: str,
) -> dict[str, Any]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for sample, score, cons in zip(samples, scores, consistency):
        grouped.setdefault(sample.item_index, []).append((float(score), float(cons)))
    out.parent.mkdir(parents=True, exist_ok=True)
    scored = 0
    with out.open("w", encoding="utf-8") as f:
        for index, item in enumerate(items):
            pairs = grouped.get(index, [])
            if pairs:
                scored += 1
                score = float(np.mean([pair[0] for pair in pairs]))
                score_max = float(np.max([pair[0] for pair in pairs]))
                action_consistency = float(np.mean([pair[1] for pair in pairs]))
            else:
                score = 0.0
                score_max = 0.0
                action_consistency = 0.0
            item = dict(item)
            meta = dict(item.get("meta") or {})
            meta[score_field] = score
            meta[f"{score_field}_max_window"] = score_max
            meta[f"{score_field}_action_consistency"] = action_consistency
            meta[f"{score_field}_num_windows"] = len(pairs)
            item["meta"] = meta
            rows = []
            for row in item.get("rows") or []:
                row = dict(row)
                row[score_field] = score
                rows.append(row)
            item["rows"] = rows
            f.write(json.dumps(item, separators=(",", ":")) + "\n")
    return {"tracklets": len(items), "scored_tracklets": scored, "score_field": score_field}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a two-branch detector-evidence + motion-action tracklet scorer.")
    parser.add_argument("--train-tracklets", type=Path, required=True)
    parser.add_argument("--test-tracklets", type=Path, required=True)
    parser.add_argument("--out-test-tracklets", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--score-field", default="two_branch_motion_action_score")
    parser.add_argument("--past-len", type=int, default=8)
    parser.add_argument("--future-len", type=int, default=2)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--max-windows-per-tracklet", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--action-loss-weight", type=float, default=0.2)
    parser.add_argument("--action-loss-positives-only", action="store_true", default=True)
    parser.add_argument("--action-loss-all", dest="action_loss_positives_only", action="store_false")
    parser.add_argument("--summary-features", choices=sorted(SUMMARY_FEATURE_GROUPS), default="full")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_items = _read_jsonl(args.train_tracklets)
    test_items = _read_jsonl(args.test_tracklets)
    summary_features = list(SUMMARY_FEATURE_GROUPS[args.summary_features])
    train_samples = build_samples(
        train_items,
        args.past_len,
        args.future_len,
        args.window_stride,
        args.max_windows_per_tracklet,
        summary_features=summary_features,
    )
    test_samples = build_samples(
        test_items,
        args.past_len,
        args.future_len,
        args.window_stride,
        args.max_windows_per_tracklet,
        summary_features=summary_features,
    )
    if not train_samples:
        raise ValueError("no training samples")
    if not test_samples:
        raise ValueError("no test samples")

    xev_train, xmo_train, xsum_train, act_train, action_mask_train, y_train = _stack(train_samples)
    xev_test, xmo_test, xsum_test, act_test, action_mask_test, y_test = _stack(test_samples)
    xev_train, xev_test, ev_mean, ev_std = _normalize(xev_train, xev_test)
    xmo_train, xmo_test, mo_mean, mo_std = _normalize(xmo_train, xmo_test)
    sum_mean = xsum_train.mean(axis=0)
    sum_std = xsum_train.std(axis=0)
    sum_std[sum_std < 1e-6] = 1.0
    xsum_train = (xsum_train - sum_mean) / sum_std
    xsum_test = (xsum_test - sum_mean) / sum_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoBranchMotionActionNet(
        evidence_dim=xev_train.shape[-1],
        motion_dim=xmo_train.shape[-1],
        summary_dim=xsum_train.shape[-1],
        past_len=args.past_len,
        future_len=args.future_len,
        d_model=args.hidden,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    cls_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device))
    action_loss_fn = torch.nn.SmoothL1Loss(reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    tensors = torch.utils.data.TensorDataset(
        torch.from_numpy(xev_train),
        torch.from_numpy(xmo_train),
        torch.from_numpy(xsum_train.astype(np.float32)),
        torch.from_numpy(act_train),
        torch.from_numpy(action_mask_train),
        torch.from_numpy(y_train),
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        tensors,
        batch_size=min(args.batch_size, len(tensors)),
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_action = 0.0
        total = 0
        for ev, mo, su, target_actions, action_mask, label in loader:
            ev = ev.to(device, non_blocking=device.type == "cuda")
            mo = mo.to(device, non_blocking=device.type == "cuda")
            su = su.to(device, non_blocking=device.type == "cuda")
            target_actions = target_actions.to(device, non_blocking=device.type == "cuda")
            action_mask = action_mask.to(device, non_blocking=device.type == "cuda")
            label = label.to(device, non_blocking=device.type == "cuda")
            logits, pred_actions = model(ev, mo, su)
            cls_loss = cls_loss_fn(logits, label)
            if args.action_loss_positives_only:
                effective_mask = action_mask * label
            else:
                effective_mask = action_mask
            raw_action_loss = action_loss_fn(pred_actions, target_actions).mean(dim=(1, 2))
            action_denom = torch.clamp(effective_mask.sum(), min=1.0)
            action_loss = (raw_action_loss * effective_mask).sum() / action_denom
            loss = cls_loss + float(args.action_loss_weight) * action_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bsz = int(label.shape[0])
            total_loss += float(loss.detach().cpu()) * bsz
            total_cls += float(cls_loss.detach().cpu()) * bsz
            total_action += float(action_loss.detach().cpu()) * bsz
            total += bsz
        history.append(
            {
                "epoch": float(epoch),
                "loss": total_loss / max(1, total),
                "cls_loss": total_cls / max(1, total),
                "action_loss": total_action / max(1, total),
            }
        )

    test_scores, test_consistency = _infer_scores(model, xev_test, xmo_test, xsum_test.astype(np.float32), act_test, action_mask_test, args.batch_size, device)
    write_summary = write_scored(test_items, test_samples, test_scores, test_consistency, args.out_test_tracklets, args.score_field)

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.cpu().state_dict(),
            "model_type": "two_branch_motion_action_transformer",
            "evidence_features": EVIDENCE_FEATURES,
            "motion_features": MOTION_FEATURES,
            "summary_features": summary_features,
            "summary_feature_group": args.summary_features,
            "evidence_mean": ev_mean,
            "evidence_std": ev_std,
            "motion_mean": mo_mean,
            "motion_std": mo_std,
            "summary_mean": sum_mean.astype(np.float32),
            "summary_std": sum_std.astype(np.float32),
            "past_len": args.past_len,
            "future_len": args.future_len,
            "hidden": args.hidden,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "score_field": args.score_field,
        },
        args.out_model,
    )
    summary = {
        "train_tracklets": str(args.train_tracklets.resolve()),
        "test_tracklets": str(args.test_tracklets.resolve()),
        "out_test_tracklets": str(args.out_test_tracklets.resolve()),
        "out_model": str(args.out_model.resolve()),
        "score_field": args.score_field,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "train_tracklets_count": len(train_items),
        "train_samples_count": len(train_samples),
        "train_positive_samples": int(y_train.sum()),
        "train_negative_samples": int(len(y_train) - y_train.sum()),
        "test_tracklets_count": len(test_items),
        "test_samples_count": len(test_samples),
        "test_positive_samples_for_audit_only": int(y_test.sum()),
        "past_len": args.past_len,
        "future_len": args.future_len,
        "window_stride": args.window_stride,
        "max_windows_per_tracklet": args.max_windows_per_tracklet,
        "action_loss_weight": args.action_loss_weight,
        "action_loss_positives_only": args.action_loss_positives_only,
        "summary_feature_group": args.summary_features,
        "summary_features": summary_features,
        "write_summary": write_summary,
        "test_score_mean": float(test_scores.mean()),
        "test_score_p50": float(np.quantile(test_scores, 0.5)),
        "test_score_p90": float(np.quantile(test_scores, 0.9)),
        "history": history,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
