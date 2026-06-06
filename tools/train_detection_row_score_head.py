from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


TRACKLET_FEATURES = [
    "num_rows",
    "mean_objectness",
    "max_objectness",
    "mean_final_score",
    "max_final_score",
    "mean_box_side",
    "std_box_side",
    "mean_center_step",
    "max_center_step",
    "std_center_step",
    "track_span_frames",
    "frame_density",
    "score_above_02_rate",
    "score_slope",
    "objectness_slope",
    "background_slope",
    "final_margin_mean",
    "final_margin_min",
    "final_margin_slope",
    "background_dominance_rate",
    "background_dominance_longest_streak",
    "score_above_02_longest_streak",
    "max_frame_gap",
    "mean_frame_gap",
    "gap_rate",
    "vatd_score",
    "motion_action_score",
    "vatd_action_consistency_score",
    "mean_vatd_action_residual_center_error",
    "median_vatd_action_residual_center_error",
    "num_action_windows",
]

ROW_FEATURES = [
    "objectness",
    "final_drone_score",
    "score",
    "vatd_score",
    "motion_action_score",
    "vatd_action_consistency_score",
    "mean_vatd_action_residual_center_error",
    "median_vatd_action_residual_center_error",
]

EXTRA_FEATURES = [
    "cx_norm",
    "cy_norm",
    "w_norm",
    "h_norm",
    "area_norm",
    "aspect_ratio",
    "row_index_norm",
]

MODEL_FEATURES = ROW_FEATURES + TRACKLET_FEATURES + EXTRA_FEATURES


def named_feature_group_indices() -> dict[str, list[int]]:
    row_offset = 0
    tracklet_offset = len(ROW_FEATURES)
    extra_offset = tracklet_offset + len(TRACKLET_FEATURES)
    row = {name: row_offset + index for index, name in enumerate(ROW_FEATURES)}
    tracklet = {name: tracklet_offset + index for index, name in enumerate(TRACKLET_FEATURES)}
    extra = {name: extra_offset + index for index, name in enumerate(EXTRA_FEATURES)}
    return {
        "row_detector": [row[name] for name in ["objectness", "final_drone_score", "score"]],
        "row_motion": [
            row[name]
            for name in [
                "vatd_score",
                "motion_action_score",
                "vatd_action_consistency_score",
                "mean_vatd_action_residual_center_error",
                "median_vatd_action_residual_center_error",
            ]
        ],
        "box_geometry": [extra[name] for name in ["cx_norm", "cy_norm", "w_norm", "h_norm", "area_norm", "aspect_ratio", "row_index_norm"]],
        "tracklet_confidence": [
            tracklet[name]
            for name in [
                "num_rows",
                "mean_objectness",
                "max_objectness",
                "mean_final_score",
                "max_final_score",
                "score_above_02_rate",
                "score_slope",
                "objectness_slope",
                "final_margin_mean",
                "final_margin_min",
                "final_margin_slope",
            ]
        ],
        "tracklet_geometry": [
            tracklet[name]
            for name in [
                "mean_box_side",
                "std_box_side",
            ]
        ],
        "tracklet_temporal": [
            tracklet[name]
            for name in [
                "mean_center_step",
                "max_center_step",
                "std_center_step",
                "track_span_frames",
                "frame_density",
                "max_frame_gap",
                "mean_frame_gap",
                "gap_rate",
            ]
        ],
        "tracklet_background": [
            tracklet[name]
            for name in [
                "background_slope",
                "background_dominance_rate",
                "background_dominance_longest_streak",
            ]
        ],
        "tracklet_motion": [
            tracklet[name]
            for name in [
                "vatd_score",
                "motion_action_score",
                "vatd_action_consistency_score",
                "mean_vatd_action_residual_center_error",
                "median_vatd_action_residual_center_error",
                "num_action_windows",
            ]
        ],
    }


def resolve_feature_indices(groups: list[str]) -> tuple[list[int], list[str]]:
    named = named_feature_group_indices()
    if not groups or groups == ["all"]:
        return list(range(len(MODEL_FEATURES))), ["all"]
    indices: list[int] = []
    resolved: list[str] = []
    for group in groups:
        if group not in named:
            raise ValueError(f"unknown feature group {group!r}; choices: all,{','.join(sorted(named))}")
        resolved.append(group)
        indices.extend(named[group])
    deduped = sorted(set(indices))
    return deduped, resolved


class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class UnifiedDetectorDynamicsNet(torch.nn.Module):
    def __init__(self, in_dim: int, detector_indices: list[int], dynamics_indices: list[int], hidden: int = 128) -> None:
        super().__init__()
        self.register_buffer("detector_indices", torch.tensor(detector_indices, dtype=torch.long), persistent=False)
        self.register_buffer("dynamics_indices", torch.tensor(dynamics_indices, dtype=torch.long), persistent=False)
        self.detector = torch.nn.Sequential(
            torch.nn.Linear(len(detector_indices), hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        self.dynamics = torch.nn.Sequential(
            torch.nn.Linear(len(dynamics_indices), hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.dynamics_head = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        det = self.detector(x.index_select(1, self.detector_indices))
        dyn = self.dynamics(x.index_select(1, self.dynamics_indices))
        score = self.head(torch.cat([det, dyn], dim=1)).squeeze(-1)
        dyn_score = self.dynamics_head(dyn).squeeze(-1)
        return score, dyn_score


def feature_group_indices() -> tuple[list[int], list[int]]:
    row_start = 0
    row_end = len(ROW_FEATURES)
    trk_start = row_end
    trk_end = trk_start + len(TRACKLET_FEATURES)
    extra_start = trk_end
    extra_end = extra_start + len(EXTRA_FEATURES)
    confidence_tracklet_names = {
        "num_rows",
        "mean_objectness",
        "max_objectness",
        "mean_final_score",
        "max_final_score",
        "mean_box_side",
        "std_box_side",
        "track_span_frames",
        "frame_density",
        "score_above_02_rate",
        "score_slope",
        "objectness_slope",
        "final_margin_mean",
        "final_margin_min",
        "final_margin_slope",
        "background_dominance_rate",
        "background_dominance_longest_streak",
        "score_above_02_longest_streak",
    }
    confidence_tracklet = [
        trk_start + index
        for index, name in enumerate(TRACKLET_FEATURES)
        if name in confidence_tracklet_names
    ]
    detector_indices = list(range(row_start, row_end)) + confidence_tracklet + list(range(extra_start, extra_end))
    dynamics_indices = list(range(trk_start, trk_end))
    return detector_indices, dynamics_indices


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if np.isfinite(out) else 0.0


def load_gt(paths: list[Path]) -> dict[tuple[str, int], np.ndarray]:
    out: dict[tuple[str, int], list[list[float]]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                seq = str(row["seq"])
                frame_id = int(float(row["frame_id"]))
                out.setdefault((seq, frame_id), []).append([
                    _float(row["x1"]),
                    _float(row["y1"]),
                    _float(row["x2"]),
                    _float(row["y2"]),
                ])
    return {key: np.asarray(value, dtype=np.float32) for key, value in out.items()}


def iou_values(box: list[float], gt: np.ndarray) -> np.ndarray:
    if gt.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1, y1, x2, y2 = [float(v) for v in box]
    ix1 = np.maximum(x1, gt[:, 0])
    iy1 = np.maximum(y1, gt[:, 1])
    ix2 = np.minimum(x2, gt[:, 2])
    iy2 = np.minimum(y2, gt[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_b = np.maximum(0.0, gt[:, 2] - gt[:, 0]) * np.maximum(0.0, gt[:, 3] - gt[:, 1])
    return inter / np.maximum(1e-9, area_a + area_b - inter)


def iou_max(box: list[float], gt: np.ndarray) -> float:
    values = iou_values(box, gt)
    return float(values.max()) if values.size else 0.0


def feature_row(row: dict[str, Any], meta: dict[str, Any], row_index: int, row_count: int) -> list[float]:
    bbox = row.get("bbox") if isinstance(row.get("bbox"), list) else [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = [_float(v) for v in bbox[:4]]
    image_w = max(1.0, _float(row.get("image_width")))
    image_h = max(1.0, _float(row.get("image_height")))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    features = [_float(row.get(key, meta.get(key))) for key in ROW_FEATURES]
    features.extend(_float(meta.get(key)) for key in TRACKLET_FEATURES)
    features.extend(
        [
            ((x1 + x2) * 0.5) / image_w,
            ((y1 + y2) * 0.5) / image_h,
            w / image_w,
            h / image_h,
            (w * h) / (image_w * image_h),
            w / max(h, 1e-6),
            float(row_index) / max(1.0, float(row_count - 1)),
        ]
    )
    return features


def load_train_rows(
    tracklet_paths: list[Path],
    gt: dict[tuple[str, int], np.ndarray],
    iou_threshold: float,
    negative_min_score: float | None = None,
    label_policy: str = "any-iou",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if label_policy not in {"any-iou", "unique-iou"}:
        raise ValueError(f"unsupported label policy: {label_policy}")
    samples: list[tuple[tuple[str, int], list[float], float, float, np.ndarray]] = []
    by_frame: dict[tuple[str, int], list[int]] = {}
    for path in tracklet_paths:
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                meta = dict(item.get("meta") or {})
                tracklet_label = 1.0 if int(float(meta.get("label", 0))) > 0 else 0.0
                rows = [dict(row) for row in item.get("rows") or []]
                for idx, row in enumerate(rows):
                    seq = str(row.get("seq") or meta.get("seq") or "")
                    frame_id = int(float(row.get("frame_id", 0) or 0))
                    bbox = row.get("bbox")
                    if not seq or not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    box = [_float(v) for v in bbox]
                    key = (seq, frame_id)
                    gt_boxes = gt.get(key, np.zeros((0, 4), dtype=np.float32))
                    ious = iou_values(box, gt_boxes)
                    is_positive = bool(ious.size and float(ious.max()) >= iou_threshold)
                    if negative_min_score is not None and not is_positive:
                        raw_score = max(_float(row.get("score")), _float(row.get("objectness")), _float(row.get("final_drone_score")))
                        if raw_score < negative_min_score:
                            continue
                    sample_index = len(samples)
                    samples.append((key, feature_row(row, meta, idx, len(rows)), 1.0 if is_positive else 0.0, tracklet_label, ious))
                    by_frame.setdefault(key, []).append(sample_index)
    if label_policy == "any-iou":
        xs = [sample[1] for sample in samples]
        ys = [sample[2] for sample in samples]
        aux = [sample[3] for sample in samples]
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(aux, dtype=np.float32)

    labels = np.zeros((len(samples),), dtype=np.float32)
    for key, indices in by_frame.items():
        gt_boxes = gt.get(key, np.zeros((0, 4), dtype=np.float32))
        if gt_boxes.size == 0:
            continue
        candidates: list[tuple[float, int, int]] = []
        for sample_index in indices:
            ious = samples[sample_index][4]
            for gt_index, iou in enumerate(ious.tolist()):
                if float(iou) >= iou_threshold:
                    candidates.append((float(iou), sample_index, gt_index))
        used_samples: set[int] = set()
        used_gt: set[int] = set()
        for _iou, sample_index, gt_index in sorted(candidates, reverse=True):
            if sample_index in used_samples or gt_index in used_gt:
                continue
            labels[sample_index] = 1.0
            used_samples.add(sample_index)
            used_gt.add(gt_index)
    xs = [sample[1] for sample in samples]
    aux = [sample[3] for sample in samples]
    return np.asarray(xs, dtype=np.float32), labels, np.asarray(aux, dtype=np.float32)


def write_scored_test(test_tracklets: Path, scores: np.ndarray, out: Path, score_field: str) -> dict[str, int]:
    out.parent.mkdir(parents=True, exist_ok=True)
    score_idx = 0
    tracklets = 0
    rows_written = 0
    with test_tracklets.open("r", encoding="utf-8-sig") as src, out.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            tracklets += 1
            item = json.loads(line)
            rows_out = []
            row_scores = []
            for row in item.get("rows") or []:
                row = dict(row)
                score = float(scores[score_idx])
                score_idx += 1
                row[score_field] = score
                rows_out.append(row)
                row_scores.append(score)
                rows_written += 1
            item["rows"] = rows_out
            meta = dict(item.get("meta") or {})
            if row_scores:
                meta[score_field] = float(np.mean(row_scores))
                meta[f"{score_field}_max"] = float(np.max(row_scores))
            item["meta"] = meta
            dst.write(json.dumps(item, separators=(",", ":")) + "\n")
    return {"test_tracklets": tracklets, "test_rows_scored": rows_written}


def load_test_features(test_tracklets: Path) -> np.ndarray:
    xs: list[list[float]] = []
    with test_tracklets.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            meta = dict(item.get("meta") or {})
            rows = [dict(row) for row in item.get("rows") or []]
            for idx, row in enumerate(rows):
                xs.append(feature_row(row, meta, idx, len(rows)))
    return np.asarray(xs, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a no-leak row-level score head for TransVisDrone detection ranking.")
    parser.add_argument("--train-tracklets", nargs="+", type=Path, required=True)
    parser.add_argument("--train-gt-csv", nargs="+", type=Path, required=True)
    parser.add_argument("--test-tracklets", type=Path, required=True)
    parser.add_argument("--out-test-tracklets", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--score-field", default="row_score_noleak")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--negative-min-score", type=float, default=None)
    parser.add_argument("--label-policy", choices=["any-iou", "unique-iou"], default="any-iou")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pairwise-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-pairs", type=int, default=8192)
    parser.add_argument("--model-kind", choices=["mlp", "unified-two-tower"], default="mlp")
    parser.add_argument("--tracklet-aux-weight", type=float, default=0.0)
    parser.add_argument("--feature-groups", nargs="+", default=["all"])
    args = parser.parse_args()

    gt = load_gt(args.train_gt_csv)
    x_train_np, y_train_np, y_tracklet_np = load_train_rows(
        args.train_tracklets,
        gt,
        float(args.iou_threshold),
        args.negative_min_score,
        args.label_policy,
    )
    x_test_np = load_test_features(args.test_tracklets)
    feature_indices, resolved_feature_groups = resolve_feature_indices(args.feature_groups)
    if args.model_kind == "unified-two-tower" and resolved_feature_groups != ["all"]:
        raise ValueError("--model-kind unified-two-tower currently requires --feature-groups all")
    x_train_np = x_train_np[:, feature_indices]
    x_test_np = x_test_np[:, feature_indices]
    mean = x_train_np.mean(axis=0)
    std = x_train_np.std(axis=0)
    std[std < 1e-6] = 1.0
    x_train_np = (x_train_np - mean) / std
    x_test_np = (x_test_np - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector_indices, dynamics_indices = feature_group_indices()
    if args.model_kind == "unified-two-tower":
        model = UnifiedDetectorDynamicsNet(
            x_train_np.shape[1],
            detector_indices=detector_indices,
            dynamics_indices=dynamics_indices,
            hidden=args.hidden,
        ).to(device)
    else:
        model = MLP(x_train_np.shape[1], hidden=args.hidden).to(device)
    pos = float(y_train_np.sum())
    neg = float(len(y_train_np) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    trk_pos = float(y_tracklet_np.sum())
    trk_neg = float(len(y_tracklet_np) - trk_pos)
    trk_pos_weight = torch.tensor([trk_neg / max(trk_pos, 1.0)], dtype=torch.float32, device=device)
    tracklet_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=trk_pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator()
    generator.manual_seed(2026)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train_np),
        torch.from_numpy(y_train_np),
        torch.from_numpy(y_tracklet_np),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for bx, by, btrk in loader:
            bx = bx.to(device)
            by = by.to(device)
            btrk = btrk.to(device)
            if args.model_kind == "unified-two-tower":
                logits, tracklet_logits = model(bx)
            else:
                logits = model(bx)
                tracklet_logits = None
            loss = loss_fn(logits, by)
            if tracklet_logits is not None and args.tracklet_aux_weight > 0.0:
                loss = loss + float(args.tracklet_aux_weight) * tracklet_loss_fn(tracklet_logits, btrk)
            if args.pairwise_weight > 0.0:
                pos_idx = torch.where(by > 0.5)[0]
                neg_idx = torch.where(by <= 0.5)[0]
                if pos_idx.numel() and neg_idx.numel():
                    pair_count = min(int(args.pairwise_pairs), int(pos_idx.numel()) * int(neg_idx.numel()))
                    pos_pick = pos_idx[torch.randint(pos_idx.numel(), (pair_count,), device=device)]
                    neg_pick = neg_idx[torch.randint(neg_idx.numel(), (pair_count,), device=device)]
                    pair_loss = torch.nn.functional.softplus(-(logits[pos_pick] - logits[neg_pick])).mean()
                    loss = loss + float(args.pairwise_weight) * pair_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch), "loss": float(np.mean(losses))})

    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x_test_np), args.batch_size):
            bx = torch.from_numpy(x_test_np[start : start + args.batch_size]).to(device)
            if args.model_kind == "unified-two-tower":
                logits, _tracklet_logits = model(bx)
            else:
                logits = model(bx)
            chunks.append(torch.sigmoid(logits).detach().cpu())
    test_scores = torch.cat(chunks).numpy()
    write_summary = write_scored_test(args.test_tracklets, test_scores, args.out_test_tracklets, args.score_field)

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "features": [MODEL_FEATURES[index] for index in feature_indices],
            "feature_groups": resolved_feature_groups,
            "feature_indices": feature_indices,
            "model_kind": args.model_kind,
            "detector_indices": detector_indices,
            "dynamics_indices": dynamics_indices,
            "mean": mean,
            "std": std,
            "score_field": args.score_field,
            "hidden": args.hidden,
        },
        args.out_model,
    )
    summary = {
        "train_tracklets": [str(path.resolve()) for path in args.train_tracklets],
        "train_gt_csv": [str(path.resolve()) for path in args.train_gt_csv],
        "test_tracklets": str(args.test_tracklets.resolve()),
        "out_test_tracklets": str(args.out_test_tracklets.resolve()),
        "out_model": str(args.out_model.resolve()),
        "score_field": args.score_field,
        "device": str(device),
        "train_rows": int(len(y_train_np)),
        "train_positive_rows": int(y_train_np.sum()),
        "train_negative_rows": int(len(y_train_np) - y_train_np.sum()),
        "train_tracklet_aux_positive_rows": int(y_tracklet_np.sum()),
        "train_tracklet_aux_negative_rows": int(len(y_tracklet_np) - y_tracklet_np.sum()),
        "negative_min_score": args.negative_min_score,
        "label_policy": args.label_policy,
        "pairwise_weight": float(args.pairwise_weight),
        "pairwise_pairs": int(args.pairwise_pairs),
        "model_kind": args.model_kind,
        "tracklet_aux_weight": float(args.tracklet_aux_weight),
        "feature_groups": resolved_feature_groups,
        "num_features": int(len(feature_indices)),
        "test_rows": int(len(test_scores)),
        "test_score_mean": float(test_scores.mean()),
        "test_score_p50": float(np.quantile(test_scores, 0.5)),
        "test_score_p90": float(np.quantile(test_scores, 0.9)),
        **write_summary,
        "history": history,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
