from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


FEATURES = [
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
    "first_final_score",
    "last_final_score",
    "first_background",
    "last_background",
    "vatd_score",
    "motion_action_score",
    "vatd_action_consistency_score",
    "mean_vatd_action_residual_center_error",
    "median_vatd_action_residual_center_error",
    "num_action_windows",
]

MODEL_FEATURES = FEATURES + [
    "has_vatd_score",
]

FEATURE_GROUPS = {
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
    "bbox_geometry": [
        "mean_box_side",
        "std_box_side",
        "first_final_score",
        "last_final_score",
    ],
    "temporal_continuity": [
        "mean_center_step",
        "max_center_step",
        "std_center_step",
        "track_span_frames",
        "frame_density",
        "max_frame_gap",
        "mean_frame_gap",
        "gap_rate",
    ],
    "background_fp": [
        "background_slope",
        "background_dominance_rate",
        "background_dominance_longest_streak",
        "final_margin_mean",
        "final_margin_min",
        "final_margin_slope",
        "first_background",
        "last_background",
    ],
    "action_motion": [
        "vatd_score",
        "motion_action_score",
        "vatd_action_consistency_score",
        "mean_vatd_action_residual_center_error",
        "median_vatd_action_residual_center_error",
        "num_action_windows",
        "has_vatd_score",
    ],
}

REQUESTED_BUT_UNSTORED = {
    "background_fp": [
        "mean_background",
    ],
}


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


def meta_value(meta: dict[str, Any], key: str) -> float:
    value = meta.get(key)
    if value is None:
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(out):
        return 0.0
    return out


def feature_row(meta: dict[str, Any]) -> list[float]:
    row = [meta_value(meta, key) for key in FEATURES]
    row.extend(
        [
            1.0 if meta.get("vatd_score") is not None else 0.0,
        ]
    )
    return row


def resolve_feature_groups(groups: list[str]) -> tuple[list[int], list[str], list[str]]:
    if not groups or "all" in groups:
        return list(range(len(MODEL_FEATURES))), MODEL_FEATURES.copy(), []

    feature_to_index = {name: index for index, name in enumerate(MODEL_FEATURES)}
    indices: list[int] = []
    selected_features: list[str] = []
    missing: list[str] = []
    for group in groups:
        if group.startswith("all_except_"):
            excluded_group = group.removeprefix("all_except_")
            if excluded_group not in FEATURE_GROUPS:
                raise ValueError(f"unknown feature group: {group}")
            excluded = set(FEATURE_GROUPS[excluded_group])
            names = [name for name in MODEL_FEATURES if name not in excluded]
            missing.extend(REQUESTED_BUT_UNSTORED.get(excluded_group, []))
        else:
            if group not in FEATURE_GROUPS:
                raise ValueError(f"unknown feature group: {group}")
            names = FEATURE_GROUPS[group]
            missing.extend(REQUESTED_BUT_UNSTORED.get(group, []))
        for name in names:
            index = feature_to_index.get(name)
            if index is None:
                missing.append(name)
                continue
            if index in indices:
                continue
            indices.append(index)
            selected_features.append(name)
    if not indices:
        raise ValueError(f"no stored features selected for groups: {groups}")
    return indices, selected_features, sorted(set(missing))


def load_items(path: Path, negative_min_max_objectness: float | None = None) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    items: list[dict[str, Any]] = []
    xs: list[list[float]] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            meta = dict(item.get("meta") or {})
            label = 1.0 if int(float(meta.get("label", 0))) > 0 else 0.0
            if negative_min_max_objectness is not None and label <= 0.0:
                if meta_value(meta, "max_objectness") < negative_min_max_objectness:
                    continue
            items.append(item)
            xs.append(feature_row(meta))
            ys.append(label)
    return items, np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def write_scored(items: list[dict[str, Any]], scores: np.ndarray, out: Path, score_field: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item, score in zip(items, scores):
            score_f = float(score)
            meta = dict(item.get("meta") or {})
            meta[score_field] = score_f
            item["meta"] = meta
            rows = []
            for row in item.get("rows") or []:
                row = dict(row)
                row[score_field] = score_f
                rows.append(row)
            item["rows"] = rows
            f.write(json.dumps(item, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tracklets", type=Path, required=True)
    parser.add_argument("--test-tracklets", type=Path, required=True)
    parser.add_argument("--out-test-tracklets", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--score-field", default="meta_score")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--negative-min-max-objectness", type=float, default=None)
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        default=["all"],
        choices=["all", *FEATURE_GROUPS.keys(), *[f"all_except_{name}" for name in FEATURE_GROUPS]],
        help="Feature groups to train on. Use all_except_<group> for leave-one-out ablations.",
    )
    args = parser.parse_args()

    train_items, x_train_np, y_train_np = load_items(args.train_tracklets, args.negative_min_max_objectness)
    test_items, x_test_np, y_test_np = load_items(args.test_tracklets)
    feature_indices, selected_features, missing_requested_features = resolve_feature_groups(args.feature_groups)
    x_train_np = x_train_np[:, feature_indices]
    x_test_np = x_test_np[:, feature_indices]

    mean = x_train_np.mean(axis=0)
    std = x_train_np.std(axis=0)
    std[std < 1e-6] = 1.0
    x_train_np = (x_train_np - mean) / std
    x_test_np = (x_test_np - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)
    model = MLP(x_train.shape[1], hidden=args.hidden).to(device)
    pos = float(y_train_np.sum())
    neg = float(len(y_train_np) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    generator = torch.Generator()
    generator.manual_seed(1337)
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            logits = model(bx)
            loss = loss_fn(logits, by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch), "loss": float(np.mean(losses))})

    model.eval()
    with torch.no_grad():
        test_logits = []
        for start in range(0, len(x_test_np), args.batch_size):
            bx = torch.from_numpy(x_test_np[start : start + args.batch_size]).to(device)
            test_logits.append(model(bx).detach().cpu())
        test_scores = torch.sigmoid(torch.cat(test_logits)).numpy()

    write_scored(test_items, test_scores, args.out_test_tracklets, args.score_field)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "features": selected_features,
            "feature_groups": args.feature_groups,
            "missing_requested_features": missing_requested_features,
            "mean": mean,
            "std": std,
            "score_field": args.score_field,
            "hidden": args.hidden,
        },
        args.out_model,
    )
    summary = {
        "train_tracklets": str(args.train_tracklets.resolve()),
        "test_tracklets": str(args.test_tracklets.resolve()),
        "out_test_tracklets": str(args.out_test_tracklets.resolve()),
        "out_model": str(args.out_model.resolve()),
        "score_field": args.score_field,
        "feature_groups": args.feature_groups,
        "features": selected_features,
        "num_features": len(selected_features),
        "missing_requested_features": missing_requested_features,
        "device": str(device),
        "train_tracklets_count": len(train_items),
        "train_positive": int(y_train_np.sum()),
        "train_negative": int(len(y_train_np) - y_train_np.sum()),
        "negative_min_max_objectness": args.negative_min_max_objectness,
        "test_tracklets_count": len(test_items),
        "test_positive_labels_for_audit_only": int(y_test_np.sum()),
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
