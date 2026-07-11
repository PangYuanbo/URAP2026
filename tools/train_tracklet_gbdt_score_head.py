from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from train_tracklet_meta_score_head import feature_row, resolve_feature_groups


def load_items(path: Path, feature_indices: list[int], negative_min_max_objectness: float | None = None) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    items: list[dict[str, Any]] = []
    features: list[list[float]] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            meta = dict(item.get("meta") or {})
            label = 1 if int(float(meta.get("label", 0))) > 0 else 0
            if negative_min_max_objectness is not None and label == 0:
                if float(meta.get("max_objectness") or 0.0) < negative_min_max_objectness:
                    continue
            full_row = feature_row(meta)
            items.append(item)
            features.append([full_row[index] for index in feature_indices])
            labels.append(label)
    return items, np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def write_scored(items: list[dict[str, Any]], scores: np.ndarray, out: Path, score_field: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for item, score in zip(items, scores, strict=True):
            score_value = float(score)
            scored = dict(item)
            meta = dict(scored.get("meta") or {})
            meta[score_field] = score_value
            scored["meta"] = meta
            rows = []
            for raw_row in scored.get("rows") or []:
                row = dict(raw_row)
                row[score_field] = score_value
                rows.append(row)
            scored["rows"] = rows
            handle.write(json.dumps(scored) + "\n")


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return None if len(np.unique(labels)) < 2 else float(roc_auc_score(labels, scores))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a nonlinear VATD tracklet score calibrator.")
    parser.add_argument("--train-tracklets", type=Path, required=True)
    parser.add_argument("--test-tracklets", type=Path, required=True)
    parser.add_argument("--out-test-tracklets", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--score-field", default="vatd_gbdt_score")
    parser.add_argument("--feature-groups", nargs="+", default=["all"])
    parser.add_argument("--negative-min-max-objectness", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=30)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=17)
    args = parser.parse_args()

    feature_indices, selected_features, missing_features = resolve_feature_groups(args.feature_groups)
    train_items, train_x, train_y = load_items(args.train_tracklets.resolve(), feature_indices, args.negative_min_max_objectness)
    test_items, test_x, test_y = load_items(args.test_tracklets.resolve(), feature_indices)
    model = HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=args.random_state,
    )
    model.fit(train_x, train_y)
    train_scores = model.predict_proba(train_x)[:, 1]
    test_scores = model.predict_proba(test_x)[:, 1]
    write_scored(test_items, test_scores, args.out_test_tracklets.resolve(), args.score_field)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": selected_features, "feature_groups": args.feature_groups, "score_field": args.score_field}, args.out_model)
    summary = {
        "train_tracklets": str(args.train_tracklets.resolve()), "test_tracklets": str(args.test_tracklets.resolve()),
        "out_test_tracklets": str(args.out_test_tracklets.resolve()), "out_model": str(args.out_model.resolve()),
        "score_field": args.score_field, "feature_groups": args.feature_groups, "features": selected_features,
        "missing_requested_features": missing_features, "train_tracklets_count": len(train_items),
        "train_positive": int(train_y.sum()), "train_negative": int(len(train_y) - train_y.sum()),
        "test_tracklets_count": len(test_items), "test_positive_labels_for_audit_only": int(test_y.sum()),
        "train_average_precision": float(average_precision_score(train_y, train_scores)), "train_roc_auc": safe_auc(train_y, train_scores),
        "test_average_precision_for_audit_only": float(average_precision_score(test_y, test_scores)), "test_roc_auc_for_audit_only": safe_auc(test_y, test_scores),
        "test_score_mean": float(test_scores.mean()), "test_score_p50": float(np.quantile(test_scores, 0.5)),
        "test_score_p90": float(np.quantile(test_scores, 0.9)), "iterations": int(model.n_iter_), "parameters": model.get_params(),
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
