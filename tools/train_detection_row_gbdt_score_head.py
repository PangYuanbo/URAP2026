from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from train_detection_row_score_head import load_gt, load_test_features, load_train_rows, resolve_feature_indices, write_scored_test


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a no-leak nonlinear row-level VATD ranking head.")
    parser.add_argument("--train-tracklets", nargs="+", type=Path, required=True)
    parser.add_argument("--train-gt-csv", nargs="+", type=Path, required=True)
    parser.add_argument("--test-tracklets", type=Path, required=True)
    parser.add_argument("--out-test-tracklets", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--score-field", default="row_gbdt_score")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--negative-min-score", type=float, default=0.005)
    parser.add_argument("--label-policy", choices=["any-iou", "unique-iou"], default="unique-iou")
    parser.add_argument("--feature-groups", nargs="+", default=["all"])
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=350)
    parser.add_argument("--max-leaf-nodes", type=int, default=63)
    parser.add_argument("--min-samples-leaf", type=int, default=40)
    parser.add_argument("--l2-regularization", type=float, default=2.0)
    parser.add_argument("--random-state", type=int, default=23)
    args = parser.parse_args()

    gt = load_gt([path.resolve() for path in args.train_gt_csv])
    train_x, train_y, _ = load_train_rows(
        [path.resolve() for path in args.train_tracklets], gt, args.iou_threshold, args.negative_min_score, args.label_policy
    )
    test_x = load_test_features(args.test_tracklets.resolve())
    feature_indices, feature_groups = resolve_feature_indices(args.feature_groups)
    train_x = train_x[:, feature_indices]
    test_x = test_x[:, feature_indices]
    model = HistGradientBoostingClassifier(
        learning_rate=args.learning_rate, max_iter=args.max_iter, max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf, l2_regularization=args.l2_regularization,
        class_weight="balanced", early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=args.random_state,
    )
    model.fit(train_x, train_y)
    train_scores = model.predict_proba(train_x)[:, 1]
    test_scores = model.predict_proba(test_x)[:, 1]
    write_summary = write_scored_test(args.test_tracklets.resolve(), test_scores, args.out_test_tracklets.resolve(), args.score_field)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_indices": feature_indices, "feature_groups": feature_groups, "score_field": args.score_field}, args.out_model)
    summary = {
        "train_tracklets": [str(path.resolve()) for path in args.train_tracklets], "train_gt_csv": [str(path.resolve()) for path in args.train_gt_csv],
        "test_tracklets": str(args.test_tracklets.resolve()), "out_test_tracklets": str(args.out_test_tracklets.resolve()),
        "out_model": str(args.out_model.resolve()), "score_field": args.score_field, "feature_groups": feature_groups,
        "train_rows": int(len(train_y)), "train_positive_rows": int(train_y.sum()), "train_negative_rows": int(len(train_y) - train_y.sum()),
        "negative_min_score": args.negative_min_score, "label_policy": args.label_policy, "test_rows": int(len(test_scores)),
        "train_average_precision": float(average_precision_score(train_y, train_scores)),
        "train_roc_auc": float(roc_auc_score(train_y, train_scores)), "test_score_mean": float(test_scores.mean()),
        "test_score_p50": float(np.quantile(test_scores, 0.5)), "test_score_p90": float(np.quantile(test_scores, 0.9)),
        "iterations": int(model.n_iter_), "write_summary": write_summary, "parameters": model.get_params(),
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
