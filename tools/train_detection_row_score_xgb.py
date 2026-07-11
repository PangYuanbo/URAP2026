from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from train_detection_row_score_head import (
    MODEL_FEATURES,
    load_gt,
    load_test_features,
    load_train_rows,
    write_scored_test,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tracklets", nargs="+", type=Path, required=True)
    parser.add_argument("--train-gt-csv", nargs="+", type=Path, required=True)
    parser.add_argument("--test-tracklets", type=Path, required=True)
    parser.add_argument("--out-test-tracklets", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--score-field", default="xgb_rank_score")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--negative-min-score", type=float, default=0.005)
    parser.add_argument("--label-policy", choices=["any-iou", "unique-iou"], default="unique-iou")
    parser.add_argument("--rounds", type=int, default=1600)
    parser.add_argument("--early-stopping-rounds", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--eta", type=float, default=0.03)
    parser.add_argument("--min-child-weight", type=float, default=5.0)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--scale-pos-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    gt = load_gt(args.train_gt_csv)
    x_train, y_train, _ = load_train_rows(
        args.train_tracklets,
        gt,
        args.iou_threshold,
        args.negative_min_score,
        args.label_policy,
    )
    x_test = load_test_features(args.test_tracklets)
    x_fit, x_valid, y_fit, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=0.18,
        random_state=args.seed,
        stratify=y_train,
    )
    dfit = xgb.QuantileDMatrix(x_fit, label=y_fit)
    dvalid = xgb.QuantileDMatrix(x_valid, label=y_valid, ref=dfit)
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["aucpr", "logloss"],
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "scale_pos_weight": args.scale_pos_weight,
        "lambda": 5.0,
        "alpha": 0.1,
        "gamma": 0.02,
        "max_bin": 256,
        "seed": args.seed,
    }
    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params,
        dfit,
        num_boost_round=args.rounds,
        evals=[(dfit, "train"), (dvalid, "valid")],
        evals_result=evals_result,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=25,
    )
    dtest = xgb.QuantileDMatrix(x_test, ref=dfit)
    scores = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    write_summary = write_scored_test(args.test_tracklets, scores, args.out_test_tracklets, args.score_field)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(args.out_model)
    summary = {
        "device": "cuda",
        "features": MODEL_FEATURES,
        "train_rows": int(len(y_train)),
        "train_positive_rows": int(y_train.sum()),
        "train_negative_rows": int(len(y_train) - y_train.sum()),
        "best_iteration": int(booster.best_iteration),
        "best_score": float(booster.best_score),
        "parameters": params,
        "test_score_mean": float(np.mean(scores)),
        "test_score_p50": float(np.quantile(scores, 0.5)),
        "test_score_p90": float(np.quantile(scores, 0.9)),
        **write_summary,
        "valid_aucpr_tail": evals_result["valid"]["aucpr"][-10:],
        "valid_logloss_tail": evals_result["valid"]["logloss"][-10:],
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
