from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.train_action_bank_motion_xgb_ranker import (
    frame_normalize, load_dataset, positive_group_view, ranking_feature_indices,
)
from tools.train_action_bank_motion_token_listwise import FEATURE_NAMES, write_score_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("train-pkl", "train-aux", "val-pkl", "val-aux", "test-pkl", "test-aux", "out-test", "out-model", "out-summary"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=264)
    parser.add_argument("--score-field", default="xgb_pairwise_refit_score")
    args = parser.parse_args()

    train_x, train_iou, _, train_groups, _ = load_dataset(args.train_pkl, args.train_aux, True)
    val_x, val_iou, _, val_groups, _ = load_dataset(args.val_pkl, args.val_aux, True)
    test_x, _, _, test_groups, test_locations = load_dataset(args.test_pkl, args.test_aux, False)
    selected = ranking_feature_indices()
    train_x, train_y, train_qid, train_frames = positive_group_view(train_x[:, selected], train_iou, train_groups)
    val_x, val_y, val_qid, val_frames = positive_group_view(val_x[:, selected], val_iou, val_groups)
    val_qid = val_qid + train_frames
    dev_x = np.concatenate((train_x, val_x))
    dev_y = np.concatenate((train_y, val_y))
    dev_qid = np.concatenate((train_qid, val_qid))

    model = xgb.XGBRanker(
        n_estimators=args.rounds, max_depth=7, learning_rate=0.035, min_child_weight=8,
        subsample=0.85, colsample_bytree=0.75, reg_lambda=8.0, reg_alpha=0.08,
        gamma=0.04, objective="rank:pairwise", tree_method="hist", device="cuda",
        max_bin=256, lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=12,
        n_jobs=8, random_state=2026,
    )
    model.fit(dev_x, dev_y, qid=dev_qid, verbose=25)
    test_scores = frame_normalize(model.predict(test_x[:, selected]), test_groups)
    write_score_jsonl(args.out_test, test_scores, test_locations, args.score_field)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.out_model)
    summary = {
        "model": "XGBRanker GPU pairwise refit Clips1-40", "features": len(selected),
        "rounds_fixed_from_validation": args.rounds, "train_frames": train_frames,
        "validation_frames_added_after_selection": val_frames, "dev_rows": len(dev_x),
        "test_rows": len(test_x), "score_field": args.score_field,
        "fusion_fixed_from_validation": {"mode": "linear-mix", "alpha": 0.2},
        "causal_inputs": True, "frame_normalized_output": True,
    }
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
