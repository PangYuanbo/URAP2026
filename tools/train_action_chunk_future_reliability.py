from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO, REPO / "tools"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.train_action_bank_motion_token_listwise import write_score_jsonl
from tools.train_action_chunk_chain_consistency import build_arrays


def valid_future_mask(sequences, timestamps):
    maximum = {}
    for sequence, timestamp in zip(sequences, timestamps):
        maximum[sequence] = max(maximum.get(sequence, timestamp), timestamp)
    return np.asarray([timestamp + 1.0 <= maximum[sequence] + 1e-6 for sequence, timestamp in zip(sequences, timestamps)], dtype=bool)


def selected_rows(features, future, valid, max_negative_ratio=8):
    positives = np.flatnonzero(valid & (future >= 0.5))
    negatives = np.flatnonzero(valid & (future < 0.5))
    if len(negatives) > max_negative_ratio * max(1, len(positives)):
        priority = features[negatives, 0]
        negatives = negatives[np.argsort(priority)[::-1][: max_negative_ratio * len(positives)]]
    keep = np.sort(np.concatenate((positives, negatives)))
    labels = (future[keep] >= 0.5).astype(np.int32)
    weights = np.where(labels > 0, 1.0 + future[keep], 1.0).astype(np.float32)
    return keep, labels, weights


def fit(features, future, valid):
    keep, labels, weights = selected_rows(features, future, valid)
    positives = max(1, int(labels.sum()))
    negatives = len(labels) - positives
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.03,
        min_child_weight=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=10,
        reg_alpha=0.12,
        gamma=0.025,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device="cuda",
        max_bin=256,
        scale_pos_weight=min(12.0, negatives / positives),
        n_jobs=8,
        random_state=2026,
    )
    model.fit(features[keep], labels, sample_weight=weights, verbose=False)
    return model, len(keep), positives


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict whether a causal Action Chunk chain will stay correct during the next second.")
    for name in ("train-pkl", "train-chain", "val-pkl", "val-chain", "test-pkl", "test-chain", "fps-json", "out-val-scores", "out-test-scores", "out-model-dir", "out-summary"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--score-field", default="action_chunk_future_reliability")
    args = parser.parse_args()
    train_x, _, _, _, _, train_sequences, train_future, train_timestamps = build_arrays(args.train_pkl, args.train_chain, args.fps_json, True)
    val_x, _, _, _, val_locations, val_sequences, val_future, val_timestamps = build_arrays(args.val_pkl, args.val_chain, args.fps_json, True)
    test_x, _, _, _, test_locations, _, _, _ = build_arrays(args.test_pkl, args.test_chain, args.fps_json, False)
    train_valid = valid_future_mask(train_sequences, train_timestamps)
    val_valid = valid_future_mask(val_sequences, val_timestamps)
    oof = np.zeros(len(val_x), np.float32)
    tests = []
    records = []
    args.out_model_dir.mkdir(parents=True, exist_ok=True)
    for held_sequence in sorted(set(val_sequences)):
        fit_x = [train_x]
        fit_future = [train_future]
        fit_valid = [train_valid]
        for sequence in sorted(set(val_sequences)):
            if sequence == held_sequence:
                continue
            mask = val_sequences == sequence
            fit_x.append(val_x[mask])
            fit_future.append(val_future[mask])
            fit_valid.append(val_valid[mask])
        combined_x = np.concatenate(fit_x)
        combined_future = np.concatenate(fit_future)
        combined_valid = np.concatenate(fit_valid)
        model, rows, positives = fit(combined_x, combined_future, combined_valid)
        held_mask = val_sequences == held_sequence
        oof[held_mask] = model.predict_proba(val_x[held_mask])[:, 1]
        tests.append(model.predict_proba(test_x)[:, 1])
        model_path = args.out_model_dir / f"action_chunk_future_without_{held_sequence}.ubj"
        model.save_model(model_path)
        record = {"excluded_validation_video": held_sequence, "rows": rows, "future_correct_rows": positives, "model": str(model_path)}
        records.append(record)
        print(json.dumps({"kind": "action_chunk_future_model", **record}), flush=True)
        del combined_x, combined_future, combined_valid, model
        gc.collect()
    test = np.mean(np.stack(tests), axis=0).astype(np.float32)
    write_score_jsonl(args.out_val_scores, oof, val_locations, args.score_field)
    write_score_jsonl(args.out_test_scores, test, test_locations, args.score_field)
    summary = {
        "model": "pure Action Chunk future-chain reliability auxiliary classifier",
        "inference_boundary": "past-only 1s/3s Action Bank and chain history",
        "training_target": "same chain has correct accepted detections in future 1 second",
        "features": int(train_x.shape[1]),
        "train_rows": len(train_x),
        "validation_rows": len(val_x),
        "test_rows": len(test_x),
        "train_valid_future_rows": int(train_valid.sum()),
        "train_future_correct_rows": int((train_valid & (train_future >= 0.5)).sum()),
        "validation_valid_future_rows": int(val_valid.sum()),
        "validation_future_correct_rows": int((val_valid & (val_future >= 0.5)).sum()),
        "models": records,
    }
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
