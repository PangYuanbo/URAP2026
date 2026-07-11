from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

REPO = Path(__file__).resolve().parents[1]
for entry in (REPO, REPO / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.train_action_bank_motion_token_listwise import write_score_jsonl
from tools.train_action_chunk_bidir_full import load_aux
from tools.train_action_chunk_neighbor_full import dataset_arrays, load_neighbor


def selected_ranking_rows(x: np.ndarray, y: np.ndarray, groups: list[tuple[int, int]], margin: float = 0.45, max_negatives: int = 40) -> tuple[np.ndarray, np.ndarray]:
    keep: list[int] = []
    qid: list[int] = []
    group_id = 0
    for start, stop in groups:
        local_y = y[start:stop]
        positives = np.flatnonzero(local_y >= 0.5)
        negatives = np.flatnonzero(local_y < 0.5)
        if not len(positives) or not len(negatives):
            continue
        raw = x[start:stop, 0]
        threshold = raw[positives].max() - margin
        hard = negatives[raw[negatives] >= threshold]
        if len(hard):
            hard = hard[np.argsort(raw[hard])[::-1][:max_negatives]]
        else:
            hard = negatives[np.argsort(raw[negatives])[::-1][:8]]
        local = np.sort(np.concatenate((positives, hard)))
        keep.extend((start + local).tolist())
        qid.extend([group_id] * len(local))
        group_id += 1
    return np.asarray(keep, dtype=np.int64), np.asarray(qid, dtype=np.int32)


def normalize_by_group(values: np.ndarray, groups: list[tuple[int, int]]) -> np.ndarray:
    output = np.zeros(len(values), dtype=np.float32)
    for start, stop in groups:
        block = np.asarray(values[start:stop], dtype=np.float32)
        if not len(block):
            continue
        center = float(np.median(block))
        scale = max(1e-3, float(np.std(block)))
        z = np.clip((block - center) / scale, -12.0, 12.0)
        output[start:stop] = 1.0 / (1.0 + np.exp(-z))
    return output


def fit_ranker(x: np.ndarray, y: np.ndarray, groups: list[tuple[int, int]], seed: int) -> tuple[xgb.XGBRanker, int, int, int]:
    keep, qid = selected_ranking_rows(x, y, groups)
    selected_y = y[keep]
    relevance = np.zeros(len(selected_y), dtype=np.float32)
    relevance[selected_y >= 0.5] = 1.0
    relevance[selected_y >= 0.6] = 2.0
    relevance[selected_y >= 0.7] = 3.0
    relevance[selected_y >= 0.8] = 4.0
    relevance[selected_y >= 0.9] = 5.0
    model = xgb.XGBRanker(
        n_estimators=1200,
        max_depth=8,
        learning_rate=0.025,
        min_child_weight=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=10.0,
        reg_alpha=0.12,
        gamma=0.02,
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=24,
        tree_method="hist",
        device="cuda",
        max_bin=256,
        n_jobs=8,
        random_state=seed,
    )
    model.fit(x[keep], relevance, qid=qid, verbose=False)
    return model, len(keep), int((relevance > 0).sum()), int(qid.max() + 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Corrected-GT neighbor Action Bank LambdaRank.")
    for name in (
        "train-pkl", "train-forward", "train-backward", "train-neighbor",
        "val-pkl", "val-forward", "val-backward", "val-neighbor",
        "test-pkl", "test-forward", "test-backward", "test-neighbor",
        "out-val-scores", "out-test-scores", "out-model-dir", "out-summary",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sequence-size-json", type=Path, required=True)
    parser.add_argument("--score-field", default="action_chunk_neighbor_lambdarank_score")
    args = parser.parse_args()
    size_map = json.loads(args.sequence_size_json.read_text(encoding="utf-8"))

    train_forward, train_backward = load_aux(args.train_forward), load_aux(args.train_backward)
    train_neighbor, feature_names = load_neighbor(args.train_neighbor)
    train_x, train_y, train_groups, _, _ = dataset_arrays(load_predictionsgt(args.train_pkl), train_forward, train_backward, train_neighbor, True, size_map)
    del train_forward, train_backward, train_neighbor
    gc.collect()

    val_forward, val_backward = load_aux(args.val_forward), load_aux(args.val_backward)
    val_neighbor, val_feature_names = load_neighbor(args.val_neighbor)
    if feature_names != val_feature_names:
        raise RuntimeError("train/validation neighbor fields differ")
    val_x, val_y, val_groups, val_locations, val_sequences = dataset_arrays(load_predictionsgt(args.val_pkl), val_forward, val_backward, val_neighbor, True, size_map)
    del val_forward, val_backward, val_neighbor
    gc.collect()

    test_forward, test_backward = load_aux(args.test_forward), load_aux(args.test_backward)
    test_neighbor, test_feature_names = load_neighbor(args.test_neighbor)
    if feature_names != test_feature_names:
        raise RuntimeError("train/test neighbor fields differ")
    test_x, _, test_groups, test_locations, _ = dataset_arrays(load_predictionsgt(args.test_pkl), test_forward, test_backward, test_neighbor, False, size_map)
    del test_forward, test_backward, test_neighbor
    gc.collect()

    val_raw = np.zeros(len(val_x), dtype=np.float32)
    test_raw: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    args.out_model_dir.mkdir(parents=True, exist_ok=True)
    sequences = sorted(set(val_sequences.tolist()))
    for fold, held in enumerate(sequences):
        parts = [train_x]
        labels = [train_y]
        groups = list(train_groups)
        cursor = len(train_x)
        for start, stop in val_groups:
            if val_sequences[start] == held:
                continue
            parts.append(val_x[start:stop])
            labels.append(val_y[start:stop])
            groups.append((cursor, cursor + stop - start))
            cursor += stop - start
        fit_x = np.concatenate(parts)
        fit_y = np.concatenate(labels)
        model, rows, positives, group_count = fit_ranker(fit_x, fit_y, groups, 2026 + fold)
        held_mask = val_sequences == held
        val_raw[held_mask] = model.predict(val_x[held_mask])
        test_raw.append(model.predict(test_x).astype(np.float32))
        model_path = args.out_model_dir / f"neighbor_lambdarank_without_{held}.ubj"
        model.save_model(model_path)
        record = {"held": held, "rank_rows": rows, "positive_rows": positives, "groups": group_count, "model": str(model_path)}
        records.append(record)
        print(json.dumps({"kind": "neighbor_lambdarank_fold", **record}), flush=True)
        del fit_x, fit_y, model
        gc.collect()

    val_scores = normalize_by_group(val_raw, val_groups)
    test_scores = normalize_by_group(np.mean(np.stack(test_raw), axis=0), test_groups)
    write_score_jsonl(args.out_val_scores, val_scores, val_locations, args.score_field)
    write_score_jsonl(args.out_test_scores, test_scores, test_locations, args.score_field)
    summary = {
        "model": "corrected-GT Action Bank neighbor LambdaRank",
        "objective": "rank:ndcg per frame",
        "features": int(train_x.shape[1]),
        "train_rows": len(train_x),
        "validation_rows": len(val_x),
        "test_rows": len(test_x),
        "neighbor_fields": feature_names,
        "folds": records,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

