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

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.train_action_bank_motion_token_listwise import (
    FEATURE_NAMES,
    dataset_arrays,
    load_auxiliary,
    write_score_jsonl,
)


def ranking_feature_indices() -> np.ndarray:
    excluded = {
        "cx_norm", "cy_norm", "w_norm", "h_norm", "area_norm", "aspect_ratio",
        "candidate_count_log", "action_bank_future_consistency",
        "online_action_bank_future_consistency", "samurai_native_object_count",
        "samurai_native_best_object_id",
    }
    return np.asarray([index for index, name in enumerate(FEATURE_NAMES) if name not in excluded], dtype=np.int32)


def motion_feature_indices() -> np.ndarray:
    excluded_exact = {
        "raw_score", "raw_logit", "raw_rank_percentile", "raw_gap_to_max",
        "cx_norm", "cy_norm", "w_norm", "h_norm", "area_norm", "aspect_ratio",
        "candidate_count_log", "aux_present", "action_bank_score",
        "action_bank_learned_score", "action_bank_future_consistency",
        "samurai_cmc_score", "online_action_bank_score",
        "online_action_bank_future_consistency", "bank_rank_percentile", "bank_gap_to_max",
        "samurai_rank_percentile", "samurai_gap_to_max", "native_present",
        "samurai_native_score", "samurai_native_object_score", "samurai_native_object_count",
        "samurai_native_best_object_id", "native_rank_percentile", "native_gap_to_max",
    }
    selected = []
    for index, name in enumerate(FEATURE_NAMES):
        if name in excluded_exact or name.endswith("_detector_score"):
            continue
        selected.append(index)
    return np.asarray(selected, dtype=np.int32)


def load_dataset(prediction_path: Path, auxiliary_path: Path, with_labels: bool):
    auxiliary, sizes = load_auxiliary(auxiliary_path)
    predictions = load_predictionsgt(prediction_path)
    arrays = dataset_arrays(predictions, auxiliary, sizes, {}, with_labels)
    del predictions, auxiliary
    return arrays


def positive_group_view(features: np.ndarray, ious: np.ndarray, groups: list[tuple[int, int]]):
    chunks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    qids: list[np.ndarray] = []
    kept = 0
    for start, stop in groups:
        overlaps = ious[start:stop]
        if not (overlaps >= 0.5).any() or not (overlaps < 0.5).any():
            continue
        chunks.append(features[start:stop])
        relevance = np.where(overlaps >= 0.5, np.clip(1 + np.floor((overlaps - 0.5) * 10.0), 1, 5), 0)
        labels.append(relevance.astype(np.int32, copy=False))
        qids.append(np.full((stop - start,), kept, dtype=np.int32))
        kept += 1
    if not chunks:
        raise RuntimeError("no trainable positive/negative frame groups")
    return np.concatenate(chunks), np.concatenate(labels), np.concatenate(qids), kept


def hard_group_view(features: np.ndarray, ious: np.ndarray, groups: list[tuple[int, int]], raw_index: int, margin: float = 0.10, max_negatives: int = 8):
    chunks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    qids: list[np.ndarray] = []
    kept = 0
    corrected_frames = 0
    for start, stop in groups:
        overlaps = ious[start:stop]
        positives = np.flatnonzero(overlaps >= 0.5)
        negatives = np.flatnonzero(overlaps < 0.5)
        if not len(positives) or not len(negatives):
            continue
        raw = features[start:stop, raw_index]
        best_positive_score = float(raw[positives].max())
        hard_negatives = negatives[raw[negatives] >= best_positive_score - margin]
        if not len(hard_negatives):
            hard_negatives = negatives[np.argsort(raw[negatives])[::-1][:2]]
        else:
            hard_negatives = hard_negatives[np.argsort(raw[hard_negatives])[::-1][:max_negatives]]
        selected_positives = positives[np.argsort(overlaps[positives])[::-1][:3]]
        selected = np.concatenate((selected_positives, hard_negatives))
        order = np.argsort(raw[selected])[::-1]
        selected = selected[order]
        local_labels = np.where(overlaps[selected] >= 0.5, np.clip(1 + np.floor((overlaps[selected] - 0.5) * 10.0), 1, 5), 0).astype(np.int32)
        chunks.append(features[start + selected])
        labels.append(local_labels)
        qids.append(np.full((len(selected),), kept, dtype=np.int32))
        corrected_frames += int(not (overlaps[int(np.argmax(raw))] >= 0.5))
        kept += 1
    if not chunks:
        raise RuntimeError("no hard candidate groups")
    return np.concatenate(chunks), np.concatenate(labels), np.concatenate(qids), kept, corrected_frames


def frame_normalize(scores: np.ndarray, groups: list[tuple[int, int]]) -> np.ndarray:
    normalized = np.full(scores.shape, 0.5, dtype=np.float32)
    for start, stop in groups:
        values = scores[start:stop].astype(np.float64, copy=False)
        count = stop - start
        if count <= 1:
            continue
        order = np.argsort(values, kind="stable")
        ranks = np.empty((count,), dtype=np.float64)
        ranks[order] = np.arange(count, dtype=np.float64) / float(count - 1)
        center = float(np.median(values))
        scale = max(float(np.median(np.abs(values - center))) * 1.4826, float(np.std(values)), 1e-3)
        calibrated = 1.0 / (1.0 + np.exp(-np.clip((values - center) / scale, -12.0, 12.0)))
        normalized[start:stop] = (0.65 * ranks + 0.35 * calibrated).astype(np.float32)
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "train-pkl", "train-aux", "val-pkl", "val-aux", "test-pkl", "test-aux",
        "out-val", "out-test", "out-model", "out-summary",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--score-field", default="xgb_pairwise_score")
    parser.add_argument("--motion-only", action="store_true")
    parser.add_argument("--hard-mining", action="store_true")
    parser.add_argument("--hard-margin", type=float, default=0.10)
    args = parser.parse_args()

    train_x, train_iou, _, train_groups, _ = load_dataset(args.train_pkl, args.train_aux, True)
    val_x, val_iou, _, val_groups, val_locations = load_dataset(args.val_pkl, args.val_aux, True)
    test_x, _, _, test_groups, test_locations = load_dataset(args.test_pkl, args.test_aux, False)
    selected = motion_feature_indices() if args.motion_only else ranking_feature_indices()
    selected_train_x = train_x[:, selected]
    selected_val_x = val_x[:, selected]
    raw_index = int(np.flatnonzero(selected == FEATURE_NAMES.index("raw_score"))[0]) if FEATURE_NAMES.index("raw_score") in selected else 0
    if args.hard_mining:
        train_x, train_iou, train_qid, train_frame_count, train_corrected_frames = hard_group_view(selected_train_x, train_iou, train_groups, raw_index, args.hard_margin)
        val_rank_x, val_rank_iou, val_qid, val_frame_count, val_corrected_frames = hard_group_view(selected_val_x, val_iou, val_groups, raw_index, args.hard_margin)
    else:
        train_x, train_iou, train_qid, train_frame_count = positive_group_view(selected_train_x, train_iou, train_groups)
        val_rank_x, val_rank_iou, val_qid, val_frame_count = positive_group_view(selected_val_x, val_iou, val_groups)
        train_corrected_frames = val_corrected_frames = 0

    model = xgb.XGBRanker(
        n_estimators=1400, max_depth=7, learning_rate=0.035, min_child_weight=8,
        subsample=0.85, colsample_bytree=0.75, reg_lambda=8.0, reg_alpha=0.08,
        gamma=0.04, objective="rank:pairwise", eval_metric="ndcg@10",
        tree_method="hist", device="cuda", max_bin=256,
        lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=12,
        n_jobs=8, random_state=2026, early_stopping_rounds=100,
    )
    model.fit(
        train_x, train_iou, qid=train_qid,
        eval_set=[(val_rank_x, val_rank_iou)], eval_qid=[val_qid], verbose=25,
    )
    val_scores = frame_normalize(model.predict(val_x[:, selected]), val_groups)
    test_scores = frame_normalize(model.predict(test_x[:, selected]), test_groups)
    write_score_jsonl(args.out_val, val_scores, val_locations, args.score_field)
    write_score_jsonl(args.out_test, test_scores, test_locations, args.score_field)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.out_model)
    selected_set = set(selected.tolist())
    summary = {
        "model": "XGBRanker GPU pairwise motion-only" if args.motion_only else "XGBRanker GPU pairwise", "all_features": len(FEATURE_NAMES),
        "selected_features": len(selected),
        "excluded_features": [FEATURE_NAMES[index] for index in range(len(FEATURE_NAMES)) if index not in selected_set],
        "train_rows": len(train_x), "train_frames": train_frame_count,
        "validation_rank_frames": val_frame_count, "validation_rows": len(val_x),
        "test_rows": len(test_x), "best_iteration": model.best_iteration,
        "score_field": args.score_field, "motion_only": args.motion_only, "hard_mining": args.hard_mining, "hard_margin": args.hard_margin, "train_raw_top_wrong_frames": train_corrected_frames, "validation_raw_top_wrong_frames": val_corrected_frames, "causal_inputs": True, "frame_normalized_output": True,
    }
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
