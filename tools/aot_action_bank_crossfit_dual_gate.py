import argparse
import copy
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

import aot_action_bank_learned_quality_gate as gate
from aot_action_bank_dual_learned_gate import detection_labels, keep_metrics


CANDIDATE_SOURCES = (
    "action_bank_track_memory_promotion",
    "action_bank_cross_segment_interpolation",
    "action_bank_edge_extrapolation",
)


def model(seed):
    return ExtraTreesClassifier(
        n_estimators=240,
        max_depth=12,
        min_samples_leaf=8,
        max_features=0.8,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def crossfit(rows, features, seed, folds):
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    groups = np.asarray([row["clip"] for row in rows])
    probabilities = np.zeros(len(rows), dtype=np.float32)
    splitter = GroupKFold(n_splits=folds)
    for fold, (train_indices, test_indices) in enumerate(splitter.split(features, labels, groups), start=1):
        estimator = model(seed + fold)
        estimator.fit(features[train_indices], labels[train_indices])
        probabilities[test_indices] = estimator.predict_proba(features[test_indices])[:, 1]
        print(json.dumps({"phase": "crossfit", "fold": fold, "folds": folds, "done": int(len(test_indices))}))
    return labels, probabilities


def crossfit_tracks(rows, features, seed, folds):
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(row["track_key"], row["source"])].append(index)
    track_keys = list(grouped)
    track_features = []
    track_labels = []
    track_groups = []
    for key in track_keys:
        indices = grouped[key]
        values = features[indices]
        track_features.append(
            np.concatenate(
                [values.mean(axis=0), values.max(axis=0), values.min(axis=0), values.std(axis=0), [np.log1p(len(indices))]]
            )
        )
        track_labels.append(max(rows[index]["label"] for index in indices))
        track_groups.append(rows[indices[0]]["clip"])
    track_features = np.asarray(track_features, dtype=np.float32)
    track_labels = np.asarray(track_labels, dtype=np.int8)
    track_groups = np.asarray(track_groups)
    track_probabilities = np.zeros(len(track_keys), dtype=np.float32)
    splitter = GroupKFold(n_splits=folds)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(track_features, track_labels, track_groups), start=1
    ):
        estimator = model(seed + fold)
        estimator.fit(track_features[train_indices], track_labels[train_indices])
        track_probabilities[test_indices] = estimator.predict_proba(track_features[test_indices])[:, 1]
        print(json.dumps({"phase": "track_crossfit", "fold": fold, "folds": folds, "done": int(len(test_indices))}))
    row_probabilities = np.zeros(len(rows), dtype=np.float32)
    for track_index, key in enumerate(track_keys):
        row_probabilities[grouped[key]] = track_probabilities[track_index]
    row_labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    return row_labels, row_probabilities, track_labels, track_probabilities


def parse_variant(value):
    candidate, base = value.split(":", 1)
    return float(candidate), float(base)


def aggregate_track_probabilities(rows, probabilities, mode):
    if mode == "none":
        return probabilities
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(row["track_key"], row["source"])].append(index)
    aggregated = probabilities.copy()
    for indices in grouped.values():
        values = probabilities[indices]
        if mode == "mean":
            score = float(np.mean(values))
        elif mode == "max":
            score = float(np.max(values))
        elif mode == "q75":
            score = float(np.quantile(values, 0.75))
        else:
            raise ValueError(f"Unknown track aggregation: {mode}")
        aggregated[indices] = score
    return aggregated


def filter_records(records, base_rows, base_probabilities, base_threshold, candidate_rows, candidate_probabilities, candidate_threshold):
    dropped = defaultdict(set)
    counters = defaultdict(int)
    for row, probability in zip(base_rows, base_probabilities):
        if probability < base_threshold:
            dropped[row["record_position"]].add(row["detection_position"])
            counters["dropped_base"] += 1
    for row, probability in zip(candidate_rows, candidate_probabilities):
        if probability < candidate_threshold:
            dropped[row["record_position"]].add(row["detection_position"])
            counters["dropped_candidate"] += 1
    output = copy.deepcopy(records)
    for record_position, positions in dropped.items():
        detections = output[record_position].get("detections") or []
        output[record_position]["detections"] = [
            detection for detection_position, detection in enumerate(detections) if detection_position not in positions
        ]
    counters["remaining_detections"] = sum(len(record.get("detections") or []) for record in output)
    return output, dict(counters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--candidate-match-folder", required=True, type=Path)
    parser.add_argument("--clean-base-match-folder", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--candidate-sources", nargs="+", default=list(CANDIDATE_SOURCES))
    parser.add_argument("--variants", nargs="+", default=["0:0.1", "0.05:0.1", "0.1:0.1", "0.1:0.15"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--candidate-track-aggregation", choices=["none", "mean", "max", "q75", "classifier"], default="none")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    records = gate.load_pickle(args.predictions)
    candidate_match_csv = gate.find_match_csv(args.candidate_match_folder)
    candidate_labels = gate.incremental_labels(candidate_match_csv, gate.find_match_csv(args.clean_base_match_folder))
    base_labels = detection_labels(args.candidate_match_folder)

    gate.SOURCES = tuple(args.candidate_sources)
    candidate_rows, candidate_features = gate.build_rows(records, candidate_labels, set())
    candidate_track_metrics = None
    if args.candidate_track_aggregation == "classifier":
        candidate_train_labels, candidate_probabilities, track_labels, track_probabilities = crossfit_tracks(
            candidate_rows, candidate_features, args.seed, args.folds
        )
        candidate_track_metrics = {
            "tracks": int(len(track_labels)),
            "positive": int(track_labels.sum()),
            "average_precision": float(average_precision_score(track_labels, track_probabilities)),
            "roc_auc": float(roc_auc_score(track_labels, track_probabilities)),
        }
    else:
        candidate_train_labels, candidate_probabilities = crossfit(
            candidate_rows, candidate_features, args.seed, args.folds
        )
        candidate_probabilities = aggregate_track_probabilities(
            candidate_rows, candidate_probabilities, args.candidate_track_aggregation
        )

    gate.SOURCES = ("base",)
    base_rows, base_features = gate.build_rows(records, base_labels, set())
    base_train_labels, base_probabilities = crossfit(base_rows, base_features, args.seed + 100, args.folds)

    candidate_thresholds = sorted({parse_variant(value)[0] for value in args.variants} | {0.05, 0.1, 0.15, 0.25})
    base_thresholds = sorted({parse_variant(value)[1] for value in args.variants} | {0.05, 0.1, 0.15, 0.2})
    summary = {
        "protocol": "strict clip-group OOF AOT ablation; every detection scored by a model trained without its clip",
        "claim_scope": "cross-fitted ablation, not zero-shot or untouched-test evidence",
        "folds": args.folds,
        "candidate_sources": args.candidate_sources,
        "candidate_track_aggregation": args.candidate_track_aggregation,
        "candidate_oof": {
            "rows": int(len(candidate_train_labels)),
            "positive": int(candidate_train_labels.sum()),
            "average_precision": float(average_precision_score(candidate_train_labels, candidate_probabilities)),
            "roc_auc": float(roc_auc_score(candidate_train_labels, candidate_probabilities)),
            "thresholds": gate.threshold_metrics(candidate_train_labels, candidate_probabilities, candidate_thresholds),
            "track_metrics": candidate_track_metrics,
        },
        "base_oof": {
            "rows": int(len(base_train_labels)),
            "positive": int(base_train_labels.sum()),
            "average_precision": float(average_precision_score(base_train_labels, base_probabilities)),
            "roc_auc": float(roc_auc_score(base_train_labels, base_probabilities)),
            "thresholds": keep_metrics(base_train_labels, base_probabilities, base_thresholds),
        },
        "variants": [],
    }

    for value in args.variants:
        candidate_threshold, base_threshold = parse_variant(value)
        output, counters = filter_records(
            records,
            base_rows,
            base_probabilities,
            base_threshold,
            candidate_rows,
            candidate_probabilities,
            candidate_threshold,
        )
        tag = f"c{candidate_threshold:.3f}_b{base_threshold:.3f}".replace(".", "p")
        root = args.out_root / tag
        prediction_dir = root / "aotpredictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        output_path = prediction_dir / "predictions_split_0.pkl"
        with output_path.open("wb") as handle:
            pickle.dump(output, handle)
        variant = {
            "candidate_threshold": candidate_threshold,
            "base_threshold": base_threshold,
            "output": str(output_path),
            "counters": counters,
        }
        summary["variants"].append(variant)
        (root / "crossfit_summary.json").write_text(json.dumps(variant, indent=2), encoding="utf-8")
    (args.out_root / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


