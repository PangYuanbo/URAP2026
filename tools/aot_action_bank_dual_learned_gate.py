import argparse
import copy
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import aot_action_bank_learned_quality_gate as gate


CANDIDATE_SOURCES = (
    "action_bank_track_memory_promotion",
    "action_bank_cross_segment_interpolation",
    "action_bank_edge_extrapolation",
)


def detection_labels(match_folder: Path) -> dict[int, int]:
    matches = pd.read_csv(gate.find_match_csv(match_folder), usecols=["index", "gt_det_match"])
    matches = matches[matches["index"].notna()]
    return {int(index): int(value) for index, value in matches.groupby("index")["gt_det_match"].max().items()}


def train_model(rows, features, seed):
    indices = np.asarray([index for index, row in enumerate(rows) if row["is_validation"]], dtype=np.int64)
    train_features = features[indices]
    labels = np.asarray([rows[index]["label"] for index in indices], dtype=np.int8)
    groups = np.asarray([rows[index]["clip"] for index in indices])
    oof = gate.oof_predictions(train_features, labels, groups, seed)
    model = gate.make_model(seed)
    model.fit(train_features, labels)
    probabilities = model.predict_proba(features)[:, 1]
    return labels, oof, probabilities


def keep_metrics(labels, probabilities, thresholds):
    output = []
    for threshold in thresholds:
        selected = probabilities >= threshold
        true_positive = int(labels[selected].sum())
        false_positive = int(selected.sum() - true_positive)
        removed_positive = int(labels.sum() - true_positive)
        removed_negative = int((labels == 0).sum() - false_positive)
        output.append(
            {
                "threshold": threshold,
                "kept": int(selected.sum()),
                "kept_positive": true_positive,
                "kept_negative": false_positive,
                "removed_positive": removed_positive,
                "removed_negative": removed_negative,
                "positive_recall": true_positive / max(int(labels.sum()), 1),
                "removed_negative_rate": removed_negative / max(int((labels == 0).sum()), 1),
            }
        )
    return output


def filter_base(records, rows, probabilities, threshold):
    dropped = defaultdict(set)
    for row, probability in zip(rows, probabilities):
        if probability < threshold:
            dropped[row["record_position"]].add(row["detection_position"])
    output = copy.deepcopy(records)
    dropped_count = 0
    for record_position, positions in dropped.items():
        detections = output[record_position].get("detections") or []
        output[record_position]["detections"] = [
            detection for detection_position, detection in enumerate(detections) if detection_position not in positions
        ]
        dropped_count += len(positions)
    return output, dropped_count


def parse_variant(value: str):
    candidate, base = value.split(":", 1)
    return float(candidate), float(base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-predictions", required=True, type=Path)
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--candidate-match-folder", required=True, type=Path)
    parser.add_argument("--base-match-folder", required=True, type=Path)
    parser.add_argument("--validation-predictions", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", default=["0:0.05", "0.05:0.05", "0.1:0.05", "0:0.1"])
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    candidate_records = gate.load_pickle(args.candidate_predictions)
    base_records = gate.load_pickle(args.base_predictions)
    validation_records = gate.load_pickle(args.validation_predictions)
    validation_clips = {
        gate.parse_image(str(record.get("img_name") or ""))[0].removeprefix("Clip_") for record in validation_records
    }

    candidate_labels = gate.incremental_labels(
        gate.find_match_csv(args.candidate_match_folder), gate.find_match_csv(args.base_match_folder)
    )
    gate.SOURCES = CANDIDATE_SOURCES
    candidate_rows, candidate_features = gate.build_rows(candidate_records, candidate_labels, validation_clips)
    candidate_train_labels, candidate_oof, candidate_probabilities = train_model(
        candidate_rows, candidate_features, args.seed
    )

    base_labels = detection_labels(args.base_match_folder)
    gate.SOURCES = ("base",)
    base_rows, base_features = gate.build_rows(base_records, base_labels, validation_clips)
    base_train_labels, base_oof, base_probabilities = train_model(base_rows, base_features, args.seed + 1)

    candidate_thresholds = sorted({parse_variant(value)[0] for value in args.variants} | {0.05, 0.1, 0.15, 0.25})
    base_thresholds = sorted({parse_variant(value)[1] for value in args.variants} | {0.02, 0.05, 0.1, 0.15, 0.2})
    summary = {
        "protocol": "part0 clip-group OOF dual gate: incremental candidate recovery plus base false-positive suppression",
        "compute": "CPU; both tabular training sets are small and ExtraTrees parallelizes across CPU cores",
        "validation_clips": sorted(validation_clips),
        "candidate_oof": {
            "rows": int(len(candidate_train_labels)),
            "positive": int(candidate_train_labels.sum()),
            "average_precision": float(average_precision_score(candidate_train_labels, candidate_oof)),
            "roc_auc": float(roc_auc_score(candidate_train_labels, candidate_oof)),
            "thresholds": gate.threshold_metrics(candidate_train_labels, candidate_oof, candidate_thresholds),
        },
        "base_oof": {
            "rows": int(len(base_train_labels)),
            "positive": int(base_train_labels.sum()),
            "average_precision": float(average_precision_score(base_train_labels, base_oof)),
            "roc_auc": float(roc_auc_score(base_train_labels, base_oof)),
            "thresholds": keep_metrics(base_train_labels, base_oof, base_thresholds),
        },
        "variants": [],
        "uses_full_test_labels_for_training": False,
        "uses_part0_labels_for_training": True,
    }

    gate.SOURCES = CANDIDATE_SOURCES
    for value in args.variants:
        candidate_threshold, base_threshold = parse_variant(value)
        filtered_base, dropped_base = filter_base(base_records, base_rows, base_probabilities, base_threshold)
        output, counters = gate.merge_candidates(
            filtered_base, candidate_rows, candidate_probabilities, candidate_threshold
        )
        tag = f"c{candidate_threshold:.3f}_b{base_threshold:.3f}".replace(".", "p")
        variant_root = args.out_root / tag
        prediction_dir = variant_root / "aotpredictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        output_path = prediction_dir / "predictions_split_0.pkl"
        with output_path.open("wb") as handle:
            pickle.dump(output, handle)
        variant = {
            "candidate_threshold": candidate_threshold,
            "base_threshold": base_threshold,
            "dropped_base": dropped_base,
            "selected_candidates": int((candidate_probabilities >= candidate_threshold).sum()),
            "output": str(output_path),
            "counters": counters,
        }
        summary["variants"].append(variant)
        (variant_root / "dual_gate_summary.json").write_text(json.dumps(variant, indent=2), encoding="utf-8")

    (args.out_root / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
