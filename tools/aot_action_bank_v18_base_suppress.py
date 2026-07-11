import argparse
import copy
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import aot_action_bank_learned_quality_gate as gate
from aot_action_bank_dual_learned_gate import detection_labels, keep_metrics, train_model


def filter_base_only(records, rows, probabilities, threshold):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--match-folder", required=True, type=Path)
    parser.add_argument("--validation-predictions", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.15, 0.2])
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    records = gate.load_pickle(args.predictions)
    validation = gate.load_pickle(args.validation_predictions)
    validation_clips = {
        gate.parse_image(str(record.get("img_name") or ""))[0].removeprefix("Clip_") for record in validation
    }
    labels = detection_labels(args.match_folder)
    gate.SOURCES = ("base",)
    rows, features = gate.build_rows(records, labels, validation_clips)
    train_labels, oof, probabilities = train_model(rows, features, args.seed)
    summary = {
        "protocol": "part0 clip-group OOF v18 base-only suppression; all action-bank recovery candidates preserved",
        "validation_clips": sorted(validation_clips),
        "training_rows": int(len(train_labels)),
        "training_positive": int(train_labels.sum()),
        "oof_average_precision": float(average_precision_score(train_labels, oof)),
        "oof_roc_auc": float(roc_auc_score(train_labels, oof)),
        "oof_thresholds": keep_metrics(train_labels, oof, args.thresholds),
        "variants": [],
        "uses_full_test_labels_for_training": False,
        "uses_part0_labels_for_training": True,
    }
    for threshold in args.thresholds:
        output, dropped = filter_base_only(records, rows, probabilities, threshold)
        tag = f"b{threshold:.3f}".replace(".", "p")
        root = args.out_root / tag
        prediction_dir = root / "aotpredictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        output_path = prediction_dir / "predictions_split_0.pkl"
        with output_path.open("wb") as handle:
            pickle.dump(output, handle)
        variant = {
            "threshold": threshold,
            "dropped_base": dropped,
            "remaining_detections": sum(len(record.get("detections") or []) for record in output),
            "output": str(output_path),
        }
        summary["variants"].append(variant)
        (root / "base_gate_summary.json").write_text(json.dumps(variant, indent=2), encoding="utf-8")
    (args.out_root / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
