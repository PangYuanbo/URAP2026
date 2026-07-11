import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import aot_action_bank_learned_quality_gate as gate
from aot_action_bank_crossfit_dual_gate import filter_records, model
from aot_action_bank_dual_learned_gate import detection_labels


FLOW_SOURCES = ("action_bank_camera_compensated_interpolation",)


def fit(rows, features, seed):
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    estimator = model(seed)
    estimator.fit(features, labels)
    probabilities = estimator.predict_proba(features)[:, 1]
    return estimator, labels, probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-predictions", required=True, type=Path)
    parser.add_argument("--train-candidate-match-folder", required=True, type=Path)
    parser.add_argument("--train-baseline-match-folder", required=True, type=Path)
    parser.add_argument("--target-predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--candidate-threshold", type=float, default=0.05)
    parser.add_argument("--base-threshold", type=float, default=0.1)
    parser.add_argument("--candidate-sources", nargs="+", default=list(FLOW_SOURCES))
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_records = gate.load_pickle(args.train_predictions)
    target_records = gate.load_pickle(args.target_predictions)
    candidate_labels = gate.incremental_labels(
        gate.find_match_csv(args.train_candidate_match_folder),
        gate.find_match_csv(args.train_baseline_match_folder),
    )
    base_labels = detection_labels(args.train_candidate_match_folder)

    gate.SOURCES = tuple(args.candidate_sources)
    train_candidate_rows, train_candidate_features = gate.build_rows(train_records, candidate_labels, set())
    candidate_model, candidate_train_labels, candidate_train_probabilities = fit(
        train_candidate_rows, train_candidate_features, args.seed
    )
    target_candidate_rows, target_candidate_features = gate.build_rows(target_records, {}, set())
    target_candidate_probabilities = candidate_model.predict_proba(target_candidate_features)[:, 1]

    gate.SOURCES = ("base",)
    train_base_rows, train_base_features = gate.build_rows(train_records, base_labels, set())
    base_model, base_train_labels, base_train_probabilities = fit(train_base_rows, train_base_features, args.seed + 100)
    target_base_rows, target_base_features = gate.build_rows(target_records, {}, set())
    target_base_probabilities = base_model.predict_proba(target_base_features)[:, 1]

    output, counters = filter_records(
        target_records,
        target_base_rows,
        target_base_probabilities,
        args.base_threshold,
        target_candidate_rows,
        target_candidate_probabilities,
        args.candidate_threshold,
    )
    prediction_dir = args.out_dir / "aotpredictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    output_path = prediction_dir / "predictions_split_0.pkl"
    with output_path.open("wb") as handle:
        pickle.dump(output, handle)
    joblib.dump(candidate_model, args.out_dir / "candidate_gate.joblib")
    joblib.dump(base_model, args.out_dir / "base_gate.joblib")
    summary = {
        "protocol": "part0-trained fixed camera-compensated Action Bank gate; full AOT application without target labels",
        "selection_source": "part0 clip-group OOF official evaluation",
        "candidate_threshold": args.candidate_threshold,
        "candidate_sources": args.candidate_sources,
        "base_threshold": args.base_threshold,
        "train_candidate_rows": len(train_candidate_rows),
        "train_candidate_positive": int(candidate_train_labels.sum()),
        "train_candidate_ap_in_sample": float(average_precision_score(candidate_train_labels, candidate_train_probabilities)),
        "train_candidate_auc_in_sample": float(roc_auc_score(candidate_train_labels, candidate_train_probabilities)),
        "train_base_rows": len(train_base_rows),
        "train_base_positive": int(base_train_labels.sum()),
        "train_base_ap_in_sample": float(average_precision_score(base_train_labels, base_train_probabilities)),
        "train_base_auc_in_sample": float(roc_auc_score(base_train_labels, base_train_probabilities)),
        "target_candidate_rows": len(target_candidate_rows),
        "target_base_rows": len(target_base_rows),
        "target_labels_used": False,
        "output": str(output_path),
        "counters": counters,
    }
    (args.out_dir / "fixed_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
