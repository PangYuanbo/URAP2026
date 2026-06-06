import csv
import json
import pickle
from pathlib import Path

import torch

from qstr_dronedet.tracking.tracklet_classifier import (
    _features,
    apply_tracklet_filter_to_infer_outputs,
    build_tracklet_classifier_official_eval_bundle,
    build_tracklet_dataset,
    evaluate_tracklet_classifier,
    evaluate_tracklet_classifier_thresholds,
    export_aot_prediction_parts_to_tracklets,
    filter_aot_prediction_parts_by_tracklets,
    rescore_aot_prediction_parts_by_tracklets,
    export_tracklet_classifier_aot_prediction_parts,
    export_tracklet_classifier_official_predictions,
    export_tracklet_jsonl_classifier_dataset,
    filter_infer_rows_with_tracklet_classifier,
    merge_tracklet_classifier_datasets,
    run_tracklet_classifier_frame_benchmark,
    run_tracklet_classifier_mixture_benchmark,
    validate_tracklet_classifier_aot_eval_inputs,
    score_tracklets_from_rows,
    train_tracklet_classifier,
    validate_tracklet_classifier_frame_benchmark_inputs,
    validate_tracklet_classifier_mixture_inputs,
)


def _row(frame_id, track_id, box, crop_drone, temp_drone, final_drone, bg, source="tracker"):
    return {
        "frame_id": frame_id,
        "bbox": box,
        "objectness": 0.3,
        "source": source,
        "final_drone_score": final_drone * 0.3,
        "track_id": track_id,
        "track_validated": True,
        "track_drift": 2.0,
        "track_speed": 1.0,
        "crop_probs": {"drone": crop_drone, "background": 1.0 - crop_drone},
        "temporal_probs": {"drone": temp_drone, "background": 1.0 - temp_drone},
        "final_probs": {"drone": final_drone, "background": bg},
    }


def test_build_train_eval_tracklet_classifier(tmp_path):
    run_dir = tmp_path / "seq001"
    run_dir.mkdir()
    diag = run_dir / "diagnostics.jsonl"
    rows = [
        _row(0, 1, [10, 10, 20, 20], 0.45, 0.65, 0.55, 0.35, "tracker+yolo_tile"),
        _row(1, 1, [11, 10, 21, 20], 0.46, 0.66, 0.56, 0.34, "tracker"),
        _row(0, 2, [100, 100, 110, 110], 0.20, 0.30, 0.25, 0.70, "tracker+yolo_tile"),
        _row(1, 2, [101, 100, 111, 110], 0.22, 0.32, 0.24, 0.72, "tracker"),
    ]
    diag.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow([str(tmp_path / "seq001" / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])
        writer.writerow([str(tmp_path / "seq001" / "visible.mp4"), 1, 11, 10, 21, 20, "drone", "tiny"])

    result = build_tracklet_dataset([diag], gt, tmp_path / "tracklets")

    assert result.summary["num_tracklets"] == 2
    assert result.summary["positives"] == 1

    weights = train_tracklet_classifier(result.csv_path, tmp_path / "tracklet.pt", epochs=2)
    metrics = evaluate_tracklet_classifier(result.csv_path, weights, tmp_path / "eval")

    assert metrics["num_tracklets"] == 2
    assert (tmp_path / "eval" / "metrics.json").exists()


def test_evaluate_tracklet_classifier_thresholds(tmp_path):
    run_dir = tmp_path / "seq001"
    run_dir.mkdir()
    diag = run_dir / "diagnostics.jsonl"
    rows = [
        _row(0, 1, [10, 10, 20, 20], 0.45, 0.65, 0.55, 0.35, "tracker+yolo_tile"),
        _row(1, 1, [11, 10, 21, 20], 0.46, 0.66, 0.56, 0.34, "tracker"),
        _row(0, 2, [100, 100, 110, 110], 0.20, 0.30, 0.25, 0.70, "tracker+yolo_tile"),
        _row(1, 2, [101, 100, 111, 110], 0.22, 0.32, 0.24, 0.72, "tracker"),
    ]
    diag.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow([str(tmp_path / "seq001" / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])

    dataset = build_tracklet_dataset([diag], gt, tmp_path / "tracklets")
    weights = train_tracklet_classifier(dataset.csv_path, tmp_path / "tracklet.pt", epochs=2)
    result = evaluate_tracklet_classifier_thresholds(dataset.csv_path, weights, tmp_path / "sweep", thresholds=[0.3, 0.5])

    assert result.csv_path.exists()
    assert result.summary_path.exists()
    assert result.summary["num_tracklets"] == 2
    assert result.summary["best"]["threshold"] in {0.3, 0.5}


def test_tracklet_dataset_includes_temporal_shape_features(tmp_path):
    run_dir = tmp_path / "seq001"
    run_dir.mkdir()
    diag = run_dir / "diagnostics.jsonl"
    rows = [
        _row(0, 1, [10, 10, 20, 20], 0.40, 0.55, 0.40, 0.60, "tracker+yolo_tile"),
        _row(1, 1, [11, 10, 21, 20], 0.42, 0.65, 0.55, 0.45, "tracker"),
        _row(3, 1, [12, 10, 22, 20], 0.44, 0.75, 0.70, 0.30, "tracker"),
    ]
    diag.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow([str(tmp_path / "seq001" / "visible.mp4"), 1, 11, 10, 21, 20, "drone", "tiny"])

    result = build_tracklet_dataset([diag], gt, tmp_path / "tracklets")
    with result.csv_path.open("r", encoding="utf-8", newline="") as f:
        out_rows = list(csv.DictReader(f))

    assert "score_slope" in out_rows[0]
    assert "background_dominance_longest_streak" in out_rows[0]
    assert "temporal_over_background_longest_streak" in out_rows[0]
    assert float(out_rows[0]["score_slope"]) > 0
    assert float(out_rows[0]["max_frame_gap"]) == 1.0
    assert float(out_rows[0]["background_dominance_longest_streak"]) == 1.0
    assert float(out_rows[0]["temporal_over_background_longest_streak"]) == 2.0


def test_tracklet_features_include_action_frame_prior_support():
    rows = [
        {
            **_row(0, 1, [10, 10, 20, 20], 0.45, 0.55, 0.50, 0.40),
            "action_frame_prior_score": 0.8,
            "action_frame_prior_num_tracklet_priors": 3,
        },
        {
            **_row(1, 1, [11, 10, 21, 20], 0.45, 0.55, 0.50, 0.40),
            "action_frame_prior_score": 0.4,
            "action_frame_prior_num_tracklet_priors": 1,
        },
        _row(2, 1, [12, 10, 22, 20], 0.45, 0.55, 0.50, 0.40),
    ]

    feats = _features(rows)

    assert abs(feats["mean_action_frame_prior_score"] - 0.4) < 1e-9
    assert abs(feats["max_action_frame_prior_score"] - 0.8) < 1e-9
    assert abs(feats["action_frame_prior_coverage_rate"] - 2 / 3) < 1e-9
    assert abs(feats["mean_action_frame_prior_tracklet_support"] - 4 / 3) < 1e-9


def test_export_tracklet_jsonl_classifier_dataset_preserves_action_features(tmp_path):
    tracklet_jsonl = tmp_path / "proposal_tracklets_with_dynamics.jsonl"
    item = {
        "meta": {"seq": "s1", "track_id": "proposal_1", "label": 1, "bucket": "positive", "dataset_source": "ard100", "best_iou": 0.8, "matched_frames": 3},
        "rows": [
            {
                **_row(0, "proposal_1", [10, 10, 20, 20], 0.4, 0.6, 0.5, 0.4),
                "action_dynamics_score": 0.7,
                "action_error_improvement_vs_cv": 0.2,
                "action_mean_learned_center_error": 1.5,
                "action_num_windows": 2,
            },
            {
                **_row(1, "proposal_1", [11, 10, 21, 20], 0.4, 0.6, 0.5, 0.4),
                "action_dynamics_score": 0.7,
                "action_error_improvement_vs_cv": 0.2,
                "action_mean_learned_center_error": 1.5,
                "action_num_windows": 2,
            },
        ],
    }
    tracklet_jsonl.write_text(json.dumps(item) + "\n", encoding="utf-8")

    result = export_tracklet_jsonl_classifier_dataset(tracklet_jsonl, tmp_path / "classifier")
    with result.csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert result.summary["num_tracklets"] == 1
    assert result.summary["tracklets_with_action_dynamics"] == 1
    assert rows[0]["seq"] == "s1"
    assert rows[0]["bucket"] == "positive"
    assert rows[0]["dataset_source"] == "ard100"
    assert float(rows[0]["mean_action_dynamics_score"]) == 0.7
    assert float(rows[0]["mean_action_error_improvement_vs_cv"]) == 0.2

    override = export_tracklet_jsonl_classifier_dataset(tracklet_jsonl, tmp_path / "classifier_override", dataset_source="nps")
    with override.csv_path.open("r", encoding="utf-8", newline="") as f:
        override_rows = list(csv.DictReader(f))
    assert override.summary["dataset_source"] == "nps"
    assert override_rows[0]["dataset_source"] == "nps"


def test_train_tracklet_classifier_records_balance_groups(tmp_path):
    csv_path = tmp_path / "tracklets.csv"
    fields = ["seq", "track_id", "label", "bucket", "dataset_source"] + [
        "num_rows",
        "mean_objectness",
        "max_objectness",
        "mean_final_score",
        "max_final_score",
        "mean_crop_drone",
        "mean_temporal_drone",
        "mean_final_drone",
        "mean_background",
        "temporal_minus_crop_mean",
        "temporal_minus_background_mean",
        "final_minus_background_mean",
        "temporal_gain_rate",
        "detector_update_rate",
        "fallback_rate",
        "validated_rate",
        "mean_track_drift",
        "max_track_drift",
        "mean_track_speed",
        "mean_box_side",
        "std_box_side",
        "mean_center_step",
        "max_center_step",
        "std_center_step",
        "track_span_frames",
        "frame_density",
        "weak_detector_temporal_signal",
        "score_above_02_rate",
        "score_slope",
        "objectness_slope",
        "temporal_drone_slope",
        "background_slope",
        "final_margin_mean",
        "final_margin_min",
        "final_margin_slope",
        "background_dominance_rate",
        "background_dominance_longest_streak",
        "temporal_over_background_rate",
        "temporal_over_background_longest_streak",
        "score_above_02_longest_streak",
        "max_frame_gap",
        "mean_frame_gap",
        "gap_rate",
        "first_final_score",
        "last_final_score",
        "first_background",
        "last_background",
        "mean_action_dynamics_score",
        "min_action_dynamics_score",
        "mean_action_error_improvement_vs_cv",
        "mean_action_learned_center_error",
        "mean_action_frame_prior_score",
        "max_action_frame_prior_score",
        "action_frame_prior_coverage_rate",
        "mean_action_frame_prior_tracklet_support",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for index in range(4):
            row = {field: 0.0 for field in fields}
            row.update(
                {
                    "seq": f"s{index}",
                    "track_id": f"t{index}",
                    "label": 1 if index == 0 else 0,
                    "bucket": "positive" if index == 0 else "easy_background",
                    "dataset_source": "nps" if index < 3 else "aot",
                    "num_rows": 2,
                    "mean_objectness": 0.8 if index == 0 else 0.1,
                    "max_objectness": 0.9 if index == 0 else 0.2,
                    "mean_final_score": 0.8 if index == 0 else 0.1,
                    "max_final_score": 0.9 if index == 0 else 0.2,
                }
            )
            writer.writerow(row)

    weights = train_tracklet_classifier(
        csv_path,
        tmp_path / "balanced_tracklet.pt",
        epochs=2,
        balance_by=["dataset_source", "bucket", "label"],
    )
    ckpt = torch.load(weights, map_location="cpu")

    assert ckpt["balance"]["enabled"] is True
    assert ckpt["balance"]["balance_by"] == ["dataset_source", "bucket", "label"]
    assert ckpt["balance"]["group_counts"]["nps|easy_background|0"] == 2
    assert ckpt["balance"]["group_counts"]["aot|easy_background|0"] == 1
    assert ckpt["balance"]["max_weight"] > ckpt["balance"]["min_weight"]


def test_merge_tracklet_classifier_datasets_writes_manifest(tmp_path):
    def write_csv(path, source, label):
        item = {
            "meta": {"seq": source, "track_id": "proposal_1", "label": label, "bucket": "positive" if label else "easy_background", "dataset_source": source},
            "rows": [_row(0, "proposal_1", [10, 10, 20, 20], 0.4, 0.6, 0.5, 0.4)],
        }
        src_jsonl = path.with_suffix(".jsonl")
        src_jsonl.write_text(json.dumps(item) + "\n", encoding="utf-8")
        return export_tracklet_jsonl_classifier_dataset(src_jsonl, path.parent / path.stem, dataset_source=source).csv_path

    nps = write_csv(tmp_path / "nps.csv", "nps", 1)
    aot = write_csv(tmp_path / "aot.csv", "aot", 0)

    result = merge_tracklet_classifier_datasets(
        [nps, aot],
        tmp_path / "merged" / "tracklets.csv",
        source_names=["nps_override", "aot_override"],
        manifest_out=tmp_path / "merged" / "manifest.json",
    )
    with result.csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.summary["rows"] == 2
    assert result.summary["positives"] == 1
    assert result.summary["dataset_source_counts"] == {"aot_override": 1, "nps_override": 1}
    assert rows[0]["dataset_source"] == "nps_override"
    assert rows[1]["dataset_source"] == "aot_override"
    assert manifest["datasets"][0]["rows"] == 1


def test_run_tracklet_classifier_mixture_benchmark(tmp_path):
    def write_csv(name, source, labels):
        items = []
        for idx, label in enumerate(labels):
            track_id = f"proposal_{idx}"
            items.append(
                {
                    "meta": {"seq": source, "track_id": track_id, "label": label, "bucket": "positive" if label else "easy_background", "dataset_source": source},
                    "rows": [_row(0, track_id, [10, 10, 20, 20], 0.7 if label else 0.1, 0.8 if label else 0.2, 0.7 if label else 0.1, 0.2 if label else 0.9)],
                }
            )
        src_jsonl = tmp_path / f"{name}.jsonl"
        src_jsonl.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
        return export_tracklet_jsonl_classifier_dataset(src_jsonl, tmp_path / name, dataset_source=source).csv_path

    train_a = write_csv("train_a", "nps", [1, 0])
    train_b = write_csv("train_b", "aot", [1, 0])
    eval_pos = write_csv("eval_pos", "heldout_pos", [1])
    eval_neg = write_csv("eval_neg", "heldout_neg", [0])
    eval_csv = merge_tracklet_classifier_datasets(
        [eval_pos, eval_neg],
        tmp_path / "eval_merged" / "tracklets.csv",
        source_names=["heldout", "heldout"],
    ).csv_path

    result = run_tracklet_classifier_mixture_benchmark(
        [train_a, train_b],
        [eval_csv],
        tmp_path / "benchmark",
        train_source_names=["nps", "aot"],
        eval_dataset_names=["heldout"],
        epochs=2,
        balance_by=["dataset_source", "label"],
        thresholds=[0.5],
    )

    assert result.out_path.exists()
    assert (tmp_path / "benchmark" / "tracklet_classifier_mixture_preflight.json").exists()
    assert (tmp_path / "benchmark" / "train" / "mixed_tracklets.manifest.json").exists()
    assert (tmp_path / "benchmark" / "train" / "joint_tracklet_classifier.pt").exists()
    assert (tmp_path / "benchmark" / "eval" / "heldout" / "tracklet_classifier_threshold_summary.json").exists()
    assert result.summary["mixed_train"]["rows"] == 4
    assert result.summary["preflight"]["valid"] is True
    assert result.summary["eval_dataset_names"] == ["heldout"]
    assert "heldout" in result.summary["best_by_dataset"]


def test_run_tracklet_classifier_mixture_benchmark_writes_baseline_report(tmp_path):
    def write_csv(name, source, labels):
        items = []
        for idx, label in enumerate(labels):
            track_id = f"proposal_{idx}"
            items.append(
                {
                    "meta": {"seq": source, "track_id": track_id, "label": label, "bucket": "positive" if label else "easy_background", "dataset_source": source},
                    "rows": [_row(0, track_id, [10, 10, 20, 20], 0.7 if label else 0.1, 0.8 if label else 0.2, 0.7 if label else 0.1, 0.2 if label else 0.9)],
                }
            )
        src_jsonl = tmp_path / f"{name}.jsonl"
        src_jsonl.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
        return export_tracklet_jsonl_classifier_dataset(src_jsonl, tmp_path / name, dataset_source=source).csv_path

    train_a = write_csv("train_a", "nps", [1, 0])
    train_b = write_csv("train_b", "aot", [1, 0])
    eval_pos = write_csv("eval_pos", "heldout_pos", [1])
    eval_neg = write_csv("eval_neg", "heldout_neg", [0])
    eval_csv = merge_tracklet_classifier_datasets(
        [eval_pos, eval_neg],
        tmp_path / "eval_merged" / "tracklets.csv",
        source_names=["heldout", "heldout"],
    ).csv_path
    baseline_csv = tmp_path / "baseline.csv"
    baseline_csv.write_text(
        "dataset,method,tracklet_best_f1,best_precision,best_recall\n"
        "heldout,YOLOMG-tracklet-baseline,-0.10,0.10,0.10\n",
        encoding="utf-8",
    )

    result = run_tracklet_classifier_mixture_benchmark(
        [train_a, train_b],
        [eval_csv],
        tmp_path / "benchmark_baseline",
        train_source_names=["nps", "aot"],
        eval_dataset_names=["heldout"],
        epochs=2,
        balance_by=["dataset_source", "label"],
        thresholds=[0.5],
        baseline_csv=baseline_csv,
        baseline_metric="tracklet_best_f1",
    )

    assert (tmp_path / "benchmark_baseline" / "tracklet_classifier_mixture_route_b_results.csv").exists()
    assert (tmp_path / "benchmark_baseline" / "baseline_report" / "route_b_tracklet_classifier_baseline_report.md").exists()
    assert result.summary["baseline_report"]["num_comparisons"] == 1
    assert result.summary["baseline_report"]["route_b_wins"] == 1
    assert result.summary["baseline_report"]["metric"] == "tracklet_best_f1"


def test_validate_tracklet_classifier_mixture_inputs_flags_sequence_overlap(tmp_path):
    def write_csv(name, source, seq, label):
        item = {
            "meta": {"seq": seq, "track_id": f"proposal_{label}", "label": label, "bucket": "positive" if label else "easy_background", "dataset_source": source},
            "rows": [_row(0, f"proposal_{label}", [10, 10, 20, 20], 0.7 if label else 0.1, 0.8 if label else 0.2, 0.7 if label else 0.1, 0.2 if label else 0.9)],
        }
        src_jsonl = tmp_path / f"{name}.jsonl"
        src_jsonl.write_text(json.dumps(item) + "\n", encoding="utf-8")
        return export_tracklet_jsonl_classifier_dataset(src_jsonl, tmp_path / name, dataset_source=source).csv_path

    train_pos = write_csv("train_pos", "nps", "shared_seq", 1)
    train_neg = write_csv("train_neg", "nps", "train_only", 0)
    eval_pos = write_csv("eval_pos", "nps", "shared_seq", 1)
    eval_neg = write_csv("eval_neg", "nps", "eval_only", 0)
    train_csv = merge_tracklet_classifier_datasets([train_pos, train_neg], tmp_path / "train" / "tracklets.csv").csv_path
    eval_csv = merge_tracklet_classifier_datasets([eval_pos, eval_neg], tmp_path / "eval" / "tracklets.csv").csv_path

    result = validate_tracklet_classifier_mixture_inputs(
        [train_csv],
        [eval_csv],
        tmp_path / "preflight.json",
        train_source_names=["nps"],
        eval_dataset_names=["nps_heldout"],
    )

    assert result.out_path.exists()
    assert result.summary["valid"] is False
    assert result.summary["combined"]["train_rows"] == 2
    assert result.summary["combined"]["eval_rows"] == 2
    assert "shared_seq" in result.summary["combined"]["train_eval_sequence_overlap"]
    assert any("train/eval sequence overlap" in error for error in result.summary["errors"])


def test_apply_tracklet_filter_rewrites_final_predictions(tmp_path):
    run_dir = tmp_path / "seq001"
    run_dir.mkdir()
    diag = run_dir / "diagnostics.jsonl"
    rows = [
        _row(0, 1, [10, 10, 20, 20], 0.55, 0.75, 0.70, 0.20, "tracker+yolo_tile"),
        _row(1, 1, [11, 10, 21, 20], 0.56, 0.76, 0.71, 0.19, "tracker"),
        _row(0, 2, [100, 100, 110, 110], 0.10, 0.10, 0.08, 0.90, "tracker+yolo_tile"),
        _row(1, 2, [101, 100, 111, 110], 0.11, 0.10, 0.08, 0.90, "tracker"),
    ]
    for r in rows:
        r["predicted_class"] = "drone"
    diag.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    pred = run_dir / "predictions.jsonl"
    pred.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow([str(tmp_path / "seq001" / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])
        writer.writerow([str(tmp_path / "seq001" / "visible.mp4"), 1, 11, 10, 21, 20, "drone", "tiny"])

    result = build_tracklet_dataset([diag], gt, tmp_path / "tracklets")
    weights = train_tracklet_classifier(result.csv_path, tmp_path / "tracklet.pt", epochs=20)
    summary = apply_tracklet_filter_to_infer_outputs(pred, diag, weights, threshold=0.99)

    assert (run_dir / "predictions_raw.jsonl").exists()
    assert summary["raw_drone_predictions"] == 4
    filtered = [json.loads(line) for line in pred.read_text(encoding="utf-8").splitlines()]
    assert any(r["predicted_class"] == "background" and r.get("raw_predicted_class") == "drone" for r in filtered)
    assert all("tracklet_classifier_prob" in r for r in filtered)


def test_tracklet_scoring_scopes_repeated_track_ids_by_sequence(tmp_path):
    train_rows = [
        _row(0, 1, [10, 10, 20, 20], 0.55, 0.75, 0.70, 0.20, "tracker+yolo_tile"),
        _row(1, 1, [11, 10, 21, 20], 0.56, 0.76, 0.71, 0.19, "tracker"),
        _row(0, 2, [100, 100, 110, 110], 0.10, 0.10, 0.08, 0.90, "tracker+yolo_tile"),
        _row(1, 2, [101, 100, 111, 110], 0.11, 0.10, 0.08, 0.90, "tracker"),
    ]
    run_dir = tmp_path / "seq_train"
    run_dir.mkdir()
    diag = run_dir / "diagnostics.jsonl"
    diag.write_text("\n".join(json.dumps(r) for r in train_rows) + "\n", encoding="utf-8")
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow([str(tmp_path / "seq_train" / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])
        writer.writerow([str(tmp_path / "seq_train" / "visible.mp4"), 1, 11, 10, 21, 20, "drone", "tiny"])

    result = build_tracklet_dataset([diag], gt, tmp_path / "tracklets")
    weights = train_tracklet_classifier(result.csv_path, tmp_path / "tracklet.pt", epochs=5)
    rows = [
        {**_row(0, 1, [10, 10, 20, 20], 0.55, 0.75, 0.70, 0.20), "seq": "seq_a"},
        {**_row(1, 1, [11, 10, 21, 20], 0.56, 0.76, 0.71, 0.19), "seq": "seq_a"},
        {**_row(0, 1, [200, 200, 210, 210], 0.10, 0.10, 0.08, 0.90), "seq": "seq_b"},
        {**_row(1, 1, [201, 200, 211, 210], 0.11, 0.10, 0.08, 0.90), "seq": "seq_b"},
    ]

    scores = score_tracklets_from_rows(rows, weights, threshold=0.5)

    assert set(scores) == {"seq_a:1", "seq_b:1"}
    assert all(score["num_rows"] == 2 for score in scores.values())


def test_run_tracklet_classifier_frame_benchmark_filters_copy_without_mutating_source(tmp_path):
    train_dir = tmp_path / "seq001"
    train_dir.mkdir()
    diag = train_dir / "diagnostics.jsonl"
    rows = [
        _row(0, 1, [10, 10, 20, 20], 0.55, 0.75, 0.70, 0.20, "tracker+yolo_tile"),
        _row(1, 1, [11, 10, 21, 20], 0.56, 0.76, 0.71, 0.19, "tracker"),
        _row(0, 2, [100, 100, 110, 110], 0.10, 0.10, 0.08, 0.90, "tracker+yolo_tile"),
        _row(1, 2, [101, 100, 111, 110], 0.11, 0.10, 0.08, 0.90, "tracker"),
    ]
    diag.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow(["seq001/visible.mp4", 0, 10, 10, 20, 20, "drone", "tiny"])
        writer.writerow(["seq001/visible.mp4", 1, 11, 10, 21, 20, "drone", "tiny"])
    dataset = build_tracklet_dataset([diag], gt, tmp_path / "tracklets")
    weights = train_tracklet_classifier(dataset.csv_path, tmp_path / "tracklet.pt", epochs=8)

    raw_rows = []
    for row in rows:
        out = dict(row)
        out["seq"] = "seq001"
        out["predicted_class"] = "drone"
        out["final_drone_score"] = 0.8 if int(out["track_id"]) == 1 else 0.7
        raw_rows.append(out)
    text = "\n".join(json.dumps(r) for r in raw_rows) + "\n"
    pred = train_dir / "predictions.jsonl"
    pred.write_text(text, encoding="utf-8")
    diag.write_text(text, encoding="utf-8")

    result = run_tracklet_classifier_frame_benchmark(
        [train_dir],
        [gt],
        weights,
        tmp_path / "frame_benchmark",
        dataset_names=["heldout"],
        threshold=0.99,
        thresholds=[0.0, 0.99],
        iou_threshold=0.3,
    )

    assert result.out_path.exists()
    assert pred.read_text(encoding="utf-8") == text
    metrics = result.summary["datasets"][0]
    assert metrics["raw_metrics"]["fp"] >= 2
    assert len(metrics["thresholds"]) == 2
    assert metrics["filtered_metrics"]["fp"] <= metrics["raw_metrics"]["fp"]
    assert (tmp_path / "frame_benchmark" / "heldout" / "threshold_0p99" / "seq001" / "predictions_raw.jsonl").exists()


def test_run_tracklet_classifier_frame_benchmark_writes_baseline_report(tmp_path):
    train_dir = tmp_path / "seq001"
    train_dir.mkdir()
    diag = train_dir / "diagnostics.jsonl"
    rows = [
        _row(0, 1, [10, 10, 20, 20], 0.55, 0.75, 0.70, 0.20, "tracker+yolo_tile"),
        _row(1, 1, [11, 10, 21, 20], 0.56, 0.76, 0.71, 0.19, "tracker"),
        _row(0, 2, [100, 100, 110, 110], 0.10, 0.10, 0.08, 0.90, "tracker+yolo_tile"),
        _row(1, 2, [101, 100, 111, 110], 0.11, 0.10, 0.08, 0.90, "tracker"),
    ]
    diag.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow(["seq001/visible.mp4", 0, 10, 10, 20, 20, "drone", "tiny"])
        writer.writerow(["seq001/visible.mp4", 1, 11, 10, 21, 20, "drone", "tiny"])
    dataset = build_tracklet_dataset([diag], gt, tmp_path / "tracklets")
    weights = train_tracklet_classifier(dataset.csv_path, tmp_path / "tracklet.pt", epochs=8)

    raw_rows = []
    for row in rows:
        out = dict(row)
        out["seq"] = "seq001"
        out["predicted_class"] = "drone"
        out["final_drone_score"] = 0.8 if int(out["track_id"]) == 1 else 0.7
        raw_rows.append(out)
    text = "\n".join(json.dumps(r) for r in raw_rows) + "\n"
    (train_dir / "predictions.jsonl").write_text(text, encoding="utf-8")
    diag.write_text(text, encoding="utf-8")
    baseline_csv = tmp_path / "frame_baseline.csv"
    baseline_csv.write_text("dataset,method,frame_best_f1\nheldout,YOLOMG-frame-smoke,0.10\n", encoding="utf-8")

    result = run_tracklet_classifier_frame_benchmark(
        [train_dir],
        [gt],
        weights,
        tmp_path / "frame_benchmark_baseline",
        dataset_names=["heldout"],
        thresholds=[0.0, 0.5],
        iou_threshold=0.3,
        baseline_csv=baseline_csv,
        baseline_metric="frame_best_f1",
    )

    assert (tmp_path / "frame_benchmark_baseline" / "tracklet_classifier_frame_route_b_results.csv").exists()
    assert (tmp_path / "frame_benchmark_baseline" / "baseline_report" / "route_b_tracklet_classifier_frame_baseline_report.md").exists()
    assert result.summary["baseline_report"]["num_comparisons"] == 1
    assert result.summary["baseline_report"]["route_b_wins"] == 1


def test_validate_tracklet_classifier_frame_benchmark_inputs_accepts_valid_smoke(tmp_path):
    run_dir = tmp_path / "seq001"
    run_dir.mkdir()
    rows = [
        {**_row(0, 1, [10, 10, 20, 20], 0.55, 0.75, 0.70, 0.20, "tracker+yolo_tile"), "seq": "seq001", "predicted_class": "drone"},
        {**_row(1, 1, [11, 10, 21, 20], 0.56, 0.76, 0.71, 0.19, "tracker"), "seq": "seq001", "predicted_class": "drone"},
        {**_row(0, 2, [100, 100, 110, 110], 0.10, 0.10, 0.08, 0.90, "tracker+yolo_tile"), "seq": "seq001", "predicted_class": "drone"},
        {**_row(1, 2, [101, 100, 111, 110], 0.11, 0.10, 0.08, 0.90, "tracker"), "seq": "seq001", "predicted_class": "drone"},
    ]
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    (run_dir / "predictions.jsonl").write_text(text, encoding="utf-8")
    (run_dir / "diagnostics.jsonl").write_text(text, encoding="utf-8")
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow(["seq001/visible.mp4", 0, 10, 10, 20, 20, "drone", "tiny"])
        writer.writerow(["seq001/visible.mp4", 1, 11, 10, 21, 20, "drone", "tiny"])
    dataset = build_tracklet_dataset([run_dir / "diagnostics.jsonl"], gt, tmp_path / "tracklets")
    weights = train_tracklet_classifier(dataset.csv_path, tmp_path / "tracklet.pt", epochs=2)
    baseline_csv = tmp_path / "frame_baseline.csv"
    baseline_csv.write_text("dataset,method,frame_best_f1\nheldout,YOLOMG-frame-smoke,0.10\n", encoding="utf-8")

    result = validate_tracklet_classifier_frame_benchmark_inputs(
        [run_dir],
        [gt],
        weights,
        tmp_path / "preflight.json",
        dataset_names=["heldout"],
        thresholds=[0.0, 0.5],
        baseline_csv=baseline_csv,
        baseline_metric="frame_best_f1",
    )

    assert result.out_path.exists()
    assert result.summary["valid"] is True
    assert result.summary["combined"]["pairs"] == 1
    assert result.summary["combined"]["prediction_rows"] == 4
    assert result.summary["combined"]["gt_boxes"] == 2
    assert result.summary["combined"]["sequence_overlap"] == ["seq001"]
    assert result.summary["baseline_validation"]["valid"] is True


def test_validate_tracklet_classifier_frame_benchmark_inputs_flags_no_gt_overlap(tmp_path):
    run_dir = tmp_path / "seq001"
    run_dir.mkdir()
    rows = [
        {**_row(0, 1, [10, 10, 20, 20], 0.55, 0.75, 0.70, 0.20, "tracker+yolo_tile"), "seq": "seq001", "predicted_class": "drone"},
        {**_row(1, 1, [11, 10, 21, 20], 0.56, 0.76, 0.71, 0.19, "tracker"), "seq": "seq001", "predicted_class": "drone"},
    ]
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    (run_dir / "predictions.jsonl").write_text(text, encoding="utf-8")
    (run_dir / "diagnostics.jsonl").write_text(text, encoding="utf-8")
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow(["other_seq/visible.mp4", 0, 10, 10, 20, 20, "drone", "tiny"])
    train_gt = tmp_path / "train_gt.csv"
    with train_gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow(["seq001/visible.mp4", 0, 10, 10, 20, 20, "drone", "tiny"])
    dataset = build_tracklet_dataset([run_dir / "diagnostics.jsonl"], train_gt, tmp_path / "tracklets")
    weights = train_tracklet_classifier(dataset.csv_path, tmp_path / "tracklet.pt", epochs=2)

    result = validate_tracklet_classifier_frame_benchmark_inputs(
        [run_dir],
        [gt],
        weights,
        tmp_path / "preflight_bad.json",
        dataset_names=["heldout"],
    )

    assert result.summary["valid"] is False
    assert result.summary["combined"]["sequence_overlap"] == []
    assert any("do not overlap GT sequences" in error for error in result.summary["errors"])


def test_build_tracklet_classifier_official_eval_bundle_copies_best_predictions(tmp_path):
    train_dir = tmp_path / "seq001"
    train_dir.mkdir()
    diag = train_dir / "diagnostics.jsonl"
    rows = [
        _row(0, 1, [10, 10, 20, 20], 0.55, 0.75, 0.70, 0.20, "tracker+yolo_tile"),
        _row(1, 1, [11, 10, 21, 20], 0.56, 0.76, 0.71, 0.19, "tracker"),
        _row(0, 2, [100, 100, 110, 110], 0.10, 0.10, 0.08, 0.90, "tracker+yolo_tile"),
        _row(1, 2, [101, 100, 111, 110], 0.11, 0.10, 0.08, 0.90, "tracker"),
    ]
    diag.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow(["seq001/visible.mp4", 0, 10, 10, 20, 20, "drone", "tiny"])
        writer.writerow(["seq001/visible.mp4", 1, 11, 10, 21, 20, "drone", "tiny"])
    dataset = build_tracklet_dataset([diag], gt, tmp_path / "tracklets")
    weights = train_tracklet_classifier(dataset.csv_path, tmp_path / "tracklet.pt", epochs=4)

    pred_rows = []
    for row in rows:
        out = dict(row)
        out["seq"] = "seq001"
        out["predicted_class"] = "drone"
        out["final_drone_score"] = 0.8 if int(out["track_id"]) == 1 else 0.7
        pred_rows.append(out)
    pred_text = "\n".join(json.dumps(row) for row in pred_rows) + "\n"
    (train_dir / "predictions.jsonl").write_text(pred_text, encoding="utf-8")
    diag.write_text(pred_text, encoding="utf-8")
    baseline_csv = tmp_path / "frame_baseline.csv"
    baseline_csv.write_text("dataset,method,frame_best_f1\nheldout,YOLOMG-frame-smoke,0.10\n", encoding="utf-8")

    frame = run_tracklet_classifier_frame_benchmark(
        [train_dir],
        [gt],
        weights,
        tmp_path / "frame_benchmark",
        dataset_names=["heldout"],
        thresholds=[0.0, 0.5],
        baseline_csv=baseline_csv,
        baseline_metric="frame_best_f1",
    )
    preflight = validate_tracklet_classifier_frame_benchmark_inputs(
        [train_dir],
        [gt],
        weights,
        tmp_path / "frame_preflight.json",
        dataset_names=["heldout"],
    )
    bundle = build_tracklet_classifier_official_eval_bundle(
        tmp_path / "frame_benchmark" / "tracklet_classifier_frame_benchmark_summary.json",
        tmp_path / "official_bundle",
        preflight_json=preflight.out_path,
        baseline_comparison_json=frame.summary["baseline_report"]["comparison_json"],
        require_baseline_comparison=True,
    )

    assert bundle.out_path.exists()
    assert bundle.summary["valid"] is True
    assert bundle.summary["combined"]["datasets"] == 1
    assert bundle.summary["combined"]["pairs"] == 1
    assert (tmp_path / "official_bundle" / "official_eval_prediction_index.csv").exists()
    copied_pred = tmp_path / "official_bundle" / "best_filtered" / "heldout" / "seq001" / "predictions.jsonl"
    assert copied_pred.exists()
    assert bundle.summary["datasets"][0]["best_threshold"] in {0.0, 0.5}
    assert bundle.summary["baseline_comparison"]["route_b_wins"] == 1


def test_export_tracklet_classifier_official_predictions_writes_flat_and_yolo(tmp_path):
    bundle_dir = tmp_path / "bundle"
    pred_dir = bundle_dir / "best_filtered" / "heldout" / "seq001"
    pred_dir.mkdir(parents=True)
    rows = [
        {"seq": "seq001", "frame_id": 0, "bbox": [10, 20, 30, 60], "predicted_class": "drone", "final_drone_score": 0.8, "track_id": 1},
        {"seq": "seq001", "frame_id": 0, "bbox": [100, 100, 120, 130], "predicted_class": "background", "final_drone_score": 0.0, "track_id": 2},
    ]
    pred = pred_dir / "predictions.jsonl"
    pred.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    diag = pred_dir / "diagnostics.jsonl"
    diag.write_text(pred.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = bundle_dir / "official_eval_bundle_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "valid": True,
                "datasets": [
                    {
                        "dataset": "heldout",
                        "pairs": [
                            {
                                "seq": "seq001",
                                "official_eval_predictions": str(pred),
                                "official_eval_diagnostics": str(diag),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = export_tracklet_classifier_official_predictions(
        manifest,
        tmp_path / "exported",
        default_image_size=(200, 100),
    )

    assert result.out_path.exists()
    assert result.summary["valid"] is True
    assert result.summary["combined"]["prediction_rows"] == 1
    flat = list(csv.DictReader((tmp_path / "exported" / "flat_xyxy_predictions.csv").open("r", encoding="utf-8")))
    assert len(flat) == 1
    assert flat[0]["image_stem"] == "seq001_000000"
    label = tmp_path / "exported" / "yolo_txt" / "heldout" / "labels" / "seq001_000000.txt"
    assert label.exists()
    parts = label.read_text(encoding="utf-8").strip().split()
    assert parts[0] == "0"
    assert abs(float(parts[1]) - 0.1) < 1e-6
    assert abs(float(parts[2]) - 0.4) < 1e-6
    assert abs(float(parts[3]) - 0.1) < 1e-6
    assert abs(float(parts[4]) - 0.4) < 1e-6
    assert abs(float(parts[5]) - 0.8) < 1e-6


def test_export_tracklet_classifier_aot_prediction_parts_writes_pickle(tmp_path):
    flat = tmp_path / "flat_xyxy_predictions.csv"
    with flat.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "seq", "frame_id", "image_stem", "class_id", "score", "x1", "y1", "x2", "y2", "track_id", "source_predictions"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "aot",
                "seq": "Clip_696",
                "frame_id": 1172,
                "image_stem": "Clip_696_01172",
                "class_id": 0,
                "score": 0.8,
                "x1": 10,
                "y1": 20,
                "x2": 30,
                "y2": 60,
                "track_id": 7,
                "source_predictions": "predictions.jsonl",
            }
        )
        writer.writerow(
            {
                "dataset": "aot",
                "seq": "Clip_696",
                "frame_id": 1172,
                "image_stem": "Clip_696_01172",
                "class_id": 0,
                "score": 0.7,
                "x1": 40,
                "y1": 50,
                "x2": 55,
                "y2": 65,
                "track_id": 8,
                "source_predictions": "predictions.jsonl",
            }
        )

    result = export_tracklet_classifier_aot_prediction_parts(
        flat,
        tmp_path / "aot_export",
        image_name_template="{image_stem}.png",
    )

    part = tmp_path / "aot_export" / "aotpredictions" / "predictions_split_0.pkl"
    assert result.out_path.exists()
    assert part.exists()
    rows = pickle.load(part.open("rb"))
    assert result.summary["valid"] is True
    assert result.summary["combined"]["detections_exported"] == 2
    assert result.summary["combined"]["result_records"] == 1
    assert rows[0]["img_name"] == "Clip_696_01172.png"
    assert len(rows[0]["detections"]) == 2
    assert rows[0]["detections"][0] == {"track_id": 7, "x": 20.0, "y": 40.0, "w": 20.0, "h": 40.0, "n": "airborne", "s": 0.8}


def test_export_tracklet_classifier_aot_prediction_parts_can_force_clip_frame_names(tmp_path):
    flat = tmp_path / "flat_xyxy_predictions.csv"
    with flat.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "seq", "frame_id", "image_stem", "class_id", "score", "x1", "y1", "x2", "y2", "track_id", "source_predictions"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "aot",
                "seq": "Clip_696",
                "frame_id": 1172,
                "image_stem": "seq001_001172",
                "class_id": 0,
                "score": 0.8,
                "x1": 10,
                "y1": 20,
                "x2": 30,
                "y2": 60,
                "track_id": 7,
                "source_predictions": "predictions.jsonl",
            }
        )

    result = export_tracklet_classifier_aot_prediction_parts(
        flat,
        tmp_path / "aot_export",
        image_name_mode="aot_clip_frame",
    )

    part = tmp_path / "aot_export" / "aotpredictions" / "predictions_split_0.pkl"
    rows = pickle.load(part.open("rb"))
    assert result.summary["valid"] is True
    assert result.summary["image_name_mode"] == "aot_clip_frame"
    assert rows[0]["img_name"] == "Clip_696_01172.png"


def test_export_aot_prediction_parts_to_tracklets_restores_xyxy_tracklets(tmp_path):
    pred_dir = tmp_path / "aotpredictions"
    pred_dir.mkdir()
    part = pred_dir / "predictions_split_0.pkl"
    pickle.dump(
        [
            {
                "img_name": "Clip_696_01172.png",
                "detections": [
                    {"track_id": 7, "x": 20.0, "y": 40.0, "w": 20.0, "h": 40.0, "n": "airborne", "s": 0.8},
                ],
            },
            {
                "img_name": "Clip_696_01173.png",
                "detections": [
                    {"track_id": 7, "x": 21.0, "y": 40.0, "w": 20.0, "h": 40.0, "n": "airborne", "s": 0.7},
                    {"track_id": 8, "x": 100.0, "y": 100.0, "w": 8.0, "h": 8.0, "n": "airborne", "s": 0.1},
                ],
            },
        ],
        part.open("wb"),
    )

    result = export_aot_prediction_parts_to_tracklets(
        pred_dir,
        tmp_path / "tracklets.jsonl",
        min_score=0.2,
        image_width=1280,
        image_height=720,
        min_tracklet_rows=2,
    )
    items = [json.loads(line) for line in result.out_path.read_text(encoding="utf-8").splitlines()]

    assert result.summary["valid"] is True
    assert result.summary["combined"]["detections_seen"] == 3
    assert result.summary["combined"]["detections_used"] == 2
    assert result.summary["combined"]["tracklets_written"] == 1
    assert items[0]["meta"]["seq"] == "Clip_696"
    assert items[0]["meta"]["track_id"] == "7"
    assert items[0]["rows"][0]["frame_id"] == 1172
    assert items[0]["rows"][0]["bbox"] == [10.0, 20.0, 30.0, 60.0]
    assert items[0]["rows"][0]["image_width"] == 1280
    assert items[0]["rows"][1]["bbox"] == [11.0, 20.0, 31.0, 60.0]


def test_export_aot_prediction_parts_to_tracklets_can_split_frame_gaps(tmp_path):
    pred_dir = tmp_path / "aotpredictions"
    pred_dir.mkdir()
    part = pred_dir / "predictions_split_0.pkl"
    pickle.dump(
        [
            {"img_name": "Clip_696_00001.png", "detections": [{"track_id": 7, "x": 20.0, "y": 40.0, "w": 20.0, "h": 40.0, "n": "airborne", "s": 0.8}]},
            {"img_name": "Clip_696_00002.png", "detections": [{"track_id": 7, "x": 21.0, "y": 40.0, "w": 20.0, "h": 40.0, "n": "airborne", "s": 0.8}]},
            {"img_name": "Clip_696_00020.png", "detections": [{"track_id": 7, "x": 80.0, "y": 90.0, "w": 20.0, "h": 40.0, "n": "airborne", "s": 0.8}]},
            {"img_name": "Clip_696_00021.png", "detections": [{"track_id": 7, "x": 81.0, "y": 90.0, "w": 20.0, "h": 40.0, "n": "airborne", "s": 0.8}]},
        ],
        part.open("wb"),
    )

    result = export_aot_prediction_parts_to_tracklets(
        pred_dir,
        tmp_path / "tracklets_gap_split.jsonl",
        min_tracklet_rows=2,
        max_frame_gap=2,
    )
    items = [json.loads(line) for line in result.out_path.read_text(encoding="utf-8").splitlines()]

    assert result.summary["combined"]["raw_tracklets"] == 1
    assert result.summary["combined"]["segments_seen"] == 2
    assert result.summary["combined"]["tracklets_written"] == 2
    assert [item["meta"]["track_id"] for item in items] == ["7:seg0", "7:seg1"]
    assert [row["frame_id"] for row in items[0]["rows"]] == [1, 2]
    assert [row["frame_id"] for row in items[1]["rows"]] == [20, 21]
    assert all(row["raw_track_id"] == "7" for item in items for row in item["rows"])


def test_export_aot_prediction_parts_to_tracklets_can_attach_aot_image_paths(tmp_path):
    pred_dir = tmp_path / "aotpredictions"
    pred_dir.mkdir()
    part = pred_dir / "predictions_split_0.pkl"
    pickle.dump(
        [
            {"img_name": "Clip_696_00000.png", "detections": [{"track_id": 6, "x": 10.0, "y": 20.0, "w": 10.0, "h": 20.0, "n": "airborne", "s": 0.7}]},
            {"img_name": "Clip_696_00001.png", "detections": [{"track_id": 7, "x": 20.0, "y": 40.0, "w": 20.0, "h": 40.0, "n": "airborne", "s": 0.8}]},
        ],
        part.open("wb"),
    )
    clip_map = tmp_path / "clip_to_flight.pkl"
    pickle.dump({696: "flight_a"}, clip_map.open("wb"))
    gt = tmp_path / "groundtruth.json"
    gt.write_text(
        json.dumps(
            {
                "samples": {
                    "flight_a": {
                        "entities": [
                            {"blob": {"frame": 1}, "img_name": "123flight_a.png"},
                            {"blob": {"frame": 3}, "img_name": "456flight_a.png"},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = export_aot_prediction_parts_to_tracklets(
        pred_dir,
        tmp_path / "tracklets_with_paths.jsonl",
        clip_id_to_flight_id_path=clip_map,
        aot_groundtruth_json=gt,
    )
    items = [json.loads(line) for line in result.out_path.read_text(encoding="utf-8").splitlines()]

    assert result.summary["combined"]["clip_id_mappings"] == 1
    assert result.summary["combined"]["frame_image_mappings"] == 3
    by_track = {item["meta"]["track_id"]: item for item in items}
    assert by_track["6"]["rows"][0]["flight_id"] == "flight_a"
    assert by_track["6"]["rows"][0]["image_name"] == "123flight_a.png"
    assert by_track["6"]["rows"][0]["image_path"] == str(Path("Images") / "flight_a" / "123flight_a.png")
    assert by_track["7"]["rows"][0]["image_name"] == "456flight_a.png"


def test_filter_aot_prediction_parts_by_tracklets_uses_raw_track_ids(tmp_path):
    pred_dir = tmp_path / "aotpredictions"
    pred_dir.mkdir()
    part = pred_dir / "predictions_split_0.pkl"
    pickle.dump(
        [
            {
                "img_name": "Clip_696_00001.png",
                "detections": [
                    {"track_id": 7, "x": 20.0, "y": 40.0, "w": 20.0, "h": 40.0, "n": "airborne", "s": 0.8},
                    {"track_id": 8, "x": 60.0, "y": 70.0, "w": 10.0, "h": 10.0, "n": "airborne", "s": 0.6},
                ],
            },
            {
                "img_name": "Clip_696_00002.png",
                "detections": [
                    {"track_id": 7, "x": 21.0, "y": 40.0, "w": 20.0, "h": 40.0, "n": "airborne", "s": 0.7},
                ],
            },
            {
                "img_name": "Clip_696_00003.png",
                "detections": [
                    {"track_id": 7, "x": 22.0, "y": 40.0, "w": 20.0, "h": 40.0, "n": "airborne", "s": 0.1},
                ],
            },
        ],
        part.open("wb"),
    )
    tracklet = {
        "meta": {"seq": "Clip_696", "track_id": "7:seg0", "raw_track_id": "7", "action_dynamics_score": 0.9},
        "rows": [
            {"seq": "Clip_696", "track_id": "7:seg0", "raw_track_id": "7", "frame_id": 1},
            {"seq": "Clip_696", "track_id": "7:seg0", "raw_track_id": "7", "frame_id": 2},
        ],
    }
    tracklets = tmp_path / "tracklets.jsonl"
    tracklets.write_text(json.dumps(tracklet) + "\n", encoding="utf-8")

    result = filter_aot_prediction_parts_by_tracklets(
        pred_dir,
        tracklets,
        tmp_path / "filtered",
        score_field="action_dynamics_score",
        min_tracklet_score=0.5,
        min_tracklet_rows=2,
    )
    rows = pickle.load((tmp_path / "filtered" / "aotpredictions" / "predictions_split_0.pkl").open("rb"))

    assert result.summary["valid"] is True
    assert result.summary["combined"]["tracklets_kept"] == 1
    assert result.summary["combined"]["detections_written"] == 2
    assert [row["img_name"] for row in rows] == ["Clip_696_00001.png", "Clip_696_00002.png"]
    assert rows[0]["detections"] == [{"track_id": 7, "x": 20.0, "y": 40.0, "w": 20.0, "h": 40.0, "n": "airborne", "s": 0.8}]


def test_rescore_aot_prediction_parts_by_tracklets_soft_suppresses_without_filtering(tmp_path):
    pred_dir = tmp_path / "aotpredictions"
    pred_dir.mkdir()
    part = pred_dir / "predictions_split_0.pkl"
    pickle.dump(
        [
            {
                "img_name": "Clip_696_00001.png",
                "detections": [
                    {"track_id": 7, "x": 20.0, "y": 40.0, "w": 20.0, "h": 40.0, "n": "airborne", "s": 0.8},
                    {"track_id": 8, "x": 60.0, "y": 70.0, "w": 10.0, "h": 10.0, "n": "airborne", "s": 0.6},
                ],
            }
        ],
        part.open("wb"),
    )
    tracklet = {
        "meta": {"seq": "Clip_696", "track_id": "7:seg0", "raw_track_id": "7", "video_action_model_fusion_score": 0.25},
        "rows": [
            {"seq": "Clip_696", "track_id": "7:seg0", "raw_track_id": "7", "frame_id": 1},
        ],
    }
    tracklets = tmp_path / "tracklets.jsonl"
    tracklets.write_text(json.dumps(tracklet) + "\n", encoding="utf-8")

    result = rescore_aot_prediction_parts_by_tracklets(
        pred_dir,
        tracklets,
        tmp_path / "rescored",
        center=0.5,
        beta=0.4,
        mode="suppress-only",
    )
    rows = pickle.load((tmp_path / "rescored" / "aotpredictions" / "predictions_split_0.pkl").open("rb"))

    assert result.summary["valid"] is True
    assert result.summary["combined"]["detections_seen"] == 2
    assert result.summary["combined"]["detections_written"] == 2
    assert result.summary["combined"]["detections_changed"] == 1
    assert abs(rows[0]["detections"][0]["s"] - 0.7) < 1e-9
    assert rows[0]["detections"][0]["tracklet_rescore_raw_s"] == 0.8
    assert rows[0]["detections"][1]["s"] == 0.6


def test_validate_tracklet_classifier_aot_eval_inputs_accepts_official_clip_names(tmp_path):
    pred_dir = tmp_path / "aotpredictions"
    pred_dir.mkdir()
    part = pred_dir / "predictions_split_0.pkl"
    pickle.dump(
        [
            {
                "img_name": "Clip_696_01172.png",
                "detections": [
                    {"track_id": 7, "x": 10.0, "y": 20.0, "w": 20.0, "h": 40.0, "n": "airborne", "s": 0.8},
                ],
            }
        ],
        part.open("wb"),
    )
    clip_map = tmp_path / "aot_clip_id_to_flight_id.pkl"
    pickle.dump({696: "flight_001"}, clip_map.open("wb"))

    result = validate_tracklet_classifier_aot_eval_inputs(
        pred_dir,
        tmp_path / "preflight_valid.json",
        clip_id_to_flight_id_path=clip_map,
    )

    assert result.out_path.exists()
    assert result.summary["valid"] is True
    assert result.summary["combined"]["parts"] == 1
    assert result.summary["combined"]["records_checked"] == 1
    assert result.summary["combined"]["detections_checked"] == 1
    assert result.summary["combined"]["clip_ids"] == [696]


def test_validate_tracklet_classifier_aot_eval_inputs_rejects_non_clip_names(tmp_path):
    pred_dir = tmp_path / "aotpredictions"
    pred_dir.mkdir()
    part = pred_dir / "predictions_split_0.pkl"
    pickle.dump(
        [
            {
                "img_name": "seq001_000000.png",
                "detections": [
                    {"track_id": 1, "x": 10.0, "y": 10.0, "w": 10.0, "h": 10.0, "n": "airborne", "s": 0.8},
                ],
            }
        ],
        part.open("wb"),
    )

    result = validate_tracklet_classifier_aot_eval_inputs(pred_dir, tmp_path / "preflight_invalid.json")

    assert result.summary["valid"] is False
    assert result.summary["combined"]["pattern_errors"] == 1
    assert any("Clip_<id>_<frame>" in error for error in result.summary["errors"])


def test_selective_tracklet_promotion_enforces_sequence_budget(tmp_path):
    train_dir = tmp_path / "seq_train"
    train_dir.mkdir()
    diag = train_dir / "diagnostics.jsonl"
    train_rows = [
        _row(0, 1, [10, 10, 20, 20], 0.55, 0.75, 0.70, 0.20, "tracker+fallback_yolo"),
        _row(1, 1, [11, 10, 21, 20], 0.56, 0.76, 0.71, 0.19, "tracker"),
        _row(0, 2, [100, 100, 110, 110], 0.10, 0.10, 0.08, 0.90, "tracker+yolo_tile"),
        _row(1, 2, [101, 100, 111, 110], 0.11, 0.10, 0.08, 0.90, "tracker"),
    ]
    diag.write_text("\n".join(json.dumps(r) for r in train_rows) + "\n", encoding="utf-8")
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow([str(tmp_path / "seq_train" / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])
        writer.writerow([str(tmp_path / "seq_train" / "visible.mp4"), 1, 11, 10, 21, 20, "drone", "tiny"])
    result = build_tracklet_dataset([diag], gt, tmp_path / "tracklets")
    weights = train_tracklet_classifier(result.csv_path, tmp_path / "tracklet.pt", epochs=8)

    rows = []
    for track_id, offset in [(1, 0), (2, 40), (3, 80)]:
        rows.extend(
            [
                {
                    **_row(0, track_id, [10 + offset, 10, 20 + offset, 20], 0.45, 0.72 - 0.05 * offset / 40, 0.52, 0.40, "tracker+fallback_yolo"),
                    "seq": "seq_a",
                    "predicted_class": "background",
                    "objectness": 0.18,
                },
                {
                    **_row(1, track_id, [11 + offset, 10, 21 + offset, 20], 0.46, 0.73 - 0.05 * offset / 40, 0.52, 0.40, "tracker"),
                    "seq": "seq_a",
                    "predicted_class": "background",
                    "objectness": 0.18,
                },
            ]
        )

    filtered, _, summary = filter_infer_rows_with_tracklet_classifier(
        rows,
        rows,
        weights,
        threshold=0.0,
        promote_positive_tracklets=True,
        promotion_score_floor=0.30,
        promotion_min_branch_drone=0.40,
        promotion_max_background=0.60,
        selective_promotion=True,
        selective_min_temporal_crop_delta=0.05,
        selective_min_temporal_background_margin=-0.05,
        selective_max_promoted_tracklets_per_sequence=1,
    )

    promoted_track_ids = {r["track_id"] for r in filtered if r["predicted_class"] == "drone"}
    assert len(promoted_track_ids) == 1
    assert summary["selective_allowed_tracklets"] == 1
