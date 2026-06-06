import csv
import json

import torch

from qstr_dronedet.tracking.action_chunk import export_action_chunk_dataset_from_tracklets
from qstr_dronedet.tracking.action_policy import (
    attach_action_dynamics_scores_to_tracklets,
    attach_tracklet_confidence_fusion_scores,
    build_route_b_baseline_report,
    collect_route_b_result_summaries,
    compare_route_b_results_to_baselines,
    evaluate_action_dynamics_thresholds,
    evaluate_action_chunk_policy,
    export_route_b_baseline_markdown_table,
    run_action_policy_ablation,
    run_action_policy_split_selection,
    run_multisource_proposal_policy_benchmark,
    run_multisource_action_policy_experiment,
    run_multisource_tracklet_action_policy_experiment,
    run_multisource_tracklet_policy_benchmark,
    run_action_dynamics_tracklet_ablation,
    run_action_dynamics_tracklet_pipeline,
    score_tracklets_with_action_policy,
    score_tracklets_with_constant_velocity,
    train_action_chunk_policy,
    validate_route_b_tracklet_inputs,
    validate_route_b_baseline_csv,
    write_route_b_baseline_template,
    write_route_b_official_baseline_seed,
)
from qstr_dronedet.tracking.tracklet_classifier import _features


def _sample(seq, track_id, start_x):
    past = [[start_x + i, 10.0, start_x + i + 4.0, 14.0] for i in range(3)]
    future = [[start_x + i, 10.0, start_x + i + 4.0, 14.0] for i in range(3, 5)]
    return {
        "seq": seq,
        "track_id": track_id,
        "anchor_frame": 2,
        "label": 1,
        "past_boxes": past,
        "past_scores": [0.8, 0.8, 0.8],
        "past_visible": [1.0, 1.0, 1.0],
        "future_actions": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        "future_boxes": future,
    }


def _tracklet_item(source, seq_index, track_index=0, label=1, rows=6, start_x=10.0):
    tracklet_rows = [
        {
            "seq": f"{source}_seq_{seq_index}",
            "track_id": f"{source}_tube_{seq_index}_{track_index}",
            "frame_id": frame_id,
            "bbox": [start_x + frame_id + seq_index, 12.0, start_x + frame_id + seq_index + 4.0, 16.0],
            "objectness": 0.8 if label else 0.2,
            "visible": True,
        }
        for frame_id in range(rows)
    ]
    return {
        "meta": {
            "seq": f"{source}_seq_{seq_index}",
            "track_id": f"{source}_tube_{seq_index}_{track_index}",
            "label": label,
            "bucket": "hard_tiny_positive" if label else "hard_negative",
            "dataset_source": source,
        },
        "rows": tracklet_rows,
    }


def _diagnostic_row(frame_id, box, score=0.8, source="fallback_yolo"):
    return {
        "frame_id": frame_id,
        "bbox": box,
        "objectness": score,
        "source": source,
        "final_drone_score": score,
        "predicted_class": "drone",
        "crop_probs": {"drone": score, "background": 1.0 - score},
        "temporal_probs": {"drone": score, "background": 1.0 - score},
        "final_probs": {"drone": score, "background": 1.0 - score},
        "alignment_quality": 0.8,
    }


def _write_proposal_run(root, profile, seq, rows):
    seq_dir = root / profile / seq
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / "diagnostics_raw.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_gt_csv(path, seq, positive_rows=6):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        for frame_id in range(positive_rows):
            x = 10.0 + frame_id
            writer.writerow([f"/tmp/{seq}/visible.mp4", frame_id, x, 12.0, x + 4.0, 16.0, "drone", "tiny"])


def _write_gt_csv_for_seqs(path, seqs, positive_rows=6):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        for seq in seqs:
            for frame_id in range(positive_rows):
                x = 10.0 + frame_id
                writer.writerow([f"/tmp/{seq}/visible.mp4", frame_id, x, 12.0, x + 4.0, 16.0, "drone", "tiny"])


def test_score_tracklets_with_constant_velocity_separates_smooth_and_jumpy(tmp_path):
    smooth = _tracklet_item("aot", 1, track_index=1, rows=5, start_x=10.0)
    jumpy = _tracklet_item("aot", 2, track_index=2, rows=5, start_x=10.0)
    jumpy["rows"][3]["bbox"] = [80.0, 12.0, 84.0, 16.0]
    jumpy["rows"][4]["bbox"] = [20.0, 12.0, 24.0, 16.0]
    tracklets = tmp_path / "tracklets.jsonl"
    tracklets.write_text(json.dumps(smooth) + "\n" + json.dumps(jumpy) + "\n", encoding="utf-8")

    result = score_tracklets_with_constant_velocity(tracklets, tmp_path / "cv_scores.jsonl", min_tracklet_rows=3)
    rows = [json.loads(line) for line in result.out_path.read_text(encoding="utf-8").splitlines()]
    by_track = {row["track_id"]: row for row in rows}

    assert result.summary["scored_tracklets"] == 2
    assert by_track["aot_tube_1_1"]["dynamics_score"] > 0.99
    assert by_track["aot_tube_2_2"]["dynamics_score"] < 0.01
    assert by_track["aot_tube_2_2"]["mean_cv_normalized_center_error"] > by_track["aot_tube_1_1"]["mean_cv_normalized_center_error"]


def test_train_and_eval_action_chunk_policy(tmp_path):
    dataset = tmp_path / "action_chunks.jsonl"
    samples = [_sample("s", str(i), float(i * 10)) for i in range(6)]
    dataset.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n", encoding="utf-8")

    weights = train_action_chunk_policy(dataset, tmp_path / "policy.pt", epochs=5, hidden=16, batch_size=4)
    result = evaluate_action_chunk_policy(dataset, weights, tmp_path / "scores.jsonl")
    rows = [json.loads(line) for line in result.out_path.read_text(encoding="utf-8").splitlines()]

    assert weights.exists()
    assert result.out_path.exists()
    assert result.summary["samples"] == 6
    assert result.summary["mean_learned_center_error"] >= 0.0
    assert result.summary["mean_constant_velocity_center_error"] == 0.0
    assert len(rows[0]["learned_boxes"]) == 2


def test_train_action_chunk_policy_records_balance_groups(tmp_path):
    dataset = tmp_path / "action_chunks.jsonl"
    samples = []
    for index in range(5):
        sample = _sample("s", str(index), float(index * 10))
        sample["dataset_source"] = "nps" if index < 4 else "aot"
        sample["bucket"] = "positive" if index < 4 else "hard_negative"
        sample["label"] = 1 if index < 4 else 0
        samples.append(sample)
    dataset.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n", encoding="utf-8")

    weights = train_action_chunk_policy(
        dataset,
        tmp_path / "balanced_policy.pt",
        epochs=2,
        hidden=16,
        batch_size=4,
        balance_by=["dataset_source", "label"],
    )
    ckpt = torch.load(weights, map_location="cpu")

    assert ckpt["balance"]["enabled"] is True
    assert ckpt["balance"]["balance_by"] == ["dataset_source", "label"]
    assert ckpt["balance"]["group_counts"] == {"nps|1": 4, "aot|0": 1}
    assert ckpt["balance"]["max_weight"] > ckpt["balance"]["min_weight"]


def test_train_and_eval_diffusion_action_chunk_policy(tmp_path):
    dataset = tmp_path / "action_chunks.jsonl"
    samples = [_sample("s", str(i), float(i * 10)) for i in range(6)]
    dataset.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n", encoding="utf-8")

    weights = train_action_chunk_policy(
        dataset,
        tmp_path / "diffusion_policy.pt",
        epochs=2,
        hidden=16,
        batch_size=4,
        model_type="diffusion",
        diffusion_steps=4,
    )
    ckpt = torch.load(weights, map_location="cpu")
    result = evaluate_action_chunk_policy(dataset, weights, tmp_path / "diffusion_scores.jsonl")

    assert ckpt["model_type"] == "diffusion"
    assert ckpt["diffusion_steps"] == 4
    assert result.summary["model_type"] == "diffusion"
    assert result.summary["diffusion_steps"] == 4
    assert result.summary["samples"] == 6
    assert result.summary["mean_learned_center_error"] >= 0.0


def test_train_and_eval_residual_action_chunk_policy(tmp_path):
    dataset = tmp_path / "action_chunks.jsonl"
    samples = [_sample("s", str(i), float(i * 10)) for i in range(6)]
    for sample in samples:
        sample["future_actions"] = [[1.25, 0.0, 0.0, 0.0], [1.25, 0.0, 0.0, 0.0]]
        sample["future_boxes"] = [[sample["past_boxes"][-1][0] + 1.25, 10.0, sample["past_boxes"][-1][2] + 1.25, 14.0], [sample["past_boxes"][-1][0] + 2.5, 10.0, sample["past_boxes"][-1][2] + 2.5, 14.0]]
    dataset.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n", encoding="utf-8")

    weights = train_action_chunk_policy(
        dataset,
        tmp_path / "residual_policy.pt",
        epochs=2,
        hidden=16,
        batch_size=4,
        model_type="residual_mlp",
    )
    ckpt = torch.load(weights, map_location="cpu")
    result = evaluate_action_chunk_policy(dataset, weights, tmp_path / "residual_scores.jsonl")

    assert ckpt["model_type"] == "residual_mlp"
    assert result.summary["model_type"] == "residual_mlp"
    assert result.summary["samples"] == 6
    assert result.summary["mean_learned_center_error"] >= 0.0


def test_run_action_policy_ablation(tmp_path):
    dataset = tmp_path / "action_chunks.jsonl"
    samples = []
    for index in range(6):
        sample = _sample("s", str(index), float(index * 10))
        sample["dataset_source"] = "nps" if index < 3 else "aot"
        sample["bucket"] = "positive"
        sample["label"] = 1
        samples.append(sample)
    dataset.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n", encoding="utf-8")

    result = run_action_policy_ablation(
        dataset,
        tmp_path / "ablation",
        model_types=["mlp", "diffusion"],
        epochs=2,
        hidden=16,
        batch_size=4,
        diffusion_steps=4,
        balance_by=["dataset_source"],
    )
    summary_json = tmp_path / "ablation" / "action_policy_ablation_summary.json"

    assert result.out_path.exists()
    assert summary_json.exists()
    assert result.summary["model_types"] == ["mlp", "diffusion"]
    assert len(result.summary["rows"]) == 2
    assert result.summary["best"]["model_type"] in {"mlp", "diffusion"}
    assert (tmp_path / "ablation" / "action_policy_mlp.pt").exists()
    assert (tmp_path / "ablation" / "action_policy_diffusion.pt").exists()
    assert result.summary["models"]["diffusion"]["model_type"] == "diffusion"


def test_run_action_policy_split_selection(tmp_path):
    train = tmp_path / "train_action_chunks.jsonl"
    calib = tmp_path / "calib_action_chunks.jsonl"
    test = tmp_path / "test_action_chunks.jsonl"
    train_samples = []
    calib_samples = []
    test_samples = []
    for index in range(8):
        sample = _sample("train_s", str(index), float(index * 10))
        sample["dataset_source"] = "nps" if index < 4 else "aot"
        sample["label"] = 1
        train_samples.append(sample)
    for index in range(4):
        sample = _sample("calib_s", str(index), float(index * 10))
        sample["dataset_source"] = "nps" if index < 2 else "aot"
        sample["label"] = 1
        calib_samples.append(sample)
    for index in range(4):
        sample = _sample("test_s", str(index), float(index * 10))
        sample["dataset_source"] = "nps" if index < 2 else "aot"
        sample["label"] = 1
        test_samples.append(sample)
    train.write_text("\n".join(json.dumps(sample) for sample in train_samples) + "\n", encoding="utf-8")
    calib.write_text("\n".join(json.dumps(sample) for sample in calib_samples) + "\n", encoding="utf-8")
    test.write_text("\n".join(json.dumps(sample) for sample in test_samples) + "\n", encoding="utf-8")

    result = run_action_policy_split_selection(
        train,
        calib,
        tmp_path / "selection",
        test_jsonl=test,
        model_types=["mlp", "diffusion"],
        epochs=2,
        hidden=16,
        batch_size=4,
        diffusion_steps=4,
        balance_by=["dataset_source"],
    )

    assert result.out_path.exists()
    assert (tmp_path / "selection" / "action_policy_split_selection_summary.json").exists()
    assert len(result.summary["rows"]) == 2
    assert result.summary["selection_metric"] == "calib_mean_learned_center_error"
    assert result.summary["best"]["model_type"] in {"mlp", "diffusion"}
    assert result.summary["best"]["calib_samples"] == 4
    assert result.summary["best"]["test_samples"] == 4
    assert (tmp_path / "selection" / "action_policy_mlp.pt").exists()
    assert (tmp_path / "selection" / "action_policy_diffusion.pt").exists()
    assert (tmp_path / "selection" / "calib_scores_mlp.jsonl").exists()
    assert (tmp_path / "selection" / "test_scores_diffusion.jsonl").exists()


def test_run_multisource_action_policy_experiment(tmp_path):
    nps = tmp_path / "nps_action_chunks.jsonl"
    aot = tmp_path / "aot_action_chunks.jsonl"
    nps_samples = []
    aot_samples = []
    for seq_index in range(4):
        for sample_index in range(2):
            sample = _sample(f"nps_seq_{seq_index}", f"nps_t{sample_index}", float(seq_index * 10 + sample_index))
            sample["dataset_source"] = "old_source"
            sample["label"] = 1
            nps_samples.append(sample)
            sample = _sample(f"aot_seq_{seq_index}", f"aot_t{sample_index}", float(seq_index * 12 + sample_index))
            sample["dataset_source"] = "old_source"
            sample["label"] = 1
            aot_samples.append(sample)
    nps.write_text("\n".join(json.dumps(sample) for sample in nps_samples) + "\n", encoding="utf-8")
    aot.write_text("\n".join(json.dumps(sample) for sample in aot_samples) + "\n", encoding="utf-8")

    result = run_multisource_action_policy_experiment(
        [nps, aot],
        tmp_path / "multisource",
        source_names=["nps", "aot"],
        calib_fraction=0.25,
        test_fraction=0.25,
        model_types=["mlp"],
        epochs=2,
        hidden=16,
        batch_size=4,
        balance_by=["dataset_source"],
    )

    assert result.out_path.exists()
    assert (tmp_path / "multisource" / "merged_action_chunks.jsonl").exists()
    assert (tmp_path / "multisource" / "split" / "action_chunk_split_manifest.json").exists()
    assert (tmp_path / "multisource" / "selection" / "action_policy_split_selection_summary.json").exists()
    assert (tmp_path / "multisource" / "multisource_action_policy_experiment_summary.json").exists()
    assert result.summary["merge"]["dataset_source_counts"] == {"nps": 8, "aot": 8}
    assert result.summary["split"]["train_samples"] > 0
    assert result.summary["split"]["calib_samples"] > 0
    assert result.summary["split"]["test_samples"] > 0
    assert result.summary["selection"]["best"]["model_type"] == "mlp"
    assert result.summary["best"]["test_samples"] == result.summary["split"]["test_samples"]
    assert "dataset_source/seq" in result.summary["leakage_guard"]


def test_run_multisource_tracklet_action_policy_experiment(tmp_path):
    tracklet_inputs = []
    for source in ["nps", "aot"]:
        items = []
        for seq_index in range(4):
            rows = [
                {
                    "seq": f"{source}_seq_{seq_index}",
                    "track_id": f"{source}_tube_{seq_index}",
                    "frame_id": frame_id,
                    "bbox": [10.0 + frame_id + seq_index, 12.0, 14.0 + frame_id + seq_index, 16.0],
                    "objectness": 0.8,
                    "visible": True,
                }
                for frame_id in range(6)
            ]
            items.append(
                {
                    "meta": {
                        "seq": f"{source}_seq_{seq_index}",
                        "track_id": f"{source}_tube_{seq_index}",
                        "label": 1,
                        "bucket": "hard_tiny_positive",
                        "dataset_source": "old_source",
                    },
                    "rows": rows,
                }
            )
        path = tmp_path / f"{source}_tracklets.jsonl"
        path.write_text("\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8")
        tracklet_inputs.append(path)

    result = run_multisource_tracklet_action_policy_experiment(
        tracklet_inputs,
        tmp_path / "tracklet_multisource",
        source_names=["nps", "aot"],
        past_len=3,
        future_len=2,
        calib_fraction=0.25,
        test_fraction=0.25,
        model_types=["mlp"],
        epochs=2,
        hidden=16,
        batch_size=4,
        balance_by=["dataset_source"],
    )

    assert result.out_path.exists()
    assert (tmp_path / "tracklet_multisource" / "action_chunks" / "nps_action_chunks.jsonl").exists()
    assert (tmp_path / "tracklet_multisource" / "action_chunks" / "aot_action_chunks.jsonl").exists()
    assert (tmp_path / "tracklet_multisource" / "policy" / "merged_action_chunks.jsonl").exists()
    assert (tmp_path / "tracklet_multisource" / "policy" / "selection" / "action_policy_split_selection_summary.json").exists()
    assert (tmp_path / "tracklet_multisource" / "multisource_tracklet_action_policy_experiment_summary.json").exists()
    assert len(result.summary["exports"]) == 2
    assert result.summary["exports"][0]["samples"] == 8
    assert result.summary["policy"]["merge"]["dataset_source_counts"] == {"nps": 8, "aot": 8}
    assert result.summary["policy"]["split"]["train_samples"] > 0
    assert result.summary["policy"]["split"]["calib_samples"] > 0
    assert result.summary["best"]["model_type"] == "mlp"
    assert "dataset_source/seq" in result.summary["leakage_guard"]


def test_run_multisource_tracklet_policy_benchmark(tmp_path):
    train_inputs = []
    for source in ["nps", "aot"]:
        path = tmp_path / f"{source}_train_tracklets.jsonl"
        items = [_tracklet_item(source, seq_index, label=1) for seq_index in range(4)]
        path.write_text("\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8")
        train_inputs.append(path)

    eval_inputs = []
    for dataset in ["nps_eval", "aot_eval"]:
        path = tmp_path / f"{dataset}_tracklets.jsonl"
        items = [
            _tracklet_item(dataset, 0, track_index=0, label=1),
            _tracklet_item(dataset, 1, track_index=1, label=0, start_x=30.0),
        ]
        path.write_text("\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8")
        eval_inputs.append(path)
    baseline_csv = tmp_path / "baselines.csv"
    baseline_csv.write_text(
        "dataset,method,best_precision,best_recall,best_f1\n"
        "nps,YOLOMG,0.3,0.3,0.3\n"
        "aot,TransVisDrone,0.3,0.3,0.3\n",
        encoding="utf-8",
    )

    result = run_multisource_tracklet_policy_benchmark(
        train_inputs,
        eval_inputs,
        tmp_path / "heldout_benchmark",
        train_source_names=["nps", "aot"],
        eval_dataset_names=["nps", "aot"],
        past_len=3,
        future_len=2,
        calib_fraction=0.25,
        test_fraction=0.25,
        model_types=["mlp"],
        epochs=2,
        hidden=16,
        batch_size=4,
        balance_by=["dataset_source"],
        thresholds=[0.5],
        baseline_csv=baseline_csv,
    )

    assert result.out_path.exists()
    assert (tmp_path / "heldout_benchmark" / "train" / "multisource_tracklet_action_policy_experiment_summary.json").exists()
    assert (tmp_path / "heldout_benchmark" / "eval" / "nps" / "action_dynamics_pipeline_summary.json").exists()
    assert (tmp_path / "heldout_benchmark" / "eval" / "aot" / "action_dynamics_pipeline_summary.json").exists()
    assert (tmp_path / "heldout_benchmark" / "collected" / "route_b_results_summary.json").exists()
    assert (tmp_path / "heldout_benchmark" / "multisource_tracklet_policy_benchmark_summary.json").exists()
    assert result.summary["selected_policy"]["model_type"] == "mlp"
    assert result.summary["eval_dataset_names"] == ["nps", "aot"]
    assert len(result.summary["eval_summaries"]) == 2
    assert result.summary["eval_summaries"][0]["tracklet_eval"]["scored_tracklets"] == 2
    assert result.summary["collected"]["num_rows"] == 2
    assert result.summary["collected"]["datasets"] == ["aot", "nps"]
    assert result.summary["baseline_report"]["num_comparisons"] == 2
    assert result.summary["baseline_report"]["route_b_wins"] == 2
    assert result.summary["baseline_report"]["markdown"].endswith("route_b_baseline_report.md")
    assert (tmp_path / "heldout_benchmark" / "baseline_report" / "route_b_baseline_report.md").exists()


def test_run_multisource_proposal_policy_benchmark(tmp_path):
    profile = "hard_recovery"
    train_roots = []
    train_gts = []
    for source in ["nps", "aot"]:
        run_root = tmp_path / f"{source}_runs"
        seqs = [f"{source}_seq_{seq_index}" for seq_index in range(4)]
        for seq in seqs:
            rows = []
            for frame_id in range(6):
                x = 10.0 + frame_id
                rows.append(_diagnostic_row(frame_id, [x, 12.0, x + 4.0, 16.0], score=0.8))
                rows.append(_diagnostic_row(frame_id, [40.0 + frame_id, 40.0, 44.0 + frame_id, 44.0], score=0.3))
            _write_proposal_run(run_root, profile, seq, rows)
        gt = tmp_path / f"{source}_gt.csv"
        _write_gt_csv_for_seqs(gt, seqs)
        train_roots.append(run_root)
        train_gts.append(gt)

    eval_root = tmp_path / "eval_runs"
    eval_seqs = [f"nps_eval_seq_{seq_index}" for seq_index in range(4)]
    for seq in eval_seqs:
        eval_rows = []
        for frame_id in range(6):
            x = 10.0 + frame_id
            eval_rows.append(_diagnostic_row(frame_id, [x, 12.0, x + 4.0, 16.0], score=0.8))
            eval_rows.append(_diagnostic_row(frame_id, [60.0 + frame_id, 60.0, 64.0 + frame_id, 64.0], score=0.3))
        _write_proposal_run(eval_root, profile, seq, eval_rows)
    eval_gt = tmp_path / "eval_gt.csv"
    _write_gt_csv_for_seqs(eval_gt, eval_seqs)
    baseline_csv = tmp_path / "baselines.csv"
    baseline_csv.write_text("dataset,method,best_f1\nnps,YOLOMG,0.3\n", encoding="utf-8")

    result = run_multisource_proposal_policy_benchmark(
        train_roots,
        train_gts,
        [eval_root],
        [eval_gt],
        tmp_path / "proposal_policy_benchmark",
        train_source_names=["nps", "aot"],
        eval_dataset_names=["nps"],
        profile=profile,
        past_len=3,
        future_len=2,
        calib_fraction=0.25,
        test_fraction=0.25,
        model_types=["mlp"],
        epochs=2,
        hidden=16,
        batch_size=4,
        thresholds=[0.5],
        balance_by=["dataset_source"],
        baseline_csv=baseline_csv,
    )

    assert result.out_path.exists()
    assert (tmp_path / "proposal_policy_benchmark" / "proposal_tracklets" / "train" / "nps" / "proposal_tracklets.jsonl").exists()
    assert (tmp_path / "proposal_policy_benchmark" / "proposal_tracklets" / "eval" / "nps" / "proposal_tracklets.jsonl").exists()
    assert (tmp_path / "proposal_policy_benchmark" / "benchmark" / "multisource_tracklet_policy_benchmark_summary.json").exists()
    assert (tmp_path / "proposal_policy_benchmark" / "multisource_proposal_policy_benchmark_summary.json").exists()
    assert result.summary["proposal_tracklets"]["train"][0]["num_tracklets"] >= 2
    assert result.summary["benchmark"]["baseline_report"]["num_comparisons"] == 1


def test_validate_route_b_tracklet_inputs(tmp_path):
    train = tmp_path / "nps_train.jsonl"
    eval_path = tmp_path / "nps_eval.jsonl"
    train.write_text(
        "\n".join(json.dumps(_tracklet_item("nps", seq_index, label=1)) for seq_index in range(2)) + "\n",
        encoding="utf-8",
    )
    eval_path.write_text(
        "\n".join(
            [
                json.dumps(_tracklet_item("nps_eval", 0, label=1)),
                json.dumps(_tracklet_item("nps_eval", 1, label=0)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_route_b_tracklet_inputs(
        [train],
        [eval_path],
        tmp_path / "preflight.json",
        train_source_names=["nps"],
        eval_dataset_names=["nps"],
        past_len=3,
        future_len=2,
    )

    assert result.out_path.exists()
    assert result.summary["valid"] is True
    assert result.summary["train_action_chunk_samples"] == 4
    assert result.summary["eval_action_chunk_samples"] == 4
    assert result.summary["train"][0]["usable_tracklets"] == 2
    assert result.summary["eval"][0]["positive_tracklets"] == 1
    assert result.summary["eval"][0]["negative_tracklets"] == 1


def test_validate_route_b_tracklet_inputs_flags_bad_inputs(tmp_path):
    train = tmp_path / "short_train.jsonl"
    eval_path = tmp_path / "unlabeled_eval.jsonl"
    short = _tracklet_item("short", 0, label=1, rows=3)
    unlabeled = _tracklet_item("unlabeled", 0, label=1)
    del unlabeled["meta"]["label"]
    train.write_text(json.dumps(short) + "\n", encoding="utf-8")
    eval_path.write_text(json.dumps(unlabeled) + "\n", encoding="utf-8")

    result = validate_route_b_tracklet_inputs(
        [train],
        [eval_path],
        tmp_path / "bad_preflight.json",
        past_len=3,
        future_len=2,
    )

    assert result.summary["valid"] is False
    assert any("no action chunks possible" in issue for issue in result.summary["issues"])
    assert any("unlabeled tracklets" in issue for issue in result.summary["issues"])


def test_score_tracklets_with_action_policy(tmp_path):
    dataset = tmp_path / "action_chunks.jsonl"
    samples = [_sample("s", str(i), float(i * 10)) for i in range(6)]
    dataset.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n", encoding="utf-8")
    weights = train_action_chunk_policy(dataset, tmp_path / "policy.pt", epochs=5, hidden=16, batch_size=4)

    tracklet_rows = [
        {"seq": "s", "track_id": "tube1", "frame_id": i, "bbox": [10.0 + i, 10.0, 14.0 + i, 14.0], "objectness": 0.8}
        for i in range(6)
    ]
    tracklet_jsonl = tmp_path / "tracklets.jsonl"
    tracklet_jsonl.write_text(
        json.dumps(
            {
                "meta": {"seq": "s", "track_id": "tube1", "label": 1, "bucket": "positive", "dataset_source": "synthetic"},
                "rows": tracklet_rows,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = score_tracklets_with_action_policy(tracklet_jsonl, weights, tmp_path / "tracklet_scores.jsonl", past_len=3, future_len=2)
    rows = [json.loads(line) for line in result.out_path.read_text(encoding="utf-8").splitlines()]

    assert result.summary["scored_tracklets"] == 1
    assert result.summary["action_windows"] == 2
    assert rows[0]["track_id"] == "tube1"
    assert rows[0]["num_action_windows"] == 2
    assert 0.0 <= rows[0]["dynamics_score"] <= 1.0


def test_score_tracklets_with_residual_policy_and_row_normalization(tmp_path):
    rows = [
        {
            "seq": "s",
            "track_id": "tube1",
            "frame_id": i,
            "bbox": [10.0 + i, 10.0, 14.0 + i, 14.0],
            "objectness": 0.8,
            "image_width": 100,
            "image_height": 100,
        }
        for i in range(6)
    ]
    tracklet_jsonl = tmp_path / "tracklets.jsonl"
    tracklet_jsonl.write_text(
        json.dumps({"meta": {"seq": "s", "track_id": "tube1", "label": 1, "dataset_source": "synthetic"}, "rows": rows}) + "\n",
        encoding="utf-8",
    )
    action_jsonl = tmp_path / "action_chunks.jsonl"
    export_action_chunk_dataset_from_tracklets(
        tracklet_jsonl,
        action_jsonl,
        past_len=3,
        future_len=2,
        normalize_by_row_image_size=True,
    )
    weights = train_action_chunk_policy(
        action_jsonl,
        tmp_path / "residual_policy.pt",
        epochs=2,
        hidden=16,
        batch_size=4,
        model_type="residual_mlp",
    )

    result = score_tracklets_with_action_policy(
        tracklet_jsonl,
        weights,
        tmp_path / "tracklet_scores.jsonl",
        past_len=3,
        future_len=2,
        normalize_by_row_image_size=True,
        dynamics_score_mode="improvement",
    )
    score_row = json.loads(result.out_path.read_text(encoding="utf-8").splitlines()[0])

    assert result.summary["model_type"] == "residual_mlp"
    assert result.summary["normalize_by_row_image_size"] is True
    assert result.summary["dynamics_score_mode"] == "improvement"
    assert result.summary["scored_tracklets"] == 1
    assert score_row["num_action_windows"] == 2
    assert score_row["dynamics_score_mode"] == "improvement"
    assert 0.0 <= score_row["improvement_score"] <= 1.0


def test_attach_action_dynamics_scores_to_tracklets(tmp_path):
    tracklet_jsonl = tmp_path / "tracklets.jsonl"
    score_jsonl = tmp_path / "scores.jsonl"
    rows = [
        {"seq": "s", "track_id": "tube1", "frame_id": i, "bbox": [10.0 + i, 10.0, 14.0 + i, 14.0], "objectness": 0.8}
        for i in range(4)
    ]
    tracklet_jsonl.write_text(
        json.dumps({"meta": {"seq": "s", "track_id": "tube1", "label": 1}, "rows": rows}) + "\n",
        encoding="utf-8",
    )
    score_jsonl.write_text(
        json.dumps(
            {
                "seq": "s",
                "track_id": "tube1",
                "dynamics_score": 0.75,
                "mean_learned_center_error": 2.0,
                "mean_constant_velocity_center_error": 4.0,
                "mean_error_improvement_vs_cv": 2.0,
                "num_action_windows": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = attach_action_dynamics_scores_to_tracklets(tracklet_jsonl, score_jsonl, tmp_path / "attached.jsonl")
    item = json.loads(result.out_path.read_text(encoding="utf-8").splitlines()[0])
    feats = _features(item["rows"])

    assert result.summary["attached_tracklets"] == 1
    assert item["meta"]["action_dynamics_score"] == 0.75
    assert item["rows"][0]["action_dynamics_score"] == 0.75
    assert feats["mean_action_dynamics_score"] == 0.75
    assert feats["mean_action_error_improvement_vs_cv"] == 2.0


def test_attach_tracklet_confidence_fusion_scores(tmp_path):
    tracklet_jsonl = tmp_path / "tracklets.jsonl"
    rows = [
        {"seq": "s", "track_id": "tube1", "frame_id": 0, "bbox": [10.0, 10.0, 14.0, 14.0], "objectness": 0.4},
        {"seq": "s", "track_id": "tube1", "frame_id": 1, "bbox": [11.0, 10.0, 15.0, 14.0], "final_drone_score": 0.8},
    ]
    tracklet_jsonl.write_text(
        json.dumps({"meta": {"seq": "s", "track_id": "tube1", "action_dynamics_score": 0.5}, "rows": rows})
        + "\n"
        + json.dumps({"meta": {"seq": "s", "track_id": "tube2"}, "rows": rows})
        + "\n",
        encoding="utf-8",
    )

    result = attach_tracklet_confidence_fusion_scores(tracklet_jsonl, tmp_path / "fused.jsonl")
    items = [json.loads(line) for line in result.out_path.read_text(encoding="utf-8").splitlines()]

    assert result.summary["total_tracklets"] == 2
    assert result.summary["fused_tracklets"] == 1
    assert result.summary["missing_action_score_tracklets"] == 1
    assert abs(items[0]["meta"]["tracklet_detection_confidence"] - 0.6) < 1e-9
    assert abs(items[0]["meta"]["video_action_conf_score"] - 0.3) < 1e-9
    assert abs(items[0]["rows"][0]["video_action_conf_score"] - 0.3) < 1e-9
    assert "video_action_conf_score" not in items[1]["meta"]


def test_attach_tracklet_confidence_fusion_scores_uses_max_reduction(tmp_path):
    tracklet_jsonl = tmp_path / "tracklets.jsonl"
    rows = [
        {"seq": "s", "track_id": "tube1", "frame_id": 0, "bbox": [10.0, 10.0, 14.0, 14.0], "objectness": 0.4},
        {"seq": "s", "track_id": "tube1", "frame_id": 1, "bbox": [11.0, 10.0, 15.0, 14.0], "objectness": 0.8},
    ]
    tracklet_jsonl.write_text(
        json.dumps({"meta": {"seq": "s", "track_id": "tube1", "my_action": 0.5}, "rows": rows}) + "\n",
        encoding="utf-8",
    )

    result = attach_tracklet_confidence_fusion_scores(
        tracklet_jsonl,
        tmp_path / "fused.jsonl",
        action_score_field="my_action",
        confidence_reduction="max",
        out_score_field="my_fused_score",
    )
    item = json.loads(result.out_path.read_text(encoding="utf-8").splitlines()[0])

    assert result.summary["mean_tracklet_detection_confidence"] == 0.8
    assert item["meta"]["my_fused_score"] == 0.4
    assert item["meta"]["tracklet_detection_confidence_reduction"] == "max"


def test_run_action_dynamics_tracklet_pipeline(tmp_path):
    tracklet_rows = [
        {"seq": "s", "track_id": "tube1", "frame_id": i, "bbox": [10.0 + i, 10.0, 14.0 + i, 14.0], "objectness": 0.8}
        for i in range(6)
    ]
    tracklet_jsonl = tmp_path / "tracklets.jsonl"
    tracklet_jsonl.write_text(
        json.dumps(
            {
                "meta": {"seq": "s", "track_id": "tube1", "label": 1, "bucket": "positive", "dataset_source": "synthetic"},
                "rows": tracklet_rows,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_action_dynamics_tracklet_pipeline(
        tracklet_jsonl,
        tmp_path / "route_b",
        past_len=3,
        future_len=2,
        epochs=5,
        hidden=16,
        batch_size=4,
        thresholds=[0.5],
        balance_by=["dataset_source", "label"],
        model_type="diffusion",
        diffusion_steps=4,
        prior_image_size=(24, 24),
        prior_split_horizon=True,
        prior_merge_mode="max",
    )
    attached = [json.loads(line) for line in result.out_path.read_text(encoding="utf-8").splitlines()]

    assert result.summary["dataset"]["samples"] == 2
    assert result.summary["tracklet_eval"]["scored_tracklets"] == 1
    assert result.summary["attach"]["attached_tracklets"] == 1
    assert result.summary["balance_by"] == ["dataset_source", "label"]
    assert result.summary["model_type"] == "diffusion"
    assert result.summary["diffusion_steps"] == 4
    assert result.summary["threshold_eval"]["num_tracklets"] == 1
    assert result.summary["threshold_eval"]["best"]["threshold"] == 0.5
    assert result.summary["action_prior_eval"]["exported"] == 4
    assert result.summary["action_prior_eval"]["split_horizon"] is True
    assert result.summary["frame_prior_eval"]["frames"] == 3
    assert result.summary["frame_prior_attach_eval"]["attached_rows"] == 3
    assert result.summary["final_tracklets"].endswith("action_dynamics_frame_prior_tracklets.jsonl")
    assert (tmp_path / "route_b" / "action_chunk_policy.pt").exists()
    assert (tmp_path / "route_b" / "action_priors" / "action_prior_heatmaps.jsonl").exists()
    assert (tmp_path / "route_b" / "frame_priors" / "frame_prior_index.jsonl").exists()
    assert (tmp_path / "route_b" / "action_dynamics_frame_prior_tracklets.jsonl").exists()
    assert (tmp_path / "route_b" / "action_dynamics_threshold_sweep.csv").exists()
    assert (tmp_path / "route_b" / "action_dynamics_threshold_summary.json").exists()
    assert (tmp_path / "route_b" / "action_dynamics_pipeline_summary.json").exists()
    assert attached[0]["meta"]["action_num_windows"] == 2
    assert attached[0]["meta"]["action_frame_prior_rows"] == 3
    assert sum("action_frame_prior" in row for row in attached[0]["rows"]) == 3


def test_run_action_dynamics_tracklet_ablation(tmp_path):
    items = []
    for track_index, label in enumerate([1, 0]):
        rows = [
            {
                "seq": "s",
                "track_id": f"tube{track_index}",
                "frame_id": i,
                "bbox": [10.0 + i + track_index * 20, 10.0, 14.0 + i + track_index * 20, 14.0],
                "objectness": 0.8 if label else 0.2,
            }
            for i in range(6)
        ]
        items.append(
            {
                "meta": {
                    "seq": "s",
                    "track_id": f"tube{track_index}",
                    "label": label,
                    "bucket": "positive" if label else "hard_negative",
                    "dataset_source": "synthetic",
                },
                "rows": rows,
            }
        )
    tracklet_jsonl = tmp_path / "tracklets.jsonl"
    tracklet_jsonl.write_text("\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8")

    result = run_action_dynamics_tracklet_ablation(
        tracklet_jsonl,
        tmp_path / "route_b_ablation",
        model_types=["mlp", "diffusion"],
        past_len=3,
        future_len=2,
        epochs=2,
        hidden=16,
        batch_size=4,
        diffusion_steps=4,
        thresholds=[0.5],
        balance_by=["dataset_source", "label"],
        prior_image_size=(24, 24),
        prior_split_horizon=True,
        prior_merge_mode="max",
    )

    assert result.out_path.exists()
    assert (tmp_path / "route_b_ablation" / "action_dynamics_tracklet_ablation_summary.json").exists()
    assert len(result.summary["rows"]) == 2
    assert result.summary["best"]["model_type"] in {"mlp", "diffusion"}
    assert result.summary["models"]["mlp"]["threshold_eval"]["num_tracklets"] == 2
    assert result.summary["models"]["diffusion"]["threshold_eval"]["num_tracklets"] == 2
    assert result.summary["models"]["mlp"]["action_prior_eval"]["exported"] == 8
    assert result.summary["models"]["diffusion"]["action_prior_eval"]["exported"] == 8
    assert result.summary["models"]["mlp"]["frame_prior_eval"]["frames"] == 3
    assert result.summary["models"]["diffusion"]["frame_prior_eval"]["frames"] == 3
    assert result.summary["models"]["mlp"]["frame_prior_attach_eval"]["attached_rows"] == 6
    assert result.summary["models"]["diffusion"]["frame_prior_attach_eval"]["attached_rows"] == 6
    assert (tmp_path / "route_b_ablation" / "mlp" / "tracklet_dynamics_scores.jsonl").exists()
    assert (tmp_path / "route_b_ablation" / "diffusion" / "tracklet_dynamics_scores.jsonl").exists()
    assert (tmp_path / "route_b_ablation" / "mlp" / "action_dynamics_frame_prior_tracklets.jsonl").exists()
    assert (tmp_path / "route_b_ablation" / "diffusion" / "action_dynamics_frame_prior_tracklets.jsonl").exists()


def test_collect_route_b_result_summaries(tmp_path):
    pipeline_summary = {
        "tracklet_jsonl": "nps_tracklets.jsonl",
        "out_dir": "nps/mlp",
        "past_len": 3,
        "future_len": 2,
        "model_type": "mlp",
        "tracklet_scores": "nps/mlp/tracklet_dynamics_scores.jsonl",
        "threshold_sweep": "nps/mlp/action_dynamics_threshold_sweep.csv",
        "tracklet_eval": {
            "scored_tracklets": 10,
            "action_windows": 40,
            "mean_learned_center_error": 1.5,
            "mean_constant_velocity_center_error": 2.0,
            "mean_error_improvement_vs_cv": 0.5,
        },
        "threshold_eval": {
            "best": {
                "threshold": 0.7,
                "precision": 0.8,
                "recall": 0.6,
                "f1": 0.685,
                "accuracy": 0.75,
                "tp": 6,
                "fp": 1,
                "fn": 4,
                "tn": 9,
            }
        },
    }
    ablation_summary = {
        "tracklet_jsonl": "aot_tracklets.jsonl",
        "out_dir": "aot/ablation",
        "past_len": 3,
        "future_len": 2,
        "rows": [
            {
                "model_type": "mlp",
                "scored_tracklets": 8,
                "action_windows": 30,
                "mean_learned_center_error": 1.0,
                "mean_constant_velocity_center_error": 1.7,
                "mean_error_improvement_vs_cv": 0.7,
                "best_threshold": 0.5,
                "best_precision": 0.75,
                "best_recall": 0.75,
                "best_f1": 0.75,
                "best_accuracy": 0.75,
                "best_tp": 3,
                "best_fp": 1,
                "best_fn": 1,
                "best_tn": 3,
                "out_dir": "aot/ablation/mlp",
                "tracklet_scores": "aot/ablation/mlp/tracklet_dynamics_scores.jsonl",
                "threshold_sweep": "aot/ablation/mlp/action_dynamics_threshold_sweep.csv",
            },
            {
                "model_type": "diffusion",
                "scored_tracklets": 8,
                "action_windows": 30,
                "mean_learned_center_error": 0.8,
                "mean_constant_velocity_center_error": 1.7,
                "mean_error_improvement_vs_cv": 0.9,
                "best_threshold": 0.6,
                "best_precision": 0.8,
                "best_recall": 0.8,
                "best_f1": 0.8,
                "best_accuracy": 0.8,
                "best_tp": 4,
                "best_fp": 1,
                "best_fn": 1,
                "best_tn": 4,
                "out_dir": "aot/ablation/diffusion",
                "tracklet_scores": "aot/ablation/diffusion/tracklet_dynamics_scores.jsonl",
                "threshold_sweep": "aot/ablation/diffusion/action_dynamics_threshold_sweep.csv",
            },
        ],
    }
    prior_sweep_summary = {
        "run_roots": ["runs"],
        "gt_csv": "gt.csv",
        "out_dir": "route_b/prior",
        "csv": "route_b/prior/action_frame_prior_fusion_run_sweep.csv",
        "profile": "hard_recovery",
        "raw": {"precision": 0.6, "recall": 0.6, "f1": 0.6, "tp": 3, "fp": 2, "fn": 2},
        "best": {
            "prior_weight": 0.5,
            "min_prior_score": 0.2,
            "promote_threshold": 0.3,
            "precision": 0.85,
            "recall": 0.85,
            "f1": 0.85,
            "tp": 5,
            "fp": 1,
            "fn": 1,
            "fused_rows": 20,
            "promoted_rows": 4,
        },
        "rows": [],
    }
    pipeline_path = tmp_path / "nps_pipeline_summary.json"
    ablation_path = tmp_path / "aot_ablation_summary.json"
    prior_path = tmp_path / "nps_action_frame_prior_fusion_run_sweep_summary.json"
    pipeline_path.write_text(json.dumps(pipeline_summary), encoding="utf-8")
    ablation_path.write_text(json.dumps(ablation_summary), encoding="utf-8")
    prior_path.write_text(json.dumps(prior_sweep_summary), encoding="utf-8")

    result = collect_route_b_result_summaries(
        [pipeline_path, ablation_path, prior_path],
        tmp_path / "collected",
        dataset_names=["nps", "aot", "nps"],
    )

    assert result.out_path.exists()
    assert (tmp_path / "collected" / "route_b_results_summary.json").exists()
    assert result.summary["num_rows"] == 4
    assert result.summary["datasets"] == ["aot", "nps"]
    assert result.summary["best"]["dataset"] == "nps"
    assert result.summary["best"]["model_type"] == "action_prior"
    assert result.summary["best"]["run_type"] == "action_prior_fusion_run_sweep"
    assert result.summary["best"]["best_f1"] == 0.85
    assert result.summary["best"]["raw_f1"] == 0.6
    assert result.summary["best"]["delta_f1"] == 0.25
    assert result.summary["best"]["prior_weight"] == 0.5
    assert result.summary["best"]["promoted_rows"] == 4


def test_compare_route_b_results_to_baselines(tmp_path):
    route_b_csv = tmp_path / "route_b_results_table.csv"
    baseline_csv = tmp_path / "baselines.csv"
    route_b_csv.write_text(
        "\n".join(
            [
                "dataset,run_type,model_type,past_len,future_len,scored_tracklets,action_windows,best_precision,best_recall,best_f1,summary_json",
                "nps,tracklet_ablation,mlp,3,2,10,40,0.80,0.70,0.74,nps_mlp.json",
                "nps,tracklet_ablation,diffusion,3,2,10,40,0.86,0.80,0.83,nps_diffusion.json",
                "aot,tracklet_ablation,mlp,3,2,8,30,0.70,0.75,0.72,aot_mlp.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_csv.write_text(
        "\n".join(
            [
                "dataset,method,best_precision,best_recall,best_f1",
                "nps,YOLOMG,0.82,0.75,0.78",
                "nps,NPS-paper,0.79,0.76,0.77",
                "aot,TransVisDrone,0.78,0.78,0.78",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = compare_route_b_results_to_baselines(route_b_csv, baseline_csv, tmp_path / "comparison")

    assert result.out_path.exists()
    assert (tmp_path / "comparison" / "route_b_baseline_ranking.csv").exists()
    assert result.summary["num_comparisons"] == 2
    assert result.summary["route_b_wins"] == 1
    nps = [row for row in result.summary["comparison_rows"] if row["dataset"] == "nps"][0]
    aot = [row for row in result.summary["comparison_rows"] if row["dataset"] == "aot"][0]
    assert nps["best_route_b_method"] == "route_b:diffusion"
    assert round(nps["delta_route_b_minus_baseline"], 2) == 0.05
    assert aot["route_b_beats_baseline"] is False


def test_write_route_b_baseline_template(tmp_path):
    result = write_route_b_baseline_template(
        tmp_path / "baseline_template.csv",
        datasets=["nps", "aot"],
        methods=["YOLOMG", "TransVisDrone"],
    )
    rows = list(csv.DictReader(result.out_path.open("r", encoding="utf-8")))

    assert result.out_path.exists()
    assert result.summary["rows"] == 4
    assert result.summary["required_for_compare"] == ["dataset", "method", "best_f1"]
    assert rows[0]["dataset"] == "nps"
    assert rows[0]["method"] == "YOLOMG"
    assert "source_notes" in rows[0]


def test_write_route_b_official_baseline_seed(tmp_path):
    result = write_route_b_official_baseline_seed(tmp_path / "official_baselines.csv")
    rows = list(csv.DictReader(result.out_path.open("r", encoding="utf-8")))

    assert result.out_path.exists()
    assert result.summary["source_backed_rows"] == 3
    assert result.summary["placeholder_rows"] == 3
    assert result.summary["strict_compare_ready"] is False
    tvd_nps = [row for row in rows if row["dataset"] == "nps" and row["method"] == "TransVisDrone"][0]
    tvd_aot = [row for row in rows if row["dataset"] == "aot" and row["method"] == "TransVisDrone"][0]
    assert tvd_nps["nps_map50"] == "0.948"
    assert round(float(tvd_nps["best_f1"]), 3) == 0.891
    assert tvd_aot["aot_hfar"] == "89.476744"
    assert any(row["needs_fill"] == "yes" for row in rows)

    draft_validation = validate_route_b_baseline_csv(
        result.out_path,
        tmp_path / "official_baselines_draft_validation.json",
        require_metric_values=False,
    )
    assert draft_validation.summary["valid"] is True
    assert draft_validation.summary["warnings"]


def test_validate_route_b_baseline_csv(tmp_path):
    valid_csv = tmp_path / "valid_baselines.csv"
    valid_csv.write_text(
        "\n".join(
            [
                "dataset,method,best_precision,best_recall,best_f1",
                "nps,YOLOMG,0.8,0.7,0.75",
                "aot,TransVisDrone,0.7,0.8,0.74",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    valid = validate_route_b_baseline_csv(valid_csv, tmp_path / "valid.json")

    assert valid.summary["valid"] is True
    assert valid.summary["issues"] == []

    invalid_csv = tmp_path / "invalid_baselines.csv"
    invalid_csv.write_text(
        "\n".join(
            [
                "dataset,method,best_precision,best_recall,best_f1",
                "nps,YOLOMG,0.8,0.7,",
                "nps,YOLOMG,1.2,0.7,0.75",
                "aot,TransVisDrone,0.7,not_number,0.74",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    invalid = validate_route_b_baseline_csv(invalid_csv, tmp_path / "invalid.json")

    assert invalid.summary["valid"] is False
    messages = [issue["message"] for issue in invalid.summary["issues"]]
    assert "best_f1 is empty" in messages
    assert any("duplicate dataset-method" in message for message in messages)
    assert "best_precision must be in [0, 1]" in messages
    assert "best_recall is not numeric" in messages


def test_export_route_b_baseline_markdown_table(tmp_path):
    summary = {
        "metric": "best_f1",
        "num_comparisons": 2,
        "route_b_wins": 1,
        "comparison_rows": [
            {
                "dataset": "aot",
                "best_baseline_method": "TransVisDrone",
                "best_baseline_value": 0.78,
                "best_route_b_method": "route_b:mlp",
                "best_route_b_value": 0.72,
                "delta_route_b_minus_baseline": -0.06,
                "route_b_beats_baseline": False,
            },
            {
                "dataset": "nps",
                "best_baseline_method": "YOLOMG",
                "best_baseline_value": 0.78,
                "best_route_b_method": "route_b:diffusion",
                "best_route_b_value": 0.83,
                "delta_route_b_minus_baseline": 0.05,
                "route_b_beats_baseline": True,
            },
        ],
    }
    summary_path = tmp_path / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = export_route_b_baseline_markdown_table(summary_path, tmp_path / "table.md", digits=2)
    text = result.out_path.read_text(encoding="utf-8")

    assert result.out_path.exists()
    assert "| Dataset | Best baseline | Baseline | Best Route B | Route B | Delta | Beats baseline |" in text
    assert "| nps | YOLOMG | 0.78 | route_b:diffusion | 0.83 | 0.05 | yes |" in text
    assert "Route B beats the best listed baseline on 1/2 dataset comparisons." in text


def test_build_route_b_baseline_report(tmp_path):
    route_summary = {
        "run_roots": ["runs"],
        "gt_csv": "gt.csv",
        "out_dir": "prior",
        "csv": "prior/action_frame_prior_fusion_run_sweep.csv",
        "raw": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "tp": 1, "fp": 1, "fn": 1},
        "best": {
            "prior_weight": 0.5,
            "min_prior_score": 0.2,
            "promote_threshold": 0.3,
            "precision": 0.8,
            "recall": 0.8,
            "f1": 0.8,
            "tp": 4,
            "fp": 1,
            "fn": 1,
            "fused_rows": 10,
            "promoted_rows": 2,
        },
        "rows": [],
    }
    summary_path = tmp_path / "action_frame_prior_fusion_run_sweep_summary.json"
    summary_path.write_text(json.dumps(route_summary), encoding="utf-8")
    baseline_csv = tmp_path / "baselines.csv"
    baseline_csv.write_text(
        "\n".join(
            [
                "dataset,method,best_precision,best_recall,best_f1",
                "nps,YOLOMG,0.7,0.7,0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_route_b_baseline_report(
        [summary_path],
        baseline_csv,
        tmp_path / "report",
        dataset_names=["nps"],
    )
    text = result.out_path.read_text(encoding="utf-8")

    assert result.out_path.name == "route_b_baseline_report.md"
    assert (tmp_path / "report" / "route_b_report_summary.json").exists()
    assert (tmp_path / "report" / "collected" / "route_b_results_table.csv").exists()
    assert (tmp_path / "report" / "comparison" / "route_b_baseline_comparison.csv").exists()
    assert result.summary["route_b_wins"] == 1
    assert result.summary["num_comparisons"] == 1
    assert result.summary["best_route_b"]["model_type"] == "action_prior"
    assert "Route B beats the best listed baseline on 1/1 dataset comparisons." in text


def test_build_route_b_baseline_report_strict_invalid_baseline(tmp_path):
    route_summary = {
        "csv": "prior/action_frame_prior_fusion_run_sweep.csv",
        "raw": {"precision": 0.5, "recall": 0.5, "f1": 0.5},
        "best": {"precision": 0.8, "recall": 0.8, "f1": 0.8},
        "rows": [],
    }
    summary_path = tmp_path / "action_frame_prior_fusion_run_sweep_summary.json"
    summary_path.write_text(json.dumps(route_summary), encoding="utf-8")
    baseline_csv = tmp_path / "bad_baselines.csv"
    baseline_csv.write_text("dataset,method,best_f1\nnps,YOLOMG,\n", encoding="utf-8")

    result = build_route_b_baseline_report(
        [summary_path],
        baseline_csv,
        tmp_path / "report",
        dataset_names=["nps"],
    )

    assert result.out_path.name == "route_b_report_summary.json"
    assert result.summary["valid"] is False
    assert (tmp_path / "report" / "baseline_validation.json").exists()
    assert not (tmp_path / "report" / "route_b_baseline_report.md").exists()


def test_evaluate_action_dynamics_thresholds(tmp_path):
    scores = tmp_path / "tracklet_scores.jsonl"
    rows = [
        {"seq": "s", "track_id": "p1", "label": 1, "dynamics_score": 0.9},
        {"seq": "s", "track_id": "p2", "label": 1, "dynamics_score": 0.8},
        {"seq": "s", "track_id": "n1", "label": 0, "dynamics_score": 0.2},
        {"seq": "s", "track_id": "n2", "label": 0, "dynamics_score": 0.1},
    ]
    scores.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = evaluate_action_dynamics_thresholds(scores, tmp_path / "sweep", thresholds=[0.5])

    assert result.out_path.exists()
    assert result.summary["best"]["threshold"] == 0.5
    assert result.summary["best"]["precision"] == 1.0
    assert result.summary["best"]["recall"] == 1.0
    assert result.summary["positives"] == 2
    assert result.summary["negatives"] == 2
