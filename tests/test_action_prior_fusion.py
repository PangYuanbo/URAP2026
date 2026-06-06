import json

import csv

from qstr_dronedet.tracking.action_prior_fusion import (
    fuse_action_frame_prior_predictions,
    sweep_action_frame_prior_fusion,
    sweep_action_frame_prior_fusion_run_root,
)


def test_fuse_action_frame_prior_predictions_boosts_and_promotes(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    rows = [
        {
            "frame_id": 0,
            "bbox": [1, 1, 3, 3],
            "predicted_class": "background",
            "final_drone_score": 0.10,
            "final_probs": {"drone": 0.10, "background": 0.80},
            "action_frame_prior_score": 0.80,
        },
        {
            "frame_id": 1,
            "bbox": [4, 4, 6, 6],
            "predicted_class": "background",
            "final_drone_score": 0.10,
            "final_probs": {"drone": 0.10, "background": 0.80},
            "action_frame_prior_score": 0.10,
        },
        {
            "frame_id": 2,
            "bbox": [7, 7, 9, 9],
            "predicted_class": "drone",
            "final_drone_score": 0.40,
            "final_probs": {"drone": 0.40, "background": 0.40},
            "action_frame_prior_score": 0.50,
        },
    ]
    pred.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = fuse_action_frame_prior_predictions(
        pred,
        tmp_path / "fused.jsonl",
        prior_weight=0.5,
        min_prior_score=0.25,
        promote_threshold=0.20,
    )
    fused = [json.loads(line) for line in result.out_path.read_text(encoding="utf-8").splitlines()]

    assert result.summary["total_rows"] == 3
    assert result.summary["fused_rows"] == 2
    assert result.summary["promoted_rows"] == 1
    assert result.summary["raw_drone_predictions"] == 1
    assert result.summary["final_drone_predictions"] == 2
    assert fused[0]["predicted_class"] == "drone"
    assert fused[0]["raw_predicted_class"] == "background"
    assert fused[0]["final_drone_score"] > 0.20
    assert fused[0]["action_prior_score_gain"] > 0.0
    assert "action_prior_fused" in fused[0]["diagnostic_cause"]
    assert "action_prior_promoted" in fused[0]["diagnostic_cause"]
    assert fused[1]["predicted_class"] == "background"
    assert "action_prior_fused_score" not in fused[1]
    assert fused[2]["predicted_class"] == "drone"
    assert fused[2]["final_drone_score"] > 0.40


def test_fuse_action_frame_prior_predictions_can_disable_promotion(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(
        json.dumps(
            {
                "frame_id": 0,
                "bbox": [1, 1, 3, 3],
                "predicted_class": "background",
                "final_drone_score": 0.10,
                "action_frame_prior_score": 0.90,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = fuse_action_frame_prior_predictions(
        pred,
        tmp_path / "fused.jsonl",
        prior_weight=0.5,
        min_prior_score=0.25,
        promote_threshold=None,
    )
    row = json.loads(result.out_path.read_text(encoding="utf-8").strip())

    assert result.summary["promoted_rows"] == 0
    assert row["predicted_class"] == "background"
    assert row["final_drone_score"] > 0.10


def test_sweep_action_frame_prior_fusion_selects_best_config(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    rows = [
        {
            "seq": "seq001",
            "frame_id": 0,
            "bbox": [10, 10, 20, 20],
            "predicted_class": "background",
            "final_drone_score": 0.05,
            "action_frame_prior_score": 0.95,
        },
        {
            "seq": "seq001",
            "frame_id": 1,
            "bbox": [40, 40, 50, 50],
            "predicted_class": "drone",
            "final_drone_score": 0.50,
            "action_frame_prior_score": 0.10,
        },
    ]
    pred.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    gt = tmp_path / "gt.csv"
    with gt.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow([str(tmp_path / "seq001" / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])

    result = sweep_action_frame_prior_fusion(
        pred,
        gt,
        tmp_path / "sweep",
        prior_weights=[0.5],
        min_prior_scores=[0.2],
        promote_thresholds=[None, 0.2],
        score_threshold=0.2,
        iou_threshold=0.3,
    )

    assert result.out_path.exists()
    assert (tmp_path / "sweep" / "action_frame_prior_fusion_sweep_summary.json").exists()
    assert result.summary["raw"]["recall"] == 0.0
    assert result.summary["best"]["recall"] == 1.0
    assert result.summary["best"]["f1"] > result.summary["raw"]["f1"]
    assert result.summary["best"]["promoted_rows"] == 1
    assert result.summary["num_configs"] == 2


def test_sweep_action_frame_prior_fusion_run_root_aggregates_sequences(tmp_path):
    run_root = tmp_path / "run"
    seq_dir = run_root / "hard_recovery" / "seq001"
    seq_dir.mkdir(parents=True)
    pred = seq_dir / "predictions.jsonl"
    pred.write_text(
        json.dumps(
            {
                "frame_id": 0,
                "bbox": [10, 10, 20, 20],
                "predicted_class": "background",
                "final_drone_score": 0.05,
                "action_frame_prior_score": 0.95,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gt = tmp_path / "gt.csv"
    with gt.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow([str(tmp_path / "seq001" / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])
        writer.writerow([str(tmp_path / "unused_seq" / "visible.mp4"), 0, 30, 30, 40, 40, "drone", "tiny"])

    result = sweep_action_frame_prior_fusion_run_root(
        [run_root],
        gt,
        tmp_path / "run_sweep",
        profile="hard_recovery",
        prior_weights=[0.5],
        min_prior_scores=[0.2],
        promote_thresholds=[None, 0.2],
        score_threshold=0.2,
        iou_threshold=0.3,
    )

    assert result.out_path.exists()
    assert (tmp_path / "run_sweep" / "action_frame_prior_fusion_run_sweep_summary.json").exists()
    assert result.summary["sequences"] == ["seq001"]
    assert result.summary["num_prediction_rows"] == 1
    assert result.summary["num_gt_rows"] == 1
    assert result.summary["raw"]["recall"] == 0.0
    assert result.summary["best"]["f1"] == 1.0
    assert result.summary["best"]["promoted_rows"] == 1
