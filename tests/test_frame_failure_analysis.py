import csv
import json

from qstr_dronedet.evaluation.frame_failure_analysis import analyze_frame_failures


def test_analyze_frame_failures_outputs_artifacts(tmp_path):
    run = tmp_path / "run"
    seq_dir = run / "hard_recovery" / "seq001"
    seq_dir.mkdir(parents=True)
    raw_rows = [
        {"frame_id": 0, "bbox": [10, 10, 20, 20], "predicted_class": "background", "final_drone_score": 0.1},
        {"frame_id": 1, "bbox": [50, 50, 60, 60], "predicted_class": "drone", "final_drone_score": 0.8},
    ]
    filtered_rows = [
        {"frame_id": 0, "bbox": [10, 10, 20, 20], "predicted_class": "drone", "final_drone_score": 0.5, "track_id": 1},
        {"frame_id": 1, "bbox": [50, 50, 60, 60], "predicted_class": "background", "final_drone_score": 0.0, "track_id": 2},
    ]
    (seq_dir / "predictions_raw.jsonl").write_text("\n".join(json.dumps(r) for r in raw_rows) + "\n", encoding="utf-8")
    (seq_dir / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in filtered_rows) + "\n", encoding="utf-8")
    gt = tmp_path / "gt.csv"
    with gt.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow([str(tmp_path / "raw_videos" / "seq001" / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])

    summary = analyze_frame_failures(run, gt, tmp_path / "analysis", max_frames=2)

    assert summary["raw"]["recall"] == 0.0
    assert summary["filtered"]["recall"] == 1.0
    assert (tmp_path / "analysis" / "per_frame_filtered.csv").exists()
    assert (tmp_path / "analysis" / "filtered_frame_timeline.png").exists()
    assert (tmp_path / "analysis" / "URAP-UAV_linear_issue.md").exists()
