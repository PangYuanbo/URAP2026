import csv
import json

from qstr_dronedet.tracking.tracklet_classifier import (
    build_tracklet_dataset,
    evaluate_tracklet_classifier,
    train_tracklet_classifier,
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
