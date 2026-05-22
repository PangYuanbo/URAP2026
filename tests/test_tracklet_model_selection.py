import csv
import json

from qstr_dronedet.evaluation.tracklet_model_selection import run_tracklet_model_selection
from qstr_dronedet.tracking.tracklet_classifier import build_tracklet_dataset


def _row(frame_id, track_id, box, crop_drone, temp_drone, final_drone, bg, predicted="drone", seq=None):
    row = {
        "frame_id": frame_id,
        "bbox": box,
        "objectness": 0.25,
        "source": "tracker+fallback_yolo",
        "final_drone_score": final_drone * 0.25,
        "predicted_class": predicted,
        "track_id": track_id,
        "track_validated": True,
        "track_drift": 1.0,
        "track_speed": 1.0,
        "crop_probs": {"drone": crop_drone, "background": 1.0 - crop_drone},
        "temporal_probs": {"drone": temp_drone, "background": 1.0 - temp_drone},
        "final_probs": {"drone": final_drone, "background": bg},
    }
    if seq is not None:
        row["seq"] = seq
    return row


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_tracklet_model_selection_uses_calibration_split(tmp_path):
    run_root = tmp_path / "runs"
    profile = "hard_recovery"
    diagnostics = []
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        for seq in ["seq_train", "seq_calib"]:
            writer.writerow([str(tmp_path / seq / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])
            writer.writerow([str(tmp_path / seq / "visible.mp4"), 1, 11, 10, 21, 20, "drone", "tiny"])

    for seq in ["seq_train", "seq_calib"]:
        seq_dir = run_root / profile / seq
        rows = [
            _row(0, 1, [10, 10, 20, 20], 0.55, 0.75, 0.70, 0.20, "drone"),
            _row(1, 1, [11, 10, 21, 20], 0.56, 0.76, 0.71, 0.19, "drone"),
            _row(0, 2, [100, 100, 110, 110], 0.35, 0.40, 0.32, 0.62, "drone"),
            _row(1, 2, [101, 100, 111, 110], 0.34, 0.39, 0.31, 0.63, "drone"),
        ]
        _write_jsonl(seq_dir / "predictions.jsonl", rows)
        _write_jsonl(seq_dir / "diagnostics.jsonl", rows)
        diagnostics.append(seq_dir / "diagnostics.jsonl")

    dataset = build_tracklet_dataset(diagnostics, gt, tmp_path / "tracklets")
    summary = run_tracklet_model_selection(
        dataset.csv_path,
        [run_root],
        gt,
        tmp_path / "selection",
        calib_seqs=["seq_calib"],
        epochs_values=[1],
        hidden_values=[8],
        lr_values=[1e-3],
        hard_tiny_positive_augments_values=[0],
        hard_negative_augments_values=[0, 1],
        classifier_thresholds=[0.5],
        promotion_enabled_values=[False],
    )

    assert summary["split"]["train_sequences"] == ["seq_train"]
    assert summary["split"]["calibration_sequences"] == ["seq_calib"]
    assert summary["num_candidates"] == 2
    assert (tmp_path / "selection" / "selected_tracklet_classifier.pt").exists()
    assert (tmp_path / "selection" / "model_selection.csv").exists()
