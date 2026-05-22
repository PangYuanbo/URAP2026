import csv
import json

from qstr_dronedet.evaluation.tracklet_sequence_model_selection import run_tracklet_sequence_model_selection
from qstr_dronedet.tracking.tracklet_classifier import build_tracklet_dataset
from qstr_dronedet.tracking.tracklet_sequence_classifier import (
    filter_infer_rows_with_tracklet_sequence_classifier,
    score_tracklets_from_rows_sequence,
    train_tracklet_sequence_classifier,
)


def _row(frame_id, track_id, box, crop_drone, temp_drone, final_drone, bg, predicted="drone", source="tracker+fallback_yolo"):
    return {
        "frame_id": frame_id,
        "bbox": box,
        "objectness": 0.25,
        "source": source,
        "final_drone_score": final_drone * 0.25,
        "predicted_class": predicted,
        "track_id": track_id,
        "track_validated": True,
        "track_drift": 1.0,
        "track_speed": 1.0,
        "crop_probs": {"drone": crop_drone, "background": 1.0 - crop_drone},
        "feature_probs": {"drone": crop_drone, "background": 1.0 - crop_drone},
        "temporal_probs": {"drone": temp_drone, "background": 1.0 - temp_drone},
        "final_probs": {"drone": final_drone, "background": bg},
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _make_dataset(tmp_path, seqs=("seq_train", "seq_calib")):
    run_root = tmp_path / "runs"
    profile = "hard_recovery"
    diagnostics = []
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        for seq in seqs:
            writer.writerow([str(tmp_path / seq / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])
            writer.writerow([str(tmp_path / seq / "visible.mp4"), 1, 11, 10, 21, 20, "drone", "tiny"])
    for seq in seqs:
        seq_dir = run_root / profile / seq
        rows = [
            _row(0, 1, [10, 10, 20, 20], 0.55, 0.75, 0.70, 0.20, "drone"),
            _row(1, 1, [11, 10, 21, 20], 0.56, 0.80, 0.72, 0.18, "drone"),
            _row(0, 2, [100, 100, 110, 110], 0.20, 0.25, 0.22, 0.78, "drone"),
            _row(1, 2, [101, 100, 111, 110], 0.20, 0.24, 0.20, 0.80, "drone"),
        ]
        _write_jsonl(seq_dir / "predictions.jsonl", rows)
        _write_jsonl(seq_dir / "diagnostics.jsonl", rows)
        diagnostics.append(seq_dir / "diagnostics.jsonl")
    dataset = build_tracklet_dataset(diagnostics, gt, tmp_path / "tracklets")
    return run_root, profile, gt, dataset


def test_train_and_filter_tracklet_sequence_classifier(tmp_path):
    _, _, _, dataset = _make_dataset(tmp_path)
    weights = train_tracklet_sequence_classifier(dataset.json_path, tmp_path / "seq.pt", epochs=2, hidden=8, max_len=4)
    rows = [
        {**_row(0, 1, [10, 10, 20, 20], 0.55, 0.75, 0.70, 0.20), "seq": "seq_a"},
        {**_row(1, 1, [11, 10, 21, 20], 0.56, 0.80, 0.72, 0.18), "seq": "seq_a"},
        {**_row(0, 2, [100, 100, 110, 110], 0.20, 0.25, 0.22, 0.78), "seq": "seq_a"},
        {**_row(1, 2, [101, 100, 111, 110], 0.20, 0.24, 0.20, 0.80), "seq": "seq_a"},
    ]
    scores = score_tracklets_from_rows_sequence(rows, weights, threshold=0.5)
    filtered, _, summary = filter_infer_rows_with_tracklet_sequence_classifier(rows, rows, weights, threshold=0.99)

    assert set(scores) == {"seq_a:1", "seq_a:2"}
    assert summary["raw_drone_predictions"] == 4
    assert any(row["predicted_class"] == "background" for row in filtered)


def test_tracklet_sequence_model_selection_uses_calibration_split(tmp_path):
    run_root, _, gt, dataset = _make_dataset(tmp_path)
    summary = run_tracklet_sequence_model_selection(
        dataset.json_path,
        [run_root],
        gt,
        tmp_path / "selection",
        calib_seqs=["seq_calib"],
        epochs_values=[1],
        hidden_values=[8],
        max_len_values=[4],
        hard_negative_augments_values=[0],
        classifier_thresholds=[0.5],
        promotion_enabled_values=[False],
    )

    assert summary["split"]["train_sequences"] == ["seq_train"]
    assert summary["split"]["calibration_sequences"] == ["seq_calib"]
    assert summary["num_candidates"] == 1
    assert (tmp_path / "selection" / "selected_tracklet_sequence_classifier.pt").exists()
