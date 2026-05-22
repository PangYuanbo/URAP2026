import csv
import json

from qstr_dronedet.tracking.tracklet_classifier import (
    apply_tracklet_filter_to_infer_outputs,
    build_tracklet_dataset,
    evaluate_tracklet_classifier,
    score_tracklets_from_rows,
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
    summary = apply_tracklet_filter_to_infer_outputs(pred, diag, weights, threshold=0.5)

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
