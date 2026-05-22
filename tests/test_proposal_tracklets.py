import csv
import json

from qstr_dronedet.tracking.proposal_tracklets import build_proposal_tracklet_dataset, merge_tracklet_jsonl


def _row(frame_id, box, score, track_id=None, source="fallback_yolo", crop=0.5, temporal=0.6, bg=0.3):
    row = {
        "frame_id": frame_id,
        "bbox": box,
        "objectness": score,
        "source": source,
        "final_drone_score": score * temporal,
        "predicted_class": "drone" if temporal > bg else "background",
        "crop_probs": {"drone": crop, "background": 1.0 - crop},
        "temporal_probs": {"drone": temporal, "background": 1.0 - temporal},
        "final_probs": {"drone": temporal, "background": bg},
        "alignment_quality": 0.8,
    }
    if track_id is not None:
        row["track_id"] = track_id
    return row


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_proposal_tracklets_relinks_untracked_rows(tmp_path):
    run_root = tmp_path / "runs"
    seq_dir = run_root / "hard_recovery" / "seq001"
    rows = [
        _row(0, [10, 10, 20, 20], 0.20),
        _row(1, [11, 10, 21, 20], 0.22),
        _row(0, [100, 100, 110, 110], 0.40, crop=0.2, temporal=0.2, bg=0.8),
        _row(1, [101, 100, 111, 110], 0.42, crop=0.2, temporal=0.2, bg=0.8),
    ]
    _write_jsonl(seq_dir / "diagnostics_raw.jsonl", rows)
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow([str(tmp_path / "seq001" / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])
        writer.writerow([str(tmp_path / "seq001" / "visible.mp4"), 1, 11, 10, 21, 20, "drone", "tiny"])

    result = build_proposal_tracklet_dataset([run_root], gt, tmp_path / "proposal_tracklets", min_tracklet_rows=2)

    assert result.summary["num_tracklets"] == 2
    assert result.summary["positives"] == 1
    assert result.summary["bucket_counts"]["hard_tiny_positive"] == 1
    assert result.summary["bucket_counts"]["high_score_detector_fp"] == 1
    assert result.json_path.exists()


def test_merge_tracklet_jsonl_tags_sources(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    item_a = {"meta": {"seq": "s", "track_id": "1", "label": 1, "bucket": "positive"}, "rows": []}
    item_b = {"meta": {"seq": "s", "track_id": "1", "label": 0, "bucket": "high_score_detector_fp"}, "rows": []}
    first.write_text(json.dumps(item_a) + "\n", encoding="utf-8")
    second.write_text(json.dumps(item_b) + "\n", encoding="utf-8")

    result = merge_tracklet_jsonl([first, second], tmp_path / "merged" / "tracklets.jsonl", source_names=["orig", "proposal"])

    lines = [json.loads(line) for line in result.json_path.read_text(encoding="utf-8").splitlines()]
    assert result.summary["num_tracklets"] == 2
    assert {line["meta"]["dataset_source"] for line in lines} == {"orig", "proposal"}
