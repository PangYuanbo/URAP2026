import json

import cv2
import numpy as np

from qstr_dronedet.evaluation.stage_b_benchmark import run_stage_b_oracle_benchmark


def test_stage_b_oracle_benchmark_smoke(tmp_path):
    video = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    boxes = []
    for i in range(8):
        frame = np.full((64, 64, 3), 180, np.uint8)
        cv2.circle(frame, (32, 32), 2, (0, 0, 0), -1)
        writer.write(frame)
        boxes.append({"frame_id": i, "bbox_xyxy": [30.0, 30.0, 34.0, 34.0]})
    writer.release()
    meta = tmp_path / "tiny.json"
    meta.write_text(json.dumps({"output": str(video), "boxes": boxes}), encoding="utf-8")

    summary = run_stage_b_oracle_benchmark([meta], tmp_path / "out", frame_stride=4, negative_per_positive=1, t=5)

    assert summary["num_samples"] == 2
    assert summary["class_counts"]["drone"] == 1
    assert summary["class_counts"]["background"] == 1
    assert (tmp_path / "out" / "stage_b_oracle_predictions.jsonl").exists()
    assert (tmp_path / "out" / "stage_b_oracle_summary.json").exists()
