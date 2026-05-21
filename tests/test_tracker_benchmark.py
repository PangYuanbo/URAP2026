import json

import cv2
import numpy as np

from qstr_dronedet.evaluation.tracker_benchmark import run_tracker_oracle_benchmark


def test_tracker_oracle_benchmark_static_survives_between_detections(tmp_path):
    video = tmp_path / "static.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    boxes = []
    for i in range(12):
        frame = np.full((64, 64, 3), 180, np.uint8)
        cv2.circle(frame, (32, 32), 2, (0, 0, 0), -1)
        writer.write(frame)
        boxes.append({"frame_id": i, "bbox_xyxy": [30.0, 30.0, 34.0, 34.0]})
    writer.release()
    meta = tmp_path / "static.json"
    meta.write_text(json.dumps({"output": str(video), "boxes": boxes}), encoding="utf-8")

    summary = run_tracker_oracle_benchmark([meta], tmp_path / "out", detection_stride=4, max_frames=12)

    assert summary["num_frames"] == 12
    assert summary["survival_without_detection_rate"] > 0.5
    assert summary["mode_counts"].get("static_or_hovering", 0) > 0
