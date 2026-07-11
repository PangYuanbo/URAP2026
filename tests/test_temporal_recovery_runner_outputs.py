from __future__ import annotations

import json

from qstr_dronedet.pipelines.temporal_recovery import TemporalRecoveryFrame
from qstr_dronedet.types import DetectionCandidate
from tools.run_temporal_recovery_pipeline import FrameRecord, write_candidate_outputs


def test_write_candidate_outputs_jsonl_and_yolo_labels(tmp_path):
    rows = [
        TemporalRecoveryFrame(
            frame_id=0,
            candidates=[
                DetectionCandidate(
                    bbox_xyxy=(10.0, 20.0, 30.0, 60.0),
                    objectness=0.72,
                    source="yolov5_dual",
                    motion_score=0.4,
                    extra={"raw_objectness": 0.51, "motion_memory_score": 0.69},
                )
            ],
            selected=None,
            memory_bbox=None,
            diagnostics={},
        )
    ]
    records = [FrameRecord(frame_id=0, path=tmp_path / "phantom109_0001.jpg", width=100, height=200)]

    write_candidate_outputs(rows, records, tmp_path, write_labels=True)

    payload = json.loads((tmp_path / "candidate_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert payload["image_path"].endswith("phantom109_0001.jpg")
    assert payload["candidates"][0]["score"] == 0.72
    assert payload["candidates"][0]["raw_objectness"] == 0.51
    assert payload["candidates"][0]["motion_memory_score"] == 0.69

    label = (tmp_path / "candidate_labels" / "phantom109_0001.txt").read_text(encoding="utf-8").strip()
    assert label == "0 0.20000000 0.20000000 0.20000000 0.20000000 0.72000000"
