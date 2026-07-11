from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_select_candidate_jsonl_prefers_detector_when_requested(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    candidate_jsonl = tmp_path / "candidates.jsonl"
    image = tmp_path / "images" / "frame_0001.jpg"
    image.parent.mkdir(parents=True)
    image.write_text("", encoding="utf-8")
    candidate_jsonl.write_text(
        json.dumps(
            {
                "frame_id": 0,
                "image_path": str(image),
                "width": 100,
                "height": 100,
                "candidates": [
                    {
                        "rank": 0,
                        "x1": 10,
                        "y1": 10,
                        "x2": 20,
                        "y2": 20,
                        "score": 0.9,
                        "raw_objectness": 0.2,
                        "motion_memory_score": 1.0,
                        "source": "gray_ncc",
                    },
                    {
                        "rank": 1,
                        "x1": 40,
                        "y1": 40,
                        "x2": 60,
                        "y2": 60,
                        "score": 0.4,
                        "raw_objectness": 0.8,
                        "motion_memory_score": 0.0,
                        "source": "yolov5_dual",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "labels"
    subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "select_candidate_jsonl_to_yolo_labels.py"),
            "--candidate-jsonl",
            str(candidate_jsonl),
            "--out-label-dir",
            str(out_dir),
            "--raw-weight",
            "1.0",
            "--score-weight",
            "0.0",
            "--prefer-detector-min-raw",
            "0.001",
        ],
        cwd=repo,
        check=True,
    )

    row = (out_dir / "frame_0001.txt").read_text(encoding="utf-8").strip().split()
    assert row[:5] == ["0", "0.50000000", "0.50000000", "0.20000000", "0.20000000"]
    assert row[5] == "0.80000000"


def test_select_candidate_jsonl_can_use_support_when_detector_filter_disabled(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    candidate_jsonl = tmp_path / "candidates.jsonl"
    image = tmp_path / "images" / "frame_0001.jpg"
    image.parent.mkdir(parents=True)
    image.write_text("", encoding="utf-8")
    candidate_jsonl.write_text(
        json.dumps(
            {
                "frame_id": 0,
                "image_path": str(image),
                "width": 100,
                "height": 100,
                "candidates": [
                    {
                        "rank": 0,
                        "x1": 10,
                        "y1": 10,
                        "x2": 20,
                        "y2": 20,
                        "score": 0.9,
                        "raw_objectness": 0.2,
                        "motion_memory_score": 1.0,
                        "source": "gray_ncc",
                    },
                    {
                        "rank": 1,
                        "x1": 40,
                        "y1": 40,
                        "x2": 60,
                        "y2": 60,
                        "score": 0.4,
                        "raw_objectness": 0.8,
                        "motion_memory_score": 0.0,
                        "source": "yolov5_dual",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "labels"
    subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "select_candidate_jsonl_to_yolo_labels.py"),
            "--candidate-jsonl",
            str(candidate_jsonl),
            "--out-label-dir",
            str(out_dir),
            "--raw-weight",
            "0.0",
            "--score-weight",
            "1.0",
        ],
        cwd=repo,
        check=True,
    )

    row = (out_dir / "frame_0001.txt").read_text(encoding="utf-8").strip().split()
    assert row[:5] == ["0", "0.15000000", "0.15000000", "0.10000000", "0.10000000"]
    assert row[5] == "0.90000000"
