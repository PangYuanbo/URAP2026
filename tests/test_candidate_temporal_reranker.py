from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_frame(f, image: Path, candidates: list[dict]) -> None:
    f.write(
        json.dumps(
            {
                "frame_id": 0,
                "image_path": str(image),
                "width": 100,
                "height": 100,
                "candidates": candidates,
            }
        )
        + "\n"
    )


def test_candidate_temporal_reranker_trains_and_emits_top1_labels(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    train_image = tmp_path / "images" / "train" / "frame_train.jpg"
    test_image = tmp_path / "images" / "val" / "frame_test.jpg"
    train_image.parent.mkdir(parents=True)
    test_image.parent.mkdir(parents=True)
    train_image.write_text("", encoding="utf-8")
    test_image.write_text("", encoding="utf-8")
    train_label = tmp_path / "labels" / "train" / "frame_train.txt"
    train_label.parent.mkdir(parents=True)
    train_label.write_text("0 0.50000000 0.50000000 0.20000000 0.20000000\n", encoding="utf-8")
    train_jsonl = tmp_path / "train_candidates.jsonl"
    test_jsonl = tmp_path / "test_candidates.jsonl"
    good = {
        "rank": 1,
        "x1": 40,
        "y1": 40,
        "x2": 60,
        "y2": 60,
        "score": 0.45,
        "raw_objectness": 0.80,
        "detector_raw_objectness": 0.80,
        "motion_memory_score": 0.75,
        "source": "yolov5_dual",
        "has_detector_member": True,
    }
    bad = {
        "rank": 0,
        "x1": 5,
        "y1": 5,
        "x2": 15,
        "y2": 15,
        "score": 0.90,
        "raw_objectness": 0.20,
        "detector_raw_objectness": 0.20,
        "motion_memory_score": 1.0,
        "source": "gray_ncc",
    }
    with train_jsonl.open("w", encoding="utf-8") as f:
        _write_frame(f, train_image, [bad, good])
    with test_jsonl.open("w", encoding="utf-8") as f:
        _write_frame(f, test_image, [bad, good])

    out_labels = tmp_path / "pred_labels"
    subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "train_candidate_temporal_reranker.py"),
            "--train-candidate-jsonl",
            str(train_jsonl),
            "--test-candidate-jsonl",
            str(test_jsonl),
            "--out-label-dir",
            str(out_labels),
            "--out-model",
            str(tmp_path / "model.npz"),
            "--out-summary",
            str(tmp_path / "summary.json"),
            "--epochs",
            "4",
            "--batch-size",
            "2",
            "--lr",
            "0.05",
        ],
        cwd=repo,
        check=True,
    )

    row = (out_labels / "frame_test.txt").read_text(encoding="utf-8").strip().split()
    assert row[:5] == ["0", "0.50000000", "0.50000000", "0.20000000", "0.20000000"]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["train_candidates"] == 2
    assert summary["train_positive"] == 1
