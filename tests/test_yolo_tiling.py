import csv
from pathlib import Path

import cv2
import numpy as np
import yaml

from qstr_dronedet.candidates.yolo_p2_train import build_tiled_class_agnostic_yolo_dataset


def test_build_tiled_yolo_dataset_with_empty_negatives(tmp_path: Path):
    frame = np.full((192, 192, 3), 180, np.uint8)
    cv2.circle(frame, (32, 32), 2, (0, 0, 0), -1)
    frame_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(frame_path), frame)
    csv_path = tmp_path / "ann.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "x1", "y1", "x2", "y2", "class"])
        writer.writeheader()
        writer.writerow({"frame_path": str(frame_path), "x1": 30, "y1": 30, "x2": 34, "y2": 34, "class": "object"})

    data_yaml = build_tiled_class_agnostic_yolo_dataset(
        csv_path,
        tmp_path / "yolo",
        tile_size=64,
        positives_per_box=1,
        negatives_per_image=1,
        val_fraction=0.0,
        min_box_px=8,
        negative_pad_px=0,
        seed=3,
    )

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert config["nc"] == 1
    labels = sorted((tmp_path / "yolo" / "labels" / "train").glob("*.txt"))
    assert len(labels) == 2
    contents = [p.read_text(encoding="utf-8").strip() for p in labels]
    assert any(c.startswith("0 ") for c in contents)
    assert any(c == "" for c in contents)
    manifest_rows = (tmp_path / "yolo" / "tile_manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(manifest_rows) == 2
