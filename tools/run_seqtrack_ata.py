#!/usr/bin/env python3
"""Run an official SeqTrack checkpoint on the fixed ATA test split."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qstr_dronedet.ata_benchmark import EXPECTED_SEQUENCES, list_images, read_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--seqtrack-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(EXPECTED_SEQUENCES), default="test")
    return parser.parse_args()


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def validate_sequences(dataset_root: Path, split: str) -> list[tuple[str, list[Path], list[tuple[float, ...]]]]:
    sequences = []
    for name in EXPECTED_SEQUENCES[split]:
        sequence_root = dataset_root / split / name
        groundtruth_path = sequence_root / "groundtruth.txt"
        if not groundtruth_path.is_file():
            raise FileNotFoundError(f"Missing annotation: {groundtruth_path}")
        boxes = read_boxes(groundtruth_path)
        images = list_images(sequence_root / "img")
        if len(images) != len(boxes):
            raise ValueError(f"{split}/{name}: {len(images)} images for {len(boxes)} annotations")
        sequences.append((name, images, boxes))
    return sequences


def read_rgb(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> int:
    args = parse_args()
    sequences = validate_sequences(args.dataset_root.resolve(), args.split)
    progress_path = args.output_root / "progress.json"
    predictions_root = args.output_root / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)
    total_frames = sum(len(images) for _, images, _ in sequences)
    progress = {
        "status": "loading-model",
        "done_sequences": 0,
        "total_sequences": len(sequences),
        "done_frames": 0,
        "total_frames": total_frames,
        "last_completed_sequence": None,
        "last_frame": 0,
        "updated_at": time.time(),
    }
    write_json(progress_path, progress)

    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, str(args.seqtrack_root.resolve()))
    from lib.config.seqtrack.config import cfg, update_config_from_file
    from lib.test.tracker.seqtrack import SEQTRACK
    from lib.test.utils import TrackerParams

    update_config_from_file(str(args.config.resolve()))
    cfg.MODEL.ENCODER.PRETRAIN_TYPE = "scratch"
    params = TrackerParams()
    params.cfg = cfg
    params.checkpoint = str(args.checkpoint.resolve())
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.debug = 0
    params.save_all_boxes = False
    tracker = SEQTRACK(params, "ata")

    started = time.perf_counter()
    done_frames = 0
    for sequence_index, (name, images, boxes) in enumerate(sequences, 1):
        predictions = [boxes[0]]
        tracker.initialize(read_rgb(images[0]), {"init_bbox": list(boxes[0])})
        for frame_index, image_path in enumerate(images[1:], 2):
            output = tracker.track(read_rgb(image_path))
            predictions.append(tuple(float(value) for value in output["target_bbox"]))
            done_frames += 1
            if frame_index % 100 == 0:
                progress.update(
                    status="running",
                    done_frames=done_frames,
                    last_sequence=name,
                    last_frame=frame_index,
                    updated_at=time.time(),
                )
                write_json(progress_path, progress)
        prediction_path = predictions_root / f"{name}.txt"
        prediction_path.write_text(
            "\n".join(",".join(f"{value:.6f}" for value in box) for box in predictions) + "\n",
            encoding="ascii",
        )
        done_frames += 1
        progress.update(
            status="running",
            done_sequences=sequence_index,
            done_frames=done_frames,
            last_completed_sequence=name,
            last_frame=len(images),
            updated_at=time.time(),
        )
        write_json(progress_path, progress)

    progress.update(status="complete", elapsed_seconds=time.perf_counter() - started, updated_at=time.time())
    write_json(progress_path, progress)
    print(json.dumps(progress, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
