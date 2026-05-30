#!/usr/bin/env python3
"""Interpolate sparse single-object frame annotations into per-frame boxes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--tag", default="linear_interpolated")
    return parser.parse_args()


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def first_box(annotation: dict) -> dict:
    boxes = annotation.get("boxes") or []
    if len(boxes) != 1:
        raise ValueError(f"Expected exactly one box at frame {annotation.get('frame_id')}, got {len(boxes)}")
    return boxes[0]


def interpolate_annotations(data: dict) -> list[dict]:
    sparse = sorted(data["annotations"], key=lambda item: int(item["frame_id"]))
    if len(sparse) < 2:
        raise ValueError("Need at least two annotated frames to interpolate")

    output: list[dict] = []
    for left, right in zip(sparse, sparse[1:]):
        left_frame = int(left["frame_id"])
        right_frame = int(right["frame_id"])
        if right_frame <= left_frame:
            raise ValueError(f"Non-increasing frame ids: {left_frame}, {right_frame}")

        left_box = first_box(left)
        right_box = first_box(right)
        gap = right_frame - left_frame
        for frame_id in range(left_frame, right_frame):
            t = (frame_id - left_frame) / gap
            box = {
                "class": left_box.get("class", right_box.get("class", "drone")),
                "x1": round(lerp(float(left_box["x1"]), float(right_box["x1"]), t), 3),
                "y1": round(lerp(float(left_box["y1"]), float(right_box["y1"]), t), 3),
                "x2": round(lerp(float(left_box["x2"]), float(right_box["x2"]), t), 3),
                "y2": round(lerp(float(left_box["y2"]), float(right_box["y2"]), t), 3),
            }
            output.append(
                {
                    "video_name": data["video_name"],
                    "frame_id": frame_id,
                    "time_sec": frame_id / float(data["fps"]),
                    "width": data["width"],
                    "height": data["height"],
                    "boxes": [box],
                    "source": "manual" if frame_id == left_frame else "linear_interpolated",
                }
            )

    last = sparse[-1]
    output.append(
        {
            "video_name": data["video_name"],
            "frame_id": int(last["frame_id"]),
            "time_sec": int(last["frame_id"]) / float(data["fps"]),
            "width": data["width"],
            "height": data["height"],
            "boxes": [first_box(last)],
            "source": "manual",
        }
    )
    return output


def write_csv(path: Path, video_path: Path, source_json: Path, annotations: list[dict], tag: str) -> None:
    fields = ["video_path", "video_filename", "annotation_source", "task", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ann in annotations:
            for box in ann["boxes"]:
                writer.writerow(
                    {
                        "video_path": str(video_path),
                        "video_filename": video_path.name,
                        "annotation_source": str(source_json),
                        "task": tag,
                        "frame_id": ann["frame_id"],
                        "x1": box["x1"],
                        "y1": box["y1"],
                        "x2": box["x2"],
                        "y2": box["y2"],
                        "class": box.get("class", "drone"),
                        "tag": ann.get("source", tag),
                    }
                )


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json)
    video_path = Path(args.video_path)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    data = json.loads(input_json.read_text())
    dense_annotations = interpolate_annotations(data)
    dense = {
        "version": 1,
        "video_name": data["video_name"],
        "fps": data["fps"],
        "frame_step": 1,
        "original_frame_step": data.get("frame_step"),
        "total_frames": data["total_frames"],
        "width": data["width"],
        "height": data["height"],
        "interpolation": {
            "method": "linear_box_coordinates",
            "source_json": str(input_json),
            "first_frame": dense_annotations[0]["frame_id"],
            "last_frame": dense_annotations[-1]["frame_id"],
        },
        "annotations": dense_annotations,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(dense, indent=2) + "\n")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, video_path, input_json, dense_annotations, args.tag)
    print(f"{output_json}: {len(dense_annotations)} per-frame annotations")
    print(f"{output_csv}: {len(dense_annotations)} rows")


if __name__ == "__main__":
    main()
