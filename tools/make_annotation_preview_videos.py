#!/usr/bin/env python3
"""Build short videos that show annotated frames with boxes overlaid."""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import imageio_ffmpeg


COLORS = {
    "drone": (0, 255, 0),
    "ground_object": (0, 180, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        default="data/annotations_all_with_local_video_paths.csv",
        help="CSV with local video_path, frame_id, box coordinates, class, and tag.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/annotated_previews",
        help="Directory for annotated preview MP4 files.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Preview playback FPS. Low FPS makes sampled-frame labels easy to inspect.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Preview video width. Height is computed from source aspect ratio.",
    )
    return parser.parse_args()


def read_annotations(path: Path) -> dict[str, dict[int, list[dict[str, str]]]]:
    by_video: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            by_video[row["video_path"]][int(float(row["frame_id"]))].append(row)
    return by_video


def draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - th - baseline - 8)
    left = max(0, x)
    cv2.rectangle(frame, (left, top), (left + tw + 8, top + th + baseline + 8), color, -1)
    cv2.putText(frame, text, (left + 4, top + th + 3), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_boxes(frame, boxes: list[dict[str, str]], x_scale: float, y_scale: float) -> None:
    for box in boxes:
        label = box["class"]
        color = COLORS.get(label, (255, 255, 0))
        x1 = int(round(float(box["x1"]) * x_scale))
        y1 = int(round(float(box["y1"]) * y_scale))
        x2 = int(round(float(box["x2"]) * x_scale))
        y2 = int(round(float(box["y2"]) * y_scale))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        draw_label(frame, label, x1, y1, color)


def encode_frames(frame_dir: Path, out_path: Path, fps: float) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%06d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def write_preview(video_path: Path, frames_to_boxes: dict[int, list[dict[str, str]]], out_path: Path, fps: float, width: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    height = int(round(width * src_height / src_width))
    if height % 2:
        height += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_scale = width / src_width
    y_scale = height / src_height
    written = 0
    with tempfile.TemporaryDirectory(prefix="annotation_preview_") as tmp:
        frame_dir = Path(tmp)
        for frame_id in sorted(frames_to_boxes):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                print(f"warning: could not read {video_path.name} frame {frame_id}")
                continue

            preview = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            draw_boxes(preview, frames_to_boxes[frame_id], x_scale, y_scale)
            stamp = f"{video_path.name} | frame {frame_id} | t={frame_id / source_fps:.2f}s"
            cv2.rectangle(preview, (0, 0), (width, 42), (0, 0, 0), -1)
            cv2.putText(preview, stamp, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(frame_dir / f"frame_{written:06d}.jpg"), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            written += 1

        if written:
            encode_frames(frame_dir, out_path, fps)

    cap.release()
    return written


def main() -> None:
    args = parse_args()
    annotations = read_annotations(Path(args.annotations))
    output_dir = Path(args.output_dir)
    for video, frames_to_boxes in sorted(annotations.items()):
        video_path = Path(video)
        out_path = output_dir / f"{video_path.stem}_annotated_preview.mp4"
        written = write_preview(video_path, frames_to_boxes, out_path, args.fps, args.width)
        print(f"{out_path}: {written} annotated frames")


if __name__ == "__main__":
    main()
