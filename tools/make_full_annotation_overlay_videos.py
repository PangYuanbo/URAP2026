#!/usr/bin/env python3
"""Build full-length videos with annotation boxes held between sampled labels."""

from __future__ import annotations

import argparse
import csv
import subprocess
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
    parser.add_argument("--annotations", default="data/annotations_all_with_local_video_paths.csv")
    parser.add_argument("--output-dir", default="data/annotated_full_videos_hold")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument(
        "--hold-after-last",
        action="store_true",
        help="Keep the final labeled box until the end of the video.",
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


def draw_boxes(frame, boxes: list[dict[str, str]], x_scale: float, y_scale: float, stale: bool) -> None:
    for box in boxes:
        label = box["class"] if not stale else f"{box['class']} hold"
        color = COLORS.get(box["class"], (255, 255, 0))
        x1 = int(round(float(box["x1"]) * x_scale))
        y1 = int(round(float(box["y1"]) * y_scale))
        x2 = int(round(float(box["x2"]) * x_scale))
        y2 = int(round(float(box["y2"]) * y_scale))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        draw_label(frame, label, x1, y1, color)


def open_encoder(out_path: Path, fps: float, width: int, height: int) -> subprocess.Popen:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    return subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def write_full_video(
    video_path: Path,
    frames_to_boxes: dict[int, list[dict[str, str]]],
    out_path: Path,
    width: int,
    hold_after_last: bool,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(round(width * src_height / src_width))
    if height % 2:
        height += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = open_encoder(out_path, fps, width, height)
    assert encoder.stdin is not None

    labeled_frames = sorted(frames_to_boxes)
    next_label_index = 0
    current_boxes: list[dict[str, str]] = []
    x_scale = width / src_width
    y_scale = height / src_height
    written = 0

    for frame_id in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break

        while next_label_index < len(labeled_frames) and labeled_frames[next_label_index] <= frame_id:
            current_boxes = frames_to_boxes[labeled_frames[next_label_index]]
            next_label_index += 1

        after_last_label = next_label_index >= len(labeled_frames)
        if after_last_label and not hold_after_last and labeled_frames and frame_id > labeled_frames[-1]:
            active_boxes = []
        else:
            active_boxes = current_boxes

        preview = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if active_boxes:
            stale = frame_id not in frames_to_boxes
            draw_boxes(preview, active_boxes, x_scale, y_scale, stale)
        stamp = f"{video_path.name} | frame {frame_id}/{frame_count} | t={frame_id / fps:.2f}s"
        cv2.rectangle(preview, (0, 0), (width, 42), (0, 0, 0), -1)
        cv2.putText(preview, stamp, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        encoder.stdin.write(preview.tobytes())
        written += 1

    encoder.stdin.close()
    return_code = encoder.wait()
    cap.release()
    if return_code:
        raise RuntimeError(f"ffmpeg failed for {out_path} with code {return_code}")
    return written


def main() -> None:
    args = parse_args()
    annotations = read_annotations(Path(args.annotations))
    output_dir = Path(args.output_dir)
    for video, frames_to_boxes in sorted(annotations.items()):
        video_path = Path(video)
        out_path = output_dir / f"{video_path.stem}_full_hold_overlay.mp4"
        written = write_full_video(video_path, frames_to_boxes, out_path, args.width, args.hold_after_last)
        print(f"{out_path}: {written} frames")


if __name__ == "__main__":
    main()
