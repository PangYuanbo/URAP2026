from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def normalize_clip_name(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("empty clip value")
    if value.lower().startswith("clip_"):
        return f"Clip_{int(value.split('_', 1)[1]):03d}"
    return f"Clip_{int(value):03d}"


def parse_clips(value: str) -> set[str]:
    return {normalize_clip_name(part) for part in value.replace(";", ",").split(",") if part.strip()}


def max_scored_frames(csv_path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            video = row["video"]
            frame = int(row["frame"])
            result[video] = max(result.get(video, 0), frame)
    return result


def even(value: int) -> int:
    return max(2, value - (value % 2))


def proxy_one(src: Path, out: Path, max_frame: int, pad_seconds: float, max_width: int, codec: str) -> int:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {src}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if src_w <= 0 or src_h <= 0:
        raise RuntimeError(f"could not read dimensions from {src}")

    target_frames = max_frame + int(round(pad_seconds * fps))
    if total_frames > 0:
        target_frames = min(total_frames, target_frames)

    scale = min(1.0, max_width / float(src_w))
    out_w = even(int(round(src_w * scale)))
    out_h = even(int(round(src_h * scale)))

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*codec), fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not create {out} with codec {codec}")

    written = 0
    while written < target_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if (out_w, out_h) != (src_w, src_h):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--video-root", type=Path, default=Path("Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Videos"))
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--clips", required=True, help="Comma-separated clip ids, e.g. 1,10,15")
    p.add_argument("--ext", default="webm")
    p.add_argument("--codec", default="VP80")
    p.add_argument("--max-width", type=int, default=1280)
    p.add_argument("--pad-seconds", type=float, default=2.0)
    args = p.parse_args()

    wanted = parse_clips(args.clips)
    max_frames = max_scored_frames(args.csv)
    ext = args.ext.lstrip(".")

    for clip in sorted(wanted):
        if clip not in max_frames:
            raise SystemExit(f"{clip} not found in {args.csv}")
        num = int(clip.split("_", 1)[1])
        src = args.video_root / f"Clip_{num}.mov"
        out = args.out_root / f"Clip_{num}.{ext}"
        written = proxy_one(src, out, max_frames[clip], args.pad_seconds, args.max_width, args.codec)
        print(f"{clip}: wrote {written} frames -> {out}")


if __name__ == "__main__":
    main()
