import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Stack original YOLOMG and NPS-flow YOLOMG videos vertically.")
    parser.add_argument("--top", type=Path, required=True)
    parser.add_argument("--bottom", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path)
    return parser.parse_args()


def title_bar(width, text, color):
    bar = np.zeros((46, width, 3), dtype=np.uint8)
    cv2.putText(bar, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
    return bar


def write_progress(path, payload):
    if path is None:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    top_capture = cv2.VideoCapture(str(args.top))
    bottom_capture = cv2.VideoCapture(str(args.bottom))
    if not top_capture.isOpened() or not bottom_capture.isOpened():
        raise RuntimeError("Could not open comparison inputs")
    top_frames = int(top_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    bottom_frames = int(bottom_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(top_frames, bottom_frames)
    fps = min(top_capture.get(cv2.CAP_PROP_FPS), bottom_capture.get(cv2.CAP_PROP_FPS)) or 29.97
    width = max(int(top_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(bottom_capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    top_height = int(top_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bottom_height = int(bottom_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_height = 46 + top_height + 46 + bottom_height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, output_height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {args.output}")
    top_bar = title_bar(width, "TOP: Original YOLOMG - PyrLK + RANSAC/H + compensated difference", (255, 255, 255))
    bottom_bar = title_bar(width, "BOTTOM: NPS Dual TV-L1 head + YOLOMG RANSAC/H + compensated difference", (0, 255, 255))
    done = 0
    started = time.time()
    while done < total:
        top_ok, top_frame = top_capture.read()
        bottom_ok, bottom_frame = bottom_capture.read()
        if not top_ok or not bottom_ok:
            break
        if top_frame.shape[1] != width:
            top_frame = cv2.resize(top_frame, (width, top_height), interpolation=cv2.INTER_AREA)
        if bottom_frame.shape[1] != width:
            bottom_frame = cv2.resize(bottom_frame, (width, bottom_height), interpolation=cv2.INTER_AREA)
        combined = np.concatenate([top_bar, top_frame, bottom_bar, bottom_frame], axis=0)
        writer.write(combined)
        done += 1
        if done % 100 == 0 or done == total:
            write_progress(args.progress_json, {"status": "merging", "done": done, "total": total, "output": str(args.output), "last_output_timestamp": time.time(), "elapsed_seconds": time.time() - started})
            print(f"{done}/{total}", flush=True)
    writer.release()
    top_capture.release()
    bottom_capture.release()
    write_progress(args.progress_json, {"status": "completed", "done": done, "total": total, "output": str(args.output), "last_output_timestamp": time.time()})


if __name__ == "__main__":
    main()