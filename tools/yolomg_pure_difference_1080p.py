import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a pure 1080p YOLOMG compensated-difference video.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--progress-json", type=Path)
    return parser.parse_args()


def resize_1080p(frame):
    return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)


def grid_points(width, height):
    spacing_x = max(32, width // 30)
    spacing_y = max(24, height // 22)
    return np.asarray(
        [(np.float32(x), np.float32(y)) for y in range(spacing_y, height - spacing_y, spacing_y) for x in range(spacing_x, width - spacing_x, spacing_x)],
        dtype=np.float32,
    ).reshape(-1, 1, 2)


def estimate_homography(source_gray, reference_gray):
    points = grid_points(reference_gray.shape[1], reference_gray.shape[0])
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        source_gray,
        reference_gray,
        points,
        None,
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003),
    )
    if tracked is None or status is None:
        return np.eye(3, dtype=np.float64), 0
    source = points[status.ravel() == 1].reshape(-1, 2)
    destination = tracked[status.ravel() == 1].reshape(-1, 2)
    if len(source) < 15:
        return np.eye(3, dtype=np.float64), len(source)
    keep = np.linalg.norm(destination - source, axis=1) < 75.0
    source = source[keep]
    destination = destination[keep]
    if len(source) < 15:
        return np.eye(3, dtype=np.float64), len(source)
    homography, _ = cv2.findHomography(destination, source, cv2.RANSAC, 3.0)
    if homography is None or not np.isfinite(homography).all():
        return np.eye(3, dtype=np.float64), len(source)
    return homography, len(source)


def compensate(source_gray, reference_gray):
    homography, tracked_points = estimate_homography(source_gray, reference_gray)
    aligned = cv2.warpPerspective(
        source_gray,
        homography,
        (reference_gray.shape[1], reference_gray.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    valid = cv2.warpPerspective(
        np.full_like(source_gray, 255),
        homography,
        (reference_gray.shape[1], reference_gray.shape[0]),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
    )
    return aligned, valid, tracked_points


def colorize_difference(gray):
    low, high = np.percentile(gray, [1.0, 99.7])
    if high <= low:
        normalized = np.zeros_like(gray)
    else:
        values = np.clip((gray.astype(np.float32) - low) / (high - low), 0.0, 1.0)
        normalized = np.uint8(np.power(values, 0.55) * 255.0)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def write_progress(path, payload):
    if path is None:
        return
    encoded = json.dumps(payload, indent=2)
    temporary = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(20):
        try:
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.25 * (attempt + 1))
    path.write_text(encoded, encoding="utf-8")


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 29.97
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(round(args.start_seconds * fps)))
    available = max(0, source_frames - start_frame)
    requested = available if args.duration_seconds <= 0 else min(available, int(round(args.duration_seconds * fps)))
    total = max(0, requested - 2)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(3):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Not enough frames in {args.input}")
        frames.append(resize_1080p(frame))
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output: {args.output}")
    done = 0
    started = time.time()
    while done < total:
        previous_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
        following_gray = cv2.cvtColor(frames[2], cv2.COLOR_BGR2GRAY)
        previous_aligned, previous_valid, previous_points = compensate(previous_gray, reference_gray)
        following_aligned, following_valid, following_points = compensate(following_gray, reference_gray)
        valid = cv2.bitwise_and(previous_valid, following_valid)
        difference = (
            cv2.absdiff(reference_gray, previous_aligned).astype(np.float32)
            + cv2.absdiff(reference_gray, following_aligned).astype(np.float32)
        ) / 2.0
        difference = difference.astype(np.uint8)
        difference[valid == 0] = 0
        writer.write(colorize_difference(difference))
        done += 1
        if done % 25 == 0 or done == total:
            write_progress(
                args.progress_json,
                {
                    "status": "running",
                    "done": done,
                    "total": total,
                    "input": str(args.input),
                    "output": str(args.output),
                    "resolution": "1920x1080",
                    "previous_points": previous_points,
                    "following_points": following_points,
                    "elapsed_seconds": time.time() - started,
                    "last_output_timestamp": time.time(),
                },
            )
            print(f"[{args.input.name}] {done}/{total}", flush=True)
        frames = frames[1:]
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(resize_1080p(frame))
    writer.release()
    capture.release()
    manifest = args.output.with_suffix(".json")
    manifest.write_text(
        json.dumps(
            {
                "method": "YOLOMG PyrLK + RANSAC homography + bidirectional compensated difference",
                "style": "pure Inferno-colored compensated difference",
                "resolution": "1920x1080",
                "input": str(args.input),
                "output": str(args.output),
                "frames": done,
                "fps": fps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_progress(args.progress_json, {"status": "completed", "done": done, "total": done, "manifest": str(manifest), "output": str(args.output), "last_output_timestamp": time.time()})
    print(f"[DONE] {args.output}", flush=True)


if __name__ == "__main__":
    main()