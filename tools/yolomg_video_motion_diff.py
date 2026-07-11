import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate YOLOMG-style compensated motion-difference videos from MP4 files.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=10.0, help="0 processes the remainder of each video.")
    parser.add_argument("--display-width", type=int, default=960)
    parser.add_argument("--process-width", type=int, default=1280)
    parser.add_argument("--progress-json", type=Path)
    return parser.parse_args()


def resize_width(image, width):
    height, source_width = image.shape[:2]
    target_height = max(2, int(round(height * width / source_width)))
    if target_height % 2:
        target_height += 1
    return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)


def grid_points(width, height):
    spacing_x = max(32, width // 30)
    spacing_y = max(24, height // 22)
    return np.asarray([(np.float32(x), np.float32(y)) for y in range(spacing_y, height - spacing_y, spacing_y) for x in range(spacing_x, width - spacing_x, spacing_x)], dtype=np.float32).reshape(-1, 1, 2)


def estimate_homography(source_gray, reference_gray):
    points = grid_points(reference_gray.shape[1], reference_gray.shape[0])
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(source_gray, reference_gray, points, None, winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003))
    if tracked is None or status is None:
        return np.eye(3, dtype=np.float64), 0
    valid_source = points[status.ravel() == 1].reshape(-1, 2)
    valid_reference = tracked[status.ravel() == 1].reshape(-1, 2)
    if len(valid_source) < 15:
        return np.eye(3, dtype=np.float64), len(valid_source)
    keep = np.linalg.norm(valid_reference - valid_source, axis=1) < 50.0
    valid_source = valid_source[keep]
    valid_reference = valid_reference[keep]
    if len(valid_source) < 15:
        return np.eye(3, dtype=np.float64), len(valid_source)
    homography, _ = cv2.findHomography(valid_reference, valid_source, cv2.RANSAC, 3.0)
    if homography is None or not np.isfinite(homography).all():
        return np.eye(3, dtype=np.float64), len(valid_source)
    return homography, len(valid_source)


def compensate(source_gray, reference_gray):
    homography, tracked_points = estimate_homography(source_gray, reference_gray)
    aligned = cv2.warpPerspective(source_gray, homography, (reference_gray.shape[1], reference_gray.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned, tracked_points


def colorize(gray):
    low, high = np.percentile(gray, [1.0, 99.7])
    if high <= low:
        enhanced = np.zeros_like(gray)
    else:
        normalized = np.clip((gray.astype(np.float32) - low) / (high - low), 0.0, 1.0)
        enhanced = np.uint8(np.power(normalized, 0.55) * 255.0)
    return cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)


def label(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(image, text, (18, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def write_progress(path, payload):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def process_video(input_path, args, video_index):
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 29.97
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(round(args.start_seconds * fps)))
    available = max(0, source_frames - start_frame)
    requested = available if args.duration_seconds <= 0 else int(round(args.duration_seconds * fps))
    output_frames = max(0, min(available, requested) - 2)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(3):
        ok, frame = capture.read()
        if ok:
            frames.append(resize_width(frame, args.process_width))
    if len(frames) < 3:
        capture.release()
        raise RuntimeError(f"Not enough frames in {input_path}")
    panel = resize_width(frames[1], args.display_width)
    output_path = args.output_dir / f"{input_path.stem}_yolomg_motion_diff.avi"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (panel.shape[1] * 3, panel.shape[0]))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output: {output_path}")
    thumbnail_path = args.output_dir / f"{input_path.stem}_yolomg_motion_diff.jpg"
    processed = 0
    started_at = time.time()
    while len(frames) == 3 and processed < output_frames:
        previous_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frames[2], cv2.COLOR_BGR2GRAY)
        aligned_previous, previous_points = compensate(previous_gray, reference_gray)
        aligned_next, next_points = compensate(next_gray, reference_gray)
        difference = ((cv2.absdiff(reference_gray, aligned_previous).astype(np.float32) + cv2.absdiff(reference_gray, aligned_next).astype(np.float32)) / 2.0).astype(np.uint8)
        rgb = label(resize_width(frames[1], args.display_width), "RGB")
        motion = resize_width(colorize(difference), args.display_width)
        overlay = cv2.addWeighted(resize_width(frames[1], args.display_width), 0.62, motion, 0.38, 0.0)
        combined = np.concatenate([rgb, label(motion, "YOLOMG compensated frame difference"), label(overlay, "Motion overlay")], axis=1)
        writer.write(combined)
        if processed == min(30, max(0, output_frames - 1)):
            cv2.imwrite(str(thumbnail_path), combined)
        processed += 1
        if processed % 10 == 0 or processed == output_frames:
            write_progress(args.progress_json, {"status": "running", "video": input_path.name, "video_index": video_index, "video_total": len(args.inputs), "done": processed, "total": output_frames, "last_output": str(output_path), "last_output_timestamp": time.time(), "tracked_points_previous": previous_points, "tracked_points_next": next_points, "elapsed_seconds": time.time() - started_at})
            print(f"[{input_path.name}] {processed}/{output_frames}", flush=True)
        frames = frames[1:]
        if processed < output_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(resize_width(frame, args.process_width))
    writer.release()
    capture.release()
    return {"input": str(input_path), "output": str(output_path), "thumbnail": str(thumbnail_path), "frames": processed, "fps": fps}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [process_video(path, args, index) for index, path in enumerate(args.inputs, start=1)]
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps({"method": "YOLOMG PyrLK + RANSAC homography + bidirectional compensated frame difference", "results": results}, indent=2), encoding="utf-8")
    write_progress(args.progress_json, {"status": "completed", "done": len(results), "total": len(results), "manifest": str(manifest), "last_output_timestamp": time.time()})
    print(f"[DONE] {manifest}", flush=True)


if __name__ == "__main__":
    main()