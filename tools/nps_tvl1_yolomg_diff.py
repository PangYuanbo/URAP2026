import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Use the NPS paper Dual TV-L1 flow head with the YOLOMG RANSAC/H/difference tail.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--process-width", type=int, default=960)
    parser.add_argument("--display-width", type=int, default=480)
    parser.add_argument("--progress-json", type=Path)
    return parser.parse_args()


def resize_width(image, width):
    height, source_width = image.shape[:2]
    target_height = max(2, int(round(height * width / source_width)))
    target_height += target_height % 2
    return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)


def create_tvl1():
    if hasattr(cv2, "DualTVL1OpticalFlow_create"):
        return cv2.DualTVL1OpticalFlow_create()
    if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
        return cv2.optflow.DualTVL1OpticalFlow_create()
    raise RuntimeError("Dual TV-L1 is unavailable; install opencv-contrib-python.")


def motion_boundary(flow):
    horizontal = flow[..., 0].astype(np.float32)
    vertical = flow[..., 1].astype(np.float32)
    horizontal_x, horizontal_y = np.gradient(horizontal)
    vertical_x, vertical_y = np.gradient(vertical)
    horizontal_magnitude = np.sqrt(np.square(horizontal_x) + np.square(horizontal_y))
    vertical_magnitude = np.sqrt(np.square(vertical_x) + np.square(vertical_y))
    boundary = np.maximum(horizontal_magnitude, vertical_magnitude)
    return cv2.normalize(boundary, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def homography_from_flow(flow):
    height, width = flow.shape[:2]
    step = max(8, min(width, height) // 45)
    ys, xs = np.mgrid[step:height-step:step, step:width-step:step]
    source = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    vectors = flow[ys, xs].reshape(-1, 2).astype(np.float32)
    destination = source + vectors
    finite = np.isfinite(destination).all(axis=1)
    source, destination = source[finite], destination[finite]
    magnitude = np.linalg.norm(vectors[finite], axis=1)
    keep = magnitude < max(width, height) * 0.15
    source, destination = source[keep], destination[keep]
    homography, mask = cv2.findHomography(destination, source, cv2.RANSAC, 3.0)
    if homography is None:
        return np.eye(3, dtype=np.float64), 0, len(source)
    return homography, int(mask.sum()) if mask is not None else 0, len(source)


def align(source_gray, reference_gray, flow):
    homography, inliers, total = homography_from_flow(flow)
    size = (reference_gray.shape[1], reference_gray.shape[0])
    aligned = cv2.warpPerspective(source_gray, homography, size, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    valid = cv2.warpPerspective(np.full_like(source_gray, 255), homography, size, flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
    return aligned, valid, inliers, total


def heat_color(gray):
    high = max(float(np.percentile(gray, 99.7)), 1.0)
    normalized = np.uint8(np.power(np.clip(gray.astype(np.float32) / high, 0.0, 1.0), 0.55) * 255.0)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def label(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 52), (0, 0, 0), -1)
    cv2.putText(image, text, (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def write_progress(path, payload):
    if path is None:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 29.97
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(args.start_seconds * fps)))
    requested = int(round(args.duration_seconds * fps)) if args.duration_seconds > 0 else int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_frames = max(0, requested - 2)
    frames = []
    for _ in range(3):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Not enough input frames")
        frames.append(resize_width(frame, args.process_width))
    sample = resize_width(frames[1], args.display_width)
    output_path = args.output_dir / f"{args.input.stem}_nps_tvl1_yolomg_diff.avi"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (sample.shape[1] * 4, sample.shape[0]))
    thumbnail_path = args.output_dir / f"{args.input.stem}_nps_tvl1_yolomg_diff.jpg"
    previous_engine = create_tvl1()
    following_engine = create_tvl1()
    done = 0
    started = time.time()
    while done < output_frames:
        previous, reference, following = frames
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        following_gray = cv2.cvtColor(following, cv2.COLOR_BGR2GRAY)
        previous_flow = previous_engine.calc(previous_gray, reference_gray, None)
        following_flow = following_engine.calc(following_gray, reference_gray, None)
        aligned_previous, valid_previous, previous_inliers, previous_total = align(previous_gray, reference_gray, previous_flow)
        aligned_following, valid_following, following_inliers, following_total = align(following_gray, reference_gray, following_flow)
        valid = cv2.bitwise_and(valid_previous, valid_following)
        raw_difference = cv2.absdiff(reference_gray, previous_gray)
        compensated = ((cv2.absdiff(reference_gray, aligned_previous).astype(np.float32) + cv2.absdiff(reference_gray, aligned_following).astype(np.float32)) / 2.0).astype(np.uint8)
        compensated[valid == 0] = 0
        boundary = motion_boundary(previous_flow)
        rgb_panel = label(resize_width(reference, args.display_width), "RGB")
        boundary_panel = label(resize_width(heat_color(boundary), args.display_width), "NPS head: Dual TV-L1 motion boundary")
        raw_panel = label(resize_width(heat_color(raw_difference), args.display_width), "Raw frame difference")
        compensated_panel = label(resize_width(heat_color(compensated), args.display_width), "NPS flow + YOLOMG RANSAC/H difference")
        combined = np.concatenate([rgb_panel, boundary_panel, raw_panel, compensated_panel], axis=1)
        writer.write(combined)
        if done == min(30, max(0, output_frames - 1)):
            cv2.imwrite(str(thumbnail_path), combined)
        done += 1
        if done % 5 == 0 or done == output_frames:
            write_progress(args.progress_json, {"status": "running", "done": done, "total": output_frames, "output": str(output_path), "last_output_timestamp": time.time(), "previous_inliers": previous_inliers, "previous_points": previous_total, "following_inliers": following_inliers, "following_points": following_total, "elapsed_seconds": time.time() - started})
            print(f"[{args.input.name}] {done}/{output_frames}", flush=True)
        frames = frames[1:]
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(resize_width(frame, args.process_width))
    writer.release()
    capture.release()
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps({"method": "NPS paper Dual TV-L1 dense flow head and motion boundary; YOLOMG RANSAC homography, warp, and bidirectional compensated difference tail", "input": str(args.input), "output": str(output_path), "thumbnail": str(thumbnail_path), "frames": done, "fps": fps}, indent=2), encoding="utf-8")
    write_progress(args.progress_json, {"status": "completed", "done": done, "total": done, "manifest": str(manifest), "last_output_timestamp": time.time()})
    print(f"[DONE] {manifest}", flush=True)


if __name__ == "__main__":
    main()