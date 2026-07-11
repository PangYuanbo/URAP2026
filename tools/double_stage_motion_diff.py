import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional


def parse_args():
    parser = argparse.ArgumentParser(description="Apply SEA-RAFT and compensated differencing to first-stage YOLOMG difference maps.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sea-raft-root", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--model-url", default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M")
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


def grid_points(width, height):
    spacing_x = max(32, width // 30)
    spacing_y = max(24, height // 22)
    return np.asarray([(np.float32(x), np.float32(y)) for y in range(spacing_y, height - spacing_y, spacing_y) for x in range(spacing_x, width - spacing_x, spacing_x)], dtype=np.float32).reshape(-1, 1, 2)


def pyr_homography(source_gray, reference_gray):
    points = grid_points(reference_gray.shape[1], reference_gray.shape[0])
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(source_gray, reference_gray, points, None, winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003))
    if tracked is None or status is None:
        return np.eye(3, dtype=np.float64)
    source = points[status.ravel() == 1].reshape(-1, 2)
    destination = tracked[status.ravel() == 1].reshape(-1, 2)
    if len(source) < 15:
        return np.eye(3, dtype=np.float64)
    keep = np.linalg.norm(destination - source, axis=1) < 50.0
    source, destination = source[keep], destination[keep]
    if len(source) < 15:
        return np.eye(3, dtype=np.float64)
    homography, _ = cv2.findHomography(destination, source, cv2.RANSAC, 3.0)
    return homography if homography is not None else np.eye(3, dtype=np.float64)


def warp_and_mask(source_gray, reference_gray, homography):
    size = (reference_gray.shape[1], reference_gray.shape[0])
    aligned = cv2.warpPerspective(source_gray, homography, size, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    valid = cv2.warpPerspective(np.full_like(source_gray, 255), homography, size, flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
    return aligned, valid


def first_stage_difference(previous_bgr, reference_bgr, following_bgr):
    previous = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
    reference = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    following = cv2.cvtColor(following_bgr, cv2.COLOR_BGR2GRAY)
    previous_h = pyr_homography(previous, reference)
    following_h = pyr_homography(following, reference)
    aligned_previous, valid_previous = warp_and_mask(previous, reference, previous_h)
    aligned_following, valid_following = warp_and_mask(following, reference, following_h)
    valid = cv2.bitwise_and(valid_previous, valid_following)
    difference = ((cv2.absdiff(reference, aligned_previous).astype(np.float32) + cv2.absdiff(reference, aligned_following).astype(np.float32)) / 2.0).astype(np.uint8)
    difference[valid == 0] = 0
    return difference


def load_model(args):
    sys.path.insert(0, str(args.sea_raft_root))
    sys.path.insert(0, str(args.sea_raft_root / "core"))
    from config.parser import json_to_args
    from raft import RAFT
    sea_args = json_to_args(str(args.cfg))
    sea_args.url = args.model_url
    return RAFT.from_pretrained(args.model_url, args=sea_args).cuda().eval(), sea_args


@torch.inference_mode()
def infer_flow(model, sea_args, first_gray, second_gray):
    first_rgb = cv2.cvtColor(first_gray, cv2.COLOR_GRAY2RGB)
    second_rgb = cv2.cvtColor(second_gray, cv2.COLOR_GRAY2RGB)
    first = torch.from_numpy(first_rgb).permute(2, 0, 1).float()[None].cuda()
    second = torch.from_numpy(second_rgb).permute(2, 0, 1).float()[None].cuda()
    scaled_first = functional.interpolate(first, scale_factor=2 ** sea_args.scale, mode="bilinear", align_corners=False)
    scaled_second = functional.interpolate(second, scale_factor=2 ** sea_args.scale, mode="bilinear", align_corners=False)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(scaled_first, scaled_second, iters=sea_args.iters, test_mode=True)
    flow = functional.interpolate(output["flow"][-1].float(), size=first.shape[-2:], mode="bilinear", align_corners=False) * (0.5 ** sea_args.scale)
    return flow[0].permute(1, 2, 0).cpu().numpy()


def flow_homography(flow):
    height, width = flow.shape[:2]
    step = max(8, min(width, height) // 45)
    ys, xs = np.mgrid[step:height-step:step, step:width-step:step]
    source = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    destination = source + flow[ys, xs].reshape(-1, 2).astype(np.float32)
    finite = np.isfinite(destination).all(axis=1)
    source, destination = source[finite], destination[finite]
    homography, mask = cv2.findHomography(destination, source, cv2.RANSAC, 2.5)
    if homography is None:
        return np.eye(3, dtype=np.float64), 0, len(source)
    return homography, int(mask.sum()) if mask is not None else 0, len(source)


def second_stage_difference(previous_diff, reference_diff, following_diff, model, sea_args):
    previous_flow = infer_flow(model, sea_args, previous_diff, reference_diff)
    following_flow = infer_flow(model, sea_args, following_diff, reference_diff)
    previous_h, previous_inliers, previous_total = flow_homography(previous_flow)
    following_h, following_inliers, following_total = flow_homography(following_flow)
    aligned_previous, valid_previous = warp_and_mask(previous_diff, reference_diff, previous_h)
    aligned_following, valid_following = warp_and_mask(following_diff, reference_diff, following_h)
    valid = cv2.bitwise_and(valid_previous, valid_following)
    difference = ((cv2.absdiff(reference_diff, aligned_previous).astype(np.float32) + cv2.absdiff(reference_diff, aligned_following).astype(np.float32)) / 2.0).astype(np.uint8)
    difference[valid == 0] = 0
    return difference, previous_flow, previous_inliers + following_inliers, previous_total + following_total


def flow_color(flow):
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    scale = max(float(np.percentile(magnitude, 99.0)), 1e-6)
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.uint8(angle * 90.0 / np.pi)
    hsv[..., 1] = 255
    hsv[..., 2] = np.uint8(np.clip(magnitude / scale, 0.0, 1.0) * 255.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def heat_color(gray):
    high = max(float(np.percentile(gray, 99.7)), 1.0)
    normalized = np.uint8(np.power(np.clip(gray.astype(np.float32) / high, 0.0, 1.0), 0.55) * 255.0)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def label(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 52), (0, 0, 0), -1)
    cv2.putText(image, text, (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def write_progress(path, payload):
    if path is None:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, sea_args = load_model(args)
    capture = cv2.VideoCapture(str(args.input))
    fps = capture.get(cv2.CAP_PROP_FPS) or 29.97
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(args.start_seconds * fps)))
    requested = int(round(args.duration_seconds * fps)) if args.duration_seconds > 0 else int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_frames = max(0, requested - 4)
    raw_frames = []
    for _ in range(5):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Not enough input frames")
        raw_frames.append(resize_width(frame, args.process_width))
    first_diffs = [first_stage_difference(raw_frames[index], raw_frames[index + 1], raw_frames[index + 2]) for index in range(3)]
    sample = resize_width(raw_frames[2], args.display_width)
    output_path = args.output_dir / f"{args.input.stem}_double_stage_flow_diff.avi"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (sample.shape[1] * 4, sample.shape[0]))
    thumbnail_path = args.output_dir / f"{args.input.stem}_double_stage_flow_diff.jpg"
    done = 0
    started = time.time()
    while done < output_frames:
        second_diff, second_flow, inliers, total = second_stage_difference(first_diffs[0], first_diffs[1], first_diffs[2], model, sea_args)
        rgb_panel = label(resize_width(raw_frames[2], args.display_width), "RGB")
        first_panel = label(resize_width(heat_color(first_diffs[1]), args.display_width), "Stage 1: YOLOMG difference")
        flow_panel = label(resize_width(flow_color(second_flow), args.display_width), "Stage 2: flow on difference maps")
        second_panel = label(resize_width(heat_color(second_diff), args.display_width), "Stage 2: compensated difference")
        combined = np.concatenate([rgb_panel, first_panel, flow_panel, second_panel], axis=1)
        writer.write(combined)
        if done == min(30, max(0, output_frames - 1)):
            cv2.imwrite(str(thumbnail_path), combined)
        done += 1
        if done % 5 == 0 or done == output_frames:
            write_progress(args.progress_json, {"status": "running", "done": done, "total": output_frames, "output": str(output_path), "last_output_timestamp": time.time(), "stage2_inliers": inliers, "stage2_points": total, "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1048576.0, "elapsed_seconds": time.time() - started})
            print(f"[{args.input.name}] {done}/{output_frames}", flush=True)
        raw_frames = raw_frames[1:]
        ok, frame = capture.read()
        if not ok:
            break
        raw_frames.append(resize_width(frame, args.process_width))
        first_diffs = first_diffs[1:]
        first_diffs.append(first_stage_difference(raw_frames[2], raw_frames[3], raw_frames[4]))
    writer.release()
    capture.release()
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps({"method": "YOLOMG PyrLK/RANSAC first-stage difference, then official SEA-RAFT/RANSAC second-stage compensated difference", "input": str(args.input), "output": str(output_path), "thumbnail": str(thumbnail_path), "frames": done, "fps": fps}, indent=2), encoding="utf-8")
    write_progress(args.progress_json, {"status": "completed", "done": done, "total": done, "manifest": str(manifest), "last_output_timestamp": time.time()})
    print(f"[DONE] {manifest}", flush=True)


if __name__ == "__main__":
    main()