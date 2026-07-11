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
    parser = argparse.ArgumentParser(description="Render SEA-RAFT optical flow and YOLOMG-style compensated frame differences.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sea-raft-root", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--model-url", default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--process-width", type=int, default=960)
    parser.add_argument("--display-width", type=int, default=480)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--progress-json", type=Path)
    return parser.parse_args()


def resize_width(image, width):
    height, source_width = image.shape[:2]
    target_height = max(2, int(round(height * width / source_width)))
    target_height += target_height % 2
    return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)


def write_progress(path, payload):
    if path is None:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_model(args):
    sys.path.insert(0, str(args.sea_raft_root))
    sys.path.insert(0, str(args.sea_raft_root / "core"))
    from config.parser import json_to_args
    from raft import RAFT

    sea_args = json_to_args(str(args.cfg))
    sea_args.url = args.model_url
    model = RAFT.from_pretrained(args.model_url, args=sea_args).cuda().eval()
    return model, sea_args


@torch.inference_mode()
def infer_flow(model, sea_args, first_bgr, second_bgr):
    first = torch.from_numpy(cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None].cuda()
    second = torch.from_numpy(cv2.cvtColor(second_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None].cuda()
    scaled_first = functional.interpolate(first, scale_factor=2 ** sea_args.scale, mode="bilinear", align_corners=False)
    scaled_second = functional.interpolate(second, scale_factor=2 ** sea_args.scale, mode="bilinear", align_corners=False)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(scaled_first, scaled_second, iters=sea_args.iters, test_mode=True)
    flow = output["flow"][-1].float()
    flow = functional.interpolate(flow, size=first.shape[-2:], mode="bilinear", align_corners=False) * (0.5 ** sea_args.scale)
    return flow[0].permute(1, 2, 0).cpu().numpy()


def estimate_homography_from_flow(flow):
    height, width = flow.shape[:2]
    step = max(8, min(width, height) // 45)
    ys, xs = np.mgrid[step:height-step:step, step:width-step:step]
    source = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    sampled = flow[ys, xs].reshape(-1, 2).astype(np.float32)
    destination = source + sampled
    finite = np.isfinite(destination).all(axis=1)
    source = source[finite]
    destination = destination[finite]
    homography, mask = cv2.findHomography(destination, source, cv2.RANSAC, 2.5)
    if homography is None:
        return np.eye(3, dtype=np.float64), 0, len(source)
    inliers = int(mask.sum()) if mask is not None else 0
    return homography, inliers, len(source)


def align_with_flow(source_gray, flow):
    homography, inliers, total = estimate_homography_from_flow(flow)
    aligned = cv2.warpPerspective(source_gray, homography, (source_gray.shape[1], source_gray.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned, inliers, total


def flow_to_color(flow):
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
    cv2.putText(image, text, (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, sea_args = load_model(args)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 29.97
    start_frame = int(round(args.start_seconds * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    requested = int(round(args.duration_seconds * fps)) if args.duration_seconds > 0 else int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame
    output_frames = max(0, (requested - 2) // args.stride)
    frames = []
    for _ in range(3):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Not enough input frames")
        frames.append(resize_width(frame, args.process_width))
    sample = resize_width(frames[1], args.display_width)
    output_path = args.output_dir / f"{args.input.stem}_sea_raft_flow_diff.avi"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"MJPG"), fps / args.stride, (sample.shape[1] * 4, sample.shape[0]))
    thumbnail_path = args.output_dir / f"{args.input.stem}_sea_raft_flow_diff.jpg"
    done = 0
    source_index = 0
    started = time.time()
    while done < output_frames:
        previous, reference, following = frames
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        following_gray = cv2.cvtColor(following, cv2.COLOR_BGR2GRAY)
        flow_previous_to_reference = infer_flow(model, sea_args, previous, reference)
        flow_following_to_reference = infer_flow(model, sea_args, following, reference)
        aligned_previous, previous_inliers, previous_total = align_with_flow(previous_gray, flow_previous_to_reference)
        aligned_following, following_inliers, following_total = align_with_flow(following_gray, flow_following_to_reference)
        raw_difference = cv2.absdiff(reference_gray, previous_gray)
        compensated_difference = ((cv2.absdiff(reference_gray, aligned_previous).astype(np.float32) + cv2.absdiff(reference_gray, aligned_following).astype(np.float32)) / 2.0).astype(np.uint8)
        rgb_panel = label(resize_width(reference, args.display_width), "RGB")
        flow_panel = label(resize_width(flow_to_color(flow_previous_to_reference), args.display_width), "SEA-RAFT optical flow")
        raw_panel = label(resize_width(heat_color(raw_difference), args.display_width), "Raw frame difference")
        compensated_panel = label(resize_width(heat_color(compensated_difference), args.display_width), "SEA-RAFT + homography difference")
        combined = np.concatenate([rgb_panel, flow_panel, raw_panel, compensated_panel], axis=1)
        writer.write(combined)
        if done == min(30, max(0, output_frames - 1)):
            cv2.imwrite(str(thumbnail_path), combined)
        done += 1
        if done % 5 == 0 or done == output_frames:
            write_progress(args.progress_json, {"status": "running", "done": done, "total": output_frames, "input": str(args.input), "output": str(output_path), "last_output_timestamp": time.time(), "previous_inliers": previous_inliers, "previous_points": previous_total, "following_inliers": following_inliers, "following_points": following_total, "elapsed_seconds": time.time() - started, "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1048576.0})
            print(f"[{args.input.name}] {done}/{output_frames}", flush=True)
        for _ in range(args.stride):
            source_index += 1
            frames = frames[1:]
            ok, frame = capture.read()
            if not ok:
                frames = []
                break
            frames.append(resize_width(frame, args.process_width))
        if len(frames) < 3:
            break
    writer.release()
    capture.release()
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps({"method": "Official SEA-RAFT M dense flow + RANSAC homography + YOLOMG bidirectional compensated difference", "input": str(args.input), "output": str(output_path), "thumbnail": str(thumbnail_path), "frames": done, "fps": fps / args.stride}, indent=2), encoding="utf-8")
    write_progress(args.progress_json, {"status": "completed", "done": done, "total": done, "manifest": str(manifest), "last_output_timestamp": time.time()})
    print(f"[DONE] {manifest}", flush=True)


if __name__ == "__main__":
    main()