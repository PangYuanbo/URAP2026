import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(r"C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load  # noqa: E402
from utils.datasets import letterbox  # noqa: E402
from utils.general import check_img_size  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


FRAME_RE = re.compile(r"^(?P<video>.+)_(?P<frame>\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="Render YOLO-MG motion-branch heatmap videos.")
    parser.add_argument(
        "--weights",
        default=r"C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\train\yolomg_ard100_e50_b4_img1280_20260221_181641\weights\best.pt",
    )
    parser.add_argument("--test-list", default=r"D:\URAP_datasets\ARD100_YOLOMG\test.txt")
    parser.add_argument("--test2-list", default=r"D:\URAP_datasets\ARD100_YOLOMG\test2.txt")
    parser.add_argument("--video-root", default=r"D:\URAP_datasets\ARD100\test_videos")
    parser.add_argument("--videos", nargs="+", default=["phantom02", "phantom05"])
    parser.add_argument(
        "--output-dir",
        default=r"C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\motion_heatmaps\test_02_05",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--hook-layer", type=int, default=3, help="Motion-branch pre-fusion layer index.")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--fps-fallback", type=float, default=29.97)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--skip-overlay", action="store_true")
    return parser.parse_args()


def read_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_frame_entry(path_str):
    name = Path(path_str).name
    m = FRAME_RE.match(name)
    if not m:
        raise ValueError(f"Cannot parse frame entry: {path_str}")
    return m.group("video"), int(m.group("frame")), path_str


def build_pairs(test_list, test2_list, allowed_videos):
    grouped = defaultdict(list)
    for img_path, mask_path in zip(test_list, test2_list):
        v1, f1, p1 = parse_frame_entry(img_path)
        v2, f2, p2 = parse_frame_entry(mask_path)
        if v1 != v2:
            continue
        if v1 not in allowed_videos:
            continue
        grouped[v1].append(
            {
                "frame": f1,
                "image_path": p1,
                "mask_path": p2,
                "mask_frame": f2,
            }
        )
    for video in grouped:
        grouped[video].sort(key=lambda x: x["frame"])
    return grouped


def load_model(weights, device_str, imgsz):
    device = select_device(device_str)
    model = attempt_load(weights, map_location=device, fuse=False)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    half = device.type != "cpu"
    if half:
        model.half()
    model.eval()
    return model, device, stride, imgsz, half


def preprocess_bgr(img, imgsz, stride, device, half):
    padded = letterbox(img, imgsz, stride=stride, auto=False)[0]
    tensor = padded.transpose((2, 0, 1))[::-1]
    tensor = np.ascontiguousarray(tensor)
    tensor = torch.from_numpy(tensor).to(device)
    tensor = tensor.half() if half else tensor.float()
    tensor /= 255.0
    tensor = tensor.unsqueeze(0)
    return tensor


def feature_to_heatmap(feature_tensor, output_size):
    feature = feature_tensor.detach().float().cpu().squeeze(0)
    heat = feature.abs().mean(dim=0).numpy()
    low, high = np.percentile(heat, [5, 99])
    if high <= low:
        heat = np.zeros_like(heat, dtype=np.float32)
    else:
        heat = np.clip((heat - low) / (high - low), 0.0, 1.0).astype(np.float32)
    heat = cv2.resize(heat, output_size, interpolation=cv2.INTER_CUBIC)
    heat_u8 = np.uint8(np.clip(heat * 255.0, 0, 255))
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return heat_u8, heat_color


def get_video_fps(video_path, fallback):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and fps > 1e-6:
        return fps
    return fallback


def ensure_video_writer(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, size)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(read_list(args.test_list), read_list(args.test2_list), set(args.videos))
    model, device, stride, imgsz, half = load_model(args.weights, args.device, args.imgsz)

    captured = {}

    def hook_fn(_module, _inputs, output):
        captured["feat"] = output

    hook_handle = model.model[args.hook_layer].register_forward_hook(hook_fn)

    try:
        for video_name in args.videos:
            items = pairs.get(video_name)
            if not items:
                print(f"[WARN] No frame pairs found for {video_name}")
                continue
            if args.max_frames and args.max_frames > 0:
                items = items[: args.max_frames]

            video_path = Path(args.video_root) / f"{video_name}.mp4"
            fps = get_video_fps(video_path, args.fps_fallback)

            video_out_dir = output_dir / video_name
            video_out_dir.mkdir(parents=True, exist_ok=True)

            heatmap_video_path = video_out_dir / f"{video_name}_motion_heatmap.mp4"
            overlay_video_path = video_out_dir / f"{video_name}_motion_overlay.mp4"
            manifest_path = video_out_dir / "manifest.txt"

            heatmap_writer = None
            overlay_writer = None

            with open(manifest_path, "w", encoding="utf-8") as manifest:
                manifest.write(f"video={video_name}\n")
                manifest.write(f"weights={args.weights}\n")
                manifest.write(f"hook_layer={args.hook_layer}\n")
                manifest.write(f"fps={fps:.4f}\n")
                manifest.write(f"frames={len(items)}\n")

                for idx, item in enumerate(items, start=1):
                    frame_bgr = cv2.imread(item["image_path"])
                    mask_bgr = cv2.imread(item["mask_path"])
                    if frame_bgr is None or mask_bgr is None:
                        print(f"[WARN] Skipping unreadable pair: {item['image_path']} | {item['mask_path']}")
                        continue

                    original_h, original_w = frame_bgr.shape[:2]
                    img1 = preprocess_bgr(frame_bgr, imgsz, stride, device, half)
                    img2 = preprocess_bgr(mask_bgr, imgsz, stride, device, half)

                    captured.clear()
                    with torch.no_grad():
                        _ = model(img1, img2, augment=False)

                    feat = captured.get("feat")
                    if feat is None:
                        raise RuntimeError(f"Hook layer {args.hook_layer} did not capture any feature map.")

                    heat_u8, heat_color = feature_to_heatmap(feat, (original_w, original_h))
                    overlay = cv2.addWeighted(frame_bgr, 1.0 - args.overlay_alpha, heat_color, args.overlay_alpha, 0.0)

                    label = f"{video_name} frame {item['frame']:04d} mask {item['mask_frame']:04d}"
                    cv2.putText(overlay, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(heat_color, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

                    if heatmap_writer is None:
                        heatmap_writer = ensure_video_writer(heatmap_video_path, fps, (original_w, original_h))
                        if not args.skip_overlay:
                            overlay_writer = ensure_video_writer(overlay_video_path, fps, (original_w, original_h))

                    heatmap_writer.write(heat_color)
                    if overlay_writer is not None:
                        overlay_writer.write(overlay)

                    if idx <= 3:
                        cv2.imwrite(str(video_out_dir / f"{video_name}_{item['frame']:04d}_heatmap.jpg"), heat_color)
                        if not args.skip_overlay:
                            cv2.imwrite(str(video_out_dir / f"{video_name}_{item['frame']:04d}_overlay.jpg"), overlay)

                    manifest.write(
                        f"{idx}\tframe={item['frame']}\tmask_frame={item['mask_frame']}\timage={item['image_path']}\tmask={item['mask_path']}\n"
                    )

                    if idx % 100 == 0 or idx == len(items):
                        print(f"[{video_name}] {idx}/{len(items)}")

            if heatmap_writer is not None:
                heatmap_writer.release()
            if overlay_writer is not None:
                overlay_writer.release()

            print(f"[DONE] {video_name} -> {video_out_dir}")
    finally:
        hook_handle.remove()


if __name__ == "__main__":
    main()
