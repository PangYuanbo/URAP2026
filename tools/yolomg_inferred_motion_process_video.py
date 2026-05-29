from __future__ import annotations

import argparse
import re
import sys
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render YOLOMG model-inferred motion process maps with Grad-CAM.")
    p.add_argument("--weights", default=str(ROOT / r"runs\train\yolomg_ard100_e50_b4_img1280_20260221_181641\weights\best.pt"))
    p.add_argument("--test-list", default=r"D:\URAP_datasets\ARD100_YOLOMG\test.txt")
    p.add_argument("--test2-list", default=r"D:\URAP_datasets\ARD100_YOLOMG\test2.txt")
    p.add_argument("--video-root", default=r"D:\URAP_datasets\ARD100\test_videos")
    p.add_argument("--videos", nargs="+", default=["phantom02"])
    p.add_argument("--output-dir", default=str(ROOT / r"runs\motion_process_gradcam\smoke"))
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--motion-layer", type=int, default=3, help="Motion-branch feature layer before RGB/motion fusion.")
    p.add_argument("--fusion-layer", type=int, default=5, help="Post-fusion feature layer.")
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--display-width", type=int, default=640)
    p.add_argument("--fps-fallback", type=float, default=29.97)
    return p.parse_args()


def read_list(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_frame(path_str: str) -> tuple[str, int, str]:
    match = FRAME_RE.match(Path(path_str).name)
    if not match:
        raise ValueError(f"Cannot parse frame entry: {path_str}")
    return match.group("video"), int(match.group("frame")), path_str


def build_pairs(test_list: list[str], test2_list: list[str], videos: set[str]) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {}
    motion_by_key: dict[tuple[str, int], str] = {}
    for motion_path in test2_list:
        v2, f2, p2 = parse_frame(motion_path)
        if v2 in videos:
            motion_by_key[(v2, f2)] = p2

    skipped = 0
    for image_path in test_list:
        v1, f1, p1 = parse_frame(image_path)
        if v1 not in videos:
            continue
        p2 = motion_by_key.get((v1, f1))
        if p2 is None:
            skipped += 1
            continue
        f2 = f1
        out.setdefault(v1, []).append({"frame": f1, "motion_frame": f2, "image_path": p1, "motion_path": p2})
    for items in out.values():
        items.sort(key=lambda x: int(x["frame"]))
    if skipped:
        print(f"[WARN] skipped {skipped} RGB frames without same-frame motion maps", flush=True)
    return out


def get_fps(video_root: Path, video: str, fallback: float) -> float:
    cap = cv2.VideoCapture(str(video_root / f"{video}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 1e-6 else fallback


def preprocess_bgr(img: np.ndarray, imgsz: int, stride: int, device: torch.device) -> torch.Tensor:
    padded = letterbox(img, imgsz, stride=stride, auto=False)[0]
    tensor = padded.transpose((2, 0, 1))[::-1]
    tensor = np.ascontiguousarray(tensor)
    tensor = torch.from_numpy(tensor).to(device).float() / 255.0
    return tensor.unsqueeze(0)


def normalize_u8(gray: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5, gamma: float = 0.7) -> np.ndarray:
    arr = gray.astype(np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    arr = np.power(arr, gamma)
    return np.uint8(arr * 255.0)


def crop_letterbox(cam: np.ndarray, orig_shape: tuple[int, int], imgsz: int) -> np.ndarray:
    orig_h, orig_w = orig_shape
    gain = min(imgsz / float(orig_h), imgsz / float(orig_w))
    new_w = int(round(orig_w * gain))
    new_h = int(round(orig_h * gain))
    pad_w = (imgsz - new_w) / 2.0
    pad_h = (imgsz - new_h) / 2.0
    x1 = max(0, int(round(pad_w - 0.1)))
    y1 = max(0, int(round(pad_h - 0.1)))
    x2 = min(imgsz, int(round(pad_w + new_w + 0.1)))
    y2 = min(imgsz, int(round(pad_h + new_h + 0.1)))
    cropped = cam[y1:y2, x1:x2]
    if cropped.size == 0:
        return cam
    return cropped


def tensor_gradcam(act: torch.Tensor, orig_shape: tuple[int, int], imgsz: int) -> np.ndarray:
    grad = act.grad
    if grad is None:
        return np.zeros(orig_shape, dtype=np.uint8)
    a = act.detach().float()
    g = grad.detach().float()
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * a).sum(dim=1, keepdim=True))
    cam = cam.squeeze().cpu().numpy()
    cam_u8 = normalize_u8(cam, lo_pct=0.0, hi_pct=99.0, gamma=0.55)
    padded = cv2.resize(cam_u8, (imgsz, imgsz), interpolation=cv2.INTER_CUBIC)
    cropped = crop_letterbox(padded, orig_shape, imgsz)
    return cv2.resize(cropped, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_CUBIC)


def colorize(gray_u8: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(gray_u8, cv2.COLORMAP_TURBO)


def resize_keep_aspect(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    target_h = max(2, int(round(h * target_w / float(w))))
    if target_h % 2:
        target_h += 1
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    scale = max(0.45, min(0.75, img.shape[1] / 900.0))
    thickness = 1 if scale < 0.7 else 2
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(out, (8, 8), (min(img.shape[1] - 1, 20 + tw), 20 + th + baseline), (0, 0, 0), -1)
    cv2.putText(out, text, (14, 16 + th), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def make_panel(frame_bgr: np.ndarray, motion_gray: np.ndarray, motion_cam: np.ndarray, fusion_cam: np.ndarray, display_w: int) -> np.ndarray:
    rgb = label(resize_keep_aspect(frame_bgr, display_w), "RGB")
    motion = label(resize_keep_aspect(cv2.cvtColor(normalize_u8(motion_gray, gamma=0.55), cv2.COLOR_GRAY2BGR), display_w), "input motion")
    motion_cam_img = label(resize_keep_aspect(colorize(motion_cam), display_w), "Grad-CAM L3 motion")
    fusion_cam_img = label(resize_keep_aspect(colorize(fusion_cam), display_w), "Grad-CAM L5 fusion")
    h = min(x.shape[0] for x in [rgb, motion, motion_cam_img, fusion_cam_img])
    return np.concatenate([rgb[:h], motion[:h], motion_cam_img[:h], fusion_cam_img[:h]], axis=1)


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(read_list(args.test_list), read_list(args.test2_list), set(args.videos))
    device = select_device(args.device)
    model = attempt_load(args.weights, map_location=device, fuse=False).float().eval()
    for p in model.parameters():
        p.requires_grad_(True)
    if hasattr(model.model[-1], "inplace"):
        model.model[-1].inplace = False
    stride = int(model.stride.max())
    imgsz = check_img_size(args.imgsz, s=stride)

    captures: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            captures[name] = output
            output.retain_grad()
        return hook

    motion_handle = model.model[args.motion_layer].register_forward_hook(make_hook("motion"))
    fusion_handle = model.model[args.fusion_layer].register_forward_hook(make_hook("fusion"))

    try:
        for video in args.videos:
            items = pairs.get(video, [])
            if args.max_frames and args.max_frames > 0:
                items = items[: args.max_frames]
            if not items:
                print(f"[WARN] no pairs for {video}", flush=True)
                continue

            video_dir = out_root / video
            video_dir.mkdir(parents=True, exist_ok=True)
            fps = get_fps(Path(args.video_root), video, args.fps_fallback)
            writer = None
            manifest = video_dir / "manifest.txt"
            with manifest.open("w", encoding="utf-8") as m:
                m.write(f"video={video}\n")
                m.write(f"weights={args.weights}\n")
                m.write(f"motion_layer={args.motion_layer}\n")
                m.write(f"fusion_layer={args.fusion_layer}\n")
                m.write(f"frames={len(items)}\n")
                for idx, item in enumerate(items, start=1):
                    frame = cv2.imread(str(item["image_path"]))
                    motion = cv2.imread(str(item["motion_path"]), cv2.IMREAD_GRAYSCALE)
                    motion_bgr = cv2.imread(str(item["motion_path"]))
                    if frame is None or motion is None or motion_bgr is None:
                        continue
                    h, w = frame.shape[:2]
                    img1 = preprocess_bgr(frame, imgsz, stride, device)
                    img2 = preprocess_bgr(motion_bgr, imgsz, stride, device)

                    model.zero_grad(set_to_none=True)
                    captures.clear()
                    pred = model(img1, img2, augment=False)
                    pred_tensor = pred[0] if isinstance(pred, (tuple, list)) else pred
                    score = pred_tensor[..., 4].max()
                    score.backward()

                    motion_cam = tensor_gradcam(captures["motion"], (h, w), imgsz)
                    fusion_cam = tensor_gradcam(captures["fusion"], (h, w), imgsz)
                    panel = make_panel(frame, motion, motion_cam, fusion_cam, args.display_width)
                    cv2.putText(panel, f"{video} frame {item['frame']:04d} score {float(score.detach().cpu()):.3f}", (16, panel.shape[0] - 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

                    if writer is None:
                        writer = cv2.VideoWriter(str(video_dir / f"{video}_model_motion_process_gradcam.avi"),
                                                 cv2.VideoWriter_fourcc(*"MJPG"), fps, (panel.shape[1], panel.shape[0]))
                    writer.write(panel)

                    if idx <= 3:
                        cv2.imwrite(str(video_dir / f"{video}_{int(item['frame']):04d}_model_motion_process.jpg"), panel)
                        cv2.imwrite(str(video_dir / f"{video}_{int(item['frame']):04d}_motion_branch_gradcam.jpg"), colorize(motion_cam))
                        cv2.imwrite(str(video_dir / f"{video}_{int(item['frame']):04d}_fusion_gradcam.jpg"), colorize(fusion_cam))
                    m.write(f"{idx}\tframe={item['frame']}\tmotion_frame={item['motion_frame']}\tscore={float(score.detach().cpu()):.6f}\n")
                    if idx % 10 == 0 or idx == len(items):
                        print(f"[{video}] {idx}/{len(items)}", flush=True)
            if writer is not None:
                writer.release()
            print(f"[DONE] {video} -> {video_dir}", flush=True)
    finally:
        motion_handle.remove()
        fusion_handle.remove()


if __name__ == "__main__":
    main()
