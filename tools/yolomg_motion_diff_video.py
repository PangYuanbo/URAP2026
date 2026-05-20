import argparse
import re
from pathlib import Path

import cv2
import numpy as np


FRAME_RE = re.compile(r"^(?P<video>.+)_(?P<frame>\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="Render YOLO-MG motion-difference-map videos.")
    parser.add_argument("--test-list", default=r"D:\URAP_datasets\ARD100_YOLOMG\test.txt")
    parser.add_argument("--test2-list", default=r"D:\URAP_datasets\ARD100_YOLOMG\test2.txt")
    parser.add_argument("--video-root", default=r"D:\URAP_datasets\ARD100\test_videos")
    parser.add_argument("--videos", nargs="+", default=["phantom02", "phantom05"])
    parser.add_argument(
        "--output-dir",
        default=r"C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\motion_diff_maps_paper\test_02_05",
    )
    parser.add_argument("--fps-fallback", type=float, default=29.97)
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
    pairs = {}
    for img_path, motion_path in zip(test_list, test2_list):
        v1, f1, p1 = parse_frame_entry(img_path)
        v2, f2, p2 = parse_frame_entry(motion_path)
        if v1 != v2 or v1 not in allowed_videos:
            continue
        pairs.setdefault(v1, []).append(
            {
                "frame": f1,
                "motion_frame": f2,
                "image_path": p1,
                "motion_path": p2,
            }
        )
    for video in pairs:
        pairs[video].sort(key=lambda x: x["frame"])
    return pairs


def get_video_fps(video_path, fallback):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and fps > 1e-6:
        return fps
    return fallback


def paper_colorize(gray):
    g = gray.astype(np.float32) / 255.0
    r = 55.0 + 200.0 * np.clip(g, 0.0, 1.0)
    gch = 255.0 * np.clip((g - 0.08) / 0.92, 0.0, 1.0) ** 1.15
    b = 255.0 * np.clip((g - 0.92) / 0.08, 0.0, 1.0)
    color = np.stack([b, gch, r], axis=-1)
    return np.clip(color, 0, 255).astype(np.uint8)


def overlay_motion(frame_bgr, motion_color, alpha=0.42):
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, motion_color, alpha, 0.0)


def ensure_writer(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(str(path), fourcc, fps, size)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(read_list(args.test_list), read_list(args.test2_list), set(args.videos))

    for video_name in args.videos:
        items = pairs.get(video_name)
        if not items:
            print(f"[WARN] No frame pairs found for {video_name}")
            continue

        video_path = Path(args.video_root) / f"{video_name}.mp4"
        fps = get_video_fps(video_path, args.fps_fallback)
        frame0 = cv2.imread(items[0]["image_path"])
        motion0 = cv2.imread(items[0]["motion_path"], cv2.IMREAD_GRAYSCALE)
        if frame0 is None or motion0 is None:
            raise RuntimeError(f"Unreadable first pair for {video_name}")
        h, w = motion0.shape[:2]

        video_out_dir = output_dir / video_name
        video_out_dir.mkdir(parents=True, exist_ok=True)
        gray_video_path = video_out_dir / f"{video_name}_motion_diff_gray.avi"
        paper_video_path = video_out_dir / f"{video_name}_motion_diff_paper.avi"
        overlay_video_path = video_out_dir / f"{video_name}_motion_diff_overlay.avi"
        manifest_path = video_out_dir / "manifest.txt"

        gray_writer = ensure_writer(gray_video_path, fps, (w, h))
        paper_writer = ensure_writer(paper_video_path, fps, (w, h))
        overlay_writer = ensure_writer(overlay_video_path, fps, (w, h))

        with open(manifest_path, "w", encoding="utf-8") as manifest:
            manifest.write(f"video={video_name}\n")
            manifest.write("source=ARD100_YOLOMG images2/test motion-difference maps\n")
            manifest.write(f"fps={fps:.4f}\n")
            manifest.write(f"frames={len(items)}\n")

            for idx, item in enumerate(items, start=1):
                frame_bgr = cv2.imread(item["image_path"])
                motion_gray = cv2.imread(item["motion_path"], cv2.IMREAD_GRAYSCALE)
                if frame_bgr is None or motion_gray is None:
                    continue

                motion_color = paper_colorize(motion_gray)
                gray_bgr = cv2.cvtColor(motion_gray, cv2.COLOR_GRAY2BGR)
                overlay = overlay_motion(frame_bgr, motion_color)

                gray_writer.write(gray_bgr)
                paper_writer.write(motion_color)
                overlay_writer.write(overlay)

                if idx <= 3:
                    cv2.imwrite(str(video_out_dir / f"{video_name}_{item['frame']:04d}_motion_gray.jpg"), motion_gray)
                    cv2.imwrite(str(video_out_dir / f"{video_name}_{item['frame']:04d}_motion_paper.jpg"), motion_color)
                    cv2.imwrite(str(video_out_dir / f"{video_name}_{item['frame']:04d}_motion_overlay.jpg"), overlay)

                manifest.write(
                    f"{idx}\tframe={item['frame']}\tmotion_frame={item['motion_frame']}\timage={item['image_path']}\tmotion={item['motion_path']}\n"
                )

                if idx % 200 == 0 or idx == len(items):
                    print(f"[{video_name}] {idx}/{len(items)}")

        gray_writer.release()
        paper_writer.release()
        overlay_writer.release()
        print(f"[DONE] {video_name} -> {video_out_dir}")


if __name__ == "__main__":
    main()
