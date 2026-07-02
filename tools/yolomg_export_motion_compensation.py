from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

cv2 = None
np = None


def load_cv_deps() -> None:
    global cv2, np
    if cv2 is not None and np is not None:
        return
    try:
        import cv2 as cv2_module
        import numpy as np_module
    except Exception as exc:
        raise SystemExit(
            "OpenCV/NumPy could not be imported in this Python environment. "
            "Run this script from a YOLOMG/ESOD/TransVisDrone venv with cv2 installed, "
            "or fix the local cv2 NumPy ABI mismatch. Original error: "
            f"{exc}"
        ) from exc
    cv2 = cv2_module
    np = np_module


@dataclass
class CompensationResult:
    compensated: np.ndarray
    homography: np.ndarray
    ok: bool
    tracks: int
    inliers: int
    inlier_ratio: float
    reproj_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export YOLOMG-style camera motion compensation diagnostics before "
            "the detector/fusion chain."
        )
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for debug frames and videos.")
    parser.add_argument("--start-frame", type=int, default=2, help="First center frame to process.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process until video end.")
    parser.add_argument("--step", type=int, default=0, help="Frame interval k. 0 derives k from fps and --target-delta-ms.")
    parser.add_argument("--target-delta-ms", type=float, default=67.0, help="Used when --step=0.")
    parser.add_argument("--save-every", type=int, default=1, help="Save image panels every N processed center frames.")
    parser.add_argument("--grid-step", type=int, default=64, help="KLT grid spacing in working pixels.")
    parser.add_argument("--align-width", type=int, default=960, help="Resize max width for homography estimation; 0 disables resize.")
    parser.add_argument("--max-track-px", type=float, default=140.0, help="Drop KLT tracks longer than this in working pixels; 0 disables.")
    parser.add_argument("--ransac-threshold", type=float, default=3.0, help="RANSAC reprojection threshold.")
    parser.add_argument("--overlay-alpha", type=float, default=0.42)
    parser.add_argument("--fps-fallback", type=float, default=29.97)
    parser.add_argument("--write-videos", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def to_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (11, 11), 0)


def make_grid_points(width: int, height: int, step: int) -> np.ndarray:
    step = max(8, int(step))
    xs = np.arange(step / 2.0, max(step / 2.0 + 1, width - step / 2.0), step, dtype=np.float32)
    ys = np.arange(step / 2.0, max(step / 2.0 + 1, height - step / 2.0), step, dtype=np.float32)
    pts = [(x, y) for y in ys for x in xs]
    return np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)


def maybe_resize(gray: np.ndarray, align_width: int) -> tuple[np.ndarray, float]:
    if align_width <= 0 or gray.shape[1] <= align_width:
        return gray, 1.0
    scale = align_width / float(gray.shape[1])
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def estimate_compensation(
    src_gray: np.ndarray,
    dst_gray: np.ndarray,
    *,
    grid_step: int,
    align_width: int,
    max_track_px: float,
    ransac_threshold: float,
) -> CompensationResult:
    src_small, scale = maybe_resize(src_gray, align_width)
    dst_small = cv2.resize(dst_gray, (src_small.shape[1], src_small.shape[0]), interpolation=cv2.INTER_AREA) if scale != 1.0 else dst_gray
    h, w = src_small.shape[:2]
    prev_pts = make_grid_points(w, h, grid_step)

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003),
    )
    next_pts, status, _err = cv2.calcOpticalFlowPyrLK(src_small, dst_small, prev_pts, None, **lk_params)
    if next_pts is None or status is None:
        return CompensationResult(src_gray.copy(), np.eye(3, dtype=np.float64), False, 0, 0, 0.0, math.inf)

    status = status.reshape(-1).astype(bool)
    src_good = prev_pts.reshape(-1, 2)[status]
    dst_good = next_pts.reshape(-1, 2)[status]

    if max_track_px > 0 and len(src_good):
        dist = np.linalg.norm(dst_good - src_good, axis=1)
        keep = dist <= max_track_px
        src_good = src_good[keep]
        dst_good = dst_good[keep]

    tracks = int(len(src_good))
    if tracks < 8:
        return CompensationResult(src_gray.copy(), np.eye(3, dtype=np.float64), False, tracks, 0, 0.0, math.inf)

    if scale != 1.0:
        src_good = src_good / scale
        dst_good = dst_good / scale

    homography, inlier_mask = cv2.findHomography(src_good, dst_good, cv2.RANSAC, ransac_threshold)
    if homography is None or inlier_mask is None:
        return CompensationResult(src_gray.copy(), np.eye(3, dtype=np.float64), False, tracks, 0, 0.0, math.inf)

    inliers_bool = inlier_mask.reshape(-1).astype(bool)
    inliers = int(inliers_bool.sum())
    inlier_ratio = inliers / max(1, tracks)

    projected = cv2.perspectiveTransform(src_good.reshape(-1, 1, 2), homography).reshape(-1, 2)
    if inliers:
        reproj_error = float(np.linalg.norm(projected[inliers_bool] - dst_good[inliers_bool], axis=1).mean())
    else:
        reproj_error = math.inf

    compensated = cv2.warpPerspective(src_gray, homography, (dst_gray.shape[1], dst_gray.shape[0]), flags=cv2.INTER_LINEAR)
    return CompensationResult(compensated, homography, True, tracks, inliers, inlier_ratio, reproj_error)


def normalize_gray(gray: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    high = float(np.percentile(gray, percentile))
    if high <= 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)
    return np.uint8(np.clip(gray.astype(np.float32) / high * 255.0, 0, 255))


def colorize(gray: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(normalize_gray(gray), cv2.COLORMAP_INFERNO)


def label_image(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_panel(images: list[tuple[str, np.ndarray]], size: tuple[int, int], cols: int = 3) -> np.ndarray:
    w, h = size
    cells = []
    for title, img in images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cell = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        cells.append(label_image(cell, title))
    rows = []
    for start in range(0, len(cells), cols):
        row = cells[start : start + cols]
        while len(row) < cols:
            row.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def ensure_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, size)


def main() -> int:
    args = parse_args()
    load_cv_deps()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / "frames"
    panel_dir = args.output_dir / "panels"
    frame_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(args.video)

    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps_fallback
    if fps <= 1e-6:
        fps = args.fps_fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = args.step if args.step > 0 else max(1, round((args.target_delta_ms / 1000.0) * fps))

    first_center = max(args.start_frame, step)
    last_center = total_frames - step - 1 if total_frames else first_center
    if args.max_frames > 0:
        last_center = min(last_center, first_center + args.max_frames - 1)

    sample = read_frame(cap, first_center)
    if sample is None:
        raise RuntimeError(f"Could not read start frame {first_center}")
    height, width = sample.shape[:2]
    panel_cell = (min(640, width), int(min(640, width) * height / width))

    map_writer = grid_writer = None
    if args.write_videos:
        map_writer = ensure_writer(args.output_dir / "motion_map.mp4", fps, (width, height))
        grid_writer = ensure_writer(args.output_dir / "diagnostic_grid.mp4", fps, (panel_cell[0] * 3, panel_cell[1] * 3))

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "center_frame",
                "prev_frame",
                "next_frame",
                "step",
                "prev_ok",
                "prev_tracks",
                "prev_inliers",
                "prev_inlier_ratio",
                "prev_reproj_error",
                "next_ok",
                "next_tracks",
                "next_inliers",
                "next_inlier_ratio",
                "next_reproj_error",
                "motion_mean",
                "motion_p99",
            ],
        )
        writer.writeheader()

        processed = 0
        for center_idx in range(first_center, last_center + 1):
            prev_idx = center_idx - step
            next_idx = center_idx + step
            prev = read_frame(cap, prev_idx)
            current = read_frame(cap, center_idx)
            nxt = read_frame(cap, next_idx)
            if prev is None or current is None or nxt is None:
                break

            prev_gray = to_gray(prev)
            cur_gray = to_gray(current)
            next_gray = to_gray(nxt)

            prev_comp = estimate_compensation(
                prev_gray,
                cur_gray,
                grid_step=args.grid_step,
                align_width=args.align_width,
                max_track_px=args.max_track_px,
                ransac_threshold=args.ransac_threshold,
            )
            next_comp = estimate_compensation(
                next_gray,
                cur_gray,
                grid_step=args.grid_step,
                align_width=args.align_width,
                max_track_px=args.max_track_px,
                ransac_threshold=args.ransac_threshold,
            )

            raw_diff_prev = cv2.absdiff(cur_gray, prev_gray)
            raw_diff_next = cv2.absdiff(cur_gray, next_gray)
            comp_diff_prev = cv2.absdiff(cur_gray, prev_comp.compensated)
            comp_diff_next = cv2.absdiff(cur_gray, next_comp.compensated)
            motion_map = np.uint8(np.clip((comp_diff_prev.astype(np.float32) + comp_diff_next.astype(np.float32)) * 0.5, 0, 255))
            motion_color = colorize(motion_map)
            overlay = cv2.addWeighted(current, 1.0 - args.overlay_alpha, motion_color, args.overlay_alpha, 0.0)

            processed += 1
            save_this = args.save_every > 0 and processed % args.save_every == 0
            if save_this:
                stem = f"frame_{center_idx:06d}"
                cv2.imwrite(str(frame_dir / f"{stem}_current.jpg"), current)
                cv2.imwrite(str(frame_dir / f"{stem}_prev_raw_diff.jpg"), normalize_gray(raw_diff_prev))
                cv2.imwrite(str(frame_dir / f"{stem}_prev_compensated.jpg"), prev_comp.compensated)
                cv2.imwrite(str(frame_dir / f"{stem}_prev_comp_diff.jpg"), normalize_gray(comp_diff_prev))
                cv2.imwrite(str(frame_dir / f"{stem}_next_raw_diff.jpg"), normalize_gray(raw_diff_next))
                cv2.imwrite(str(frame_dir / f"{stem}_next_compensated.jpg"), next_comp.compensated)
                cv2.imwrite(str(frame_dir / f"{stem}_next_comp_diff.jpg"), normalize_gray(comp_diff_next))
                cv2.imwrite(str(frame_dir / f"{stem}_motion_map.jpg"), normalize_gray(motion_map))
                cv2.imwrite(str(frame_dir / f"{stem}_motion_overlay.jpg"), overlay)

            panel = make_panel(
                [
                    (f"current t={center_idx}", current),
                    (f"raw diff t-{step}", colorize(raw_diff_prev)),
                    ("prev compensated", prev_comp.compensated),
                    ("comp diff prev", colorize(comp_diff_prev)),
                    (f"raw diff t+{step}", colorize(raw_diff_next)),
                    ("next compensated", next_comp.compensated),
                    ("comp diff next", colorize(comp_diff_next)),
                    ("3-frame motion map", motion_color),
                    ("motion overlay", overlay),
                ],
                panel_cell,
            )
            if save_this:
                cv2.imwrite(str(panel_dir / f"frame_{center_idx:06d}_panel.jpg"), panel)

            if map_writer is not None:
                map_writer.write(motion_color)
            if grid_writer is not None:
                grid_writer.write(panel)

            writer.writerow(
                {
                    "center_frame": center_idx,
                    "prev_frame": prev_idx,
                    "next_frame": next_idx,
                    "step": step,
                    "prev_ok": int(prev_comp.ok),
                    "prev_tracks": prev_comp.tracks,
                    "prev_inliers": prev_comp.inliers,
                    "prev_inlier_ratio": f"{prev_comp.inlier_ratio:.4f}",
                    "prev_reproj_error": f"{prev_comp.reproj_error:.4f}",
                    "next_ok": int(next_comp.ok),
                    "next_tracks": next_comp.tracks,
                    "next_inliers": next_comp.inliers,
                    "next_inlier_ratio": f"{next_comp.inlier_ratio:.4f}",
                    "next_reproj_error": f"{next_comp.reproj_error:.4f}",
                    "motion_mean": f"{float(motion_map.mean()):.4f}",
                    "motion_p99": f"{float(np.percentile(motion_map, 99)):.4f}",
                }
            )

            if processed % 100 == 0:
                print(f"processed={processed} center_frame={center_idx}")

    cap.release()
    if map_writer is not None:
        map_writer.release()
    if grid_writer is not None:
        grid_writer.release()

    print(f"output_dir={args.output_dir}")
    print(f"manifest={manifest_path}")
    print(f"frames_processed={processed}")
    print(f"step={step} fps={fps:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
