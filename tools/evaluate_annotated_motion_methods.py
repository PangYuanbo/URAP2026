#!/usr/bin/env python3
"""Evaluate motion maps against one manually annotated DJI video.

The goal is not to train a detector. It is a small, inspectable testbed for:

* YOLOMG-style frame-difference motion maps.
* NPS/Purdue U2U-D&T-style sparse-flow global motion compensation and residuals.

The script reports whether motion evidence concentrates inside the annotated
drone box and whether simple contour proposals overlap the annotation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


METHODS = ("yolomg_diff_k1", "yolomg_diff_k5", "nps_sparse_flow")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/dji_522_motion_methods"))
    p.add_argument("--process-width", type=int, default=1280)
    p.add_argument("--max-frames", type=int, default=0, help="0 means all annotated frames.")
    p.add_argument("--stride", type=int, default=1, help="Use every Nth annotation after sorting.")
    p.add_argument("--write-video", action="store_true")
    p.add_argument("--sample-count", type=int, default=12)
    p.add_argument("--proposal-percentile", type=float, default=99.6)
    return p.parse_args()


def load_annotations(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    anns = [a for a in data["annotations"] if a.get("boxes")]
    return sorted(anns, key=lambda item: int(item["frame_id"]))


def first_box_xyxy(annotation: dict) -> tuple[float, float, float, float]:
    box = annotation["boxes"][0]
    return float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])


def scale_box(box: tuple[float, float, float, float], sx: float, sy: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def resize_for_processing(frame: np.ndarray, process_width: int) -> tuple[np.ndarray, float, float]:
    h, w = frame.shape[:2]
    if process_width <= 0 or w <= process_width:
        return frame, 1.0, 1.0
    scale = process_width / float(w)
    out_h = int(round(h * scale))
    resized = cv2.resize(frame, (process_width, out_h), interpolation=cv2.INTER_AREA)
    return resized, scale, scale


def read_frame(cap: cv2.VideoCapture, frame_id: int, process_width: int) -> np.ndarray | None:
    if frame_id < 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    if not ok:
        return None
    return resize_for_processing(frame, process_width)[0]


def gray(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.GaussianBlur(g, (5, 5), 0)


def normalize_motion(arr: np.ndarray, gamma: float = 0.65) -> np.ndarray:
    arr_f = arr.astype(np.float32)
    hi = float(np.percentile(arr_f, 99.7))
    lo = float(np.percentile(arr_f, 1.0))
    if hi <= lo + 1e-6:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    norm = np.clip((arr_f - lo) / (hi - lo), 0.0, 1.0)
    norm = np.power(norm, gamma)
    return np.uint8(norm * 255.0)


def clean_motion(motion: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    out = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out


def yolomg_diff(prev: np.ndarray | None, curr: np.ndarray) -> np.ndarray:
    if prev is None:
        return np.zeros(curr.shape[:2], dtype=np.uint8)
    diff = cv2.absdiff(gray(prev), gray(curr))
    return clean_motion(normalize_motion(diff))


def nps_sparse_flow_residual(prev: np.ndarray | None, curr: np.ndarray) -> tuple[np.ndarray, dict]:
    if prev is None:
        return np.zeros(curr.shape[:2], dtype=np.uint8), {"tracked": 0, "inliers": 0, "homography": False}

    prev_gray = gray(prev)
    curr_gray = gray(curr)
    features = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=2500,
        qualityLevel=0.01,
        minDistance=12,
        blockSize=7,
    )
    if features is None or len(features) < 12:
        return yolomg_diff(prev, curr), {"tracked": 0, "inliers": 0, "homography": False, "fallback": "diff"}

    nxt, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        features,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if nxt is None or st is None:
        return yolomg_diff(prev, curr), {"tracked": 0, "inliers": 0, "homography": False, "fallback": "diff"}

    src = features.reshape(-1, 2)
    dst = nxt.reshape(-1, 2)
    valid = st.reshape(-1) == 1
    if err is not None:
        valid &= err.reshape(-1) < 30.0
    src = src[valid].astype(np.float32)
    dst = dst[valid].astype(np.float32)
    if len(src) < 12:
        return yolomg_diff(prev, curr), {"tracked": int(len(src)), "inliers": 0, "homography": False, "fallback": "diff"}

    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if homography is None or mask is None:
        return yolomg_diff(prev, curr), {"tracked": int(len(src)), "inliers": 0, "homography": False, "fallback": "diff"}

    h, w = curr_gray.shape[:2]
    warped = cv2.warpPerspective(prev_gray, homography, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    residual = cv2.absdiff(curr_gray, warped)

    flow = dst - src
    if int(mask.sum()) >= 8:
        bg_flow = np.median(flow[mask.reshape(-1) == 1], axis=0)
    else:
        bg_flow = np.median(flow, axis=0)
    flow_residual = np.linalg.norm(flow - bg_flow, axis=1)

    # The paper pipeline extracts salient points in the stabilized difference
    # image and uses optical-flow motion traits to prune target candidates. We
    # encode that idea by boosting residual pixels around points whose sparse
    # flow differs from the estimated background motion.
    point_boost = np.zeros_like(residual)
    strong = flow_residual >= max(1.5, float(np.percentile(flow_residual, 90.0)))
    for x, y in dst[strong].astype(np.int32):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(point_boost, (int(x), int(y)), 5, 255, -1)

    combined = cv2.addWeighted(residual, 0.75, point_boost, 0.25, 0)
    motion = clean_motion(normalize_motion(combined, gamma=0.58))
    return motion, {
        "tracked": int(len(src)),
        "inliers": int(mask.sum()),
        "homography": True,
        "median_bg_dx": float(bg_flow[0]),
        "median_bg_dy": float(bg_flow[1]),
        "strong_flow_points": int(strong.sum()),
    }


def bbox_mask(shape: tuple[int, int], box: tuple[float, float, float, float], pad: int = 0) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = box
    x1i = max(0, int(np.floor(x1)) - pad)
    y1i = max(0, int(np.floor(y1)) - pad)
    x2i = min(w, int(np.ceil(x2)) + pad)
    y2i = min(h, int(np.ceil(y2)) + pad)
    mask = np.zeros((h, w), dtype=np.uint8)
    if x2i > x1i and y2i > y1i:
        mask[y1i:y2i, x1i:x2i] = 1
    return mask


def annulus_mask(shape: tuple[int, int], box: tuple[float, float, float, float], outer_pad: int = 42) -> np.ndarray:
    outer = bbox_mask(shape, box, pad=outer_pad)
    inner = bbox_mask(shape, box, pad=4)
    return np.clip(outer - inner, 0, 1).astype(np.uint8)


def score_motion(motion: np.ndarray, box: tuple[float, float, float, float]) -> dict:
    target = bbox_mask(motion.shape[:2], box, pad=4)
    context = annulus_mask(motion.shape[:2], box)
    target_vals = motion[target > 0]
    context_vals = motion[context > 0]
    all_vals = motion.reshape(-1)
    target_mean = float(target_vals.mean() / 255.0) if target_vals.size else 0.0
    context_mean = float(context_vals.mean() / 255.0) if context_vals.size else 0.0
    global_p99 = float(np.percentile(all_vals, 99.0) / 255.0) if all_vals.size else 0.0
    contrast = target_mean / (context_mean + 1e-6)
    return {
        "target_mean": target_mean,
        "context_mean": context_mean,
        "global_p99": global_p99,
        "contrast": float(contrast),
    }


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def proposal_hit(motion: np.ndarray, box: tuple[float, float, float, float], percentile: float) -> dict:
    thresh = max(10, int(np.percentile(motion, percentile)))
    _, binary = cv2.threshold(motion, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_iou = 0.0
    best_overlap = 0.0
    proposals = 0
    gt_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    for c in contours:
        area = cv2.contourArea(c)
        if area < 2 or area > 6000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        prop = (float(x), float(y), float(x + w), float(y + h))
        proposals += 1
        best_iou = max(best_iou, iou(prop, box))
        ix1, iy1 = max(prop[0], box[0]), max(prop[1], box[1])
        ix2, iy2 = min(prop[2], box[2]), min(prop[3], box[3])
        overlap = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) / gt_area
        best_overlap = max(best_overlap, float(overlap))
    return {
        "threshold": thresh,
        "proposal_count": proposals,
        "best_iou": best_iou,
        "best_gt_overlap": best_overlap,
        "hit_iou_0p1": best_iou >= 0.1,
        "hit_overlap_0p25": best_overlap >= 0.25,
    }


def colorize(motion: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(motion, cv2.COLORMAP_TURBO)


def draw_box(img: np.ndarray, box: tuple[float, float, float, float], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def make_panel(frame: np.ndarray, box: tuple[float, float, float, float], maps: dict[str, np.ndarray], frame_id: int) -> np.ndarray:
    rgb = frame.copy()
    draw_box(rgb, box, (0, 255, 0), "GT")
    tiles = [rgb]
    for name in METHODS:
        tile = colorize(maps[name])
        overlay = cv2.addWeighted(frame, 0.60, tile, 0.40, 0)
        draw_box(overlay, box, (0, 255, 0), name)
        tiles.append(overlay)
    target_h = 360
    resized = []
    for tile in tiles:
        h, w = tile.shape[:2]
        target_w = int(round(w * target_h / h))
        resized.append(cv2.resize(tile, (target_w, target_h), interpolation=cv2.INTER_AREA))
    panel = np.concatenate(resized, axis=1)
    cv2.putText(panel, f"frame {frame_id}", (16, panel.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def summarize(rows: list[dict], source: dict, args: argparse.Namespace) -> dict:
    frame_ids = sorted({int(r["frame_id"]) for r in rows})
    summary = {
        "source": source,
        "settings": {
            "process_width": args.process_width,
            "proposal_percentile": args.proposal_percentile,
            "stride": args.stride,
            "max_frames": args.max_frames,
        },
        "frames_evaluated": len(frame_ids),
        "methods": {},
    }
    for method in METHODS:
        part = [r for r in rows if r["method"] == method]
        if not part:
            continue
        summary["methods"][method] = {
            "mean_target_motion": float(np.mean([r["target_mean"] for r in part])),
            "median_target_motion": float(np.median([r["target_mean"] for r in part])),
            "mean_context_motion": float(np.mean([r["context_mean"] for r in part])),
            "median_contrast": float(np.median([r["contrast"] for r in part])),
            "hit_rate_iou_0p1": float(np.mean([r["hit_iou_0p1"] for r in part])),
            "hit_rate_gt_overlap_0p25": float(np.mean([r["hit_overlap_0p25"] for r in part])),
            "mean_best_iou": float(np.mean([r["best_iou"] for r in part])),
            "mean_best_gt_overlap": float(np.mean([r["best_gt_overlap"] for r in part])),
            "mean_proposal_count": float(np.mean([r["proposal_count"] for r in part])),
        }
    return summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(args.annotations)
    annotations = annotations[:: max(1, args.stride)]
    if args.max_frames > 0:
        annotations = annotations[: args.max_frames]

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.97)
    sx = min(1.0, args.process_width / float(video_w)) if args.process_width > 0 else 1.0
    sy = sx

    rows: list[dict] = []
    sample_every = max(1, len(annotations) // max(1, args.sample_count))
    writer = None
    panel_video = args.out_dir / "motion_method_comparison.mp4"

    frame_cache: dict[int, np.ndarray | None] = {}

    def cached_frame(fid: int) -> np.ndarray | None:
        if fid not in frame_cache:
            frame_cache[fid] = read_frame(cap, fid, args.process_width)
            if len(frame_cache) > 12:
                for key in sorted(frame_cache)[: max(1, len(frame_cache) - 12)]:
                    frame_cache.pop(key, None)
        return frame_cache[fid]

    for idx, ann in enumerate(annotations):
        frame_id = int(ann["frame_id"])
        curr = cached_frame(frame_id)
        if curr is None:
            continue
        prev1 = cached_frame(frame_id - 1)
        prev5 = cached_frame(frame_id - 5)
        box = scale_box(first_box_xyxy(ann), sx, sy)

        nps_map, nps_debug = nps_sparse_flow_residual(prev1, curr)
        maps = {
            "yolomg_diff_k1": yolomg_diff(prev1, curr),
            "yolomg_diff_k5": yolomg_diff(prev5, curr),
            "nps_sparse_flow": nps_map,
        }

        for method, motion in maps.items():
            score = score_motion(motion, box)
            hit = proposal_hit(motion, box, args.proposal_percentile)
            rows.append(
                {
                    "frame_id": frame_id,
                    "method": method,
                    **score,
                    **hit,
                    "nps_tracked": nps_debug.get("tracked", 0) if method == "nps_sparse_flow" else "",
                    "nps_inliers": nps_debug.get("inliers", 0) if method == "nps_sparse_flow" else "",
                    "nps_strong_flow_points": nps_debug.get("strong_flow_points", 0) if method == "nps_sparse_flow" else "",
                }
            )

        if idx % sample_every == 0:
            panel = make_panel(curr, box, maps, frame_id)
            cv2.imwrite(str(args.out_dir / f"sample_frame_{frame_id:06d}.jpg"), panel)
            if args.write_video:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(panel_video), fourcc, min(12.0, fps), (panel.shape[1], panel.shape[0]))
                writer.write(panel)

        if (idx + 1) % 100 == 0 or idx + 1 == len(annotations):
            print(f"[motion-eval] {idx + 1}/{len(annotations)} annotated frames", flush=True)

    if writer is not None:
        writer.release()
    cap.release()

    csv_path = args.out_dir / "motion_method_metrics.csv"
    fields = [
        "frame_id",
        "method",
        "target_mean",
        "context_mean",
        "global_p99",
        "contrast",
        "threshold",
        "proposal_count",
        "best_iou",
        "best_gt_overlap",
        "hit_iou_0p1",
        "hit_overlap_0p25",
        "nps_tracked",
        "nps_inliers",
        "nps_strong_flow_points",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    source = {
        "video": str(args.video),
        "annotations": str(args.annotations),
        "video_width": video_w,
        "video_height": video_h,
        "fps": fps,
    }
    summary = summarize(rows, source, args)
    summary_path = args.out_dir / "motion_method_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"csv={csv_path}", flush=True)
    print(f"summary={summary_path}", flush=True)
    if args.write_video:
        print(f"video={panel_video}", flush=True)


if __name__ == "__main__":
    main()
