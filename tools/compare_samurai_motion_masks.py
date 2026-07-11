#!/usr/bin/env python3
"""Export and visualize stock SAM2 versus SAMURAI selected masks on one NPS track."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SAM2_ROOT = ROOT / "third_party" / "samurai" / "sam2"
for path in (ROOT, SAM2_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stock-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--samurai-config", default="configs/samurai/sam2.1_hiera_b+.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--fps", type=float, default=15.0)
    return parser.parse_args()


def read_gt(path: Path) -> np.ndarray:
    return np.asarray([[float(value) for value in line.split(",")] for line in path.read_text().splitlines() if line.strip()], dtype=np.float32)


def mask_box(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask)
    if not len(points):
        return np.zeros(4, dtype=np.float32)
    y1, x1 = points.min(axis=0); y2, x2 = points.max(axis=0)
    return np.asarray((x1, y1, x2 - x1 + 1, y2 - y1 + 1), dtype=np.float32)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    if min(a[2], a[3], b[2], b[3]) <= 0:
        return 0.0
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    intersection = max(0.0, min(ax2, bx2) - max(a[0], b[0])) * max(0.0, min(ay2, by2) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return float(intersection / union) if union > 0 else 0.0


def center_error(a: np.ndarray, b: np.ndarray) -> float:
    if min(a[2], a[3], b[2], b[3]) <= 0:
        return float("inf")
    return float(math.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2, a[1] + a[3] / 2 - b[1] - b[3] / 2))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def mask_inside_box(mask: np.ndarray, box: np.ndarray) -> float:
    area = int(mask.sum())
    if not area:
        return 0.0
    height, width = mask.shape
    x1 = int(np.clip(math.floor(box[0]), 0, width)); y1 = int(np.clip(math.floor(box[1]), 0, height))
    x2 = int(np.clip(math.ceil(box[0] + box[2]), 0, width)); y2 = int(np.clip(math.ceil(box[1] + box[3]), 0, height))
    return float(mask[y1:y2, x1:x2].sum() / area)


def run_model(config: str, checkpoint: Path, frame_root: Path, initial: np.ndarray, frames: int, device: str, dtype: str):
    predictor = build_sam2_video_predictor(config, str(checkpoint), device=device)
    state = predictor.init_state(str(frame_root), offload_video_to_cpu=True, offload_state_to_cpu=False)
    prompt = np.asarray((initial[0], initial[1], initial[0] + initial[2], initial[1] + initial[3]), dtype=np.float32)
    predictor.add_new_points_or_box(state, box=prompt, frame_idx=0, obj_id=0)
    output = [None] * frames
    autocast_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype, enabled=device.startswith("cuda")):
        for frame_index, object_ids, masks in predictor.propagate_in_video(state, max_frame_num_to_track=frames):
            if frame_index >= frames:
                break
            object_index = list(object_ids).index(0)
            output[frame_index] = masks[object_index][0].detach().cpu().numpy() > 0.0
    if any(mask is None for mask in output):
        raise RuntimeError("Model did not return every frame")
    del state, predictor
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.stack(output)


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    x, y, w, h = [int(round(value)) for value in box]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    result = image.copy(); layer = np.zeros_like(image); layer[mask] = color
    cv2.addWeighted(layer, alpha, result, 1.0, 0.0, dst=result)
    return result


def crop_panel(image: np.ndarray, mask: np.ndarray, gt: np.ndarray, prediction: np.ndarray, color: tuple[int, int, int], label: str) -> np.ndarray:
    height, width = image.shape[:2]; boxes = [gt, prediction]
    valid = [box for box in boxes if box[2] > 0 and box[3] > 0]
    x1 = min(box[0] for box in valid); y1 = min(box[1] for box in valid)
    x2 = max(box[0] + box[2] for box in valid); y2 = max(box[1] + box[3] for box in valid)
    span = max(x2 - x1, y2 - y1, 32.0); margin = max(24.0, span * 1.5)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; radius = span / 2 + margin
    left, top = int(max(0, math.floor(cx - radius))), int(max(0, math.floor(cy - radius)))
    right, bottom = int(min(width, math.ceil(cx + radius))), int(min(height, math.ceil(cy + radius)))
    crop = overlay_mask(image[top:bottom, left:right], mask[top:bottom, left:right], color)
    local_gt = gt.copy(); local_gt[:2] -= (left, top)
    local_pred = prediction.copy(); local_pred[:2] -= (left, top)
    draw_box(crop, local_gt, (0, 255, 0), 2); draw_box(crop, local_pred, color, 2)
    panel = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return panel


def main() -> int:
    args = parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    frame_root = args.sequence_root / "img"; frame_paths = sorted(frame_root.glob("*.jpg"))
    gt = read_gt(args.sequence_root / "groundtruth.txt")[:len(frame_paths)]
    if len(gt) != len(frame_paths): raise ValueError("Frame and ground-truth counts differ")
    stock_masks = run_model(args.stock_config, args.checkpoint, frame_root, gt[0], len(gt), args.device, args.dtype)
    samurai_masks = run_model(args.samurai_config, args.checkpoint, frame_root, gt[0], len(gt), args.device, args.dtype)
    np.savez_compressed(args.output_root / "masks.npz", stock=stock_masks, samurai=samurai_masks, gt_xywh=gt)
    rows = []
    for frame_index in range(len(gt)):
        stock_box, samurai_box = mask_box(stock_masks[frame_index]), mask_box(samurai_masks[frame_index])
        rows.append({"frame_index": frame_index, "stock_iou": box_iou(stock_box, gt[frame_index]), "samurai_iou": box_iou(samurai_box, gt[frame_index]), "stock_center_error": center_error(stock_box, gt[frame_index]), "samurai_center_error": center_error(samurai_box, gt[frame_index]), "stock_area": int(stock_masks[frame_index].sum()), "samurai_area": int(samurai_masks[frame_index].sum()), "stock_inside_gt": mask_inside_box(stock_masks[frame_index], gt[frame_index]), "samurai_inside_gt": mask_inside_box(samurai_masks[frame_index], gt[frame_index]), "stock_samurai_mask_iou": mask_iou(stock_masks[frame_index], samurai_masks[frame_index]), **{f"stock_{key}": float(value) for key, value in zip(("x", "y", "w", "h"), stock_box)}, **{f"samurai_{key}": float(value) for key, value in zip(("x", "y", "w", "h"), samurai_box)}})
    with (args.output_root / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    first = cv2.imread(str(frame_paths[0])); height, width = first.shape[:2]
    video_path = args.output_root / "comparison.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1200, 600))
    deltas = np.asarray([row["samurai_iou"] - row["stock_iou"] for row in rows])
    keyframes = set(np.argsort(deltas)[:8].tolist()) | {0, len(rows) - 1}
    for frame_index, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path))
        stock_box = np.asarray([rows[frame_index][f"stock_{key}"] for key in ("x", "y", "w", "h")])
        samurai_box = np.asarray([rows[frame_index][f"samurai_{key}"] for key in ("x", "y", "w", "h")])
        full = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA); scale = np.asarray((800/width,600/height,800/width,600/height))
        draw_box(full, gt[frame_index]*scale, (0,255,0), 2); draw_box(full, stock_box*scale, (255,120,0), 2); draw_box(full, samurai_box*scale, (0,0,255), 2)
        cv2.putText(full, f"frame {frame_index+1}/{len(gt)}", (15,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(full, f"stock IoU {rows[frame_index]['stock_iou']:.3f}  samurai IoU {rows[frame_index]['samurai_iou']:.3f}", (15,58), cv2.FONT_HERSHEY_SIMPLEX, .62, (255,255,255), 2, cv2.LINE_AA)
        side = np.vstack((crop_panel(image, stock_masks[frame_index], gt[frame_index], stock_box, (255,120,0), "Stock SAM2"), crop_panel(image, samurai_masks[frame_index], gt[frame_index], samurai_box, (0,0,255), "SAMURAI motion")))
        canvas = np.hstack((full, cv2.resize(side, (400,600), interpolation=cv2.INTER_AREA))); writer.write(canvas)
        if frame_index in keyframes: cv2.imwrite(str(args.output_root / f"frame_{frame_index:04d}.jpg"), canvas)
    writer.release()
    stock_errors = np.asarray([row["stock_center_error"] for row in rows], dtype=float); samurai_errors = np.asarray([row["samurai_center_error"] for row in rows], dtype=float)
    summary = {"sequence": args.sequence_root.name, "frames": len(rows), "stock_mean_box_iou": float(np.mean([row["stock_iou"] for row in rows])), "samurai_mean_box_iou": float(np.mean([row["samurai_iou"] for row in rows])), "stock_median_center_error": float(np.median(stock_errors[np.isfinite(stock_errors)])), "samurai_median_center_error": float(np.median(samurai_errors[np.isfinite(samurai_errors)])), "stock_median_mask_area": float(np.median([row["stock_area"] for row in rows])), "samurai_median_mask_area": float(np.median([row["samurai_area"] for row in rows])), "stock_mean_inside_gt": float(np.mean([row["stock_inside_gt"] for row in rows])), "samurai_mean_inside_gt": float(np.mean([row["samurai_inside_gt"] for row in rows])), "mean_stock_samurai_mask_iou": float(np.mean([row["stock_samurai_mask_iou"] for row in rows])), "first_major_divergence_frame": next((row["frame_index"] for row in rows if row["stock_iou"]-row["samurai_iou"] >= .3), None), "worst_frames": sorted(rows, key=lambda row: row["samurai_iou"]-row["stock_iou"])[:10], "video": str(video_path)}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
