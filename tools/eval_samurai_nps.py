#!/usr/bin/env python
"""Evaluate official SAMURAI on first-frame-prompt NPS sequences."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SAM2_ROOT = ROOT / "third_party" / "samurai" / "sam2"
for path in (ROOT, SAM2_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sam2.build_sam import build_sam2, build_sam2_video_predictor  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", default="configs/samurai/sam2.1_hiera_t.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--sequence-shard-count", type=int, default=1)
    parser.add_argument("--sequence-shard-index", type=int, default=0)
    parser.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--async-loading-frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--propagation-mode",
        choices=("video", "image-box"),
        default="video",
        help="video uses SAM2 memory; image-box re-prompts each image with the previous predicted box",
    )
    parser.add_argument(
        "--feature-output",
        type=Path,
        help="Optional NPZ export of per-frame SAM2 object-pointer features for a frozen bbox readout",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse complete per-sequence CSV/feature outputs")
    return parser.parse_args()


def read_gt(path: Path) -> list[tuple[float, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip():
            rows.append(tuple(float(value) for value in line.split(",")))
    return rows


def mask_to_xywh(mask: np.ndarray) -> tuple[float, float, float, float]:
    points = np.argwhere(mask)
    if not len(points):
        return 0.0, 0.0, 0.0, 0.0
    y1, x1 = points.min(axis=0)
    y2, x2 = points.max(axis=0)
    return float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if a[2] <= 0 or a[3] <= 0 or b[2] <= 0 or b[3] <= 0:
        return 0.0
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    intersection = max(0.0, min(ax2, bx2) - max(a[0], b[0])) * max(0.0, min(ay2, by2) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0.0


def center_error(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if a[2] <= 0 or a[3] <= 0 or b[2] <= 0 or b[3] <= 0:
        return float("inf")
    return math.hypot((a[0] + a[2] / 2) - (b[0] + b[2] / 2), (a[1] + a[3] / 2) - (b[1] + b[3] / 2))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        for attempt in range(100):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 99:
                    raise
                time.sleep(0.05)
    finally:
        temporary.unlink(missing_ok=True)


def read_prediction_csv(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, object] = {"sequence": row["sequence"]}
            parsed["frame_index"] = int(row["frame_index"])
            parsed["visible"] = int(row["visible"])
            for key in ("pred_x", "pred_y", "pred_w", "pred_h", "gt_x", "gt_y", "gt_w", "gt_h"):
                parsed[key] = float(row[key])
            parsed["iou"] = None if row["iou"] in ("", "None") else float(row["iou"])
            parsed["center_error"] = None if row["center_error"] in ("", "None") else float(row["center_error"])
            rows.append(parsed)
    return rows


def summarize_sequence(name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    visible_rows = [row for row in rows if row["visible"]]
    ious = [float(row["iou"]) for row in visible_rows]
    errors = [float(row["center_error"]) for row in visible_rows]
    thresholds = np.linspace(0.0, 1.0, 21)
    return {
        "sequence": name,
        "frames": len(rows),
        "visible_frames": len(visible_rows),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "success_auc": float(np.mean([np.mean(np.asarray(ious) >= threshold) for threshold in thresholds])) if ious else 0.0,
        "success_50": float(np.mean(np.asarray(ious) >= 0.5)) if ious else 0.0,
        "precision_5": float(np.mean(np.asarray(errors) <= 5.0)) if errors else 0.0,
        "precision_10": float(np.mean(np.asarray(errors) <= 10.0)) if errors else 0.0,
        "precision_20": float(np.mean(np.asarray(errors) <= 20.0)) if errors else 0.0,
    }


def read_rgb(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_box_predictions(predictor, frame_paths: list[Path], initial_box: tuple[float, ...]) -> list[tuple[float, ...]]:
    predictions = []
    prompt_box = initial_box
    for frame_index, frame_path in enumerate(frame_paths):
        predictor.set_image(read_rgb(frame_path))
        x, y, width, height = prompt_box
        prompt_xyxy = np.asarray([x, y, x + width, y + height], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=prompt_xyxy, multimask_output=True)
        best_index = int(np.argmax(scores))
        prediction = mask_to_xywh(np.asarray(masks[best_index], dtype=bool))
        predictions.append(prediction)
        if prediction[2] > 0 and prediction[3] > 0:
            prompt_box = prediction
        elif frame_index == 0:
            prompt_box = initial_box
    return predictions


def current_object_pointer(state: dict[str, object], frame_index: int, object_id: int = 0) -> np.ndarray:
    object_index = state["obj_id_to_idx"][object_id]
    output = state["output_dict_per_obj"][object_index]
    frame_output = output["cond_frame_outputs"].get(frame_index)
    if frame_output is None:
        frame_output = output["non_cond_frame_outputs"][frame_index]
    return frame_output["obj_ptr"].detach().float().cpu().reshape(-1).numpy()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    all_names = [line.strip() for line in (args.dataset_root / f"{args.split}_set.txt").read_text().splitlines() if line.strip()]
    if args.sequence_shard_count < 1 or not 0 <= args.sequence_shard_index < args.sequence_shard_count:
        raise ValueError("Invalid sequence shard")
    global_sequence_indices = {name: index for index, name in enumerate(all_names)}
    names = [name for index, name in enumerate(all_names) if index % args.sequence_shard_count == args.sequence_shard_index]
    if args.max_sequences is not None:
        names = names[: args.max_sequences]
    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions_root = args.output_root / "predictions"
    predictions_root.mkdir(exist_ok=True)
    feature_chunks_root = args.output_root / "feature_chunks"
    if args.feature_output is not None:
        feature_chunks_root.mkdir(exist_ok=True)
    progress_path = args.output_root / "progress.json"
    progress = {"status": "initializing", "done_sequences": 0, "total_sequences": len(names), "done_frames": 0, "last_update": utc_now()}
    write_json_atomic(progress_path, progress)

    if args.propagation_mode == "video":
        predictor = build_sam2_video_predictor(args.model_config, str(args.checkpoint), device=args.device)
    else:
        predictor = SAM2ImagePredictor(build_sam2(args.model_config, str(args.checkpoint), device=args.device))
    autocast_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    all_rows: list[dict[str, object]] = []
    sequence_summaries: list[dict[str, object]] = []
    feature_rows: list[np.ndarray] = []
    feature_targets: list[tuple[float, ...]] = []
    feature_previous_boxes: list[tuple[float, ...]] = []
    feature_mask_boxes: list[tuple[float, ...]] = []
    feature_sequence_ids: list[int] = []
    feature_frame_indices: list[int] = []
    feature_image_sizes: list[tuple[int, int]] = []
    progress["status"] = "running"

    for sequence_index, name in enumerate(names, 1):
        sequence_root = args.dataset_root / "lasot" / "uav" / name
        frame_root = sequence_root / "img"
        gt = read_gt(sequence_root / "groundtruth.txt")
        frame_limit = min(len(gt), args.max_frames) if args.max_frames else len(gt)
        gt = gt[:frame_limit]
        initial = gt[0]
        prediction_path = predictions_root / f"{name}.csv"
        feature_chunk_path = feature_chunks_root / f"{name}.npz"
        can_resume = args.resume and prediction_path.is_file()
        if args.feature_output is not None:
            can_resume = can_resume and feature_chunk_path.is_file()
        if can_resume:
            sequence_rows = read_prediction_csv(prediction_path)
            if len(sequence_rows) == frame_limit:
                all_rows.extend(sequence_rows)
                sequence_summaries.append(summarize_sequence(name, sequence_rows))
                if args.feature_output is not None:
                    chunk = np.load(feature_chunk_path)
                    feature_rows.extend(chunk["object_pointer"])
                    feature_targets.extend(chunk["target_xywh"])
                    feature_previous_boxes.extend(chunk["previous_xywh"])
                    feature_mask_boxes.extend(chunk["mask_xywh"])
                    feature_sequence_ids.extend(chunk["sequence_id"])
                    feature_frame_indices.extend(chunk["frame_index"])
                    feature_image_sizes.extend(chunk["image_wh"])
                progress.update(
                    done_sequences=sequence_index,
                    done_frames=len(all_rows),
                    last_completed_sequence=name,
                    last_update=utc_now(),
                )
                write_json_atomic(progress_path, progress)
                continue
        sequence_rows: list[dict[str, object]] = []
        feature_start = len(feature_rows)
        if args.propagation_mode == "video":
            prompt = np.array([initial[0], initial[1], initial[0] + initial[2], initial[1] + initial[3]], dtype=np.float32)
            state = predictor.init_state(
                str(frame_root),
                offload_video_to_cpu=args.offload_video_to_cpu,
                offload_state_to_cpu=args.offload_state_to_cpu,
                async_loading_frames=args.async_loading_frames,
                max_frames=frame_limit,
            )
            predictor.add_new_points_or_box(state, box=prompt, frame_idx=0, obj_id=0)
            with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype, enabled=args.device.startswith("cuda")):
                generated = []
                previous_box = initial
                for frame_idx, object_ids, masks in predictor.propagate_in_video(state, max_frame_num_to_track=frame_limit):
                    if frame_idx >= frame_limit:
                        break
                    object_position = list(object_ids).index(0)
                    mask = masks[object_position][0].detach().cpu().numpy() > 0.0
                    prediction = mask_to_xywh(mask)
                    generated.append((frame_idx, prediction))
                    if args.feature_output is not None:
                        height, width = mask.shape
                        feature_rows.append(current_object_pointer(state, frame_idx))
                        feature_targets.append(gt[frame_idx])
                        feature_previous_boxes.append(previous_box)
                        feature_mask_boxes.append(prediction)
                        feature_sequence_ids.append(global_sequence_indices[name])
                        feature_frame_indices.append(frame_idx)
                        feature_image_sizes.append((width, height))
                    if prediction[2] > 0 and prediction[3] > 0:
                        previous_box = prediction
        else:
            frame_paths = sorted(frame_root.glob("*.jpg"))[:frame_limit]
            with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype, enabled=args.device.startswith("cuda")):
                generated = list(enumerate(image_box_predictions(predictor, frame_paths, initial)))

        for frame_idx, prediction in generated:
                target = gt[frame_idx]
                visible = target[2] > 0 and target[3] > 0
                iou = box_iou(prediction, target) if visible else None
                error = center_error(prediction, target) if visible else None
                row = {
                    "sequence": name,
                    "frame_index": frame_idx,
                    "pred_x": prediction[0],
                    "pred_y": prediction[1],
                    "pred_w": prediction[2],
                    "pred_h": prediction[3],
                    "gt_x": target[0],
                    "gt_y": target[1],
                    "gt_w": target[2],
                    "gt_h": target[3],
                    "visible": int(visible),
                    "iou": iou,
                    "center_error": error,
                }
                sequence_rows.append(row)
                all_rows.append(row)
                progress["done_frames"] += 1
                progress.update({"last_sequence": name, "last_frame": frame_idx, "last_update": utc_now()})
                if frame_idx % 100 == 0 or frame_idx == frame_limit - 1:
                    write_json_atomic(progress_path, progress)

        summary = summarize_sequence(name, sequence_rows)
        sequence_summaries.append(summary)
        with prediction_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sequence_rows[0]))
            writer.writeheader()
            writer.writerows(sequence_rows)
        if args.feature_output is not None:
            np.savez_compressed(
                feature_chunk_path,
                object_pointer=np.asarray(feature_rows[feature_start:], dtype=np.float32),
                target_xywh=np.asarray(feature_targets[feature_start:], dtype=np.float32),
                previous_xywh=np.asarray(feature_previous_boxes[feature_start:], dtype=np.float32),
                mask_xywh=np.asarray(feature_mask_boxes[feature_start:], dtype=np.float32),
                sequence_id=np.asarray(feature_sequence_ids[feature_start:], dtype=np.int32),
                frame_index=np.asarray(feature_frame_indices[feature_start:], dtype=np.int32),
                image_wh=np.asarray(feature_image_sizes[feature_start:], dtype=np.int32),
            )
        progress["done_sequences"] = sequence_index
        progress.update({"last_completed_sequence": name, "last_update": utc_now()})
        write_json_atomic(progress_path, progress)
        if args.propagation_mode == "video":
            del state
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    visible_rows = [row for row in all_rows if row["visible"]]
    ious = np.asarray([row["iou"] for row in visible_rows], dtype=np.float64)
    errors = np.asarray([row["center_error"] for row in visible_rows], dtype=np.float64)
    thresholds = np.linspace(0.0, 1.0, 21)
    report = {
        "model_config": args.model_config,
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "dtype": args.dtype,
        "propagation_mode": args.propagation_mode,
        "sequences": len(sequence_summaries),
        "frames": len(all_rows),
        "visible_frames": len(visible_rows),
        "mean_iou": float(ious.mean()) if len(ious) else 0.0,
        "success_auc": float(np.mean([(ious >= threshold).mean() for threshold in thresholds])) if len(ious) else 0.0,
        "success_50": float((ious >= 0.5).mean()) if len(ious) else 0.0,
        "precision_5": float((errors <= 5.0).mean()) if len(errors) else 0.0,
        "precision_10": float((errors <= 10.0).mean()) if len(errors) else 0.0,
        "precision_20": float((errors <= 20.0).mean()) if len(errors) else 0.0,
        "sequence_results": sequence_summaries,
    }
    write_json_atomic(args.output_root / "metrics.json", report)
    if args.feature_output is not None:
        if args.propagation_mode != "video":
            raise ValueError("--feature-output is supported only with --propagation-mode video")
        args.feature_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.feature_output,
            object_pointer=np.asarray(feature_rows, dtype=np.float32),
            target_xywh=np.asarray(feature_targets, dtype=np.float32),
            previous_xywh=np.asarray(feature_previous_boxes, dtype=np.float32),
            mask_xywh=np.asarray(feature_mask_boxes, dtype=np.float32),
            sequence_id=np.asarray(feature_sequence_ids, dtype=np.int32),
            frame_index=np.asarray(feature_frame_indices, dtype=np.int32),
            image_wh=np.asarray(feature_image_sizes, dtype=np.int32),
        )
    progress.update({"status": "completed", "last_update": utc_now()})
    write_json_atomic(progress_path, progress)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
