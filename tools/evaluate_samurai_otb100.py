from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SAMURAI_ROOT = REPO / "third_party/samurai"
SAM2_ROOT = SAMURAI_ROOT / "sam2"
for path in (REPO, SAM2_ROOT, SAMURAI_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sam2.build_sam import build_sam2_video_predictor
from qstr_dronedet.camera_motion import estimate_background_homography
from qstr_dronedet.tracking.online_action_bank import OnlineActionTrack


def parse_rect(line: str) -> list[float]:
    values = [float(value) for value in re.split(r"[,\t ]+", line.strip()) if value]
    if len(values) != 4:
        raise ValueError(f"expected xywh rectangle, got {line!r}")
    return values


def load_groundtruth(path: Path, start: int, stop: int | None) -> list[list[float]]:
    rows = [parse_rect(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    first = max(1, start)
    last = stop if stop is not None else first + len(rows) - 1
    expected = last - first + 1
    if len(rows) != expected:
        last = first + len(rows) - 1
    return rows[: last - first + 1]


def prepare_frames(source: Path, target: Path, start: int, count: int) -> None:
    marker = target / ".ready"
    if marker.is_file() and len(list(target.glob("*.jpg"))) == count:
        return
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for offset in range(count):
        frame_number = start + offset
        source_path = next(
            (
                candidate
                for width in (4, 5, 6, 8)
                if (candidate := source / f"{frame_number:0{width}d}.jpg").is_file()
            ),
            None,
        )
        if source_path is None:
            raise FileNotFoundError(f"missing frame {frame_number} under {source}")
        os.link(source_path, target / f"{offset:08d}.jpg")
    marker.write_text(str(count), encoding="utf-8")


def mask_box(mask_logits: torch.Tensor) -> list[float]:
    mask = mask_logits[0].detach().cpu().numpy() > 0.0
    positions = np.argwhere(mask)
    if not len(positions):
        return [0.0, 0.0, 0.0, 0.0]
    y_min, x_min = positions.min(axis=0)
    y_max, x_max = positions.max(axis=0)
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, width, height = box
    return float(x), float(y), float(x + width), float(y + height)


def xyxy_to_xywh(box: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def blend_xyxy(raw_box: tuple[float, float, float, float], predicted_box: tuple[float, float, float, float], weight: float) -> tuple[float, float, float, float]:
    weight = float(np.clip(weight, 0.0, 1.0))
    return tuple((1.0 - weight) * raw + weight * predicted for raw, predicted in zip(raw_box, predicted_box))


def valid_box(box: list[float]) -> bool:
    return len(box) == 4 and all(np.isfinite(box)) and box[2] > 0.0 and box[3] > 0.0


def scalar(value: Any, sigmoid: bool = False) -> float:
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        value = float(value.detach().float().mean().cpu())
    result = float(value)
    if sigmoid:
        result = 1.0 / (1.0 + np.exp(-np.clip(result, -30.0, 30.0)))
    return float(np.clip(result, 0.0, 1.0))


def output_quality(state: dict[str, Any], frame_index: int) -> tuple[float, float]:
    outputs = state["output_dict"]
    current = outputs["cond_frame_outputs"].get(frame_index) or outputs["non_cond_frame_outputs"].get(frame_index)
    if current is None:
        return 1.0, 1.0
    return scalar(current.get("best_iou_score")), scalar(current.get("object_score_logits"), sigmoid=True)


def iou_xywh(left: list[float], right: list[float]) -> float:
    ax1, ay1, aw, ah = left
    bx1, by1, bw, bh = right
    ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
    bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - intersection
    return intersection / max(1e-9, union)


def center_error(left: list[float], right: list[float]) -> float:
    return float(np.hypot(left[0] + 0.5 * left[2] - right[0] - 0.5 * right[2], left[1] + 0.5 * left[3] - right[1] - 0.5 * right[3]))


def sequence_metrics(predictions: list[list[float]], groundtruth: list[list[float]]) -> dict[str, float]:
    overlaps = np.asarray([iou_xywh(prediction, target) for prediction, target in zip(predictions, groundtruth)], dtype=np.float64)
    errors = np.asarray([center_error(prediction, target) for prediction, target in zip(predictions, groundtruth)], dtype=np.float64)
    thresholds = np.linspace(0.0, 1.0, 101)
    success_auc = float(np.mean([(overlaps >= threshold).mean() for threshold in thresholds]))
    return {
        "success_auc": success_auc,
        "precision_20px": float((errors <= 20.0).mean()),
        "mean_iou": float(overlaps.mean()),
        "frames": len(groundtruth),
    }


def summarize(results: list[dict[str, Any]], target: float) -> dict[str, Any]:
    frames = sum(int(result["frames"]) for result in results)
    weights = np.asarray([result["frames"] for result in results], dtype=np.float64)
    aucs = np.asarray([result["success_auc"] for result in results], dtype=np.float64)
    precisions = np.asarray([result["precision_20px"] for result in results], dtype=np.float64)
    return {
        "sequences": len(results),
        "frames": frames,
        "success_auc_percent_sequence_mean": float(100.0 * aucs.mean()) if len(aucs) else 0.0,
        "success_auc_percent_frame_weighted": float(100.0 * np.average(aucs, weights=weights)) if len(aucs) else 0.0,
        "precision_20px_percent": float(100.0 * np.average(precisions, weights=weights)) if len(precisions) else 0.0,
        "target_success_auc_percent": target,
        "target_met": bool(len(aucs) and 100.0 * aucs.mean() >= target),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SAMURAI CMC time-memory zero-shot tracking on OTB100.")
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\URAP_local_datasets\OTB100"))
    parser.add_argument("--metadata", type=Path, default=REPO / "data_templates/otb100_sequences.json")
    parser.add_argument("--checkpoint", type=Path, default=SAM2_ROOT / "checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--config", default="configs/samurai_cmc/sam2.1_hiera_b+.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\URAP_vatd_rank_results\otb100_samurai_cmc_timebank_v2"))
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--target", type=float, default=70.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--short-seconds", type=float, default=1.0)
    parser.add_argument("--long-seconds", type=float, default=3.0)
    parser.add_argument("--bank-min-native-quality", type=float, default=0.20)
    parser.add_argument("--bank-min-motion-score", type=float, default=0.42)
    parser.add_argument("--bank-blend-weight", type=float, default=0.10)
    parser.add_argument("--camera-max-size", type=int, default=512)
    parser.add_argument("--max-sequences", type=int)
    args = parser.parse_args()

    metadata: dict[str, dict[str, Any]] = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    selected = list(metadata.items())[: args.max_sequences]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    result_dir = args.output_dir / "sequence_results"
    frame_cache = args.output_dir / "frame_views"
    for path in (prediction_dir, result_dir, frame_cache):
        path.mkdir(parents=True, exist_ok=True)

    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda:0")
    results: list[dict[str, Any]] = []
    for sequence_index, (name, attributes) in enumerate(selected, start=1):
        result_path = result_dir / f"{name}.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if "raw_samurai" in result and int(result.get("frames", 0)) > 0:
                results.append(result)
                continue
        base = str(attributes.get("base", name))
        sequence_root = args.dataset_root / base
        groundtruth_path = sequence_root / str(attributes.get("groundtruth", "groundtruth_rect.txt"))
        start = int(attributes.get("start", 1))
        stop = int(attributes["stop"]) if attributes.get("stop") is not None else None
        groundtruth = load_groundtruth(groundtruth_path, start, stop)
        if not groundtruth:
            raise RuntimeError(f"{name}: no groundtruth rows loaded from {groundtruth_path}")
        frame_view = frame_cache / name
        prepare_frames(sequence_root / "img", frame_view, start, len(groundtruth))
        initial = groundtruth[0]
        initial_xyxy = [initial[0], initial[1], initial[0] + initial[2], initial[1] + initial[3]]
        raw_predictions: list[list[float]] = []
        predictions: list[list[float]] = []
        initial_box = xywh_to_xyxy(initial)
        bank = OnlineActionTrack(0, 0.0, initial_box, 1.0)
        identity = np.eye(3, dtype=np.float64)
        previous_image: np.ndarray | None = None
        state = None
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(str(frame_view), offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)
                predictor.add_new_points_or_box(state, box=initial_xyxy, frame_idx=0, obj_id=0)
                for frame_index, object_ids, masks in predictor.propagate_in_video(state):
                    if len(object_ids) != 1 or len(masks) != 1:
                        raise RuntimeError("OTB evaluator supports exactly one object")
                    current_image = cv2.imread(str(frame_view / f"{frame_index:08d}.jpg"), cv2.IMREAD_COLOR)
                    if current_image is None:
                        raise FileNotFoundError(frame_view / f"{frame_index:08d}.jpg")
                    if previous_image is None:
                        camera_transform = identity
                        camera_validity = 1.0
                    else:
                        camera = estimate_background_homography(previous_image, current_image, max_size=args.camera_max_size)
                        camera_transform = camera.matrix if camera.valid else identity
                        camera_validity = camera.inlier_ratio if camera.valid else 0.0
                    previous_image = current_image
                    raw_prediction = mask_box(masks[0])
                    raw_predictions.append(raw_prediction)
                    if args.progress_json and ((frame_index + 1) % 100 == 0 or frame_index + 1 == len(groundtruth)):
                        frame_payload = {"stage": "otb100_sequence_frames", "done": sequence_index - 1, "total": len(selected), "last_sequence": name, "sequence_frame": frame_index + 1, "sequence_total": len(groundtruth), "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 3)}
                        args.progress_json.parent.mkdir(parents=True, exist_ok=True)
                        args.progress_json.write_text(json.dumps(frame_payload, indent=2), encoding="utf-8")
                        print(json.dumps(frame_payload), flush=True)
                    if frame_index == 0:
                        predictions.append(list(initial))
                        continue
                    timestamp = frame_index / args.fps
                    predicted_box = bank.predict(timestamp, camera_transform, args.short_seconds, args.long_seconds)
                    predicted_iou, object_score = output_quality(state, frame_index)
                    native_quality = predicted_iou * object_score
                    if valid_box(raw_prediction):
                        raw_box = xywh_to_xyxy(raw_prediction)
                        motion = bank.score_candidate(raw_box, timestamp, camera_transform, camera_validity, args.short_seconds, args.long_seconds)
                    else:
                        raw_box = None
                        motion = None
                    reliable = raw_box is not None and native_quality >= args.bank_min_native_quality and motion.score >= args.bank_min_motion_score
                    if raw_box is not None:
                        if reliable:
                            agreement = float(np.clip((motion.score - 0.5) * 2.0, 0.0, 1.0))
                            blend_weight = args.bank_blend_weight * native_quality * agreement
                            output_box = blend_xyxy(raw_box, predicted_box, blend_weight)
                            bank.update(frame_index, timestamp, raw_box, native_quality, motion.score, camera_transform, args.long_seconds)
                        else:
                            output_box = raw_box
                        predictions.append(xyxy_to_xywh(output_box))
                    else:
                        predictions.append(xyxy_to_xywh(predicted_box))
        finally:
            if state is not None:
                predictor.reset_state(state)
            gc.collect()
            torch.cuda.empty_cache()
        if len(predictions) != len(groundtruth) or len(raw_predictions) != len(groundtruth):
            raise RuntimeError(f"{name}: prediction length {len(predictions)} raw {len(raw_predictions)} != groundtruth {len(groundtruth)}")
        raw_metrics = sequence_metrics(raw_predictions, groundtruth)
        metrics = {"sequence": name, **sequence_metrics(predictions, groundtruth), "raw_samurai": raw_metrics}
        result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with (prediction_dir / f"{name}.txt").open("w", encoding="utf-8") as handle:
            for prediction in predictions:
                handle.write(",".join(f"{value:.4f}" for value in prediction) + "\n")
        with (prediction_dir / f"{name}_raw_samurai.txt").open("w", encoding="utf-8") as handle:
            for prediction in raw_predictions:
                handle.write(",".join(f"{value:.4f}" for value in prediction) + "\n")
        results.append(metrics)
        payload = {"stage": "otb100_zero_shot", "done": sequence_index, "total": len(selected), "last_sequence": name, "last_result": metrics}
        if args.progress_json:
            args.progress_json.parent.mkdir(parents=True, exist_ok=True)
            args.progress_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"kind": "otb100_sequence_done", **payload, "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 3)}), flush=True)

    summary = summarize(results, args.target)
    raw_summary = summarize([{"success_auc": result["raw_samurai"]["success_auc"], "precision_20px": result["raw_samurai"]["precision_20px"], "frames": result["raw_samurai"]["frames"]} for result in results], args.target)
    summary["raw_samurai"] = raw_summary
    benchmark_complete = args.max_sequences is None and len(metadata) >= 100 and len(results) == len(metadata)
    summary["benchmark_complete"] = benchmark_complete
    summary["target_met"] = bool(benchmark_complete and summary["target_met"])
    summary.update({"benchmark": "OTB100" if benchmark_complete else "OTB100 partial/smoke", "tracker": "SAMURAI optical-flow CMC + causal 1-second/3-second Action Bank", "checkpoint": str(args.checkpoint), "action_bank": {"short_seconds": args.short_seconds, "long_seconds": args.long_seconds, "fps": args.fps, "min_native_quality": args.bank_min_native_quality, "min_motion_score": args.bank_min_motion_score, "blend_weight": args.bank_blend_weight, "camera_compensation": "background LK optical flow + RANSAC homography", "camera_max_size": args.camera_max_size}, "results": results})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"kind": "otb100_zero_shot_done", **summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


