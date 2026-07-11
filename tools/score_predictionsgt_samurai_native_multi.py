from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.score_predictionsgt_samurai_native import (
    SAM2_ROOT,
    build_sam2_video_predictor,
    finite,
    image_key,
    iou,
    load_predictionsgt,
    mask_box,
    prepare_frames,
)


def select_distinct_detections(detections: list[dict[str, Any]], max_objects: int, nms_iou: float, score_gate: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda row: finite(row.get("score")), reverse=True):
        if finite(detection.get("score")) < score_gate:
            break
        if any(iou(detection["bbox"], kept["bbox"]) >= nms_iou for kept in selected):
            continue
        selected.append(detection)
        if len(selected) >= max_objects:
            break
    return selected


def tensor_values(value: Any, count: int, sigmoid: bool = False) -> list[float]:
    if count <= 0:
        return []
    if value is None:
        return [1.0] * count
    if isinstance(value, torch.Tensor):
        array = value.detach().float().cpu().numpy().reshape(-1)
    else:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    if sigmoid:
        array = 1.0 / (1.0 + np.exp(-np.clip(array, -30.0, 30.0)))
    array = np.clip(array, 0.0, 1.0)
    if not len(array):
        return [1.0] * count
    if len(array) < count:
        array = np.pad(array, (0, count - len(array)), constant_values=float(array[-1]))
    return [float(value) for value in array[:count]]


def output_qualities(state: dict[str, Any], frame_index: int, object_count: int) -> tuple[list[float], list[float]]:
    output = state["output_dict"]
    current = output["cond_frame_outputs"].get(frame_index) or output["non_cond_frame_outputs"].get(frame_index)
    if current is None:
        return [1.0] * object_count, [1.0] * object_count
    return tensor_values(current.get("best_iou_score"), object_count), tensor_values(current.get("object_score_logits"), object_count, sigmoid=True)


def initialize_objects(predictor: Any, state: dict[str, Any], frame_index: int, detections: list[dict[str, Any]], max_objects: int, nms_iou: float, score_gate: float) -> tuple[dict[int, tuple[float, float, float, float]], list[dict[str, Any]]]:
    selected = select_distinct_detections(detections, max_objects, nms_iou, score_gate)
    current_boxes: dict[int, tuple[float, float, float, float]] = {}
    for obj_id, detection in enumerate(selected):
        _, object_ids, masks = predictor.add_new_points_or_box(state, box=detection["bbox"], frame_idx=frame_index, obj_id=obj_id)
        for returned_id, mask in zip(object_ids, masks):
            resolved = mask_box(mask)
            if resolved is not None:
                current_boxes[int(returned_id)] = resolved
    return current_boxes, selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Score TVD candidates with multi-object native causal SAMURAI memory.")
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=SAM2_ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt")
    parser.add_argument("--config", default="configs/samurai_cmc/sam2.1_hiera_b+.yaml")
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--object-nms-iou", type=float, default=0.55)
    parser.add_argument("--start-gate", type=float, default=0.45)
    parser.add_argument("--reset-gate", type=float, default=0.70)
    parser.add_argument("--reset-iou", type=float, default=0.05)
    parser.add_argument("--object-gate", type=float, default=0.20)
    parser.add_argument("--reset-patience", type=int, default=2)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--sequence-fps-json", type=Path)
    parser.add_argument("--long-seconds", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    predictions = load_predictionsgt(args.predictionsgt_pkl)
    sequence_fps = json.loads(args.sequence_fps_json.read_text(encoding="utf-8-sig")) if args.sequence_fps_json else {}
    grouped: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
    for image_id, item in predictions.items():
        sequence, frame_id, _ = image_key(str(image_id), 0)
        if args.sequences and sequence not in args.sequences:
            continue
        grouped.setdefault(sequence, []).append((frame_id, str(image_id), item))
    for frames in grouped.values():
        frames.sort(key=lambda item: item[0])
        if args.max_frames:
            del frames[args.max_frames:]

    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda:0")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    completed = []
    total_frame_count = sum(len(frames) for frames in grouped.values())
    completed_frame_count = 0
    with args.output_jsonl.open("w", encoding="utf-8") as target:
        for sequence_index, (sequence, frames) in enumerate(sorted(grouped.items()), start=1):
            fps = float(sequence_fps.get(sequence, args.fps))
            frame_ids = [frame_id for frame_id, _, _ in frames]
            frame_view = args.frame_cache / sequence
            prepare_frames(args.frame_root, sequence, frame_ids, frame_view)
            state = None
            current_boxes: dict[int, tuple[float, float, float, float]] = {}
            low_quality_frames = 0
            disagreement_frames = 0
            sequence_rows = 0
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    state = predictor.init_state(str(frame_view), offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)
                    for frame_index, (frame_id, image_id, item) in enumerate(frames):
                        detections = list(item.get("detections") or [])
                        reset_event = False
                        if not current_boxes:
                            current_boxes, selected = initialize_objects(predictor, state, frame_index, detections, args.max_objects, args.object_nms_iou, args.start_gate)
                            predicted_ious = [1.0] * len(current_boxes)
                            object_scores = [1.0] * len(current_boxes)
                        else:
                            outputs = list(predictor.propagate_in_video(state, start_frame_idx=frame_index, max_frame_num_to_track=0))
                            object_ids: list[int] = []
                            if outputs:
                                _, returned_ids, masks = outputs[-1]
                                object_ids = [int(value) for value in returned_ids]
                                current_boxes = {}
                                for obj_id, mask in zip(object_ids, masks):
                                    resolved = mask_box(mask)
                                    if resolved is not None:
                                        current_boxes[obj_id] = resolved
                            predicted_ious, object_scores = output_qualities(state, frame_index, len(object_ids))
                            best_native_quality = max((a * b for a, b in zip(predicted_ious, object_scores)), default=0.0)
                            strong = select_distinct_detections(detections, args.max_objects, args.object_nms_iou, args.reset_gate)
                            disagreement = bool(strong) and any(all(iou(detection["bbox"], tracked) < args.reset_iou for tracked in current_boxes.values()) for detection in strong)
                            low_quality_frames = low_quality_frames + 1 if not current_boxes or best_native_quality < args.object_gate else 0
                            disagreement_frames = disagreement_frames + 1 if disagreement else 0
                            if low_quality_frames >= args.reset_patience or disagreement_frames >= args.reset_patience:
                                predictor.reset_state(state)
                                current_boxes, selected = initialize_objects(predictor, state, frame_index, detections, args.max_objects, args.object_nms_iou, args.start_gate)
                                predicted_ious = [1.0] * len(current_boxes)
                                object_scores = [1.0] * len(current_boxes)
                                low_quality_frames = 0
                                disagreement_frames = 0
                                reset_event = True
                        object_ids = list(current_boxes)
                        if len(predicted_ious) != len(object_ids):
                            predicted_ious = (predicted_ious + [1.0] * len(object_ids))[: len(object_ids)]
                        if len(object_scores) != len(object_ids):
                            object_scores = (object_scores + [1.0] * len(object_ids))[: len(object_ids)]
                        rows = []
                        for candidate_index, detection in enumerate(detections):
                            best_score = best_overlap = best_iou_quality = best_object_quality = 0.0
                            best_object_id = -1
                            for index, obj_id in enumerate(object_ids):
                                overlap = iou(current_boxes[obj_id], detection["bbox"])
                                iou_quality = predicted_ious[index]
                                object_quality = object_scores[index]
                                score = overlap * (0.35 + 0.65 * iou_quality) * (0.35 + 0.65 * object_quality)
                                if score > best_score:
                                    best_score, best_overlap = score, overlap
                                    best_iou_quality, best_object_quality, best_object_id = iou_quality, object_quality, obj_id
                            rows.append({
                                "seq": sequence, "frame_id": frame_id, "prediction_index": candidate_index,
                                "samurai_native_score": float(np.clip(best_score, 0.0, 1.0)),
                                "samurai_native_iou": best_overlap,
                                "samurai_native_predicted_iou": best_iou_quality,
                                "samurai_native_object_score": best_object_quality,
                                "samurai_native_active": float(bool(current_boxes)),
                                "samurai_native_reset_event": float(reset_event),
                                "samurai_native_low_object_frames": low_quality_frames,
                                "samurai_native_disagreement_frames": disagreement_frames,
                                "samurai_native_object_count": len(current_boxes),
                                "samurai_native_best_object_id": best_object_id,
                            })
                        target.write(json.dumps({"meta": {"seq": sequence, "image_id": image_id}, "rows": rows}, separators=(",", ":")) + "\n")
                        sequence_rows += len(rows)
                        if args.progress_json and (frame_index == 0 or (frame_index + 1) % 100 == 0 or frame_index + 1 == len(frames)):
                            payload = {"stage": "samurai_native_multi_frames", "done": completed_frame_count + frame_index + 1, "total": total_frame_count, "sequence": sequence, "frame_id": frame_id, "sequence_frame": frame_index + 1, "sequence_total": len(frames), "objects": len(current_boxes), "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 3)}
                            args.progress_json.parent.mkdir(parents=True, exist_ok=True)
                            args.progress_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                            print(json.dumps(payload), flush=True)
                completed.append({"sequence": sequence, "fps": fps, "frames": len(frames), "rows": sequence_rows})
                completed_frame_count += len(frames)
            finally:
                if state is not None:
                    predictor.reset_state(state)
                gc.collect()
                torch.cuda.empty_cache()
    summary = {"kind": "samurai_native_multi_scoring_done", "sequences": completed, "output_jsonl": str(args.output_jsonl), "max_objects": args.max_objects, "object_nms_iou": args.object_nms_iou, "start_gate": args.start_gate, "reset_gate": args.reset_gate, "reset_iou": args.reset_iou, "object_gate": args.object_gate, "reset_patience": args.reset_patience}
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
