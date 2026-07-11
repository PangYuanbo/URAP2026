from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SAMURAI_ROOT = REPO / "third_party" / "samurai"
SAM2_ROOT = SAMURAI_ROOT / "sam2"
for entry in (str(REPO), str(SAM2_ROOT), str(SAMURAI_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from sam2.build_sam import build_sam2_video_predictor
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import image_key


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def iou(left, right) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in left]
    bx1, by1, bx2, by2 = [float(value) for value in right]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / max(1e-9, union)


def mask_box(mask_logits: torch.Tensor) -> tuple[float, float, float, float] | None:
    mask = mask_logits[0].detach().cpu().numpy() > 0.0
    positions = np.argwhere(mask)
    if not len(positions):
        return None
    y_min, x_min = positions.min(axis=0)
    y_max, x_max = positions.max(axis=0)
    return float(x_min), float(y_min), float(x_max + 1), float(y_max + 1)


def scalar(value: Any, sigmoid: bool = False) -> float:
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        value = float(value.detach().float().mean().cpu())
    result = finite(value)
    if sigmoid:
        result = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, result))))
    return float(np.clip(result, 0.0, 1.0))


def output_quality(state, frame_index: int) -> tuple[float, float]:
    output = state["output_dict"]
    current = output["cond_frame_outputs"].get(frame_index) or output["non_cond_frame_outputs"].get(frame_index)
    if current is None:
        return 1.0, 1.0
    return scalar(current.get("best_iou_score")), scalar(current.get("object_score_logits"), sigmoid=True)


def prepare_frames(frame_root: Path, sequence: str, frame_ids: list[int], target: Path) -> None:
    marker = target / ".ready"
    expected = "\n".join(str(frame_id) for frame_id in frame_ids)
    if marker.is_file() and marker.read_text(encoding="utf-8") == expected and len(list(target.glob("*.jpg"))) == len(frame_ids):
        return
    if target.exists():
        for path in target.iterdir():
            path.unlink()
    else:
        target.mkdir(parents=True)
    for index, frame_id in enumerate(frame_ids):
        source = frame_root / f"{sequence}_{frame_id:05d}.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = target / f"{index:08d}.jpg"
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    marker.write_text(expected, encoding="utf-8")


def top_detection(detections):
    return max(detections, key=lambda row: finite(row.get("score"))) if detections else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Score TVD candidates with native causal SAMURAI memory and predicted IoU.")
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=SAM2_ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt")
    parser.add_argument("--config", default="configs/samurai_cmc/sam2.1_hiera_b+.yaml")
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--start-gate", type=float, default=0.55)
    parser.add_argument("--reset-gate", type=float, default=0.70)
    parser.add_argument("--reset-iou", type=float, default=0.05)
    parser.add_argument("--object-gate", type=float, default=0.20)
    parser.add_argument("--reset-policy", choices=["any", "quality-only", "persistent"], default="persistent")
    parser.add_argument("--reset-patience", type=int, default=3)
    parser.add_argument("--disagreement-reset-gate", type=float, default=0.90)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--sequence-fps-json", type=Path)
    parser.add_argument("--long-seconds", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    predictions = load_predictionsgt(args.predictionsgt_pkl)
    sequence_fps = json.loads(args.sequence_fps_json.read_text(encoding="utf-8-sig")) if args.sequence_fps_json else {}
    grouped = {}
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
            active = False
            low_object_frames = 0
            disagreement_frames = 0
            current_box = None
            sequence_rows = 0
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    state = predictor.init_state(str(frame_view), offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)
                    for frame_index, (frame_id, image_id, item) in enumerate(frames):
                        detections = list(item.get("detections") or [])
                        top = top_detection(detections)
                        top_score = finite(top.get("score")) if top else 0.0
                        predicted_iou = object_score = 0.0
                        reset_event = False
                        if not active and top is not None and top_score >= args.start_gate:
                            _, _, masks = predictor.add_new_points_or_box(state, box=top["bbox"], frame_idx=frame_index, obj_id=0)
                            current_box = mask_box(masks[0]) or tuple(float(value) for value in top["bbox"])
                            predicted_iou, object_score = 1.0, 1.0
                            active = True
                            low_object_frames = 0
                            disagreement_frames = 0
                        elif active:
                            outputs = list(predictor.propagate_in_video(state, start_frame_idx=frame_index, max_frame_num_to_track=0))
                            if outputs:
                                _, object_ids, masks = outputs[-1]
                                if len(object_ids) == 1 and len(masks) == 1:
                                    current_box = mask_box(masks[0])
                            predicted_iou, object_score = output_quality(state, frame_index)
                            quality_bad = current_box is None or object_score < args.object_gate
                            disagreement = (
                                current_box is not None
                                and top is not None
                                and top_score >= args.disagreement_reset_gate
                                and iou(current_box, top["bbox"]) < args.reset_iou
                            )
                            low_object_frames = low_object_frames + 1 if quality_bad else 0
                            disagreement_frames = disagreement_frames + 1 if disagreement else 0
                            if args.reset_policy == "any":
                                should_reset = top is not None and top_score >= args.reset_gate and (quality_bad or disagreement)
                            elif args.reset_policy == "quality-only":
                                should_reset = top is not None and top_score >= args.reset_gate and low_object_frames >= args.reset_patience
                            else:
                                should_reset = (
                                    top is not None
                                    and top_score >= args.reset_gate
                                    and (low_object_frames >= args.reset_patience or disagreement_frames >= args.reset_patience)
                                )
                            reset_event = False
                            if should_reset:
                                predictor.reset_state(state)
                                _, _, masks = predictor.add_new_points_or_box(state, box=top["bbox"], frame_idx=frame_index, obj_id=0)
                                current_box = mask_box(masks[0]) or tuple(float(value) for value in top["bbox"])
                                predicted_iou, object_score = 1.0, 1.0
                                low_object_frames = 0
                                disagreement_frames = 0
                                reset_event = True
                            if low_object_frames > int(round(args.long_seconds * fps)):
                                predictor.reset_state(state)
                                active = False
                                current_box = None
                                low_object_frames = 0
                                disagreement_frames = 0
                        rows = []
                        for candidate_index, detection in enumerate(detections):
                            overlap = iou(current_box, detection["bbox"]) if current_box is not None else 0.0
                            score = overlap * (0.35 + 0.65 * predicted_iou) * (0.35 + 0.65 * object_score)
                            rows.append({
                                "seq": sequence, "frame_id": frame_id, "prediction_index": candidate_index,
                                "samurai_native_score": float(np.clip(score, 0.0, 1.0)),
                                "samurai_native_iou": overlap, "samurai_native_predicted_iou": predicted_iou,
                                "samurai_native_object_score": object_score, "samurai_native_active": float(active),
                                "samurai_native_reset_event": float(reset_event),
                                "samurai_native_low_object_frames": low_object_frames,
                                "samurai_native_disagreement_frames": disagreement_frames,
                            })
                        target.write(json.dumps({"meta": {"seq": sequence, "image_id": image_id}, "rows": rows}, separators=(",", ":")) + "\n")
                        sequence_rows += len(rows)
                        if args.progress_json and (frame_index == 0 or (frame_index + 1) % 100 == 0 or frame_index + 1 == len(frames)):
                            frame_payload = {"stage": "samurai_native_frames", "done": completed_frame_count + frame_index + 1, "total": total_frame_count, "sequence": sequence, "frame_id": frame_id, "sequence_frame": frame_index + 1, "sequence_total": len(frames), "fps": fps, "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 3)}
                            args.progress_json.parent.mkdir(parents=True, exist_ok=True)
                            args.progress_json.write_text(json.dumps(frame_payload, indent=2), encoding="utf-8")
                            print(json.dumps(frame_payload), flush=True)
                completed.append({"sequence": sequence, "fps": fps, "frames": len(frames), "rows": sequence_rows})
                payload = {"stage": "samurai_native_scoring", "done": sequence_index, "total": len(grouped), "last_sequence": sequence, "last_completed_unit": completed[-1], "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 3)}
                if args.progress_json:
                    args.progress_json.parent.mkdir(parents=True, exist_ok=True)
                    args.progress_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(json.dumps(payload), flush=True)
                completed_frame_count += len(frames)
            finally:
                if state is not None:
                    predictor.reset_state(state)
                gc.collect()
                torch.cuda.empty_cache()
    summary = {"kind": "samurai_native_scoring_done", "sequences": completed, "output_jsonl": str(args.output_jsonl), "checkpoint": str(args.checkpoint), "config": args.config, "start_gate": args.start_gate, "reset_gate": args.reset_gate, "reset_iou": args.reset_iou, "object_gate": args.object_gate, "reset_policy": args.reset_policy, "reset_patience": args.reset_patience, "disagreement_reset_gate": args.disagreement_reset_gate, "sequence_fps_json": str(args.sequence_fps_json) if args.sequence_fps_json else None}
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
