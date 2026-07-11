from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.score_tracklets_samurai_cmc import HomographyCache
from tools.sweep_tvd_predictionsgt_action_rescore import image_key


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def valid_box(row: dict[str, Any]) -> bool:
    box = row.get("bbox")
    return isinstance(box, list) and len(box) == 4 and finite(box[2]) > finite(box[0]) and finite(box[3]) > finite(box[1])


def transform_boxes(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return boxes
    corners = np.stack((
        boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]],
    ), axis=1).astype(np.float64, copy=False)
    transformed = cv2.perspectiveTransform(corners, matrix)
    return np.concatenate((transformed.min(axis=1), transformed.max(axis=1)), axis=1).astype(np.float32)


@dataclass
class MemoryFrame:
    frame_id: int
    timestamp: float
    boxes: np.ndarray
    scores: np.ndarray
    camera_validity: float = 1.0


def support_against(current: np.ndarray, memory: MemoryFrame, now: float) -> np.ndarray:
    if not len(current) or not len(memory.boxes):
        return np.zeros((len(current),), dtype=np.float32)
    past = memory.boxes
    current_center = 0.5 * (current[:, :2] + current[:, 2:])
    past_center = 0.5 * (past[:, :2] + past[:, 2:])
    current_size = np.maximum(1.0, current[:, 2:] - current[:, :2])
    past_size = np.maximum(1.0, past[:, 2:] - past[:, :2])
    delta = current_center[:, None, :] - past_center[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    reference = np.maximum(5.0, 0.25 * (current_size[:, None, :].sum(axis=2) + past_size[None, :, :].sum(axis=2)))
    age = max(1e-3, now - memory.timestamp)
    center_similarity = np.exp(-distance / (reference * (1.0 + 1.4 * age)))
    scale_error = np.abs(np.log(current_size[:, None, :] / past_size[None, :, :])).sum(axis=2)
    scale_similarity = np.exp(-scale_error)
    left_top = np.maximum(current[:, None, :2], past[None, :, :2])
    right_bottom = np.minimum(current[:, None, 2:], past[None, :, 2:])
    intersection = np.maximum(0.0, right_bottom - left_top).prod(axis=2)
    current_area = current_size.prod(axis=1)[:, None]
    past_area = past_size.prod(axis=1)[None, :]
    iou = intersection / np.maximum(1e-6, current_area + past_area - intersection)
    compatibility = 0.50 * center_similarity + 0.30 * scale_similarity + 0.20 * iou
    evidence = compatibility * np.sqrt(np.clip(memory.scores[None, :], 0.0, 1.0))
    return (evidence.max(axis=1) * (0.75 + 0.25 * memory.camera_validity)).astype(np.float32)


def sampled_memory(history: list[MemoryFrame], now: float, short_seconds: float, long_seconds: float, short_tokens: int, long_tokens: int):
    if not history:
        return [], []
    short_targets = np.linspace(short_seconds / short_tokens, short_seconds, short_tokens)
    long_targets = np.linspace(short_seconds + (long_seconds - short_seconds) / long_tokens, long_seconds, long_tokens)
    ages = np.asarray([now - item.timestamp for item in history], dtype=np.float64)
    def choose(targets):
        chosen = []
        used = set()
        for target in targets:
            index = int(np.argmin(np.abs(ages - target)))
            if index not in used and 0.0 < ages[index] <= long_seconds + 1e-6:
                chosen.append(history[index]); used.add(index)
        return chosen
    return choose(short_targets), choose(long_targets)


def summarize_support(values: list[np.ndarray], count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not values:
        zeros = np.zeros((count,), dtype=np.float32)
        return zeros, zeros, zeros
    array = np.stack(values)
    return array.mean(axis=0), (array >= 0.45).mean(axis=0).astype(np.float32), array.max(axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--homography-cache", type=Path, required=True)
    parser.add_argument("--sequence-fps-json", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--short-seconds", type=float, default=1.0)
    parser.add_argument("--long-seconds", type=float, default=3.0)
    parser.add_argument("--short-tokens", type=int, default=8)
    parser.add_argument("--long-tokens", type=int, default=16)
    parser.add_argument("--memory-top-k", type=int, default=12)
    args = parser.parse_args()

    data = load_predictionsgt(args.predictionsgt_pkl)
    fps_map = json.loads(args.sequence_fps_json.read_text(encoding="utf-8-sig"))
    cache = HomographyCache(args.frame_root, args.homography_cache, 320)
    grouped: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
    for image_id, item in data.items():
        sequence, frame_id, _ = image_key(str(image_id), 0)
        grouped.setdefault(sequence, []).append((frame_id, str(image_id), item))
    for frames in grouped.values():
        frames.sort(key=lambda item: item[0])

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    summaries = []
    with args.out_jsonl.open("w", encoding="utf-8") as output:
        for sequence, frames in sorted(grouped.items()):
            fps = float(fps_map.get(sequence, 30.0))
            history: list[MemoryFrame] = []
            previous_frame_id = None
            sequence_rows = 0
            for frame_id, image_id, item in frames:
                now = frame_id / fps
                if previous_frame_id is not None:
                    matrix, validity = cache.between(sequence, previous_frame_id, frame_id)
                    for memory in history:
                        memory.boxes = transform_boxes(memory.boxes, matrix)
                        memory.camera_validity *= float(validity)
                history = [memory for memory in history if now - memory.timestamp <= args.long_seconds + 1e-6]
                detections = [row for row in item.get("detections", []) if valid_box(row)]
                boxes = np.asarray([row["bbox"] for row in detections], dtype=np.float32) if detections else np.zeros((0, 4), dtype=np.float32)
                short_memory, long_memory = sampled_memory(history, now, args.short_seconds, args.long_seconds, args.short_tokens, args.long_tokens)
                short_values = [support_against(boxes, memory, now) for memory in short_memory]
                long_values = [support_against(boxes, memory, now) for memory in long_memory]
                short_mean, short_density, short_max = summarize_support(short_values, len(boxes))
                long_mean, long_density, long_max = summarize_support(long_values, len(boxes))
                score = np.clip(0.34 * short_mean + 0.24 * short_density + 0.12 * short_max + 0.16 * long_mean + 0.10 * long_density + 0.04 * long_max, 0.0, 1.0)
                rows = []
                for index in range(len(detections)):
                    rows.append({
                        "seq": sequence, "frame_id": frame_id, "prediction_index": index,
                        "memory_support_score": float(score[index]),
                        "memory_short_mean": float(short_mean[index]), "memory_short_density": float(short_density[index]),
                        "memory_short_max": float(short_max[index]), "memory_long_mean": float(long_mean[index]),
                        "memory_long_density": float(long_density[index]), "memory_long_max": float(long_max[index]),
                        "memory_short_frames": len(short_memory), "memory_long_frames": len(long_memory),
                    })
                output.write(json.dumps({"meta": {"seq": sequence, "image_id": image_id, "fps": fps}, "rows": rows}, separators=(",", ":")) + "\n")
                raw_scores = np.asarray([finite(row.get("score")) for row in detections], dtype=np.float32)
                if len(raw_scores):
                    keep = np.argsort(raw_scores)[::-1][: args.memory_top_k]
                    history.append(MemoryFrame(frame_id, now, boxes[keep].copy(), raw_scores[keep].copy()))
                previous_frame_id = frame_id
                total_rows += len(rows); sequence_rows += len(rows)
            summary = {"sequence": sequence, "frames": len(frames), "rows": sequence_rows, "fps": fps}
            summaries.append(summary); print(json.dumps({"kind": "candidate_memory_sequence", **summary}), flush=True)
    summary = {"kind": "candidate_memory_done", "rows": total_rows, "short_seconds": args.short_seconds, "long_seconds": args.long_seconds, "short_tokens": args.short_tokens, "long_tokens": args.long_tokens, "memory_top_k": args.memory_top_k, "sequences": summaries}
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
