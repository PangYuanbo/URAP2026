from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qstr_dronedet.tracking.action_bank import ActionBankConfig, OnlineActionBankTracker, box_iou


def center_error(left: list[float], right: list[float]) -> float:
    left_center = ((left[0] + left[2]) * 0.5, (left[1] + left[3]) * 0.5)
    right_center = ((right[0] + right[2]) * 0.5, (right[1] + right[3]) * 0.5)
    return float(np.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]))


def evaluate_sequence(sequence: dict, config: ActionBankConfig) -> dict:
    frames = sequence.get("frames") or []
    if not frames:
        return {"seq": sequence.get("seq", ""), "frames": 0, "ious": [], "center_errors": []}
    tracker = OnlineActionBankTracker(config=config, match_threshold=0.2, max_dormant_seconds=config.long_seconds)
    first = frames[0]
    initial = {"bbox": first["gt_bbox"], "timestamp_sec": first.get("timestamp_sec", 0.0), "score": 1.0, "visible": True, **(first.get("image_meta") or {})}
    target_track_id = tracker.update([initial])[0]["action_bank_track_id"]
    ious, errors = [1.0], [0.0]
    last_prediction = list(first["gt_bbox"])
    for frame_index, frame in enumerate(frames[1:], start=1):
        timestamp = float(frame.get("timestamp_sec", frame_index / config.fps_fallback))
        candidates = []
        for candidate in frame.get("candidates") or []:
            row = dict(candidate)
            row.setdefault("timestamp_sec", timestamp)
            row.setdefault("visible", True)
            candidates.append(row)
        associated = tracker.update(candidates, timestamp=timestamp)
        target = next((row for row in associated if row["action_bank_track_id"] == target_track_id), None)
        if target is not None:
            last_prediction = list(target["bbox"])
        gt = list(frame["gt_bbox"])
        ious.append(box_iou(last_prediction, gt))
        errors.append(center_error(last_prediction, gt))
    return {"seq": sequence.get("seq", ""), "frames": len(frames), "ious": ious, "center_errors": errors}


def summarize(results: list[dict], target_success: float) -> dict:
    ious = np.asarray([value for result in results for value in result["ious"]], dtype=np.float32)
    errors = np.asarray([value for result in results for value in result["center_errors"]], dtype=np.float32)
    thresholds = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    success_curve = [float(np.mean(ious >= threshold)) if len(ious) else 0.0 for threshold in thresholds]
    success_auc = float(np.mean(success_curve))
    precision_20 = float(np.mean(errors <= 20.0)) if len(errors) else 0.0
    return {
        "sequences": len(results),
        "frames": int(len(ious)),
        "success_auc": success_auc,
        "success_auc_percent": 100.0 * success_auc,
        "precision_20px": precision_20,
        "precision_20px_percent": 100.0 * precision_20,
        "mean_iou": float(np.mean(ious)) if len(ious) else 0.0,
        "target_success_percent": target_success,
        "target_met": 100.0 * success_auc >= target_success,
        "per_sequence": [{"seq": result["seq"], "frames": result["frames"], "success_auc": float(np.mean([np.mean(np.asarray(result["ious"]) >= threshold) for threshold in thresholds]))} for result in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Zero-shot arbitrary-object Action Bank tracking benchmark.")
    parser.add_argument("--input", required=True, help="JSONL sequences with gt_bbox and detector candidates per frame")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-success", type=float, default=70.0)
    parser.add_argument("--fps-fallback", type=float, default=29.97)
    args = parser.parse_args()
    config = ActionBankConfig(fps_fallback=args.fps_fallback)
    with Path(args.input).open("r", encoding="utf-8-sig") as source:
        sequences = [json.loads(line) for line in source if line.strip()]
    summary = summarize([evaluate_sequence(sequence, config) for sequence in sequences], args.target_success)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["target_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
