from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qstr_dronedet.tracking.online_action_bank import OnlineActionTrack, OnlineCandidateScore
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.score_tracklets_samurai_cmc import HomographyCache, bbox_iou
from tools.sweep_tvd_predictionsgt_action_rescore import image_key


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in row["bbox"])



def future_box_similarity(predicted: tuple[float, float, float, float], target: tuple[float, float, float, float]) -> float:
    px1, py1, px2, py2 = predicted
    tx1, ty1, tx2, ty2 = target
    predicted_width = max(1e-6, px2 - px1)
    predicted_height = max(1e-6, py2 - py1)
    target_width = max(1e-6, tx2 - tx1)
    target_height = max(1e-6, ty2 - ty1)
    center_error = math.hypot(0.5 * (px1 + px2 - tx1 - tx2), 0.5 * (py1 + py2 - ty1 - ty2))
    reference_side = max(5.0, 0.5 * (predicted_width + predicted_height + target_width + target_height))
    center_similarity = math.exp(-center_error / reference_side)
    scale_error = abs(math.log(target_width / predicted_width)) + abs(math.log(target_height / predicted_height))
    return float(np.clip(center_similarity * math.exp(-scale_error), 0.0, 1.0))

def logit(value: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, value))
    return math.log(clipped / (1.0 - clipped))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def choose_updates(tracks, detections, candidate_scores, transforms, frame_id, timestamp, internal_alpha, update_gate):
    ranked_pairs = []
    for track_index, _track in enumerate(tracks):
        for candidate_index, detection in enumerate(detections):
            raw = finite(detection.get("score"))
            motion = candidate_scores[candidate_index][track_index].score
            combined = sigmoid(logit(raw) + internal_alpha * logit(motion))
            if combined >= update_gate or motion >= 0.72:
                ranked_pairs.append((combined, motion, raw, track_index, candidate_index))
    proposals = []
    assigned_tracks = set()
    assigned_candidates = set()
    for _combined, motion, raw, track_index, candidate_index in sorted(ranked_pairs, reverse=True):
        if track_index in assigned_tracks or candidate_index in assigned_candidates:
            continue
        updated = tracks[track_index].clone()
        updated.update(frame_id, timestamp, box(detections[candidate_index]), raw, motion, transforms[track_index])
        proposals.append(updated)
        assigned_tracks.add(track_index)
        assigned_candidates.add(candidate_index)
    return proposals


def prune_tracks(tracks, beam_size):
    output = []
    for track in sorted(tracks, key=lambda item: item.quality * (0.10 + 0.90 * min(1.0, max(0, item.observations - 1) / 8.0)), reverse=True):
        if any(bbox_iou(track.bbox, kept.bbox) >= 0.72 for kept in output):
            continue
        output.append(track)
        if len(output) >= beam_size:
            break
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Score all TVD candidates against causal 1s/3s online Action Bank hypotheses.")
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--homography-cache", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--sequence-fps-json", type=Path)
    parser.add_argument("--short-seconds", type=float, default=1.0)
    parser.add_argument("--long-seconds", type=float, default=3.0)
    parser.add_argument("--beam-size", type=int, default=6)
    parser.add_argument("--start-gate", type=float, default=0.12)
    parser.add_argument("--update-gate", type=float, default=0.08)
    parser.add_argument("--internal-alpha", type=float, default=2.5)
    parser.add_argument("--future-supervision-seconds", type=float, default=1.0)
    parser.add_argument("--short-token-count", type=int, default=8)
    parser.add_argument("--long-token-count", type=int, default=16)
    args = parser.parse_args()

    predictions = load_predictionsgt(args.predictionsgt_pkl)
    sequence_fps = json.loads(args.sequence_fps_json.read_text()) if args.sequence_fps_json else {}
    cache = HomographyCache(args.frame_root, args.homography_cache, 320)
    grouped = {}
    for image_id, item in predictions.items():
        sequence, frame_id, _ = image_key(str(image_id), 0)
        grouped.setdefault(sequence, []).append((frame_id, str(image_id), item))
    for frames in grouped.values():
        frames.sort(key=lambda item: item[0])

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    total_frames = total_candidates = frames_with_tracks = 0
    with args.out_jsonl.open("w", encoding="utf-8") as target:
        for sequence, frames in sorted(grouped.items()):
            fps = float(sequence_fps.get(sequence, args.fps))
            tracks = []
            sequence_candidates = sequence_track_frames = 0
            for sequence_frame_index, (frame_id, image_id, item) in enumerate(frames):
                timestamp = frame_id / fps
                detections = list(item.get("detections") or [])
                active = [track for track in tracks if timestamp - track.timestamp <= args.long_seconds]
                transforms, validities = [], []
                for track in active:
                    transform, validity = cache.between(sequence, track.frame_id, frame_id)
                    transforms.append(transform)
                    validities.append(validity)
                future_offset = max(1, int(round(args.future_supervision_seconds * fps)))
                future_entry = frames[sequence_frame_index + future_offset] if sequence_frame_index + future_offset < len(frames) else None
                future_labels = []
                future_frame_id = None
                future_timestamp = None
                future_camera_transform = np.eye(3, dtype=np.float64)
                if future_entry is not None:
                    future_frame_id, _future_image_id, future_item = future_entry
                    future_timestamp = future_frame_id / fps
                    future_labels = [box(label) for label in future_item.get("labels", []) if isinstance(label.get("bbox"), list) and len(label["bbox"]) == 4]
                    future_camera_transform, _future_camera_validity = cache.between(sequence, frame_id, future_frame_id)
                all_scores, rows = [], []
                for candidate_index, detection in enumerate(detections):
                    candidate = box(detection)
                    scores = [track.score_candidate(candidate, timestamp, transforms[index], validities[index], args.short_seconds, args.long_seconds) for index, track in enumerate(active)]
                    all_scores.append(scores)
                    best_track_index = max(range(len(scores)), key=lambda index: scores[index].score) if scores else None
                    best = scores[best_track_index] if best_track_index is not None else OnlineCandidateScore(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    future_consistency = 0.0
                    if future_labels and future_timestamp is not None:
                        if best_track_index is not None:
                            hypothetical = active[best_track_index].clone()
                            hypothetical.update(frame_id, timestamp, candidate, finite(detection.get("score")), best.score, transforms[best_track_index])
                        else:
                            hypothetical = OnlineActionTrack(frame_id, timestamp, candidate, finite(detection.get("score")))
                        future_prediction = hypothetical.predict(future_timestamp, future_camera_transform, args.short_seconds, args.long_seconds)
                        future_consistency = max(future_box_similarity(future_prediction, target) for target in future_labels)
                    rows.append({
                        "seq": sequence, "frame_id": frame_id, "prediction_index": candidate_index,
                        "online_action_bank_score": best.score,
                        "online_action_bank_predicted_iou": best.predicted_iou,
                        "online_action_bank_center_similarity": best.center_similarity,
                        "online_action_bank_direction_similarity": best.direction_similarity,
                        "online_action_bank_scale_similarity": best.scale_similarity,
                        "online_action_bank_track_quality": best.track_quality,
                        "online_action_bank_track_age_seconds": best.track_age_seconds,
                        "online_action_bank_acceleration_similarity": best.acceleration_similarity,
                        "online_action_bank_motion_stability": best.motion_stability,
                        "online_action_bank_short_tokens": active[best_track_index].candidate_action_tokens(candidate, timestamp, transforms[best_track_index], args.short_seconds, args.short_token_count) if best_track_index is not None else [0.0] * (2 * args.short_token_count),
                        "online_action_bank_long_tokens": active[best_track_index].candidate_action_tokens(candidate, timestamp, transforms[best_track_index], args.long_seconds, args.long_token_count) if best_track_index is not None else [0.0] * (2 * args.long_token_count),
                        "online_action_bank_hypotheses": len(active),
                        "online_action_bank_future_consistency": float(future_consistency),
                        "online_action_bank_future_seconds": float(args.future_supervision_seconds),
                    })
                proposals = choose_updates(active, detections, all_scores, transforms, frame_id, timestamp, args.internal_alpha, args.update_gate)
                for detection in sorted(detections, key=lambda row: finite(row.get("score")), reverse=True):
                    raw = finite(detection.get("score"))
                    if raw < args.start_gate:
                        break
                    proposals.append(OnlineActionTrack(frame_id, timestamp, box(detection), raw))
                    if len(proposals) >= args.beam_size + 3:
                        break
                tracks = prune_tracks(proposals, args.beam_size)
                target.write(json.dumps({"meta": {"seq": sequence, "image_id": image_id, "fps": fps}, "rows": rows}, separators=(",", ":")) + "\n")
                total_frames += 1
                total_candidates += len(detections)
                sequence_candidates += len(detections)
                if active:
                    frames_with_tracks += 1
                    sequence_track_frames += 1
            summary = {"sequence": sequence, "fps": fps, "frames": len(frames), "candidates": sequence_candidates, "frames_with_active_bank": sequence_track_frames}
            summaries.append(summary)
            print(json.dumps({"kind": "online_action_bank_sequence", **summary}), flush=True)
    summary = {
        "kind": "online_action_bank_done", "predictionsgt_pkl": str(args.predictionsgt_pkl),
        "frame_root": str(args.frame_root), "homography_cache": str(args.homography_cache),
        "out_jsonl": str(args.out_jsonl), "short_seconds": args.short_seconds,
        "long_seconds": args.long_seconds, "beam_size": args.beam_size, "start_gate": args.start_gate,
        "update_gate": args.update_gate, "internal_alpha": args.internal_alpha,
        "frames": total_frames, "candidates": total_candidates,
        "frames_with_active_bank": frames_with_tracks, "sequences": summaries,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
