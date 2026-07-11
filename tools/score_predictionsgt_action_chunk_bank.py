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

from qstr_dronedet.tracking.action_chunk_bank import ActionChunkTrack, ActionChunkCandidateScore
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from qstr_dronedet.action_chunk_camera_motion import ActionChunkCameraMotionCache
from qstr_dronedet.candidates.merge import bbox_iou
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.train_action_chunk_bidir_full import token_summary


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in row["bbox"])



def logit(value: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, value))
    return math.log(clipped / (1.0 - clipped))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def choose_updates(tracks, detections, candidate_scores, transforms, frame_id, timestamp, internal_alpha, update_gate, return_assignments=False, return_assignment_map=False):
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
    assignment_map = {}
    for _combined, motion, raw, track_index, candidate_index in sorted(ranked_pairs, reverse=True):
        if track_index in assigned_tracks or candidate_index in assigned_candidates:
            continue
        updated = tracks[track_index].clone()
        updated.update(frame_id, timestamp, box(detections[candidate_index]), raw, motion, transforms[track_index])
        proposals.append(updated)
        assigned_tracks.add(track_index)
        assigned_candidates.add(candidate_index)
        assignment_map[candidate_index] = track_index
    if return_assignments and return_assignment_map:
        return proposals, assigned_tracks, assigned_candidates, assignment_map
    if return_assignments:
        return proposals, assigned_tracks, assigned_candidates
    return proposals


def prune_tracks(tracks, beam_size, timestamp=None, long_seconds=3.0, projected_boxes=None):
    output = []
    output_boxes = []
    if projected_boxes is not None and len(projected_boxes) != len(tracks):
        raise ValueError('projected_boxes must align with tracks')
    def rank(index):
        item = tracks[index]
        maturity = 0.10 + 0.90 * min(1.0, max(0, item.observations - 1) / 8.0)
        age = max(0.0, float(timestamp) - item.timestamp) if timestamp is not None else 0.0
        recency = math.exp(-age / max(1e-6, long_seconds))
        return item.quality * maturity * recency
    for index in sorted(range(len(tracks)), key=rank, reverse=True):
        track = tracks[index]
        projected = projected_boxes[index] if projected_boxes is not None else track.bbox
        duplicate = any(bbox_iou(projected, kept_box) >= 0.72 for kept_box in output_boxes) if projected_boxes is not None else any(track.frame_id == kept.frame_id and bbox_iou(track.bbox, kept.bbox) >= 0.72 for kept in output)
        if duplicate:
            continue
        output.append(track)
        output_boxes.append(projected)
        if len(output) >= beam_size:
            break
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Score TVD candidates with the original real-time 1s/3s Action Chunk Bank.")
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
    parser.add_argument("--short-token-count", type=int, default=8)
    parser.add_argument("--long-token-count", type=int, default=16)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--compact-chain-output", action="store_true")
    parser.add_argument("--compact-model-output", action="store_true", help="Store exact token summaries required by learned rankers instead of full token arrays.")
    args = parser.parse_args()

    predictions = load_predictionsgt(args.predictionsgt_pkl)
    sequence_fps = json.loads(args.sequence_fps_json.read_text()) if args.sequence_fps_json else {}
    cache = ActionChunkCameraMotionCache(args.frame_root, args.homography_cache, 320)
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
            next_track_id = 0
            sequence_candidates = sequence_track_frames = 0
            ordered_frames = list(reversed(frames)) if args.reverse else frames
            maximum_frame_id = frames[-1][0]
            for sequence_frame_index, (frame_id, image_id, item) in enumerate(ordered_frames):
                timestamp = (maximum_frame_id - frame_id) / fps if args.reverse else frame_id / fps
                detections = list(item.get("detections") or [])
                active = [track for track in tracks if timestamp - track.timestamp <= args.long_seconds]
                transforms, validities = [], []
                for track in active:
                    transform, validity = cache.between(sequence, track.frame_id, frame_id)
                    transforms.append(transform)
                    validities.append(validity)
                all_scores, rows = [], []
                for candidate_index, detection in enumerate(detections):
                    candidate = box(detection)
                    scores = [track.score_candidate(candidate, timestamp, transforms[index], validities[index], args.short_seconds, args.long_seconds) for index, track in enumerate(active)]
                    all_scores.append(scores)
                    best_track_index = max(range(len(scores)), key=lambda index: scores[index].score) if scores else None
                    best = scores[best_track_index] if best_track_index is not None else ActionChunkCandidateScore(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    row = {
                        "seq": sequence, "frame_id": frame_id, "prediction_index": candidate_index,
                        "action_chunk_bank_score": best.score,
                        "action_chunk_bank_predicted_iou": best.predicted_iou,
                        "action_chunk_bank_center_similarity": best.center_similarity,
                        "action_chunk_bank_direction_similarity": best.direction_similarity,
                        "action_chunk_bank_scale_similarity": best.scale_similarity,
                        "action_chunk_bank_track_quality": best.track_quality,
                        "action_chunk_bank_track_age_seconds": best.track_age_seconds,
                        "action_chunk_bank_track_id": active[best_track_index].track_id if best_track_index is not None else -1,
                        "action_chunk_bank_track_observations": active[best_track_index].observations if best_track_index is not None else 0,
                        "action_chunk_bank_chain_duration_seconds": max(0.0, timestamp - float(active[best_track_index].born_timestamp)) if best_track_index is not None else 0.0,
                        "action_chunk_bank_acceleration_similarity": best.acceleration_similarity,
                        "action_chunk_bank_motion_stability": best.motion_stability,
                        "action_chunk_bank_hypotheses": len(active),
                    }
                    if not args.compact_chain_output:
                        short_tokens = active[best_track_index].candidate_motion_tokens(candidate, timestamp, transforms[best_track_index], args.short_seconds, args.short_token_count, finite(detection.get("score"))) if best_track_index is not None else [0.0] * (12 * args.short_token_count)
                        long_tokens = active[best_track_index].candidate_motion_tokens(candidate, timestamp, transforms[best_track_index], args.long_seconds, args.long_token_count, finite(detection.get("score"))) if best_track_index is not None else [0.0] * (12 * args.long_token_count)
                        if args.compact_model_output:
                            row["action_chunk_bank_short_token_summary"] = token_summary(short_tokens)
                            row["action_chunk_bank_long_token_summary"] = token_summary(long_tokens)
                        else:
                            row["action_chunk_bank_short_tokens"] = short_tokens
                            row["action_chunk_bank_long_tokens"] = long_tokens
                    rows.append(row)
                proposals, assigned_tracks, assigned_candidates, assignment_map = choose_updates(active, detections, all_scores, transforms, frame_id, timestamp, args.internal_alpha, args.update_gate, return_assignments=True, return_assignment_map=True)
                for candidate_index, row in enumerate(rows):
                    assigned_track_index = assignment_map.get(candidate_index)
                    row["action_chunk_bank_assigned"] = int(assigned_track_index is not None)
                    row["action_chunk_bank_assigned_track_id"] = active[assigned_track_index].track_id if assigned_track_index is not None else -1
                for track_index, track in enumerate(active):
                    if track_index not in assigned_tracks and track.observations >= 2:
                        proposals.append(track.clone())
                ranked_detections = sorted(enumerate(detections), key=lambda item: finite(item[1].get("score")), reverse=True)
                for candidate_index, detection in ranked_detections:
                    if candidate_index in assigned_candidates:
                        continue
                    raw = finite(detection.get("score"))
                    if raw < args.start_gate:
                        break
                    candidate_box = box(detection)
                    if any(bbox_iou(candidate_box, proposal.bbox) >= 0.72 for proposal in proposals if proposal.frame_id == frame_id):
                        continue
                    proposals.append(ActionChunkTrack(frame_id, timestamp, candidate_box, raw, track_id=next_track_id, born_timestamp=timestamp))
                    next_track_id += 1
                    if len(proposals) >= args.beam_size + 3:
                        break
                projected_boxes = []
                for proposal in proposals:
                    if proposal.frame_id == frame_id:
                        projected_boxes.append(proposal.bbox)
                    else:
                        proposal_transform, _ = cache.between(sequence, proposal.frame_id, frame_id)
                        projected_boxes.append(proposal.predict(timestamp, proposal_transform, args.short_seconds, args.long_seconds))
                tracks = prune_tracks(proposals, args.beam_size, timestamp, args.long_seconds, projected_boxes)
                target.write(json.dumps({"meta": {"seq": sequence, "image_id": image_id, "fps": fps}, "rows": rows}, separators=(",", ":")) + "\n")
                total_frames += 1
                total_candidates += len(detections)
                sequence_candidates += len(detections)
                if active:
                    frames_with_tracks += 1
                    sequence_track_frames += 1
            summary = {"sequence": sequence, "fps": fps, "frames": len(frames), "candidates": sequence_candidates, "frames_with_active_bank": sequence_track_frames}
            summaries.append(summary)
            print(json.dumps({"kind": "action_chunk_bank_sequence", **summary}), flush=True)
    summary = {
        "kind": "action_chunk_bank_done", "predictionsgt_pkl": str(args.predictionsgt_pkl),
        "frame_root": str(args.frame_root), "homography_cache": str(args.homography_cache),
        "out_jsonl": str(args.out_jsonl), "short_seconds": args.short_seconds,
        "long_seconds": args.long_seconds, "beam_size": args.beam_size, "start_gate": args.start_gate,
        "compact_chain_output": args.compact_chain_output, "compact_model_output": args.compact_model_output,
        "update_gate": args.update_gate, "internal_alpha": args.internal_alpha, "reverse": args.reverse,
        "frames": total_frames, "candidates": total_candidates,
        "frames_with_active_bank": frames_with_tracks, "sequences": summaries,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
