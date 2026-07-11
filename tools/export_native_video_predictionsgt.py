from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qstr_dronedet.native_video_detector import NPSClipDataset, NativeVideoDetector, collate_nps_clips
from qstr_dronedet.native_video_detector.data import load_gt_csv


def cxcywh_to_xyxy_pixels(box: torch.Tensor, image_w: float, image_h: float) -> list[float]:
    cx, cy, w, h = [float(v) for v in box.tolist()]
    x1 = (cx - w * 0.5) * image_w
    y1 = (cy - h * 0.5) * image_h
    x2 = (cx + w * 0.5) * image_w
    y2 = (cy + h * 0.5) * image_h
    return [max(0.0, x1), max(0.0, y1), min(image_w, x2), min(image_h, y2)]


def bbox_iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def xyxy_to_cxcywh_pixels(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [(x1 + x2) * 0.5, (y1 + y2) * 0.5, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def cxcywh_to_xyxy_list(box: list[float]) -> list[float]:
    cx, cy, w, h = [float(v) for v in box]
    return [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]


def center_motion_score(candidate: list[float], predicted: list[float], sigma_pixels: float) -> float:
    cand = xyxy_to_cxcywh_pixels(candidate)
    pred = xyxy_to_cxcywh_pixels(predicted)
    sigma = max(float(sigma_pixels), 1e-6)
    dx = cand[0] - pred[0]
    dy = cand[1] - pred[1]
    return float(math.exp(-0.5 * ((dx * dx + dy * dy) / (sigma * sigma))))


def fused_samurai_score(
    appearance_score: float,
    motion_iou: float,
    center_score: float,
    appearance_weight: float,
    motion_iou_weight: float,
    center_weight: float,
) -> float:
    weights = [
        max(0.0, float(appearance_weight)),
        max(0.0, float(motion_iou_weight)),
        max(0.0, float(center_weight)),
    ]
    denom = sum(weights)
    if denom <= 0.0:
        return float(appearance_score)
    score = (
        weights[0] * float(appearance_score)
        + weights[1] * float(motion_iou)
        + weights[2] * float(center_score)
    ) / denom
    return float(max(0.0, min(1.0, score)))


def predict_motion_box(state: dict[str, object], frame_id: int) -> list[float] | None:
    box = state.get("box")
    last_frame = state.get("last_frame")
    velocity = state.get("velocity")
    if not isinstance(box, list) or not isinstance(velocity, list) or last_frame is None:
        return None
    dt = max(1, int(frame_id) - int(last_frame))
    predicted = [float(box[idx]) + float(velocity[idx]) * dt for idx in range(4)]
    return cxcywh_to_xyxy_list(predicted)


def update_motion_state(
    state: dict[str, object],
    bbox: list[float],
    frame_id: int,
    velocity_momentum: float,
) -> None:
    measurement = xyxy_to_cxcywh_pixels(bbox)
    previous = state.get("box")
    previous_frame = state.get("last_frame")
    old_velocity = state.get("velocity")
    if isinstance(previous, list) and previous_frame is not None:
        dt = max(1, int(frame_id) - int(previous_frame))
        observed_velocity = [(measurement[idx] - float(previous[idx])) / dt for idx in range(4)]
        if isinstance(old_velocity, list):
            momentum = max(0.0, min(1.0, float(velocity_momentum)))
            velocity = [
                momentum * float(old_velocity[idx]) + (1.0 - momentum) * float(observed_velocity[idx])
                for idx in range(4)
            ]
        else:
            velocity = observed_velocity
    else:
        velocity = [0.0, 0.0, 0.0, 0.0]
    state["box"] = measurement
    state["velocity"] = velocity
    state["last_frame"] = int(frame_id)
    state["stable_updates"] = int(state.get("stable_updates", 0)) + 1
    state["lost_frames"] = 0


def samurai_motion_rerank_sequence(
    detections_by_frame: list[tuple[str, int, list[dict[str, object]]]],
    appearance_weight: float,
    motion_iou_weight: float,
    center_weight: float,
    center_sigma_pixels: float,
    update_score_threshold: float,
    update_motion_iou_threshold: float,
    lost_tau: int,
    velocity_momentum: float,
) -> None:
    state: dict[str, object] = {"box": None, "velocity": None, "last_frame": None, "stable_updates": 0, "lost_frames": 0}
    for _, frame_id, detections in sorted(detections_by_frame, key=lambda item: item[1]):
        if not detections:
            state["lost_frames"] = int(state.get("lost_frames", 0)) + 1
            continue
        use_motion = (
            state.get("box") is not None
            and int(state.get("stable_updates", 0)) > 0
            and int(state.get("lost_frames", 0)) <= int(lost_tau)
        )
        predicted = predict_motion_box(state, frame_id) if use_motion else None
        for rank, row in enumerate(detections):
            bbox = row.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            appearance = float(row.get("score", 0.0))
            row.setdefault("appearance_score", appearance)
            motion_iou = bbox_iou_xyxy(bbox, predicted) if predicted is not None else 0.0
            center_score = center_motion_score(bbox, predicted, center_sigma_pixels) if predicted is not None else 0.0
            fused = fused_samurai_score(
                appearance,
                motion_iou,
                center_score,
                appearance_weight,
                motion_iou_weight if predicted is not None else 0.0,
                center_weight if predicted is not None else 0.0,
            )
            row["raw_rank"] = int(row.get("raw_rank", rank))
            row["samurai_motion_iou"] = motion_iou
            row["samurai_center_score"] = center_score
            row["samurai_score"] = fused
            row["score"] = fused
        detections.sort(key=lambda row: float(row.get("samurai_score", row.get("score", 0.0))), reverse=True)
        selected = detections[0]
        selected_bbox = selected.get("bbox")
        selected_score = float(selected.get("appearance_score", selected.get("score", 0.0)))
        selected_motion_iou = float(selected.get("samurai_motion_iou", 0.0))
        motion_ok = predicted is None or selected_motion_iou >= float(update_motion_iou_threshold)
        if isinstance(selected_bbox, list) and selected_score >= float(update_score_threshold) and motion_ok:
            update_motion_state(state, selected_bbox, frame_id, velocity_momentum)
        else:
            state["lost_frames"] = int(state.get("lost_frames", 0)) + 1


def samurai_motion_rerank(
    out: dict[str, dict[str, list[dict[str, object]]]],
    frame_order: list[tuple[str, str, int]],
    appearance_weight: float,
    motion_iou_weight: float,
    center_weight: float,
    center_sigma_pixels: float,
    update_score_threshold: float,
    update_motion_iou_threshold: float,
    lost_tau: int,
    velocity_momentum: float,
) -> None:
    by_seq: dict[str, list[tuple[str, int, list[dict[str, object]]]]] = {}
    for image_id, seq, frame_id in frame_order:
        item = out.get(str(image_id))
        if item is None:
            continue
        by_seq.setdefault(seq, []).append((str(image_id), int(frame_id), item.get("detections", [])))
    for sequence_items in by_seq.values():
        samurai_motion_rerank_sequence(
            sequence_items,
            appearance_weight=appearance_weight,
            motion_iou_weight=motion_iou_weight,
            center_weight=center_weight,
            center_sigma_pixels=center_sigma_pixels,
            update_score_threshold=update_score_threshold,
            update_motion_iou_threshold=update_motion_iou_threshold,
            lost_tau=lost_tau,
            velocity_momentum=velocity_momentum,
        )


def motion_affinity(candidate: list[float], predicted: list[float], center_sigma_pixels: float) -> float:
    center_score = center_motion_score(candidate, predicted, center_sigma_pixels)
    motion_iou = bbox_iou_xyxy(candidate, predicted)
    return float(max(0.0, min(1.0, 0.75 * center_score + 0.25 * motion_iou)))


def _track_predicted_box(track: dict[str, object], frame_id: int) -> list[float] | None:
    box = track.get("box")
    velocity = track.get("velocity")
    last_frame = track.get("last_frame")
    if not isinstance(box, list) or not isinstance(velocity, list) or last_frame is None:
        return None
    dt = max(1, int(frame_id) - int(last_frame))
    predicted = [float(box[idx]) + float(velocity[idx]) * dt for idx in range(4)]
    return cxcywh_to_xyxy_list(predicted)


def _update_track(
    track: dict[str, object],
    row: dict[str, object],
    frame_id: int,
    motion_score: float,
    velocity_momentum: float,
) -> None:
    bbox = row.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    measurement = xyxy_to_cxcywh_pixels(bbox)
    previous = track.get("box")
    previous_frame = track.get("last_frame")
    old_velocity = track.get("velocity")
    if isinstance(previous, list) and previous_frame is not None:
        dt = max(1, int(frame_id) - int(previous_frame))
        observed_velocity = [(measurement[idx] - float(previous[idx])) / dt for idx in range(4)]
        if isinstance(old_velocity, list):
            momentum = max(0.0, min(1.0, float(velocity_momentum)))
            velocity = [
                momentum * float(old_velocity[idx]) + (1.0 - momentum) * float(observed_velocity[idx])
                for idx in range(4)
            ]
        else:
            velocity = observed_velocity
    else:
        velocity = [0.0, 0.0, 0.0, 0.0]
    appearance = float(row.get("appearance_score", row.get("score", 0.0)))
    row["samurai_track_id"] = int(track["id"])
    row["samurai_track_motion_score"] = float(motion_score)
    track["box"] = measurement
    track["velocity"] = velocity
    track["last_frame"] = int(frame_id)
    track["length"] = int(track.get("length", 0)) + 1
    track["score_sum"] = float(track.get("score_sum", 0.0)) + appearance
    track["motion_sum"] = float(track.get("motion_sum", 0.0)) + float(motion_score)
    rows = track.setdefault("rows", [])
    if isinstance(rows, list):
        rows.append(row)


def _new_track(track_id: int, row: dict[str, object], frame_id: int) -> dict[str, object]:
    bbox = row.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("track row must contain a valid bbox")
    appearance = float(row.get("appearance_score", row.get("score", 0.0)))
    row["samurai_track_id"] = int(track_id)
    row["samurai_track_motion_score"] = 0.0
    return {
        "id": int(track_id),
        "box": xyxy_to_cxcywh_pixels(bbox),
        "velocity": [0.0, 0.0, 0.0, 0.0],
        "last_frame": int(frame_id),
        "length": 1,
        "score_sum": appearance,
        "motion_sum": 0.0,
        "rows": [row],
    }


def samurai_tracklet_rerank_sequence(
    detections_by_frame: list[tuple[str, int, list[dict[str, object]]]],
    candidate_topk: int,
    center_sigma_pixels: float,
    match_threshold: float,
    max_gap: int,
    spawn_score_threshold: float,
    length_norm: float,
    appearance_weight: float,
    tracklet_weight: float,
    unmatched_scale: float,
    velocity_momentum: float,
) -> None:
    tracks: list[dict[str, object]] = []
    next_track_id = 0
    candidate_topk = max(1, int(candidate_topk))
    max_gap = max(0, int(max_gap))
    for _, frame_id, detections in sorted(detections_by_frame, key=lambda item: item[1]):
        detections.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        candidates = detections[:candidate_topk]
        for rank, row in enumerate(detections):
            row.setdefault("appearance_score", float(row.get("score", 0.0)))
            row.setdefault("raw_rank", rank)
            row["samurai_tracklet_score"] = 0.0
            row["samurai_track_motion_score"] = 0.0
        active_tracks = [
            track for track in tracks
            if int(frame_id) - int(track.get("last_frame", frame_id)) <= max_gap + 1
        ]
        pair_scores: list[tuple[float, int, int]] = []
        for track_idx, track in enumerate(active_tracks):
            predicted = _track_predicted_box(track, frame_id)
            if predicted is None:
                continue
            for det_idx, row in enumerate(candidates):
                bbox = row.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                affinity = motion_affinity(bbox, predicted, center_sigma_pixels)
                if affinity >= float(match_threshold):
                    appearance = float(row.get("appearance_score", row.get("score", 0.0)))
                    pair_scores.append((0.8 * affinity + 0.2 * appearance, track_idx, det_idx))
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()
        for score, track_idx, det_idx in sorted(pair_scores, reverse=True):
            if track_idx in assigned_tracks or det_idx in assigned_dets:
                continue
            track = active_tracks[track_idx]
            row = candidates[det_idx]
            _update_track(track, row, frame_id, score, velocity_momentum)
            assigned_tracks.add(track_idx)
            assigned_dets.add(det_idx)
        for det_idx, row in enumerate(candidates):
            if det_idx in assigned_dets:
                continue
            appearance = float(row.get("appearance_score", row.get("score", 0.0)))
            if appearance < float(spawn_score_threshold):
                continue
            tracks.append(_new_track(next_track_id, row, frame_id))
            next_track_id += 1

    length_norm = max(1e-6, float(length_norm))
    appearance_weight = max(0.0, float(appearance_weight))
    tracklet_weight = max(0.0, float(tracklet_weight))
    score_denom = max(appearance_weight + tracklet_weight, 1e-6)
    unmatched_scale = max(0.0, min(1.0, float(unmatched_scale)))
    for track in tracks:
        rows = track.get("rows")
        if not isinstance(rows, list) or not rows:
            continue
        length = int(track.get("length", len(rows)))
        if length < 2:
            for row in rows:
                row["samurai_tracklet_length"] = length
                row["samurai_tracklet_score"] = 0.0
            continue
        avg_score = float(track.get("score_sum", 0.0)) / max(1, length)
        avg_motion = float(track.get("motion_sum", 0.0)) / max(1, length - 1)
        length_score = min(1.0, float(length) / length_norm)
        tracklet_score = max(0.0, min(1.0, avg_score * (0.5 + 0.5 * length_score) * (0.5 + 0.5 * avg_motion)))
        for row in rows:
            appearance = float(row.get("appearance_score", row.get("score", 0.0)))
            row["samurai_tracklet_length"] = length
            row["samurai_tracklet_score"] = tracklet_score
            row["score"] = max(0.0, min(1.0, (appearance_weight * appearance + tracklet_weight * tracklet_score) / score_denom))
    for _, _, detections in detections_by_frame:
        for row in detections:
            if float(row.get("samurai_tracklet_score", 0.0)) <= 0.0:
                appearance = float(row.get("appearance_score", row.get("score", 0.0)))
                row["score"] = appearance * unmatched_scale
        detections.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)


def samurai_tracklet_rerank(
    out: dict[str, dict[str, list[dict[str, object]]]],
    frame_order: list[tuple[str, str, int]],
    candidate_topk: int,
    center_sigma_pixels: float,
    match_threshold: float,
    max_gap: int,
    spawn_score_threshold: float,
    length_norm: float,
    appearance_weight: float,
    tracklet_weight: float,
    unmatched_scale: float,
    velocity_momentum: float,
) -> None:
    by_seq: dict[str, list[tuple[str, int, list[dict[str, object]]]]] = {}
    for image_id, seq, frame_id in frame_order:
        item = out.get(str(image_id))
        if item is None:
            continue
        by_seq.setdefault(seq, []).append((str(image_id), int(frame_id), item.get("detections", [])))
    for sequence_items in by_seq.values():
        samurai_tracklet_rerank_sequence(
            sequence_items,
            candidate_topk=candidate_topk,
            center_sigma_pixels=center_sigma_pixels,
            match_threshold=match_threshold,
            max_gap=max_gap,
            spawn_score_threshold=spawn_score_threshold,
            length_norm=length_norm,
            appearance_weight=appearance_weight,
            tracklet_weight=tracklet_weight,
            unmatched_scale=unmatched_scale,
            velocity_momentum=velocity_momentum,
        )


def nms_detections(detections: list[dict[str, object]], iou_threshold: float) -> list[dict[str, object]]:
    if iou_threshold < 0 or len(detections) <= 1:
        return detections
    kept: list[dict[str, object]] = []
    for det in sorted(detections, key=lambda row: float(row.get("score", 0.0)), reverse=True):
        bbox = det.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if all(bbox_iou_xyxy(bbox, kept_det["bbox"]) <= iou_threshold for kept_det in kept if isinstance(kept_det.get("bbox"), list)):
            kept.append(det)
    return kept


def build_export_frame_maps(
    dataset: NPSClipDataset,
) -> tuple[dict[tuple[str, int], int], set[tuple[str, int]]]:
    frame_index_by_key: dict[tuple[str, int], int] = {}
    for seq, items in dataset.by_seq.items():
        for idx, ref in enumerate(items):
            frame_index_by_key[(seq, int(ref.frame_id))] = idx
    export_keys: set[tuple[str, int]] = set()
    for seq, idx in dataset.index:
        ref = dataset.by_seq[seq][idx]
        export_keys.add((seq, int(ref.frame_id)))
    return frame_index_by_key, export_keys


def ensure_prediction_item(
    out: dict[str, dict[str, list[dict[str, object]]]],
    image_id: str,
    seq: str,
    frame_id: int,
    gt: dict[tuple[str, int], list[list[float]]],
) -> dict[str, list[dict[str, object]]]:
    labels = [{"bbox": box, "category_id": 0} for box in gt.get((seq, int(frame_id)), [])]
    item = out.setdefault(str(image_id), {"detections": [], "labels": labels})
    item["labels"] = labels
    return item


def append_action_chunk_detections(
    out: dict[str, dict[str, list[dict[str, object]]]],
    batch: dict[str, object],
    chunk_boxes: torch.Tensor,
    chunk_logits: torch.Tensor,
    dataset: NPSClipDataset,
    frame_index_by_key: dict[tuple[str, int], int],
    export_keys: set[tuple[str, int]],
    gt: dict[tuple[str, int], list[list[float]]],
    max_step: int,
    top_k: int,
    score_threshold: float,
    score_decay: float,
) -> int:
    if max_step <= 0 or chunk_boxes.shape[2] <= 1:
        return 0
    seqs = batch["seq"]
    frame_ids = batch["frame_id"]
    image_ids = batch["image_id"]
    image_sizes = batch["image_size"]
    inserted = 0
    step_limit = min(int(max_step), int(chunk_boxes.shape[2]) - 1)
    top_k = max(1, int(top_k))
    score_decay = max(0.0, float(score_decay))
    for batch_idx, seq in enumerate(seqs):
        seq = str(seq)
        frame_id = int(frame_ids[batch_idx])
        center_idx = frame_index_by_key.get((seq, frame_id))
        if center_idx is None:
            continue
        items = dataset.by_seq.get(seq, [])
        if not items:
            continue
        image_w, image_h = [float(v) for v in image_sizes[batch_idx].tolist()]
        for step in range(1, step_limit + 1):
            target_idx = min(len(items) - 1, center_idx + step)
            if target_idx == center_idx:
                continue
            target_ref = items[target_idx]
            target_key = (seq, int(target_ref.frame_id))
            if target_key not in export_keys:
                continue
            step_scores = torch.sigmoid(chunk_logits[batch_idx, :, step]) * (score_decay ** step)
            top_count = min(top_k, int(step_scores.numel()))
            top_scores, top_indices = step_scores.topk(top_count)
            item = ensure_prediction_item(out, target_ref.path.stem, seq, int(target_ref.frame_id), gt)
            detections = item["detections"]
            for score, query_idx in zip(top_scores.tolist(), top_indices.tolist()):
                if float(score) < float(score_threshold):
                    continue
                xyxy = cxcywh_to_xyxy_pixels(chunk_boxes[batch_idx, int(query_idx), step, :], image_w, image_h)
                if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                    continue
                detections.append(
                    {
                        "bbox": xyxy,
                        "score": float(score),
                        "category_id": 0,
                        "raw_rank": len(detections),
                        "source": "action_chunk",
                        "action_chunk_step": int(step),
                        "action_chunk_source_image_id": str(image_ids[batch_idx]),
                    }
                )
                inserted += 1
    return inserted


def merge_action_chunk_support(
    out: dict[str, dict[str, list[dict[str, object]]]],
    support_iou: float,
    support_weight: float,
    keep_unmatched: bool,
) -> int:
    support_iou = max(0.0, min(1.0, float(support_iou)))
    support_weight = max(0.0, min(1.0, float(support_weight)))
    supported = 0
    for item in out.values():
        detections = item.get("detections", [])
        base_rows = [row for row in detections if row.get("source") != "action_chunk"]
        action_rows = [row for row in detections if row.get("source") == "action_chunk"]
        if not action_rows:
            continue
        unmatched_action_rows: list[dict[str, object]] = []
        for action_row in action_rows:
            action_bbox = action_row.get("bbox")
            if not isinstance(action_bbox, list) or len(action_bbox) != 4:
                continue
            best_row = None
            best_iou = 0.0
            for base_row in base_rows:
                base_bbox = base_row.get("bbox")
                if not isinstance(base_bbox, list) or len(base_bbox) != 4:
                    continue
                iou = bbox_iou_xyxy(action_bbox, base_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_row = base_row
            if best_row is None or best_iou < support_iou:
                if keep_unmatched:
                    unmatched_action_rows.append(action_row)
                continue
            raw_score = float(best_row.get("score", 0.0))
            action_score = float(action_row.get("score", 0.0))
            boosted = max(raw_score, (1.0 - support_weight) * raw_score + support_weight * action_score)
            best_row["pre_action_chunk_score"] = raw_score
            best_row["score"] = max(0.0, min(1.0, boosted))
            best_row["action_chunk_support_count"] = int(best_row.get("action_chunk_support_count", 0)) + 1
            best_row["action_chunk_support_score"] = max(
                float(best_row.get("action_chunk_support_score", 0.0)),
                action_score,
            )
            best_row["action_chunk_support_iou"] = max(
                float(best_row.get("action_chunk_support_iou", 0.0)),
                best_iou,
            )
            supported += 1
        item["detections"] = base_rows + unmatched_action_rows
        item["detections"].sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return supported


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(device_arg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export native video detector outputs as TransVisDrone predictionsgt pkl.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--gt-csv", nargs="+", type=Path, required=True)
    parser.add_argument("--out-pkl", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--proposal-prefilter-topk", type=int, default=0)
    parser.add_argument("--proposal-score-weight", type=float, default=0.0)
    parser.add_argument("--quality-score-weight", type=float, default=0.0)
    parser.add_argument("--nms-iou-threshold", type=float, default=-1.0)
    parser.add_argument("--samurai-motion-rerank", action="store_true")
    parser.add_argument("--samurai-appearance-weight", type=float, default=0.60)
    parser.add_argument("--samurai-motion-iou-weight", type=float, default=0.30)
    parser.add_argument("--samurai-center-weight", type=float, default=0.05)
    parser.add_argument("--samurai-center-sigma-pixels", type=float, default=96.0)
    parser.add_argument("--samurai-update-score-threshold", type=float, default=0.05)
    parser.add_argument("--samurai-update-motion-iou-threshold", type=float, default=0.0)
    parser.add_argument("--samurai-lost-tau", type=int, default=8)
    parser.add_argument("--samurai-velocity-momentum", type=float, default=0.6)
    parser.add_argument("--samurai-tracklet-rerank", action="store_true")
    parser.add_argument("--samurai-tracklet-candidate-topk", type=int, default=32)
    parser.add_argument("--samurai-tracklet-match-threshold", type=float, default=0.15)
    parser.add_argument("--samurai-tracklet-max-gap", type=int, default=2)
    parser.add_argument("--samurai-tracklet-spawn-score-threshold", type=float, default=0.02)
    parser.add_argument("--samurai-tracklet-length-norm", type=float, default=4.0)
    parser.add_argument("--samurai-tracklet-appearance-weight", type=float, default=0.65)
    parser.add_argument("--samurai-tracklet-weight", type=float, default=0.35)
    parser.add_argument("--samurai-tracklet-unmatched-scale", type=float, default=0.5)
    parser.add_argument("--action-chunk-backfill", action="store_true")
    parser.add_argument("--action-chunk-max-step", type=int, default=0)
    parser.add_argument("--action-chunk-top-k", type=int, default=0)
    parser.add_argument("--action-chunk-score-decay", type=float, default=0.85)
    parser.add_argument("--action-chunk-merge-mode", choices=["add", "support"], default="add")
    parser.add_argument("--action-chunk-support-iou", type=float, default=0.3)
    parser.add_argument("--action-chunk-support-weight", type=float, default=0.4)
    parser.add_argument("--action-chunk-keep-unmatched", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu")
    config = ckpt["config"]
    dataset = NPSClipDataset(
        args.frames_dir,
        args.gt_csv,
        clip_len=int(config["clip_len"]),
        future_len=int(config["future_len"]),
        image_size=int(config["image_size"]),
        max_samples=args.max_samples if args.max_samples > 0 else None,
        cache_dir=args.cache_dir,
    )
    frame_index_by_key, export_keys = build_export_frame_maps(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_nps_clips)
    model = NativeVideoDetector(
        clip_len=int(config["clip_len"]),
        future_len=int(config["future_len"]),
        num_queries=int(config["num_queries"]),
        d_model=int(config["d_model"]),
        nhead=int(config.get("nhead", 4)),
        encoder_layers=int(config["encoder_layers"]),
        decoder_layers=int(config["decoder_layers"]),
        encoder_mode=str(config.get("encoder_mode", "global")),
        patch_stride=int(config.get("patch_stride", 16)),
        spatial_refine_layers=int(config.get("spatial_refine_layers", 0)),
        spatial_refine_kernel=int(config.get("spatial_refine_kernel", 7)),
        spatial_refine_expansion=float(config.get("spatial_refine_expansion", 2.0)),
        motion_channels=bool(config.get("motion_channels", False)),
        memory_mode=str(config.get("memory_mode", "last")),
        box_size_scale=float(config.get("box_size_scale", 1.0)),
        query_mode=str(config.get("query_mode", "learned")),
        anchor_offset_cells=float(config.get("anchor_offset_cells", 4.0)),
        dense_obj_source=str(config.get("dense_obj_source", "token")),
        memory_attention=str(config.get("memory_attention", "none")),
        memory_slots=int(config.get("memory_slots", 64)),
        memory_match_mode=str(config.get("memory_match_mode", "none")),
        memory_match_weight=float(config.get("memory_match_weight", 0.0)),
        memory_match_temperature=float(config.get("memory_match_temperature", 5.0)),
        motion_score_mode=str(config.get("motion_score_mode", "none")),
        motion_score_weight=float(config.get("motion_score_weight", 1.0)),
        proposal_mode=str(config.get("proposal_mode", "none")),
        quality_score_mode=str(config.get("quality_score_mode", "none")),
    )
    state_key = "ema_model" if (not args.no_ema and "ema_model" in ckpt) else "model"
    model.load_state_dict(ckpt[state_key])
    device = resolve_device(args.device)
    model.to(device)
    model.eval()
    gt = load_gt_csv(args.gt_csv)
    out: dict[str, dict[str, list[dict[str, object]]]] = {}
    frame_order: list[tuple[str, str, int]] = []
    suppressed_by_nms = 0
    action_chunk_backfilled = 0
    action_chunk_supported = 0
    processed = 0
    batches_seen = 0
    with torch.no_grad():
        for batch in loader:
            batches_seen += 1
            outputs = model(batch["clip"].to(device))
            chunk_logits = outputs["logits"].cpu()
            chunk_boxes = outputs["boxes"].cpu()
            final_logits = chunk_logits[:, :, 0]
            proposal_logits = outputs.get("proposal_logits")
            if proposal_logits is not None:
                proposal_logits = proposal_logits.cpu()
                score_logits = final_logits + (float(args.proposal_score_weight) * proposal_logits)
                proposal_scores = torch.sigmoid(proposal_logits)
            else:
                score_logits = final_logits
                proposal_scores = None
            quality_logits = outputs.get("quality_logits")
            if quality_logits is not None:
                quality_logits = quality_logits.cpu()
                score_logits = score_logits + (float(args.quality_score_weight) * quality_logits)
                quality_scores = torch.sigmoid(quality_logits)
            else:
                quality_scores = None
            memory_match_logits = outputs.get("memory_match_logits")
            if memory_match_logits is not None:
                memory_match_scores = torch.sigmoid(memory_match_logits.cpu())
            else:
                memory_match_scores = None
            scores = torch.sigmoid(score_logits)
            boxes = chunk_boxes[:, :, 0, :]
            for idx, image_id in enumerate(batch["image_id"]):
                image_w, image_h = [float(v) for v in batch["image_size"][idx].tolist()]
                detections = []
                if proposal_scores is not None and args.proposal_prefilter_topk > 0:
                    prefilter_count = min(int(args.proposal_prefilter_topk), scores.shape[1])
                    candidate_indices = proposal_scores[idx].topk(prefilter_count).indices
                    candidate_scores = scores[idx, candidate_indices]
                    top_count = min(args.top_k, int(candidate_indices.numel()))
                    top_scores, local_indices = candidate_scores.topk(top_count)
                    top_indices = candidate_indices[local_indices]
                else:
                    top_scores, top_indices = scores[idx].topk(min(args.top_k, scores.shape[1]))
                for score, query_idx in zip(top_scores.tolist(), top_indices.tolist()):
                    if float(score) < args.score_threshold:
                        continue
                    xyxy = cxcywh_to_xyxy_pixels(boxes[idx, int(query_idx)], image_w, image_h)
                    if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                        continue
                    row = {"bbox": xyxy, "score": float(score), "category_id": 0, "raw_rank": len(detections)}
                    if proposal_scores is not None:
                        row["proposal_score"] = float(proposal_scores[idx, int(query_idx)])
                    if memory_match_scores is not None:
                        row["memory_match_score"] = float(memory_match_scores[idx, int(query_idx)])
                    if quality_scores is not None:
                        row["quality_score"] = float(quality_scores[idx, int(query_idx)])
                    detections.append(row)
                before_nms = len(detections)
                detections = nms_detections(detections, args.nms_iou_threshold)
                suppressed_by_nms += before_nms - len(detections)
                item = ensure_prediction_item(out, str(image_id), str(batch["seq"][idx]), int(batch["frame_id"][idx]), gt)
                item["detections"].extend(detections)
                frame_order.append((str(image_id), str(batch["seq"][idx]), int(batch["frame_id"][idx])))
                processed += 1
            if args.action_chunk_backfill:
                action_chunk_backfilled += append_action_chunk_detections(
                    out,
                    batch,
                    chunk_boxes=chunk_boxes,
                    chunk_logits=chunk_logits,
                    dataset=dataset,
                    frame_index_by_key=frame_index_by_key,
                    export_keys=export_keys,
                    gt=gt,
                    max_step=args.action_chunk_max_step if args.action_chunk_max_step > 0 else int(config["future_len"]),
                    top_k=args.action_chunk_top_k if args.action_chunk_top_k > 0 else args.top_k,
                    score_threshold=args.score_threshold,
                    score_decay=args.action_chunk_score_decay,
                )
            if args.log_every > 0 and (batches_seen == 1 or batches_seen % args.log_every == 0 or processed >= len(dataset)):
                print(
                    json.dumps(
                        {
                            "kind": "native_video_export_progress",
                            "batches": batches_seen,
                            "processed": processed,
                            "total": len(dataset),
                            "detections": sum(len(item["detections"]) for item in out.values()),
                            "action_chunk_backfilled": action_chunk_backfilled,
                            "action_chunk_supported": action_chunk_supported,
                        }
                    ),
                    flush=True,
                )
    if args.action_chunk_backfill and args.action_chunk_merge_mode == "support":
        action_chunk_supported = merge_action_chunk_support(
            out,
            support_iou=args.action_chunk_support_iou,
            support_weight=args.action_chunk_support_weight,
            keep_unmatched=bool(args.action_chunk_keep_unmatched),
        )
    if args.samurai_motion_rerank:
        samurai_motion_rerank(
            out,
            frame_order,
            appearance_weight=args.samurai_appearance_weight,
            motion_iou_weight=args.samurai_motion_iou_weight,
            center_weight=args.samurai_center_weight,
            center_sigma_pixels=args.samurai_center_sigma_pixels,
            update_score_threshold=args.samurai_update_score_threshold,
            update_motion_iou_threshold=args.samurai_update_motion_iou_threshold,
            lost_tau=args.samurai_lost_tau,
            velocity_momentum=args.samurai_velocity_momentum,
        )
    if args.samurai_tracklet_rerank:
        samurai_tracklet_rerank(
            out,
            frame_order,
            candidate_topk=args.samurai_tracklet_candidate_topk,
            center_sigma_pixels=args.samurai_center_sigma_pixels,
            match_threshold=args.samurai_tracklet_match_threshold,
            max_gap=args.samurai_tracklet_max_gap,
            spawn_score_threshold=args.samurai_tracklet_spawn_score_threshold,
            length_norm=args.samurai_tracklet_length_norm,
            appearance_weight=args.samurai_tracklet_appearance_weight,
            tracklet_weight=args.samurai_tracklet_weight,
            unmatched_scale=args.samurai_tracklet_unmatched_scale,
            velocity_momentum=args.samurai_velocity_momentum,
        )
    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pkl.open("wb") as f:
        pickle.dump(out, f)
    summary = {
        "predictionsgt_pkl": str(args.out_pkl.resolve()),
        "images": len(out),
        "detections": sum(len(item["detections"]) for item in out.values()),
        "labels": sum(len(item["labels"]) for item in out.values()),
        "max_samples": int(args.max_samples),
        "full_split": args.max_samples <= 0,
        "score_threshold": args.score_threshold,
        "top_k": args.top_k,
        "proposal_prefilter_topk": args.proposal_prefilter_topk,
        "proposal_score_weight": args.proposal_score_weight,
        "quality_score_weight": args.quality_score_weight,
        "nms_iou_threshold": args.nms_iou_threshold,
        "suppressed_by_nms": suppressed_by_nms,
        "samurai_motion_rerank": bool(args.samurai_motion_rerank),
        "samurai_appearance_weight": args.samurai_appearance_weight,
        "samurai_motion_iou_weight": args.samurai_motion_iou_weight,
        "samurai_center_weight": args.samurai_center_weight,
        "samurai_center_sigma_pixels": args.samurai_center_sigma_pixels,
        "samurai_update_score_threshold": args.samurai_update_score_threshold,
        "samurai_update_motion_iou_threshold": args.samurai_update_motion_iou_threshold,
        "samurai_lost_tau": args.samurai_lost_tau,
        "samurai_velocity_momentum": args.samurai_velocity_momentum,
        "samurai_tracklet_rerank": bool(args.samurai_tracklet_rerank),
        "samurai_tracklet_candidate_topk": args.samurai_tracklet_candidate_topk,
        "samurai_tracklet_match_threshold": args.samurai_tracklet_match_threshold,
        "samurai_tracklet_max_gap": args.samurai_tracklet_max_gap,
        "samurai_tracklet_spawn_score_threshold": args.samurai_tracklet_spawn_score_threshold,
        "samurai_tracklet_length_norm": args.samurai_tracklet_length_norm,
        "samurai_tracklet_appearance_weight": args.samurai_tracklet_appearance_weight,
        "samurai_tracklet_weight": args.samurai_tracklet_weight,
        "samurai_tracklet_unmatched_scale": args.samurai_tracklet_unmatched_scale,
        "action_chunk_backfill": bool(args.action_chunk_backfill),
        "action_chunk_backfilled_detections": int(action_chunk_backfilled),
        "action_chunk_supported_matches": int(action_chunk_supported),
        "action_chunk_max_step": args.action_chunk_max_step if args.action_chunk_max_step > 0 else int(config["future_len"]),
        "action_chunk_top_k": args.action_chunk_top_k if args.action_chunk_top_k > 0 else args.top_k,
        "action_chunk_score_decay": args.action_chunk_score_decay,
        "action_chunk_merge_mode": args.action_chunk_merge_mode,
        "action_chunk_support_iou": args.action_chunk_support_iou,
        "action_chunk_support_weight": args.action_chunk_support_weight,
        "action_chunk_keep_unmatched": bool(args.action_chunk_keep_unmatched),
        "state_key": state_key,
        "future_len": int(config["future_len"]),
        "output_chunk_len": int(config["future_len"]) + 1,
        "motion_channels": bool(config.get("motion_channels", False)),
        "spatial_refine_layers": int(config.get("spatial_refine_layers", 0)),
        "spatial_refine_kernel": int(config.get("spatial_refine_kernel", 7)),
        "spatial_refine_expansion": float(config.get("spatial_refine_expansion", 2.0)),
        "memory_mode": str(config.get("memory_mode", "last")),
        "box_size_scale": float(config.get("box_size_scale", 1.0)),
        "query_mode": str(config.get("query_mode", "learned")),
        "anchor_offset_cells": float(config.get("anchor_offset_cells", 4.0)),
        "dense_obj_source": str(config.get("dense_obj_source", "token")),
        "memory_attention": str(config.get("memory_attention", "none")),
        "memory_slots": int(config.get("memory_slots", 64)),
        "memory_match_mode": str(config.get("memory_match_mode", "none")),
        "memory_match_weight": float(config.get("memory_match_weight", 0.0)),
        "memory_match_temperature": float(config.get("memory_match_temperature", 5.0)),
        "motion_score_mode": str(config.get("motion_score_mode", "none")),
        "motion_score_weight": float(config.get("motion_score_weight", 1.0)),
        "proposal_mode": str(config.get("proposal_mode", "none")),
        "quality_score_mode": str(config.get("quality_score_mode", "none")),
        "device": str(device),
        "device_arg": args.device,
    }
    args.out_pkl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
