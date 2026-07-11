import argparse
import json
import pickle
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


IMAGE_RE = re.compile(r"^(?P<seq>Clip_\d+)_(?P<frame>\d+)\.png$")


def row_box(row):
    return np.asarray(row["bbox"][:4], dtype=np.float64)


def row_score(row):
    return max(float(row.get("objectness", 0) or 0), float(row.get("final_drone_score", 0) or 0))


def detection_box(detection):
    x = float(detection["x"])
    y = float(detection["y"])
    width = float(detection["w"])
    height = float(detection["h"])
    return np.asarray([x - width / 2, y - height / 2, x + width / 2, y + height / 2], dtype=np.float64)


def iou(left, right):
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def make_detection(box, confidence, track_id, image_width, image_height, metadata=None):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(image_width), float(x1)))
    x2 = max(0.0, min(float(image_width), float(x2)))
    y1 = max(0.0, min(float(image_height), float(y1)))
    y2 = max(0.0, min(float(image_height), float(y2)))
    detection = {
        "track_id": int(track_id),
        "x": (x1 + x2) / 2,
        "y": (y1 + y2) / 2,
        "w": max(0.0, x2 - x1),
        "h": max(0.0, y2 - y1),
        "n": "airborne",
        "s": min(0.999, max(0.2001, float(confidence))),
        "source": "action_bank_camera_compensated_interpolation",
    }
    if metadata:
        detection.update(metadata)
    return detection


def warp_box(box, affine):
    points = np.asarray(
        [[box[0], box[1], 1.0], [box[2], box[1], 1.0], [box[2], box[3], 1.0], [box[0], box[3], 1.0]],
        dtype=np.float64,
    )
    warped = (affine @ points.T).T
    return np.asarray(
        [warped[:, 0].min(), warped[:, 1].min(), warped[:, 0].max(), warped[:, 1].max()], dtype=np.float64
    )


def crop_patch(image, box, patch_size):
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 2.0)
    height = max(y2 - y1, 2.0)
    x1 = max(0, int(np.floor(x1 - width * 0.25)))
    y1 = max(0, int(np.floor(y1 - height * 0.25)))
    x2 = min(image.shape[1], int(np.ceil(x2 + width * 0.25)))
    y2 = min(image.shape[0], int(np.ceil(y2 + height * 0.25)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    patch = cv2.resize(image[y1:y2, x1:x2], (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    patch = patch.astype(np.float32)
    patch -= patch.mean()
    norm = float(np.linalg.norm(patch))
    return patch / norm if norm > 1e-6 else None


def patch_similarity(left, right):
    if left is None or right is None:
        return -1.0
    return float(np.clip(np.sum(left * right), -1.0, 1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-folder", required=True, type=Path)
    parser.add_argument("--prediction-part", type=Path)
    parser.add_argument("--output-name", default="predictions_split_0.pkl")
    parser.add_argument("--tracklets", required=True, type=Path)
    parser.add_argument("--frames-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--score-field", default="vatd_score")
    parser.add_argument("--min-action-score", type=float, default=0.6)
    parser.add_argument("--max-gap", type=int, default=30)
    parser.add_argument("--flow-width", type=int, default=612)
    parser.add_argument("--promotion-iou", type=float, default=0.3)
    parser.add_argument("--duplicate-iou", type=float, default=0.5)
    parser.add_argument("--appearance-patch-size", type=int, default=24)
    parser.add_argument("--appearance-search-fraction", type=float, default=0.5)
    args = parser.parse_args()

    frame_paths = {}
    for path in args.frames_root.rglob("*.png"):
        match = IMAGE_RE.match(path.name)
        if match:
            frame_paths[(match.group("seq"), int(match.group("frame")))] = path
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {args.frames_root}")
    frame_sequences = {key[0] for key in frame_paths}

    records = []
    record_by_key = {}
    existing_boxes = defaultdict(list)
    existing_by_track = defaultdict(dict)
    prediction_parts = [args.prediction_part] if args.prediction_part is not None else sorted(args.results_folder.glob("*.pkl"))
    for part in prediction_parts:
        for source_record in pickle.load(part.open("rb")):
            record = dict(source_record)
            record["detections"] = [dict(item) for item in source_record.get("detections") or []]
            match = IMAGE_RE.match(str(record.get("img_name") or ""))
            if match:
                key = (match.group("seq"), int(match.group("frame")))
                record_by_key[key] = record
                for item in record["detections"]:
                    existing_boxes[key].append(detection_box(item))
                    existing_by_track[key][int(item.get("track_id", -1))] = item
            records.append(record)
    for key, path in frame_paths.items():
        if key not in record_by_key:
            record = {"img_name": path.name, "detections": []}
            record_by_key[key] = record
            records.append(record)

    groups = defaultdict(dict)
    actions = defaultdict(float)
    for line in args.tracklets.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows = item.get("rows") or []
        meta = item.get("meta") or {}
        if not rows:
            continue
        seq = str(meta.get("seq") or rows[0].get("seq") or "")
        if seq not in frame_sequences:
            continue
        raw_track = str(meta.get("raw_track_id") or rows[0].get("raw_track_id") or meta.get("track_id") or "")
        action = float(meta.get(args.score_field, 0) or 0)
        actions[(seq, raw_track)] = max(actions[(seq, raw_track)], action)
        for row in rows:
            groups[(seq, raw_track)][int(float(row.get("frame_id", 0) or 0))] = row

    @lru_cache(maxsize=12)
    def gray(key):
        image = cv2.imread(str(frame_paths[key]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(frame_paths[key])
        scale = args.flow_width / image.shape[1]
        resized = cv2.resize(image, (args.flow_width, max(32, int(round(image.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        return resized, scale

    @lru_cache(maxsize=6)
    def full_gray(key):
        image = cv2.imread(str(frame_paths[key]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(frame_paths[key])
        return image

    def refine_with_appearance(seq, left_frame, right_frame, frame, left_box, right_box, predicted):
        left_template = crop_patch(full_gray((seq, left_frame)), left_box, args.appearance_patch_size)
        right_template = crop_patch(full_gray((seq, right_frame)), right_box, args.appearance_patch_size)
        current_image = full_gray((seq, frame))
        width = max(predicted[2] - predicted[0], 2.0)
        height = max(predicted[3] - predicted[1], 2.0)
        fraction = args.appearance_search_fraction
        offsets = (-fraction, 0.0, fraction)
        best_box = predicted
        best_left = -1.0
        best_right = -1.0
        best_score = -2.0
        best_shift = 0.0
        for offset_y in offsets:
            for offset_x in offsets:
                shifted = predicted + np.asarray(
                    [offset_x * width, offset_y * height, offset_x * width, offset_y * height], dtype=np.float64
                )
                candidate_patch = crop_patch(current_image, shifted, args.appearance_patch_size)
                left_score = patch_similarity(left_template, candidate_patch)
                right_score = patch_similarity(right_template, candidate_patch)
                score = 0.5 * (left_score + right_score)
                if score > best_score:
                    best_box = shifted
                    best_left = left_score
                    best_right = right_score
                    best_score = score
                    best_shift = float(np.hypot(offset_x, offset_y))
        raw_patch = crop_patch(current_image, best_box, args.appearance_patch_size)
        raw_std = float(np.std(raw_patch)) if raw_patch is not None else 0.0
        return best_box, {
            "appearance_left_ncc": best_left,
            "appearance_right_ncc": best_right,
            "appearance_mean_ncc": 0.5 * (best_left + best_right),
            "appearance_min_ncc": min(best_left, best_right),
            "appearance_endpoint_ncc": patch_similarity(left_template, right_template),
            "appearance_search_shift": best_shift,
            "appearance_patch_std": raw_std,
        }

    transform_cache = {}
    counters = defaultdict(int)

    def pair_transform(seq, frame):
        key = (seq, frame)
        if key in transform_cache:
            return transform_cache[key]
        left_key = (seq, frame)
        right_key = (seq, frame + 1)
        if left_key not in frame_paths or right_key not in frame_paths:
            transform_cache[key] = np.eye(3, dtype=np.float64)
            counters["missing_frame_pair"] += 1
            return transform_cache[key]
        left, scale = gray(left_key)
        right, right_scale = gray(right_key)
        if abs(scale - right_scale) > 1e-6:
            raise ValueError("Frame scale mismatch")
        points = cv2.goodFeaturesToTrack(left, maxCorners=700, qualityLevel=0.01, minDistance=7, blockSize=7)
        affine = None
        if points is not None and len(points) >= 12:
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                left, right, points, None, winSize=(31, 31), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            valid = status.reshape(-1).astype(bool)
            if valid.sum() >= 12:
                affine_small, inliers = cv2.estimateAffinePartial2D(
                    points.reshape(-1, 2)[valid], tracked.reshape(-1, 2)[valid], method=cv2.RANSAC,
                    ransacReprojThreshold=2.5, maxIters=2000, confidence=0.995,
                )
                if affine_small is not None and np.isfinite(affine_small).all():
                    affine = np.eye(3, dtype=np.float64)
                    affine[:2, :2] = affine_small[:, :2]
                    affine[:2, 2] = affine_small[:, 2] / scale
                    counters["flow_pair_success"] += 1
                    if inliers is not None:
                        counters["flow_inliers"] += int(inliers.sum())
        if affine is None:
            affine = np.eye(3, dtype=np.float64)
            counters["flow_pair_fallback_identity"] += 1
        transform_cache[key] = affine
        return affine

    def compose_forward(seq, start, end):
        transform = np.eye(3, dtype=np.float64)
        for frame in range(start, end):
            transform = pair_transform(seq, frame) @ transform
        return transform

    processed_groups = 0
    total_groups = len(groups)
    for (seq, raw_track), rows_by_frame in groups.items():
        processed_groups += 1
        if args.progress is not None and (processed_groups == 1 or processed_groups % 5 == 0 or processed_groups == total_groups):
            args.progress.parent.mkdir(parents=True, exist_ok=True)
            progress = {"status": "running", "done": processed_groups - 1, "total": total_groups, "last_completed_unit": f"group_{processed_groups - 1}", "counters": dict(counters)}
            args.progress.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(json.dumps(progress), flush=True)
        if actions[(seq, raw_track)] < args.min_action_score:
            continue
        try:
            track_id = int(raw_track)
        except ValueError:
            continue
        frames = sorted(rows_by_frame)
        for left_frame, right_frame in zip(frames, frames[1:]):
            gap = right_frame - left_frame
            if gap <= 1 or gap > args.max_gap:
                continue
            if any((seq, frame) not in frame_paths for frame in range(left_frame, right_frame + 1)):
                counters["skipped_incomplete_gap"] += 1
                continue
            counters["selected_gaps"] += 1
            left_row = rows_by_frame[left_frame]
            right_row = rows_by_frame[right_frame]
            left_box = row_box(left_row)
            right_box = row_box(right_row)
            full_transform = compose_forward(seq, left_frame, right_frame)
            inverse_full = np.linalg.inv(full_transform)
            confidence = max(0.2001, min(row_score(left_row), row_score(right_row)))
            action_score = actions[(seq, raw_track)]
            image_width = int(left_row.get("image_width") or 2448)
            image_height = int(left_row.get("image_height") or 2048)
            for frame in range(left_frame + 1, right_frame):
                counters["candidate_frames"] += 1
                left_to_frame = compose_forward(seq, left_frame, frame)
                frame_to_right = compose_forward(seq, frame, right_frame)
                right_to_frame = np.linalg.inv(frame_to_right)
                left_warped = warp_box(left_box, left_to_frame)
                right_warped = warp_box(right_box, right_to_frame)
                alpha = (frame - left_frame) / gap
                predicted = (1 - alpha) * left_warped + alpha * right_warped
                predicted, appearance_metadata = refine_with_appearance(
                    seq, left_frame, right_frame, frame, left_box, right_box, predicted
                )
                left_width = max(left_warped[2] - left_warped[0], 1e-6)
                left_height = max(left_warped[3] - left_warped[1], 1e-6)
                right_width = max(right_warped[2] - right_warped[0], 1e-6)
                right_height = max(right_warped[3] - right_warped[1], 1e-6)
                left_center = np.asarray([(left_warped[0] + left_warped[2]) / 2, (left_warped[1] + left_warped[3]) / 2])
                right_center = np.asarray([(right_warped[0] + right_warped[2]) / 2, (right_warped[1] + right_warped[3]) / 2])
                reference_diagonal = max(np.hypot((left_width + right_width) / 2, (left_height + right_height) / 2), 1e-6)
                metadata = {
                    "flow_endpoint_iou": float(iou(left_warped, right_warped)),
                    "flow_center_disagreement": float(np.linalg.norm(left_center - right_center) / reference_diagonal),
                    "flow_width_ratio": float(min(left_width, right_width) / max(left_width, right_width)),
                    "flow_height_ratio": float(min(left_height, right_height) / max(left_height, right_height)),
                    "flow_area_ratio": float(min(left_width * left_height, right_width * right_height) / max(left_width * left_height, right_width * right_height)),
                    "flow_gap": int(gap),
                    "flow_alpha": float(alpha),
                    "flow_action_score": float(action_score),
                    "flow_left_score": float(row_score(left_row)),
                    "flow_right_score": float(row_score(right_row)),
                    "video_action_model_fusion_score_tracklet_score": float(action_score),
                    **appearance_metadata,
                }
                record = record_by_key[(seq, frame)]
                existing = existing_by_track[(seq, frame)].get(track_id)
                if existing is not None:
                    if float(existing.get("s", 0) or 0) < 0.2001 and iou(predicted, detection_box(existing)) >= args.promotion_iou:
                        existing["s"] = confidence
                        existing["source"] = "action_bank_camera_compensated_promotion"
                        counters["promoted_existing"] += 1
                    else:
                        counters["existing_not_promoted"] += 1
                    continue
                if any(iou(predicted, old) >= args.duplicate_iou for old in existing_boxes[(seq, frame)]):
                    counters["duplicate"] += 1
                    continue
                item = make_detection(predicted, confidence, track_id, image_width, image_height, metadata)
                if item["w"] < 1 or item["h"] < 1:
                    counters["bad_geometry"] += 1
                    continue
                record["detections"].append(item)
                existing_boxes[(seq, frame)].append(predicted)
                existing_by_track[(seq, frame)][track_id] = item
                counters["added_new"] += 1

    output_dir = args.out_dir / "aotpredictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / args.output_name
    with output.open("wb") as handle:
        pickle.dump(records, handle)
    summary = {
        "protocol": "camera-motion-compensated action-bank gap recovery",
        "results_folder": str(args.results_folder),
        "frames_root": str(args.frames_root),
        "tracklets": str(args.tracklets),
        "output": str(output),
        "parameters": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "records": len(records),
        "frame_paths": len(frame_paths),
        "groups": len(groups),
        "counters": dict(counters),
        "uses_labels": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "flow_recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.progress is not None:
        args.progress.write_text(json.dumps({"status": "complete", "done": total_groups, "total": total_groups, "last_completed_unit": "flow_recovery_summary", "counters": dict(counters)}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()





