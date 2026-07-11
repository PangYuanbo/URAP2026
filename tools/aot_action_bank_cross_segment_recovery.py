import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path


IMAGE_RE = re.compile(r"^(?P<seq>Clip_\d+)_(?P<frame>\d+)\.png$")


def box(row):
    return tuple(float(value) for value in row["bbox"][:4])


def score(row):
    return max(float(row.get("objectness", 0) or 0), float(row.get("final_drone_score", 0) or 0))


def iou(left, right):
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-9, left_area + right_area - intersection)


def detection(interpolated, confidence, track_id, width, height, source="action_bank_cross_segment_interpolation"):
    x1, y1, x2, y2 = interpolated
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    return {
        "track_id": track_id,
        "x": (x1 + x2) / 2,
        "y": (y1 + y2) / 2,
        "w": max(0.0, x2 - x1),
        "h": max(0.0, y2 - y1),
        "n": "airborne",
        "s": min(0.999, max(0.2001, confidence)),
        "source": source,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-folder", required=True, type=Path)
    parser.add_argument("--tracklets", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--frames-root", type=Path)
    parser.add_argument("--score-field", default="vatd_score")
    parser.add_argument("--min-action-score", type=float, default=0.7)
    parser.add_argument("--max-gap", type=int, default=10)
    parser.add_argument("--edge-horizon", type=int, default=0)
    parser.add_argument("--edge-min-action-score", type=float)
    parser.add_argument("--use-track-memory-score", action="store_true")
    parser.add_argument("--promote-existing-track-score", action="store_true")
    parser.add_argument("--duplicate-iou", type=float, default=0.5)
    args = parser.parse_args()

    records = []
    record_by_key = {}
    existing = defaultdict(list)
    existing_track_ids = defaultdict(set)
    existing_detection_by_track = defaultdict(dict)
    max_track_id = 0
    for part in sorted(args.results_folder.glob("*.pkl")):
        for source_record in pickle.load(part.open("rb")):
            record = dict(source_record)
            record["detections"] = [dict(item) for item in source_record.get("detections") or []]
            match = IMAGE_RE.match(str(record.get("img_name") or ""))
            if match:
                key = (match.group("seq"), int(match.group("frame")))
                record_by_key[key] = record
                for item in record["detections"]:
                    cx, cy, width, height = map(float, (item["x"], item["y"], item["w"], item["h"]))
                    existing[key].append((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2))
                    try:
                        track_id = int(item.get("track_id", 0))
                        existing_track_ids[key].add(track_id)
                        existing_detection_by_track[key][track_id] = item
                        max_track_id = max(max_track_id, track_id)
                    except (TypeError, ValueError):
                        pass
            records.append(record)

    created_empty_records = 0
    if args.frames_root is not None:
        for image_path in args.frames_root.rglob("*.png"):
            match = IMAGE_RE.match(image_path.name)
            if match is None:
                continue
            key = (match.group("seq"), int(match.group("frame")))
            if key in record_by_key:
                continue
            record = {"detections": [], "img_name": image_path.name}
            record_by_key[key] = record
            records.append(record)
            created_empty_records += 1

    groups = defaultdict(dict)
    group_scores = defaultdict(dict)
    for line in args.tracklets.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        meta = item.get("meta") or {}
        rows = item.get("rows") or []
        if not rows:
            continue
        seq = str(meta.get("seq") or rows[0].get("seq") or "")
        raw_track = str(meta.get("raw_track_id") or rows[0].get("raw_track_id") or meta.get("track_id") or "")
        action_score = float(meta.get(args.score_field, 0) or 0)
        for row in rows:
            frame = int(float(row.get("frame_id", 0) or 0))
            groups[(seq, raw_track)][frame] = row
            group_scores[(seq, raw_track)][frame] = action_score

    sequence_frame_bounds = {}
    for seq, frame in record_by_key:
        if seq not in sequence_frame_bounds:
            sequence_frame_bounds[seq] = [frame, frame]
        else:
            sequence_frame_bounds[seq][0] = min(sequence_frame_bounds[seq][0], frame)
            sequence_frame_bounds[seq][1] = max(sequence_frame_bounds[seq][1], frame)

    candidates = 0
    added = 0
    duplicates = 0
    duplicate_track_ids = 0
    promoted_existing_tracks = 0
    missing_records = 0
    selected_gaps = 0
    edge_candidates = 0
    edge_added = 0
    examples = []
    for (seq, raw_track), rows_by_frame in groups.items():
        frames = sorted(rows_by_frame)
        track_memory_score = max(group_scores[(seq, raw_track)].values())
        try:
            output_track_id = int(raw_track)
        except ValueError:
            max_track_id += 1
            output_track_id = max_track_id
        for left_frame, right_frame in zip(frames, frames[1:]):
            gap = right_frame - left_frame
            if gap <= 1 or gap > args.max_gap:
                continue
            left_action = group_scores[(seq, raw_track)][left_frame]
            right_action = group_scores[(seq, raw_track)][right_frame]
            gap_action = track_memory_score if args.use_track_memory_score else max(left_action, right_action)
            if gap_action < args.min_action_score:
                continue
            selected_gaps += 1
            left_row, right_row = rows_by_frame[left_frame], rows_by_frame[right_frame]
            left_box, right_box = box(left_row), box(right_row)
            confidence = max(0.2001, min(score(left_row), score(right_row)))
            width = int(left_row.get("image_width") or right_row.get("image_width") or 2448)
            height = int(left_row.get("image_height") or right_row.get("image_height") or 2048)
            for frame in range(left_frame + 1, right_frame):
                candidates += 1
                record = record_by_key.get((seq, frame))
                if record is None:
                    missing_records += 1
                    continue
                if output_track_id in existing_track_ids[(seq, frame)]:
                    existing_detection = existing_detection_by_track[(seq, frame)].get(output_track_id)
                    if args.promote_existing_track_score and existing_detection is not None and float(existing_detection.get("s", 0) or 0) < 0.2001:
                        existing_detection["s"] = confidence
                        existing_detection["source"] = "action_bank_track_memory_promotion"
                        promoted_existing_tracks += 1
                    duplicate_track_ids += 1
                    continue
                alpha = (frame - left_frame) / gap
                interpolated = tuple((1 - alpha) * left + alpha * right for left, right in zip(left_box, right_box))
                if any(iou(interpolated, old) >= args.duplicate_iou for old in existing[(seq, frame)]):
                    duplicates += 1
                    continue
                record["detections"].append(detection(interpolated, confidence, output_track_id, width, height))
                existing[(seq, frame)].append(interpolated)
                existing_track_ids[(seq, frame)].add(output_track_id)
                existing_detection_by_track[(seq, frame)][output_track_id] = record["detections"][-1]
                added += 1
                if len(examples) < 25:
                    examples.append({"seq": seq, "raw_track": raw_track, "frame": frame, "gap": [left_frame, right_frame], "action": gap_action, "confidence": confidence})

        edge_threshold = args.edge_min_action_score if args.edge_min_action_score is not None else args.min_action_score
        if args.edge_horizon <= 0 or len(frames) < 2 or seq not in sequence_frame_bounds:
            continue
        sequence_first, sequence_last = sequence_frame_bounds[seq]
        edge_specs = (
            (frames[0], frames[1], -1, sequence_first),
            (frames[-1], frames[-2], 1, sequence_last),
        )
        for endpoint_frame, reference_frame, direction, sequence_limit in edge_specs:
            endpoint_action = track_memory_score if args.use_track_memory_score else group_scores[(seq, raw_track)][endpoint_frame]
            if endpoint_action < edge_threshold:
                continue
            endpoint_row = rows_by_frame[endpoint_frame]
            reference_row = rows_by_frame[reference_frame]
            endpoint_box = box(endpoint_row)
            reference_box = box(reference_row)
            frame_delta = endpoint_frame - reference_frame
            if frame_delta == 0:
                continue
            velocity = tuple((endpoint - reference) / frame_delta for endpoint, reference in zip(endpoint_box, reference_box))
            confidence = max(0.2001, score(endpoint_row))
            width = int(endpoint_row.get("image_width") or reference_row.get("image_width") or 2448)
            height = int(endpoint_row.get("image_height") or reference_row.get("image_height") or 2048)
            for step in range(1, args.edge_horizon + 1):
                frame = endpoint_frame + direction * step
                if (direction < 0 and frame < sequence_limit) or (direction > 0 and frame > sequence_limit):
                    break
                candidates += 1
                edge_candidates += 1
                record = record_by_key.get((seq, frame))
                if record is None:
                    missing_records += 1
                    continue
                if output_track_id in existing_track_ids[(seq, frame)]:
                    existing_detection = existing_detection_by_track[(seq, frame)].get(output_track_id)
                    if args.promote_existing_track_score and existing_detection is not None and float(existing_detection.get("s", 0) or 0) < 0.2001:
                        existing_detection["s"] = max(0.2001, confidence * (0.98 ** step))
                        existing_detection["source"] = "action_bank_track_memory_promotion"
                        promoted_existing_tracks += 1
                    duplicate_track_ids += 1
                    continue
                extrapolated = tuple(endpoint + direction * step * delta for endpoint, delta in zip(endpoint_box, velocity))
                clipped_width = min(float(width), extrapolated[2]) - max(0.0, extrapolated[0])
                clipped_height = min(float(height), extrapolated[3]) - max(0.0, extrapolated[1])
                if clipped_width < 1.0 or clipped_height < 1.0:
                    continue
                if any(iou(extrapolated, old) >= args.duplicate_iou for old in existing[(seq, frame)]):
                    duplicates += 1
                    continue
                record["detections"].append(detection(
                    extrapolated,
                    confidence * (0.98 ** step),
                    output_track_id,
                    width,
                    height,
                    source="action_bank_edge_extrapolation",
                ))
                existing[(seq, frame)].append(extrapolated)
                existing_track_ids[(seq, frame)].add(output_track_id)
                existing_detection_by_track[(seq, frame)][output_track_id] = record["detections"][-1]
                added += 1
                edge_added += 1

    prediction_dir = args.out_dir / "aotpredictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    output = prediction_dir / "predictions_split_0.pkl"
    pickle.dump(records, output.open("wb"))
    summary = {
        "results_folder": str(args.results_folder),
        "tracklets": str(args.tracklets),
        "out": str(output),
        "parameters": {
            "min_action_score": args.min_action_score,
            "max_gap": args.max_gap,
            "edge_horizon": args.edge_horizon,
            "edge_min_action_score": args.edge_min_action_score,
            "use_track_memory_score": args.use_track_memory_score,
            "promote_existing_track_score": args.promote_existing_track_score,
            "duplicate_iou": args.duplicate_iou,
        },
        "records": len(records),
        "created_empty_records": created_empty_records,
        "groups": len(groups),
        "selected_gaps": selected_gaps,
        "edge_candidates": edge_candidates,
        "edge_added": edge_added,
        "candidate_boxes": candidates,
        "added_boxes": added,
        "duplicates": duplicates,
        "duplicate_track_ids": duplicate_track_ids,
        "promoted_existing_tracks": promoted_existing_tracks,
        "missing_records": missing_records,
        "uses_labels": False,
        "examples": examples,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
