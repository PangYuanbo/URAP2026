from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
from pathlib import Path
from typing import Any


IMG_RE = re.compile(r"^(?P<seq>Clip_\d+)_(?P<frame>\d+)\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add conservative internal-gap pseudo detections to AOT prediction PKLs.")
    parser.add_argument("--results-folder", type=Path, required=True)
    parser.add_argument("--tracklet-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--score-field", default="video_action_model_fusion_score")
    parser.add_argument("--min-score", type=float, default=0.9)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--min-tracklet-rows", type=int, default=8)
    parser.add_argument("--duplicate-iou", type=float, default=0.5)
    parser.add_argument("--conf-scale", type=float, default=1.0)
    parser.add_argument("--pseudo-track-offset", type=int, default=1000000)
    return parser.parse_args()


def row_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    value = row.get("bbox") or row.get("bbox_xyxy")
    if value is None:
        value = [row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")]
    if value is None or len(value) != 4:
        raise ValueError("tracklet row missing bbox")
    return tuple(float(v) for v in value)  # type: ignore[return-value]


def row_score(row: dict[str, Any]) -> float:
    return float(max(row.get("objectness", 0.0), row.get("final_drone_score", 0.0), row.get("score", 0.0)))


def xyxy_to_aot_det(
    box: tuple[float, float, float, float],
    score: float,
    track_id: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    x1 = min(float(image_width), max(0.0, x1))
    x2 = min(float(image_width), max(0.0, x2))
    y1 = min(float(image_height), max(0.0, y1))
    y2 = min(float(image_height), max(0.0, y2))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return {
        "track_id": int(track_id),
        "x": float(x1 + w * 0.5),
        "y": float(y1 + h * 0.5),
        "w": float(w),
        "h": float(h),
        "n": "airborne",
        "s": float(min(0.999999, max(0.001, score))),
        "source": "tracklet_gap_interpolation",
    }


def det_box(det: dict[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        cx = float(det["x"])
        cy = float(det["y"])
        w = float(det["w"])
        h = float(det["h"])
    except (KeyError, TypeError, ValueError):
        return None
    return cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(aa + ba - inter, 1e-9)


def parse_img_name(name: str) -> tuple[str, int] | None:
    match = IMG_RE.match(name)
    if match is None:
        return None
    return match.group("seq"), int(match.group("frame"))


def load_records(parts: list[Path]) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    record_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for part in parts:
        with part.open("rb") as f:
            part_records = pickle.load(f)
        if not isinstance(part_records, list):
            raise ValueError(f"{part}: PKL root is not a list")
        for record in part_records:
            if not isinstance(record, dict):
                continue
            out_record = dict(record)
            out_record["detections"] = [dict(det) for det in (record.get("detections") or []) if isinstance(det, dict)]
            parsed = parse_img_name(str(record.get("img_name") or ""))
            if parsed is not None:
                record_by_key[parsed] = out_record
            records.append(out_record)
    return records, record_by_key


def main() -> None:
    args = parse_args()
    if args.max_gap < 2:
        raise ValueError("--max-gap must be >= 2 to fill internal gaps")
    if not args.results_folder.exists():
        raise FileNotFoundError(args.results_folder)
    if not args.tracklet_jsonl.exists():
        raise FileNotFoundError(args.tracklet_jsonl)
    parts = sorted(args.results_folder.glob("*.pkl"))
    if not parts:
        raise FileNotFoundError(f"no .pkl prediction parts found: {args.results_folder}")

    records, record_by_key = load_records(parts)
    existing_boxes: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
    max_track_id = 0
    for key, record in record_by_key.items():
        boxes = []
        for det in record.get("detections") or []:
            box = det_box(det)
            if box is not None:
                boxes.append(box)
            try:
                max_track_id = max(max_track_id, int(det.get("track_id", 0)))
            except (TypeError, ValueError):
                pass
        existing_boxes[key] = boxes

    candidates = 0
    added = 0
    skipped_duplicate = 0
    skipped_missing_frame = 0
    skipped_bad_geometry = 0
    selected_tracklets = 0
    pseudo_index = 0

    with args.tracklet_jsonl.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            meta = dict(item.get("meta") or {})
            rows = sorted([dict(row) for row in item.get("rows") or []], key=lambda r: int(float(r.get("frame_id", 0) or 0)))
            if len(rows) < args.min_tracklet_rows:
                continue
            raw_score = meta.get(args.score_field)
            if raw_score is None and rows:
                raw_score = rows[0].get(args.score_field)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if score < args.min_score:
                continue
            selected_tracklets += 1
            for left, right in zip(rows, rows[1:]):
                seq = str(left.get("seq") or meta.get("seq") or "")
                try:
                    left_frame = int(float(left.get("frame_id", 0) or 0))
                    right_frame = int(float(right.get("frame_id", 0) or 0))
                except (TypeError, ValueError):
                    continue
                gap = right_frame - left_frame
                if gap <= 1 or gap > args.max_gap:
                    continue
                left_box = row_box(left)
                right_box = row_box(right)
                image_width = int(left.get("image_width") or right.get("image_width") or 2448)
                image_height = int(left.get("image_height") or right.get("image_height") or 2048)
                left_score = row_score(left)
                right_score = row_score(right)
                for frame_id in range(left_frame + 1, right_frame):
                    candidates += 1
                    key = (seq, frame_id)
                    record = record_by_key.get(key)
                    if record is None:
                        skipped_missing_frame += 1
                        continue
                    alpha = (frame_id - left_frame) / max(1, gap)
                    box = tuple((1.0 - alpha) * a + alpha * b for a, b in zip(left_box, right_box))
                    if box[2] <= box[0] or box[3] <= box[1]:
                        skipped_bad_geometry += 1
                        continue
                    boxes = existing_boxes.setdefault(key, [])
                    if any(box_iou(box, old) >= args.duplicate_iou for old in boxes):
                        skipped_duplicate += 1
                        continue
                    pseudo_index += 1
                    conf = score * args.conf_scale * min(max(left_score, 0.001), max(right_score, 0.001))
                    track_id = max_track_id + args.pseudo_track_offset + pseudo_index
                    det = xyxy_to_aot_det(box, conf, track_id, image_width=image_width, image_height=image_height)
                    record.setdefault("detections", []).append(det)
                    boxes.append(box)
                    added += 1

    out_pred_dir = args.out_dir / "aotpredictions"
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    out_part = out_pred_dir / "predictions_split_0.pkl"
    with out_part.open("wb") as f:
        pickle.dump(records, f)

    copied_extra_parts = 0
    for extra in args.results_folder.iterdir():
        if extra.suffix.lower() == ".pkl":
            continue
        dst = out_pred_dir / extra.name
        if extra.is_file():
            shutil.copyfile(extra, dst)
            copied_extra_parts += 1

    summary = {
        "results_folder": str(args.results_folder),
        "tracklet_jsonl": str(args.tracklet_jsonl),
        "out_dir": str(args.out_dir),
        "aotpredictions_dir": str(out_pred_dir),
        "part_path": str(out_part),
        "score_field": args.score_field,
        "min_score": args.min_score,
        "max_gap": args.max_gap,
        "min_tracklet_rows": args.min_tracklet_rows,
        "duplicate_iou": args.duplicate_iou,
        "conf_scale": args.conf_scale,
        "source_parts": len(parts),
        "records_written": len(records),
        "selected_tracklets": selected_tracklets,
        "candidate_gap_boxes": candidates,
        "added_boxes": added,
        "skipped_duplicate": skipped_duplicate,
        "skipped_missing_frame": skipped_missing_frame,
        "skipped_bad_geometry": skipped_bad_geometry,
        "copied_extra_parts": copied_extra_parts,
    }
    summary_path = args.out_dir / "aot_gap_augment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
