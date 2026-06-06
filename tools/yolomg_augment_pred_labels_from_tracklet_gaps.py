from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


STEM_RE = re.compile(r"^(?P<prefix>.+_)(?P<frame>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add conservative internal-gap pseudo boxes from high-scoring tracklets to YOLO labels.")
    parser.add_argument("--images-list", type=Path, required=True)
    parser.add_argument("--pred-label-dir", type=Path, required=True)
    parser.add_argument("--tracklet-jsonl", type=Path, required=True)
    parser.add_argument("--out-label-dir", type=Path, required=True)
    parser.add_argument("--score-field", default="vatd_score")
    parser.add_argument("--min-score", type=float, default=0.9)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--min-tracklet-rows", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--conf-scale", type=float, default=1.0)
    parser.add_argument("--duplicate-iou", type=float, default=0.5)
    return parser.parse_args()


def read_image_list(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def row_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    value = row.get("bbox") or row.get("bbox_xyxy")
    if value is None:
        value = [row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")]
    if value is None or len(value) != 4:
        raise ValueError("tracklet row missing bbox")
    return tuple(float(v) for v in value)  # type: ignore[return-value]


def row_score(row: dict[str, Any]) -> float:
    return float(max(row.get("objectness", 0.0), row.get("final_drone_score", 0.0), row.get("score", 0.0)))


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(aa + ba - inter, 1e-9)


def xyxy_to_yolo(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = min(float(width), max(0.0, x1))
    x2 = min(float(width), max(0.0, x2))
    y1 = min(float(height), max(0.0, y1))
    y2 = min(float(height), max(0.0, y2))
    cx = ((x1 + x2) * 0.5) / max(1, width)
    cy = ((y1 + y2) * 0.5) / max(1, height)
    bw = max(0.0, x2 - x1) / max(1, width)
    bh = max(0.0, y2 - y1) / max(1, height)
    return cx, cy, bw, bh


def yolo_to_xyxy(parts: list[str], width: int, height: int) -> tuple[float, float, float, float]:
    cx, cy, bw, bh = [float(v) for v in parts[1:5]]
    x = cx * width
    y = cy * height
    w = bw * width
    h = bh * height
    return x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5


def image_stem_for_gap(left_row: dict[str, Any], frame_id: int) -> str | None:
    image_path = left_row.get("image_path")
    if image_path:
        stem = Path(str(image_path)).stem
        match = STEM_RE.match(stem)
        if match:
            return f"{match.group('prefix')}{frame_id:0{len(match.group('frame'))}d}"
    seq = left_row.get("seq")
    if seq:
        return f"{seq}_{frame_id:05d}"
    return None


def copy_labels(images: list[Path], pred_label_dir: Path, out_label_dir: Path) -> tuple[int, int]:
    out_label_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    for image_path in images:
        src = pred_label_dir / f"{image_path.stem}.txt"
        dst = out_label_dir / f"{image_path.stem}.txt"
        if src.exists():
            shutil.copyfile(src, dst)
            copied += 1
        else:
            dst.write_text("", encoding="utf-8")
            missing += 1
    return copied, missing


def main() -> None:
    args = parse_args()
    if args.max_gap < 2:
        raise ValueError("--max-gap must be >= 2 to fill internal gaps")
    images = read_image_list(args.images_list)
    copied, missing = copy_labels(images, args.pred_label_dir, args.out_label_dir)
    existing_boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    for image_path in images:
        label_path = args.out_label_dir / f"{image_path.stem}.txt"
        boxes = []
        for line in label_path.read_text(encoding="utf-8-sig").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append(yolo_to_xyxy(parts, args.image_width, args.image_height))
        existing_boxes[image_path.stem] = boxes

    candidates = 0
    added = 0
    skipped_duplicate = 0
    skipped_missing_frame = 0
    selected_tracklets = 0
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
                left_frame = int(float(left.get("frame_id", 0) or 0))
                right_frame = int(float(right.get("frame_id", 0) or 0))
                gap = right_frame - left_frame
                if gap <= 1 or gap > args.max_gap:
                    continue
                left_box = row_box(left)
                right_box = row_box(right)
                left_score = row_score(left)
                right_score = row_score(right)
                for frame_id in range(left_frame + 1, right_frame):
                    candidates += 1
                    stem = image_stem_for_gap(left, frame_id)
                    if stem is None or stem not in existing_boxes:
                        skipped_missing_frame += 1
                        continue
                    alpha = (frame_id - left_frame) / max(1, right_frame - left_frame)
                    box = tuple((1.0 - alpha) * a + alpha * b for a, b in zip(left_box, right_box))
                    if any(box_iou(box, old) >= args.duplicate_iou for old in existing_boxes[stem]):
                        skipped_duplicate += 1
                        continue
                    conf = min(0.999999, max(0.001, score * args.conf_scale * min(max(left_score, 0.001), max(right_score, 0.001))))
                    cx, cy, bw, bh = xyxy_to_yolo(box, args.image_width, args.image_height)
                    with (args.out_label_dir / f"{stem}.txt").open("a", encoding="utf-8") as out_f:
                        out_f.write(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f} {conf:.8f}\n")
                    existing_boxes[stem].append(box)
                    added += 1

    summary = {
        "images_list": str(args.images_list),
        "pred_label_dir": str(args.pred_label_dir),
        "tracklet_jsonl": str(args.tracklet_jsonl),
        "out_label_dir": str(args.out_label_dir),
        "score_field": args.score_field,
        "min_score": args.min_score,
        "max_gap": args.max_gap,
        "min_tracklet_rows": args.min_tracklet_rows,
        "copied_label_files": copied,
        "missing_input_label_files": missing,
        "selected_tracklets": selected_tracklets,
        "candidate_gap_boxes": candidates,
        "added_boxes": added,
        "skipped_duplicate": skipped_duplicate,
        "skipped_missing_frame": skipped_missing_frame,
    }
    summary_path = args.out_label_dir.parent / f"{args.out_label_dir.name}_gap_augment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
