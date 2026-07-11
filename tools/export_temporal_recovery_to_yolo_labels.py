from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def _remap_path(path: Path) -> Path:
    if path.exists():
        return path
    text = str(path)
    for old, new in (("D:\\URAP_datasets\\", "U:\\URAP_datasets\\"), ("D:/URAP_datasets/", "U:/URAP_datasets/")):
        if text.startswith(old):
            candidate = Path(new + text[len(old) :])
            if candidate.exists():
                return candidate
    return path


def _read_images(path: Path, limit: int = 0) -> list[Path]:
    out = [_remap_path(Path(line.strip())) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return out[:limit] if limit else out


def _size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return int(img.shape[1]), int(img.shape[0])


def _xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = min(max(0.0, x1), float(width))
    x2 = min(max(0.0, x2), float(width))
    y1 = min(max(0.0, y1), float(height))
    y2 = min(max(0.0, y2), float(height))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / width, cy / height, bw / width, bh / height


def main() -> None:
    parser = argparse.ArgumentParser(description="Export temporal recovery trajectory.csv to YOLO prediction labels.")
    parser.add_argument("--trajectory-csv", type=Path, required=True)
    parser.add_argument("--images-list", type=Path, required=True)
    parser.add_argument("--out-label-dir", type=Path, required=True)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    args.out_label_dir.mkdir(parents=True, exist_ok=True)
    images = _read_images(args.images_list, args.max_frames)
    rows = list(csv.DictReader(args.trajectory_csv.open("r", encoding="utf-8-sig")))
    written = 0
    skipped = 0
    for index, image_path in enumerate(images):
        out_path = args.out_label_dir / f"{image_path.stem}.txt"
        if index >= len(rows):
            out_path.write_text("", encoding="utf-8")
            skipped += 1
            continue
        row = rows[index]
        if not row.get("x1"):
            out_path.write_text("", encoding="utf-8")
            skipped += 1
            continue
        width, height = _size(image_path)
        x1, y1, x2, y2 = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
        cx, cy, bw, bh = _xyxy_to_yolo(x1, y1, x2, y2, width, height)
        score = float(row["score"]) if row.get("score") else 1.0
        out_path.write_text(f"{args.class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f} {score:.8f}\n", encoding="utf-8")
        written += 1

    summary = {
        "trajectory_csv": str(args.trajectory_csv),
        "images_list": str(args.images_list),
        "out_label_dir": str(args.out_label_dir),
        "images": len(images),
        "rows": len(rows),
        "written": written,
        "skipped": skipped,
    }
    (args.out_label_dir.parent / "temporal_recovery_yolo_label_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
