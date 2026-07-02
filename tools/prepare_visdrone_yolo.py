from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

VISDRONE_URLS = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    "test-dev": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
}

CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def _download(url: str, out_path: Path) -> None:
    if out_path.is_file() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as response, tmp_path.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp_path.replace(out_path)


def _extract(zip_path: Path, root: Path, subset_dir: Path) -> None:
    if subset_dir.is_dir() and any((subset_dir / "images").glob("*.jpg")):
        return
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)


def _parse_annotation_line(line: str) -> tuple[int, int, int, int, int, int, int, int] | None:
    line = line.strip().rstrip(",")
    if not line:
        return None
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 8:
        raise ValueError(f"Bad VisDrone annotation row: {line!r}")
    x, y, w, h, score, category, truncation, occlusion = (int(float(part)) for part in parts[:8])
    return x, y, w, h, score, category, truncation, occlusion


def _to_yolo(x: int, y: int, w: int, h: int, image_w: int, image_h: int) -> tuple[float, float, float, float] | None:
    x = max(0, min(x, image_w - 1))
    y = max(0, min(y, image_h - 1))
    w = max(0, min(w, image_w - x))
    h = max(0, min(h, image_h - y))
    if w <= 0 or h <= 0:
        return None
    cx = x + w / 2.0
    cy = y + h / 2.0
    return cx / image_w, cy / image_h, w / image_w, h / image_h


def convert_subset(subset_dir: Path, keep_truncated: bool = False) -> dict[str, Any]:
    images_dir = subset_dir / "images"
    ann_dir = subset_dir / "annotations"
    labels_dir = subset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No VisDrone images found in {images_dir}")

    converted_images = 0
    converted_boxes = 0
    missing_annotations = 0
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        ann_path = ann_dir / f"{image_path.stem}.txt"
        if not ann_path.is_file():
            label_path.write_text("", encoding="utf-8")
            missing_annotations += 1
            converted_images += 1
            continue

        with Image.open(image_path) as img:
            image_w, image_h = img.size

        lines: list[str] = []
        for raw in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parsed = _parse_annotation_line(raw)
            if parsed is None:
                continue
            x, y, w, h, _score, category, truncation, _occlusion = parsed
            if category < 1 or category > 10:
                continue
            if not keep_truncated and truncation >= 2:
                continue
            yolo = _to_yolo(x, y, w, h, image_w=image_w, image_h=image_h)
            if yolo is None:
                continue
            cls = category - 1
            cx, cy, bw, bh = yolo
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        converted_images += 1
        converted_boxes += len(lines)

    return {
        "subset_dir": str(subset_dir),
        "images": converted_images,
        "boxes": converted_boxes,
        "missing_annotations": missing_annotations,
        "labels_dir": str(labels_dir),
    }


def write_split(root: Path, split: str, subset_dir: Path) -> Path:
    split_dir = root / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / f"{split}.txt"
    image_paths = sorted((subset_dir / "images").glob("*.jpg"))
    out_path.write_text("".join(f"{path.resolve()}\n" for path in image_paths), encoding="utf-8")
    return out_path


def write_yaml(root: Path, yaml_out: Path) -> Path:
    split_dir = root / "split"
    lines = [
        f"train: {str((split_dir / 'train.txt').resolve())}",
        f"val: {str((split_dir / 'val.txt').resolve())}",
        f"test: {str((split_dir / 'test-dev.txt').resolve())}",
        "",
        "nc: 10",
        "names: [" + ", ".join(repr(name) for name in CLASS_NAMES) + "]",
        "",
    ]
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    yaml_out.write_text("\n".join(lines), encoding="utf-8")
    return yaml_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and convert VisDrone DET splits to YOLO labels/list files.")
    parser.add_argument("--root", type=Path, default=ROOT / "datasets" / "VisDrone")
    parser.add_argument("--split", action="append", choices=sorted(VISDRONE_URLS), default=[])
    parser.add_argument("--download", action="store_true", help="Download missing split zips from the configured public mirrors.")
    parser.add_argument("--keep-truncated", action="store_true", help="Keep VisDrone objects with truncation >= 2.")
    parser.add_argument("--yaml-out", type=Path, default=ROOT / "runs" / "window_accuracy" / "papers" / "visdrone_esod.yaml")
    parser.add_argument("--summary-json", type=Path, default=ROOT / "runs" / "window_accuracy" / "papers" / "visdrone_prepare_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    splits = args.split or ["val"]
    root.mkdir(parents=True, exist_ok=True)
    zip_dir = root / "zips"

    summary: dict[str, Any] = {"root": str(root), "splits": []}
    for split in splits:
        subset_dir = root / f"VisDrone2019-DET-{split}"
        if args.download:
            zip_path = zip_dir / f"VisDrone2019-DET-{split}.zip"
            _download(VISDRONE_URLS[split], zip_path)
            _extract(zip_path, root, subset_dir)
        if not subset_dir.is_dir():
            raise FileNotFoundError(f"Missing subset directory: {subset_dir}")
        converted = convert_subset(subset_dir, keep_truncated=args.keep_truncated)
        split_file = write_split(root, split, subset_dir)
        converted["split_file"] = str(split_file)
        summary["splits"].append(converted)

    yaml_path = write_yaml(root, args.yaml_out.resolve())
    summary["yaml"] = str(yaml_path)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
