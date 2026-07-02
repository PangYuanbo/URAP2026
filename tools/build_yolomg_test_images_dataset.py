from __future__ import annotations

import argparse
import os
import shutil
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int, float]]:
    ys, xs = np.nonzero(mask)
    active = set(zip(xs.tolist(), ys.tolist()))
    out: list[tuple[int, int, int, int, int, float]] = []
    while active:
        start = active.pop()
        q: deque[tuple[int, int]] = deque([start])
        min_x = max_x = start[0]
        min_y = max_y = start[1]
        area = 0
        while q:
            x, y = q.popleft()
            area += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if nx == x and ny == y:
                        continue
                    if (nx, ny) in active:
                        active.remove((nx, ny))
                        q.append((nx, ny))
        score = float(mask[min_y : max_y + 1, min_x : max_x + 1].sum())
        out.append((min_x, min_y, max_x, max_y, area, score))
    return out


def mask_to_yolo_label(mask_path: Path, min_area: int = 4, pad: int = 3) -> list[str]:
    arr = np.asarray(Image.open(mask_path).convert("L"))
    h, w = arr.shape
    if arr.max() <= 0:
        return []
    threshold = max(20.0, float(arr.max()) * 0.35)
    binary = arr >= threshold
    boxes = [box for box in _components(binary) if box[4] >= min_area]
    if not boxes:
        binary = arr >= max(5.0, float(np.percentile(arr, 99.95)))
        boxes = [box for box in _components(binary) if box[4] >= min_area]
    boxes = sorted(boxes, key=lambda item: item[5], reverse=True)[:3]

    lines = []
    for min_x, min_y, max_x, max_y, _area, _score in boxes:
        x1 = max(0, min_x - pad)
        y1 = max(0, min_y - pad)
        x2 = min(w - 1, max_x + pad)
        y2 = min(h - 1, max_y + pad)
        width = max(1, x2 - x1 + 1)
        height = max(1, y2 - y1 + 1)
        cx = (x1 + x2 + 1) / 2.0 / w
        cy = (y1 + y2 + 1) / 2.0 / h
        lines.append(f"0 {cx:.8f} {cy:.8f} {width / w:.8f} {height / h:.8f}")
    return lines


def build_dataset(src_root: Path, out_root: Path) -> dict[str, Path | int]:
    image_dir = src_root / "images"
    mask_dir = src_root / "mask"
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No jpg images found under {image_dir}")
    out_images = out_root / "images"
    out_images2 = out_root / "images2"
    out_labels = out_root / "labels"
    out_root.mkdir(parents=True, exist_ok=True)
    for cache in (out_root / "images.cache", out_root / "images2.cache"):
        if cache.exists():
            cache.unlink()

    image_list: list[str] = []
    image2_list: list[str] = []
    label_count = 0
    box_count = 0
    for img_path in images:
        mask_path = mask_dir / img_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask for {img_path.name}: {mask_path}")
        dst_img = out_images / img_path.name
        dst_mask = out_images2 / img_path.name
        _link_or_copy(img_path, dst_img)
        _link_or_copy(mask_path, dst_mask)

        label_lines = mask_to_yolo_label(mask_path)
        label_file = out_labels / f"{img_path.stem}.txt"
        label_file.parent.mkdir(parents=True, exist_ok=True)
        label_file.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        if label_lines:
            label_count += 1
            box_count += len(label_lines)
        image_list.append(str(dst_img.absolute()))
        image2_list.append(str(dst_mask.absolute()))

    images_txt = out_root / "images.txt"
    images2_txt = out_root / "images2.txt"
    data_yaml = out_root / "yolomg_test_images.yaml"
    images_txt.write_text("\n".join(image_list) + "\n", encoding="utf-8")
    images2_txt.write_text("\n".join(image2_list) + "\n", encoding="utf-8")
    yaml_text = "\n".join(
        [
            f"train: {images_txt.resolve()}",
            f"train2: {images2_txt.resolve()}",
            f"val: {images_txt.resolve()}",
            f"val2: {images2_txt.resolve()}",
            f"test: {images_txt.resolve()}",
            f"test2: {images2_txt.resolve()}",
            "nc: 1",
            "names: ['Drone']",
            "",
        ]
    )
    data_yaml.write_text(yaml_text, encoding="utf-8")
    return {
        "images": len(images),
        "labeled_images": label_count,
        "boxes": box_count,
        "data_yaml": data_yaml,
        "labels": out_labels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiny YOLOMG dataset from the repository Test_images fixture.")
    parser.add_argument("--src-root", type=Path, default=ROOT / "papers" / "YOLOMG" / "data" / "Test_images")
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs" / "window_accuracy" / "yolomg_test_images_dataset")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_dataset(args.src_root, args.out_root)
    print(f"images={report['images']}")
    print(f"labeled_images={report['labeled_images']}")
    print(f"boxes={report['boxes']}")
    print(f"data_yaml={report['data_yaml']}")
    print(f"labels={report['labels']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
