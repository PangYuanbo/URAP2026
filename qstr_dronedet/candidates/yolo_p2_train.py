from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import yaml


def write_yolov8_p2_model_yaml(path: str | Path, nc: int = 1) -> Path:
    """Write a small YOLOv8-style P2/P3/P4/P5 model yaml for candidate detection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = {
        "nc": nc,
        "scales": {"n": [0.33, 0.25, 1024], "s": [0.33, 0.50, 1024]},
        "backbone": [
            [-1, 1, "Conv", [64, 3, 2]],
            [-1, 1, "Conv", [128, 3, 2]],
            [-1, 3, "C2f", [128, True]],
            [-1, 1, "Conv", [256, 3, 2]],
            [-1, 6, "C2f", [256, True]],
            [-1, 1, "Conv", [512, 3, 2]],
            [-1, 6, "C2f", [512, True]],
            [-1, 1, "Conv", [1024, 3, 2]],
            [-1, 3, "C2f", [1024, True]],
            [-1, 1, "SPPF", [1024, 5]],
        ],
        "head": [
            [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
            [[-1, 6], 1, "Concat", [1]],
            [-1, 3, "C2f", [512]],
            [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
            [[-1, 4], 1, "Concat", [1]],
            [-1, 3, "C2f", [256]],
            [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
            [[-1, 2], 1, "Concat", [1]],
            [-1, 3, "C2f", [128]],
            [-1, 1, "Conv", [128, 3, 2]],
            [[-1, 15], 1, "Concat", [1]],
            [-1, 3, "C2f", [256]],
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 12], 1, "Concat", [1]],
            [-1, 3, "C2f", [512]],
            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 9], 1, "Concat", [1]],
            [-1, 3, "C2f", [1024]],
            [[18, 21, 24, 27], 1, "Detect", ["nc"]],
        ],
    }
    path.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _image_size(path: Path) -> tuple[int, int]:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    return w, h


def _yolo_line(row: dict[str, str], width: int, height: int, min_box_px: float = 1.0) -> str:
    x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
    x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(float(min_box_px), x2 - x1)
    bh = max(float(min_box_px), y2 - y1)
    x1 = max(0.0, cx - bw / 2.0)
    x2 = min(float(width), cx + bw / 2.0)
    y1 = max(0.0, cy - bh / 2.0)
    y2 = min(float(height), cy + bh / 2.0)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"0 {cx / width:.8f} {cy / height:.8f} {bw / width:.8f} {bh / height:.8f}"


def build_class_agnostic_yolo_dataset(
    annotations_csv: str | Path,
    out: str | Path,
    images_root: str | Path | None = None,
    val_fraction: float = 0.2,
    seed: int = 7,
    min_box_px: float = 1.0,
) -> Path:
    rows = _read_csv(Path(annotations_csv))
    out = Path(out)
    images_root_p = Path(images_root) if images_root else None
    grouped: dict[Path, list[dict[str, str]]] = {}
    for row in rows:
        img_path = Path(row["frame_path"])
        if not img_path.is_absolute() and images_root_p is not None:
            img_path = images_root_p / img_path
        grouped.setdefault(img_path, []).append(row)

    image_paths = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(image_paths)
    split_at = max(1, int(len(image_paths) * (1.0 - val_fraction))) if len(image_paths) > 1 else len(image_paths)
    split_map = {p: ("train" if i < split_at else "val") for i, p in enumerate(image_paths)}

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for img_path, img_rows in grouped.items():
        split = split_map[img_path]
        width, height = _image_size(img_path)
        dst_img = out / "images" / split / img_path.name
        shutil.copy2(img_path, dst_img)
        label_lines = [_yolo_line(r, width, height, min_box_px=min_box_px) for r in img_rows]
        (out / "labels" / split / f"{img_path.stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "object"},
                "nc": 1,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def _resolve_frame_path(row: dict[str, str], images_root: Path | None) -> Path:
    img_path = Path(row["frame_path"])
    if not img_path.is_absolute() and images_root is not None:
        img_path = images_root / img_path
    return img_path


def _clip_tile_origin(cx: float, cy: float, width: int, height: int, tile_size: int, rng: random.Random, jitter: float) -> tuple[int, int]:
    half = tile_size / 2.0
    ox = int(round(cx - half + rng.uniform(-jitter, jitter)))
    oy = int(round(cy - half + rng.uniform(-jitter, jitter)))
    ox = max(0, min(max(0, width - tile_size), ox))
    oy = max(0, min(max(0, height - tile_size), oy))
    return ox, oy


def _box_intersects_tile(box: tuple[float, float, float, float], ox: int, oy: int, tile_size: int, pad: float = 0.0) -> bool:
    x1, y1, x2, y2 = box
    tx1, ty1, tx2, ty2 = ox - pad, oy - pad, ox + tile_size + pad, oy + tile_size + pad
    return max(x1, tx1) < min(x2, tx2) and max(y1, ty1) < min(y2, ty2)


def _box_to_tile_row(box: tuple[float, float, float, float], ox: int, oy: int) -> dict[str, str]:
    x1, y1, x2, y2 = box
    return {
        "frame_path": "",
        "x1": str(x1 - ox),
        "y1": str(y1 - oy),
        "x2": str(x2 - ox),
        "y2": str(y2 - oy),
        "class": "object",
    }


def _photometric_variant(tile, variant_idx: int):
    import cv2
    import numpy as np

    idx = int(variant_idx)
    if idx % 4 == 0:
        alpha, beta = 0.65, -8.0
    elif idx % 4 == 1:
        alpha, beta = 0.8, 8.0
    elif idx % 4 == 2:
        alpha, beta = 1.25, -6.0
    else:
        alpha, beta = 0.55, 4.0
    out = cv2.convertScaleAbs(tile, alpha=alpha, beta=beta)
    if idx % 4 == 3:
        mean = out.reshape(-1, out.shape[-1]).mean(axis=0)
        out = np.clip(mean + 0.45 * (out.astype(np.float32) - mean), 0, 255).astype(np.uint8)
    return out


def _inject_low_contrast_target(
    tile,
    tile_box: tuple[float, float, float, float],
    variant_idx: int,
    rng: random.Random,
    min_render_px: float,
):
    import cv2
    import numpy as np

    out = tile.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = tile_box
    cx = int(round((x1 + x2) / 2.0 + rng.uniform(-0.5, 0.5)))
    cy = int(round((y1 + y2) / 2.0 + rng.uniform(-0.5, 0.5)))
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    bw = max(float(min_render_px), x2 - x1)
    bh = max(float(min_render_px), y2 - y1)
    radius = max(1, int(round(max(bw, bh) * rng.uniform(0.25, 0.45))))
    bg_radius = max(radius * 4, 12)
    rx1, rx2 = max(0, cx - bg_radius), min(w, cx + bg_radius + 1)
    ry1, ry2 = max(0, cy - bg_radius), min(h, cy + bg_radius + 1)
    local = out[ry1:ry2, rx1:rx2].astype(np.float32)
    if local.size == 0:
        base = np.array([128.0, 128.0, 128.0], dtype=np.float32)
    else:
        base = np.median(local.reshape(-1, 3), axis=0)

    deltas = [3.0, 5.0, 7.0, 10.0, 13.0, 16.0]
    delta = deltas[int(variant_idx) % len(deltas)]
    sign = -1.0 if (int(variant_idx) // len(deltas)) % 2 == 0 else 1.0
    color = np.clip(base + sign * delta, 0, 255).astype(np.uint8).tolist()
    axes = (max(1, radius), max(1, int(round(radius * rng.uniform(0.65, 1.15)))))
    angle = rng.uniform(0.0, 180.0)
    cv2.ellipse(out, (cx, cy), axes, angle, 0, 360, color, -1, lineType=cv2.LINE_AA)

    # Low-contrast targets should be visible enough to learn but still near the
    # background. A tiny blur and sensor-like noise prevent a single hard edge.
    patch_pad = max(2, radius + 2)
    px1, px2 = max(0, cx - patch_pad), min(w, cx + patch_pad + 1)
    py1, py2 = max(0, cy - patch_pad), min(h, cy + patch_pad + 1)
    patch = out[py1:py2, px1:px2]
    if patch.size:
        blurred = cv2.GaussianBlur(patch, (3, 3), 0)
        noise = rng.uniform(0.0, 1.5)
        if noise > 0:
            np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
            noisy = blurred.astype(np.float32) + np_rng.normal(0.0, noise, blurred.shape)
            blurred = np.clip(noisy, 0, 255).astype(np.uint8)
        out[py1:py2, px1:px2] = blurred
    return out


def build_tiled_class_agnostic_yolo_dataset(
    annotations_csv: str | Path,
    out: str | Path,
    images_root: str | Path | None = None,
    tile_size: int = 256,
    positives_per_box: int = 2,
    negatives_per_image: int = 2,
    val_fraction: float = 0.2,
    seed: int = 11,
    min_box_px: float = 8.0,
    negative_pad_px: float = 32.0,
    photometric_augmentations: int = 0,
    low_contrast_injections: int = 0,
    positive_repeat_patterns: tuple[str, ...] = (),
    positive_repeat_factor: int = 1,
) -> Path:
    """Build a tiled YOLO dataset where tiny full-frame objects occupy more pixels.

    Positive tiles are centered near each object. Negative tiles are empty-label crops
    from the same source frames that avoid all known target boxes.
    """
    import cv2

    rows = _read_csv(Path(annotations_csv))
    out = Path(out)
    images_root_p = Path(images_root) if images_root else None
    rng = random.Random(seed)
    grouped: dict[Path, list[tuple[float, float, float, float]]] = {}
    for row in rows:
        img_path = _resolve_frame_path(row, images_root_p)
        box = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
        grouped.setdefault(img_path, []).append(box)

    image_paths = sorted(grouped)
    rng.shuffle(image_paths)
    split_at = max(1, int(len(image_paths) * (1.0 - val_fraction))) if len(image_paths) > 1 else len(image_paths)
    split_map = {p: ("train" if i < split_at else "val") for i, p in enumerate(image_paths)}

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        height, width = img.shape[:2]
        if width < tile_size or height < tile_size:
            raise ValueError(f"Image {img_path} is smaller than tile_size={tile_size}")
        split = split_map[img_path]
        boxes = grouped[img_path]
        stem = img_path.stem
        tile_idx = 0
        repeat_key = img_path.name.lower()
        repeat_factor = max(1, int(positive_repeat_factor)) if any(p.lower() in repeat_key for p in positive_repeat_patterns) else 1

        for box in boxes:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            for _ in range(max(1, int(positives_per_box)) * repeat_factor):
                ox, oy = _clip_tile_origin(cx, cy, width, height, tile_size, rng, jitter=tile_size * 0.18)
                tile = img[oy : oy + tile_size, ox : ox + tile_size]
                tile_row = _box_to_tile_row(box, ox, oy)
                tile_box = tuple(float(tile_row[k]) for k in ("x1", "y1", "x2", "y2"))
                label = _yolo_line(tile_row, tile_size, tile_size, min_box_px=min_box_px)
                variants = [("orig", tile)] + [(f"photo{j}", _photometric_variant(tile, j)) for j in range(max(0, int(photometric_augmentations)))]
                variants.extend(
                    (
                        f"lowcontrast{j}",
                        _inject_low_contrast_target(tile, tile_box, j, rng, min_render_px=max(2.0, min_box_px * 0.5)),
                    )
                    for j in range(max(0, int(low_contrast_injections)))
                )
                for variant_name, variant_tile in variants:
                    tile_name = f"{stem}_pos_{tile_idx:04d}_{variant_name}.jpg"
                    cv2.imwrite(str(out / "images" / split / tile_name), variant_tile)
                    (out / "labels" / split / f"{Path(tile_name).stem}.txt").write_text(label + "\n", encoding="utf-8")
                    manifest.append(
                        {
                            "tile_path": str(out / "images" / split / tile_name),
                            "label_path": str(out / "labels" / split / f"{Path(tile_name).stem}.txt"),
                            "source_frame": str(img_path),
                            "split": split,
                            "kind": "positive",
                            "variant": variant_name,
                            "positive_repeat_factor": repeat_factor,
                            "tile_origin_xy": [ox, oy],
                            "tile_size": tile_size,
                            "source_bbox_xyxy": list(box),
                        }
                    )
                tile_idx += 1

        made_neg = 0
        attempts = 0
        while made_neg < max(0, int(negatives_per_image)) and attempts < max(50, int(negatives_per_image) * 80):
            attempts += 1
            ox = rng.randint(0, width - tile_size)
            oy = rng.randint(0, height - tile_size)
            if any(_box_intersects_tile(box, ox, oy, tile_size, pad=negative_pad_px) for box in boxes):
                continue
            tile = img[oy : oy + tile_size, ox : ox + tile_size]
            tile_name = f"{stem}_neg_{made_neg:04d}.jpg"
            cv2.imwrite(str(out / "images" / split / tile_name), tile)
            (out / "labels" / split / f"{Path(tile_name).stem}.txt").write_text("", encoding="utf-8")
            manifest.append(
                {
                    "tile_path": str(out / "images" / split / tile_name),
                    "label_path": str(out / "labels" / split / f"{Path(tile_name).stem}.txt"),
                    "source_frame": str(img_path),
                    "split": split,
                    "kind": "negative",
                    "tile_origin_xy": [ox, oy],
                    "tile_size": tile_size,
                    "source_bbox_xyxy": None,
                }
            )
            made_neg += 1

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "object"},
                "nc": 1,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (out / "tile_manifest.jsonl").write_text("\n".join(json.dumps(r) for r in manifest) + "\n", encoding="utf-8")
    return data_yaml


def train_yolo_p2(
    data_yaml: str | Path,
    out: str | Path,
    model_yaml: str | Path | None = None,
    pretrained: str | Path | None = None,
    epochs: int = 50,
    imgsz: int = 1280,
    batch: int = 8,
    device: str | None = None,
) -> Path:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError("ultralytics is required for YOLO-P2 training. Install it with: python -m pip install ultralytics") from exc

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if model_yaml is None:
        model_yaml = write_yolov8_p2_model_yaml(out / "yolov8_p2_candidate.yaml")
    model = YOLO(str(model_yaml))
    if pretrained:
        model = model.load(str(pretrained))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(out),
        name="yolo_p2_candidate",
        device=device,
        workers=0,
        mosaic=0.0,
        close_mosaic=0,
        erasing=0.0,
        scale=0.0,
        translate=0.0,
        fliplr=0.0,
    )
    return out / "yolo_p2_candidate"
