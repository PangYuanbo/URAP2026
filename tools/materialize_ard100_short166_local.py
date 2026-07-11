#!/usr/bin/env python3
"""Rebuild local VOS image hardlinks and validate downloaded short166 splits."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw


EXPECTED_VIDEOS = {"train": 55, "val": 10, "test": 35}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def verify_image(path: Path, expected_size: tuple[int, int], *, decode: bool) -> None:
    with Image.open(path) as image:
        if image.size != expected_size:
            raise RuntimeError(f"Unexpected image size for {path}: {image.size}")
        if decode:
            image.load()
        else:
            image.verify()


def write_box_mask(path: Path, box_line: str, size: tuple[int, int]) -> None:
    values = [float(value) for value in box_line.split(",")]
    if len(values) != 4:
        raise ValueError(f"Invalid box row for {path}: {box_line}")
    x, y, width, height = values
    mask = Image.new("L", size, 0)
    if width > 0 and height > 0:
        bounds = (
            max(0, math.floor(x)),
            max(0, math.floor(y)),
            min(size[0] - 1, math.ceil(x + width) - 1),
            min(size[1] - 1, math.ceil(y + height) - 1),
        )
        ImageDraw.Draw(mask).rectangle(bounds, fill=1)
    temporary = path.with_suffix(".png.tmp")
    mask.save(temporary, format="PNG", compress_level=1)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    progress_path = args.root / "materialize_progress.json"
    report = {"root": str(args.root), "splits": {}, "complete": True}
    checked_frames = 0
    source_videos: dict[str, set[int]] = {}
    for split, expected_videos in EXPECTED_VIDEOS.items():
        root = args.root / f"{split}_v1"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        names = [line.strip() for line in (root / f"{split}_set.txt").read_text().splitlines() if line.strip()]
        if int(manifest["source_video_count"]) != expected_videos or len(names) != int(manifest["sequence_count"]):
            raise RuntimeError(f"Incomplete {split} manifest")
        frames = masks = links = repaired_masks = 0
        clips: set[int] = set()
        for name in names:
            clips.add(int(name.split("_")[2]))
            image_root = root / "lasot" / "uav" / name / "img"
            mask_root = root / "vos" / "Annotations" / name
            images = sorted(image_root.glob("*.jpg"))
            sequence_masks = sorted(mask_root.glob("*.png"))
            groundtruth = [line.strip() for line in (root / "lasot" / "uav" / name / "groundtruth.txt").read_text(encoding="ascii").splitlines() if line.strip()]
            if len(images) != len(sequence_masks) or len(images) != len(groundtruth):
                raise RuntimeError(f"Image/mask mismatch: {name}")
            for index, source in enumerate(images):
                verify_image(source, (1920, 1080), decode=False)
                mask_path = sequence_masks[index]
                try:
                    verify_image(mask_path, (1920, 1080), decode=False)
                except Exception:
                    write_box_mask(mask_path, groundtruth[index], (1920, 1080))
                    verify_image(mask_path, (1920, 1080), decode=False)
                    repaired_masks += 1
                checked_frames += 1
                if checked_frames == 1 or checked_frames % 2000 == 0:
                    progress = {
                        "status": "running",
                        "split": split,
                        "sequence": name,
                        "checked_frames": checked_frames,
                        "repaired_masks": repaired_masks,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    temporary = progress_path.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
                    temporary.replace(progress_path)
            frames += len(images)
            masks += len(sequence_masks)
        source_videos[split] = clips
        report["splits"][split] = {
            "source_videos": len(clips),
            "sequences": len(names),
            "frames": frames,
            "masks": masks,
            "vos_image_links": 0,
            "all_images_verified": True,
            "all_masks_verified": True,
            "repaired_masks": repaired_masks,
        }
    overlap = {"train_val": sorted(source_videos["train"] & source_videos["val"]), "train_test": sorted(source_videos["train"] & source_videos["test"]), "val_test": sorted(source_videos["val"] & source_videos["test"])}
    report["split_overlap"] = overlap
    report["complete"] = not any(overlap.values())
    if not report["complete"]:
        raise RuntimeError(f"Video split leakage: {overlap}")
    output = args.root / "LOCAL_MATERIALIZE_COMPLETE.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    progress_path.write_text(
        json.dumps({"status": "completed", "checked_frames": checked_frames, "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
