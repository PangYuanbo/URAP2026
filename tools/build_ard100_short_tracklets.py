#!/usr/bin/env python3
"""Build NPS-length first-frame-prompt tracklets from TVD-format ARD100."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from build_ard100_samurai_dataset import extract_video_frames, load_xml_annotations, select_track, utc_now, write_json_atomic


@dataclass(frozen=True)
class Tracklet:
    clip_id: int
    frame_ids: tuple[int, ...]
    boxes: tuple[np.ndarray | None, ...]

    @property
    def visible_frames(self) -> int:
        return sum(box is not None for box in self.boxes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--raw-video-root", type=Path, required=True)
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=166)
    parser.add_argument("--min-visible-frames", type=int, default=8)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--image-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def split_tracklets(
    clip_id: int,
    selected_by_frame: dict[int, np.ndarray | None],
    *,
    max_gap: int,
    max_frames: int,
    min_visible_frames: int,
    min_visibility: float,
) -> list[Tracklet]:
    visible_ids = sorted(frame_id for frame_id, box in selected_by_frame.items() if box is not None)
    if not visible_ids:
        return []
    groups: list[list[int]] = [[visible_ids[0]]]
    for frame_id in visible_ids[1:]:
        if frame_id - groups[-1][-1] <= max_gap + 1:
            groups[-1].append(frame_id)
        else:
            groups.append([frame_id])
    result: list[Tracklet] = []
    for group in groups:
        cursor = group[0]
        while cursor <= group[-1]:
            end = min(group[-1], cursor + max_frames - 1)
            frame_ids = tuple(range(cursor, end + 1))
            boxes = tuple(selected_by_frame.get(frame_id) for frame_id in frame_ids)
            visible = sum(box is not None for box in boxes)
            if visible >= min_visible_frames and visible / len(frame_ids) >= min_visibility:
                result.append(Tracklet(clip_id, frame_ids, boxes))
            cursor = end + 1
            while cursor <= group[-1] and selected_by_frame.get(cursor) is None:
                cursor += 1
    return result


def place_image(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return "existing"
    if mode == "symlink":
        destination.symlink_to(source)
        return "symlink"
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


def is_complete(root: Path, name: str, expected: int) -> bool:
    sequence = root / "lasot" / "uav" / name
    gt = sequence / "groundtruth.txt"
    if not gt.is_file():
        return False
    rows = [line for line in gt.read_text(encoding="ascii").splitlines() if line.strip()]
    images = list((sequence / "img").glob("*.jpg"))
    masks = list((root / "vos" / "Annotations" / name).glob("*.png"))
    return len(rows) == len(images) == len(masks) == expected


def load_tracklets(args: argparse.Namespace) -> tuple[list[int], list[Tracklet]]:
    index_path = args.source_root / "Videos" / args.split / "video_length_dict.pkl"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    with index_path.open("rb") as handle:
        clip_ids = sorted(int(value) for value in pickle.load(handle))
    if args.max_videos is not None:
        clip_ids = clip_ids[: args.max_videos]
    annotations = load_xml_annotations(args.annotations_zip, set(clip_ids))
    tracklets: list[Tracklet] = []
    for clip_id in clip_ids:
        frames = annotations.get(clip_id, {})
        if not frames:
            raise ValueError(f"No XML annotations for clip {clip_id}")
        frame_ids = sorted(frames)
        selected = select_track([frames[frame_id] for frame_id in frame_ids], 1920, 1080)
        tracklets.extend(
            split_tracklets(
                clip_id,
                dict(zip(frame_ids, selected)),
                max_gap=args.max_gap,
                max_frames=args.max_frames,
                min_visible_frames=args.min_visible_frames,
                min_visibility=args.min_visibility,
            )
        )
    return clip_ids, tracklets


def write_tracklet(
    root: Path,
    source_images: Path,
    raw_video_root: Path,
    split: str,
    name: str,
    tracklet: Tracklet,
    mode: str,
) -> list[dict[str, object]]:
    sequence = root / "lasot" / "uav" / name
    gt_lines: list[str] = []
    occlusion: list[str] = []
    rows: list[dict[str, object]] = []
    missing: dict[int, Path] = {}
    for local_index, frame_id in enumerate(tracklet.frame_ids, 1):
        source = source_images / f"Clip_{tracklet.clip_id}_{frame_id:05d}.jpg"
        output = sequence / "img" / f"{local_index:08d}.jpg"
        if not source.is_file() and not output.is_file():
            missing[frame_id] = output
    raw_folder = "test_videos" if split == "test" else "train_videos"
    if missing:
        extract_video_frames(raw_video_root / raw_folder / f"phantom{tracklet.clip_id:02d}.mp4", missing)

    for local_index, (frame_id, box) in enumerate(zip(tracklet.frame_ids, tracklet.boxes), 1):
        source = source_images / f"Clip_{tracklet.clip_id}_{frame_id:05d}.jpg"
        output_name = f"{local_index:08d}.jpg"
        image_path = sequence / "img" / output_name
        storage = place_image(source.resolve(), image_path, mode) if source.is_file() else "decoded"
        # VOS JPEGImages is materialized as local hardlinks after download. Keeping
        # a second remote symlink per frame exhausts the Modal volume inode quota.
        if box is None:
            gt_lines.append("0,0,0,0")
            occlusion.append("1")
        else:
            x, y, width, height = box
            gt_lines.append(f"{x:.3f},{y:.3f},{width:.3f},{height:.3f}")
            occlusion.append("0")
        mask_path = root / "vos" / "Annotations" / name / f"{local_index:08d}.png"
        if not mask_path.exists():
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask = Image.new("L", (1920, 1080), 0)
            if box is not None:
                x, y, width, height = box
                bounds = (max(0, math.floor(x)), max(0, math.floor(y)), min(1919, math.ceil(x + width) - 1), min(1079, math.ceil(y + height) - 1))
                ImageDraw.Draw(mask).rectangle(bounds, fill=1)
            mask.save(mask_path, compress_level=1)
        rows.append({"derived_sequence": name, "sequence_index": local_index, "source_clip": tracklet.clip_id, "source_frame": frame_id, "source_path": str(source if source.is_file() else image_path), "visible": int(box is not None), "image_storage": storage})
    sequence.mkdir(parents=True, exist_ok=True)
    (sequence / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="ascii")
    (sequence / "full_occlusion.txt").write_text(",".join(occlusion) + "\n", encoding="ascii")
    (sequence / "out_of_view.txt").write_text(",".join(occlusion) + "\n", encoding="ascii")
    return rows


def main() -> int:
    args = parse_args()
    if args.max_gap < 0 or args.max_frames < 1 or args.min_visible_frames < 1:
        raise ValueError("Invalid tracklet bounds")
    if not 0.0 < args.min_visibility <= 1.0:
        raise ValueError("min-visibility must be in (0, 1]")
    source_images = args.source_root / "AllFrames" / args.split
    if not source_images.is_dir():
        raise FileNotFoundError(source_images)
    clip_ids, tracklets = load_tracklets(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.json"
    progress = {"status": "running", "split": args.split, "done_sequences": 0, "total_sequences": len(tracklets), "done_frames": 0, "last_update": utc_now()}
    write_json_atomic(progress_path, progress)
    names: list[str] = []
    source_rows: list[dict[str, object]] = []
    total_frames = visible_frames = 0
    per_clip: dict[int, int] = {}
    for sequence_index, tracklet in enumerate(tracklets, 1):
        local_index = per_clip.get(tracklet.clip_id, 0) + 1
        per_clip[tracklet.clip_id] = local_index
        name = f"ARD_Clip_{tracklet.clip_id:03d}_T{local_index:04d}"
        names.append(name)
        if not (args.resume and is_complete(args.output_root, name, len(tracklet.frame_ids))):
            source_rows.extend(write_tracklet(args.output_root, source_images, args.raw_video_root, args.split, name, tracklet, args.image_mode))
        total_frames += len(tracklet.frame_ids)
        visible_frames += tracklet.visible_frames
        progress.update(done_sequences=sequence_index, done_frames=total_frames, last_completed_sequence=name, last_update=utc_now())
        write_json_atomic(progress_path, progress)
        if sequence_index == 1 or sequence_index % 25 == 0 or sequence_index == len(tracklets):
            print(json.dumps({"kind": "tracklet_build_progress", **progress}), flush=True)
    (args.output_root / f"{args.split}_set.txt").write_text("\n".join(names) + "\n", encoding="ascii")
    if source_rows:
        with (args.output_root / "source_map.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
            writer.writeheader()
            writer.writerows(source_rows)
    lengths = [len(tracklet.frame_ids) for tracklet in tracklets]
    manifest = {
        "format_version": 1,
        "source": "TVD-format ARD100",
        "protocol": "NPS-style short first-frame-prompt tracklets",
        "split": args.split,
        "source_video_count": len(clip_ids),
        "sequence_count": len(tracklets),
        "frame_count": total_frames,
        "visible_frame_count": visible_frames,
        "occluded_frame_count": total_frames - visible_frames,
        "max_gap": args.max_gap,
        "max_frames": args.max_frames,
        "min_visible_frames": args.min_visible_frames,
        "min_visibility": args.min_visibility,
        "mean_sequence_frames": float(np.mean(lengths)) if lengths else 0.0,
        "median_sequence_frames": float(np.median(lengths)) if lengths else 0.0,
        "min_sequence_frames": min(lengths) if lengths else 0,
        "max_sequence_frames": max(lengths) if lengths else 0,
        "first_frame_prompt": "groundtruth row 1, xywh; no later correction",
        "vos_masks": "weak rectangular masks generated from continuity-selected ARD100 boxes",
        "image_mode": args.image_mode,
    }
    write_json_atomic(args.output_root / "manifest.json", manifest)
    progress.update(status="completed", last_update=utc_now())
    write_json_atomic(progress_path, progress)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
