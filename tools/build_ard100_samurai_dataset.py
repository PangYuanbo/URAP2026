#!/usr/bin/env python3
"""Convert TVD-format ARD100 videos into SAMURAI/LaSOT and weak-mask VOS layouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

FRAME_RE = re.compile(r"^Clip_(?P<clip>\d+)_(?P<frame>\d+)\.jpg$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--raw-video-root", type=Path, required=True)
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-frames-per-video", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def place_file(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


def load_xml_annotations(archive_path: Path, clip_ids: set[int]) -> dict[int, dict[int, list[np.ndarray]]]:
    annotations: dict[int, dict[int, list[np.ndarray]]] = defaultdict(dict)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            match = re.search(r"phantom(?P<clip>\d+)_?(?P<frame>\d+)\.xml$", member, re.IGNORECASE)
            if not match:
                continue
            clip_id, frame_id = int(match.group("clip")), int(match.group("frame"))
            if clip_id not in clip_ids:
                continue
            root = ET.fromstring(archive.read(member))
            boxes = []
            for obj in root.findall("object"):
                bounds = obj.find("bndbox")
                if bounds is None:
                    continue
                x1 = float(bounds.findtext("xmin")); y1 = float(bounds.findtext("ymin"))
                x2 = float(bounds.findtext("xmax")); y2 = float(bounds.findtext("ymax"))
                boxes.append(np.asarray((x1, y1, x2 - x1, y2 - y1), dtype=np.float32))
            annotations[clip_id][frame_id] = boxes
    return annotations


def extract_video_frames(video_path: Path, requested: dict[int, Path]) -> None:
    if not requested:
        return
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    remaining = dict(requested)
    frame_id = 0
    try:
        while remaining:
            ok, frame = capture.read()
            if not ok:
                break
            frame_id += 1
            destination = remaining.pop(frame_id, None)
            if destination is not None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Failed to write decoded frame: {destination}")
    finally:
        capture.release()
    if remaining:
        raise RuntimeError(f"Video ended before requested frames {sorted(remaining)[:10]} in {video_path}")


def box_distance(left: np.ndarray, right: np.ndarray, image_width: int, image_height: int) -> float:
    left_center = left[:2] + left[2:] / 2
    right_center = right[:2] + right[2:] / 2
    center = ((left_center[0] - right_center[0]) / image_width) ** 2 + ((left_center[1] - right_center[1]) / image_height) ** 2
    size = float(np.square(np.log(np.maximum(left[2:], 1.0) / np.maximum(right[2:], 1.0))).sum())
    return float(center + 0.0025 * size)


def select_track(candidates: list[list[np.ndarray]], image_width: int, image_height: int) -> list[np.ndarray | None]:
    selected: list[np.ndarray | None] = []
    previous = None
    for index, frame_candidates in enumerate(candidates):
        if not frame_candidates:
            selected.append(None)
            continue
        if len(frame_candidates) == 1:
            choice = frame_candidates[0]
        else:
            next_single = next((future[0] for future in candidates[index + 1 : index + 31] if len(future) == 1), None)
            references = [box for box in (previous, next_single) if box is not None]
            if references:
                choice = min(frame_candidates, key=lambda box: sum(box_distance(box, reference, image_width, image_height) for reference in references))
            else:
                choice = max(frame_candidates, key=lambda box: float(box[2] * box[3]))
        selected.append(choice)
        previous = choice
    return selected


def complete_sequence(output_root: Path, split: str, sequence_name: str, expected_frames: int) -> bool:
    sequence_root = output_root / "lasot" / "uav" / sequence_name
    gt_path = sequence_root / "groundtruth.txt"
    if not gt_path.is_file():
        return False
    gt_count = sum(1 for line in gt_path.read_text(encoding="ascii").splitlines() if line.strip())
    image_count = len(list((sequence_root / "img").glob("*.jpg")))
    mask_count = len(list((output_root / "vos" / "Annotations" / sequence_name).glob("*.png")))
    return gt_count == image_count == mask_count == expected_frames


def main() -> int:
    args = parse_args()
    source_images = args.source_root / "AllFrames" / args.split
    video_index_path = args.source_root / "Videos" / args.split / "video_length_dict.pkl"
    if not source_images.is_dir() or not video_index_path.is_file():
        raise FileNotFoundError(f"Missing TVD-format ARD100 split under {args.source_root}")
    import pickle
    with video_index_path.open("rb") as handle:
        video_index = pickle.load(handle)
    clip_ids = sorted(int(value) for value in video_index)
    if args.max_videos is not None:
        clip_ids = clip_ids[: args.max_videos]
    annotations = load_xml_annotations(args.annotations_zip, set(clip_ids))
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.json"
    progress = {"status": "running", "split": args.split, "done_sequences": 0, "total_sequences": len(clip_ids), "done_frames": 0, "last_update": utc_now()}
    write_json_atomic(progress_path, progress)
    sequence_names = []
    source_rows = []
    total_frames = visible_frames = multi_candidate_frames = 0
    for sequence_index, clip_id in enumerate(clip_ids, 1):
        frame_annotations = annotations.get(clip_id, {})
        if not frame_annotations:
            raise ValueError(f"No XML annotations for clip {clip_id}")
        visible_ids = [frame_id for frame_id, boxes in frame_annotations.items() if boxes]
        if not visible_ids:
            raise ValueError(f"No visible target in clip {clip_id}")
        first_frame, last_frame = min(visible_ids), max(frame_annotations)
        frame_ids = list(range(first_frame, last_frame + 1))
        if args.max_frames_per_video is not None:
            frame_ids = frame_ids[: args.max_frames_per_video]
        sequence_name = f"ARD_Clip_{clip_id:03d}"
        sequence_names.append(sequence_name)
        if args.resume and complete_sequence(args.output_root, args.split, sequence_name, len(frame_ids)):
            sequence_root = args.output_root / "lasot" / "uav" / sequence_name
            gt_lines = [line for line in (sequence_root / "groundtruth.txt").read_text().splitlines() if line.strip()]
            visible_frames += sum(line != "0,0,0,0" for line in gt_lines)
            total_frames += len(frame_ids)
            progress.update(done_sequences=sequence_index, done_frames=total_frames, last_completed_sequence=sequence_name, last_update=utc_now())
            write_json_atomic(progress_path, progress)
            continue
        image_width, image_height = 1920, 1080
        frame_candidates = [frame_annotations.get(frame_id, []) for frame_id in frame_ids]
        multi_candidate_frames += sum(len(boxes) > 1 for boxes in frame_candidates)
        selected = select_track(frame_candidates, image_width, image_height)
        sequence_root = args.output_root / "lasot" / "uav" / sequence_name
        image_root = sequence_root / "img"
        vos_image_root = args.output_root / "vos" / "JPEGImages" / sequence_name
        vos_mask_root = args.output_root / "vos" / "Annotations" / sequence_name
        gt_lines, occlusion, out_of_view = [], [], []
        missing_frames: dict[int, Path] = {}
        for local_index, source_frame in enumerate(frame_ids, 1):
            existing_source = source_images / f"Clip_{clip_id}_{source_frame:05d}.jpg"
            output_path = image_root / f"{local_index:08d}.jpg"
            if not existing_source.is_file() and not output_path.is_file():
                missing_frames[source_frame] = output_path
        raw_folder = "test_videos" if args.split == "test" else "train_videos"
        extract_video_frames(args.raw_video_root / raw_folder / f"phantom{clip_id:02d}.mp4", missing_frames)
        for local_index, (source_frame, box) in enumerate(zip(frame_ids, selected), 1):
            output_name = f"{local_index:08d}.jpg"
            existing_source = source_images / f"Clip_{clip_id}_{source_frame:05d}.jpg"
            output_path = image_root / output_name
            if existing_source.is_file():
                storage = place_file(existing_source, output_path, args.image_mode)
            else:
                storage = "decoded"
            place_file(output_path, vos_image_root / output_name, "hardlink")
            if box is None:
                gt_lines.append("0,0,0,0"); occlusion.append("1"); out_of_view.append("1")
            else:
                x, y, width, height = box
                gt_lines.append(f"{x:.3f},{y:.3f},{width:.3f},{height:.3f}")
                occlusion.append("0"); out_of_view.append("0"); visible_frames += 1
            mask_path = vos_mask_root / f"{local_index:08d}.png"
            if not mask_path.exists():
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                mask = Image.new("L", (image_width, image_height), 0)
                if box is not None:
                    x, y, width, height = box
                    ImageDraw.Draw(mask).rectangle((max(0, math.floor(x)), max(0, math.floor(y)), min(image_width - 1, math.ceil(x + width) - 1), min(image_height - 1, math.ceil(y + height) - 1)), fill=1)
                mask.save(mask_path, optimize=True)
            source_rows.append({"derived_sequence": sequence_name, "sequence_index": local_index, "source_clip": clip_id, "source_frame": source_frame, "source_path": str(existing_source if existing_source.is_file() else output_path), "candidate_count": len(frame_candidates[local_index - 1]), "visible": int(box is not None), "image_storage": storage})
            total_frames += 1
        sequence_root.mkdir(parents=True, exist_ok=True)
        (sequence_root / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="ascii")
        (sequence_root / "full_occlusion.txt").write_text(",".join(occlusion) + "\n", encoding="ascii")
        (sequence_root / "out_of_view.txt").write_text(",".join(out_of_view) + "\n", encoding="ascii")
        progress.update(done_sequences=sequence_index, done_frames=total_frames, last_completed_sequence=sequence_name, last_update=utc_now())
        write_json_atomic(progress_path, progress)
    (args.output_root / f"{args.split}_set.txt").write_text("\n".join(sequence_names) + "\n", encoding="ascii")
    if source_rows:
        with (args.output_root / "source_map.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(source_rows[0])); writer.writeheader(); writer.writerows(source_rows)
    manifest = {"format_version": 1, "source": "TVD-format ARD100", "source_root": str(args.source_root), "raw_video_root": str(args.raw_video_root), "annotations_zip": str(args.annotations_zip), "timeline": "all XML frames from first visible through final annotated frame; missing positive-frame cache entries decoded from MP4", "split": args.split, "sequence_count": len(sequence_names), "frame_count": total_frames, "visible_frame_count": visible_frames, "occluded_frame_count": total_frames - visible_frames, "multi_candidate_frames": multi_candidate_frames, "image_mode": args.image_mode, "vos_masks": "weak rectangular masks generated from continuity-selected YOLO boxes", "first_frame_prompt": "groundtruth row 1, xywh"}
    write_json_atomic(args.output_root / "manifest.json", manifest)
    progress.update(status="completed", done_sequences=len(sequence_names), done_frames=total_frames, last_update=utc_now())
    write_json_atomic(progress_path, progress)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
