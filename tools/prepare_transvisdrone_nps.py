import argparse
import glob
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm


@dataclass(frozen=True)
class SplitSpec:
    name: str
    clip_ids: List[int]


SPLITS = [
    SplitSpec("train", list(range(1, 37))),  # 1..36
    SplitSpec("val", list(range(37, 41))),   # 37..40
    SplitSpec("test", list(range(41, 51))),  # 41..50
]


def clip_to_split(clip_id: int) -> str:
    for s in SPLITS:
        if clip_id in s.clip_ids:
            return s.name
    raise ValueError(f"Unexpected clip id {clip_id} (expected 1..50)")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def parse_frame_index_from_name(path: str) -> int:
    # Clip_14_00001.png -> 1
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    return int(parts[-1])


def safe_ints_from_csv_line(line: str) -> List[int]:
    # expected: "11,2,1026,395,1040,409,1127,542,1139,553"
    # strip spaces; allow stray commas.
    items = [x.strip() for x in line.strip().split(",") if x.strip() != ""]
    return [int(x) for x in items]


def bboxes_to_yolo_lines(
    bboxes_xyxy: List[Tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
) -> List[str]:
    lines: List[str] = []
    for (x1, y1, x2, y2) in bboxes_xyxy:
        # Clamp to image bounds (defensive).
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0
        lines.append(
            "0 "
            + f"{cx / img_w:.6f} {cy / img_h:.6f} {bw / img_w:.6f} {bh / img_h:.6f}"
        )
    return lines


def parse_clip_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    clip_ids: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Bad clip range: {part}")
            clip_ids.update(range(start, end + 1))
        else:
            clip_ids.add(int(part))
    for clip_id in sorted(clip_ids):
        clip_to_split(clip_id)
    return clip_ids


def extract_frames(
    video_path: str,
    out_dir: str,
    clip_id: int,
    image_ext: str,
    png_compression: int,
    jpg_quality: int,
) -> Tuple[int, Tuple[int, int]]:
    """Extract all frames to out_dir as Clip_{id}_{frame:05}.{image_ext}.

    Frame numbering starts at 1 to match TransVisDrone's NPS label mapping
    (labels are indexed from 0, images from 1).
    """
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ext = image_ext.lower().lstrip(".")
    if ext not in {"png", "jpg", "jpeg"}:
        raise ValueError(f"Unsupported image extension: {image_ext}")
    pattern = os.path.join(out_dir, f"Clip_{clip_id}_*.{ext}")
    existing = glob.glob(pattern)
    if len(existing) == n_frames:
        cap.release()
        return n_frames, (img_w, img_h)

    start_idx = 1
    if existing:
        max_existing = max(parse_frame_index_from_name(p) for p in existing)
        # Resume from next frame index.
        start_idx = max_existing + 1
        # cap frame position is 0-indexed; our saved index is 1-indexed.
        cap.set(cv2.CAP_PROP_POS_FRAMES, max_existing)

    pbar = tqdm(
        total=n_frames,
        initial=start_idx - 1,
        desc=f"Extract clip {clip_id} -> {os.path.basename(out_dir)}",
        unit="frame",
    )
    idx = start_idx
    while idx <= n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        out_path = os.path.join(out_dir, f"Clip_{clip_id}_{idx:05d}.{ext}")
        if not os.path.exists(out_path):
            write_params = (
                [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]
                if ext == "png"
                else [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]
            )
            ok_write = cv2.imwrite(
                out_path,
                frame,
                write_params,
            )
            if not ok_write:
                cap.release()
                raise RuntimeError(f"Failed to write frame: {out_path}")
        idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # Verify completeness.
    existing2 = glob.glob(pattern)
    if len(existing2) != n_frames:
        raise RuntimeError(
            f"Frame extraction incomplete for clip {clip_id}: "
            f"expected {n_frames}, found {len(existing2)} in {out_dir}"
        )
    return n_frames, (img_w, img_h)


def write_yolo_labels_for_clip(
    anno_path: str,
    labels_dir: str,
    clip_id: int,
    img_w: int,
    img_h: int,
) -> int:
    """Write YOLO labels in TransVisDrone expected naming:

    Label files are named Clip_{id}_{frame:05}.txt where frame is 0-indexed
    and corresponds to image Clip_{id}_{frame+1:05}.png.
    """
    ensure_dir(labels_dir)
    if not os.path.exists(anno_path):
        raise RuntimeError(f"Annotation file missing: {anno_path}")

    written = 0
    with open(anno_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = safe_ints_from_csv_line(line)
            if len(parts) < 2:
                continue
            frame_no = int(parts[0])  # 0-indexed
            num_obj = int(parts[1])
            coords = parts[2:]
            if num_obj <= 0:
                # Create an empty file to explicitly mark as no-target frame.
                out_txt = os.path.join(labels_dir, f"Clip_{clip_id}_{frame_no:05d}.txt")
                if not os.path.exists(out_txt):
                    open(out_txt, "w", encoding="utf-8").close()
                    written += 1
                continue
            if len(coords) != num_obj * 4:
                raise RuntimeError(
                    f"Bad annotation row in {anno_path}: {raw!r} "
                    f"(num_obj={num_obj}, coords={len(coords)})"
                )
            bboxes: List[Tuple[int, int, int, int]] = []
            for i in range(num_obj):
                x1, y1, x2, y2 = coords[i * 4 : i * 4 + 4]
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))

            yolo_lines = bboxes_to_yolo_lines(bboxes, img_w=img_w, img_h=img_h)
            out_txt = os.path.join(labels_dir, f"Clip_{clip_id}_{frame_no:05d}.txt")
            with open(out_txt, "w", encoding="utf-8") as out:
                out.write("\n".join(yolo_lines))
                if yolo_lines:
                    out.write("\n")
            written += 1

    return written


def build_video_length_dict(split_dir: str, clip_to_len: Dict[int, int]) -> None:
    ensure_dir(split_dir)
    out_path = os.path.join(split_dir, "video_length_dict.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(dict(clip_to_len), f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--videos-dir",
        required=True,
        help="Directory containing Clip_*.mov videos (NPS).",
    )
    ap.add_argument(
        "--annos-dir",
        required=True,
        help="Directory containing NPS annotation txt files (Clip_###.txt).",
    )
    ap.add_argument(
        "--out-root",
        required=True,
        help="Output root directory (will create AllFrames/, NPSvisdroneStyle/, Videos/).",
    )
    ap.add_argument(
        "--png-compression",
        type=int,
        default=3,
        help="OpenCV PNG compression level [0..9]. Higher is smaller but slower.",
    )
    ap.add_argument(
        "--image-ext",
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="Frame image extension. Use jpg/jpeg for lower disk use on local NPS subsets.",
    )
    ap.add_argument(
        "--jpg-quality",
        type=int,
        default=90,
        help="OpenCV JPEG quality [0..100] when --image-ext is jpg/jpeg.",
    )
    ap.add_argument(
        "--only-split",
        choices=["train", "val", "test"],
        default=None,
        help="If set, only process the given split.",
    )
    ap.add_argument(
        "--only-clips",
        default=None,
        help="Optional clip ids/ranges such as '37-40,45'. Applied after --only-split.",
    )
    args = ap.parse_args()

    videos_dir = os.path.abspath(args.videos_dir)
    annos_dir = os.path.abspath(args.annos_dir)
    out_root = os.path.abspath(args.out_root)

    allframes_root = os.path.join(out_root, "AllFrames")
    labels_root = os.path.join(out_root, "NPSvisdroneStyle")
    videos_root = os.path.join(out_root, "Videos")

    split_to_lens: Dict[str, Dict[int, int]] = {"train": {}, "val": {}, "test": {}}

    only_clips = parse_clip_ids(args.only_clips)
    splits = [s for s in SPLITS if args.only_split in (None, s.name)]
    for split in splits:
        clip_ids = [clip_id for clip_id in split.clip_ids if only_clips is None or clip_id in only_clips]
        if not clip_ids:
            continue
        frames_dir = os.path.join(allframes_root, split.name)
        labels_dir = os.path.join(labels_root, split.name, "labels")
        vids_dir = os.path.join(videos_root, split.name)
        ensure_dir(frames_dir)
        ensure_dir(labels_dir)
        ensure_dir(vids_dir)

        for clip_id in clip_ids:
            video_path = os.path.join(videos_dir, f"Clip_{clip_id}.mov")
            if not os.path.exists(video_path):
                raise RuntimeError(f"Missing video file: {video_path}")

            # Extract frames.
            n_frames, (img_w, img_h) = extract_frames(
                video_path=video_path,
                out_dir=frames_dir,
                clip_id=clip_id,
                image_ext=args.image_ext,
                png_compression=args.png_compression,
                jpg_quality=args.jpg_quality,
            )
            split_to_lens[split.name][clip_id] = int(n_frames)

            # Write labels (from Dogfight annotations).
            anno_path = os.path.join(annos_dir, f"Clip_{clip_id:03d}.txt")
            write_yolo_labels_for_clip(
                anno_path=anno_path,
                labels_dir=labels_dir,
                clip_id=clip_id,
                img_w=img_w,
                img_h=img_h,
            )

        # Write video_length_dict.pkl for this split.
        build_video_length_dict(vids_dir, split_to_lens[split.name])

    # Summary.
    total_frames = sum(sum(d.values()) for d in split_to_lens.values())
    print(f"Done. Total frames: {total_frames}")
    for name in ("train", "val", "test"):
        if split_to_lens[name]:
            print(f"{name}: {len(split_to_lens[name])} clips, {sum(split_to_lens[name].values())} frames")


if __name__ == "__main__":
    main()
