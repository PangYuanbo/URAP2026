from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from qstr_dronedet.features.roi import crop_with_context
from qstr_dronedet.candidates.motion_candidates import candidates_from_motion
from qstr_dronedet.candidates.merge import bbox_iou, center_distance, nms_candidates
from qstr_dronedet.candidates.yolo_wrapper import candidates_from_yolo_tiled
from qstr_dronedet.motion.difference import compute_multik_motion


def apply_linear_motion_blur(frame: np.ndarray, kernel_size: int = 0, direction: str = "horizontal") -> np.ndarray:
    if kernel_size is None or kernel_size <= 1:
        return frame
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    if direction == "vertical":
        kernel[:, kernel_size // 2] = 1.0
    elif direction == "diagonal":
        np.fill_diagonal(kernel, 1.0)
    else:
        kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(frame, -1, kernel)


def make_speed_augmented_videos(
    input_path: str | Path,
    out_dir: str | Path,
    strides: list[int],
    keep_fps: bool = True,
    motion_blur: int = 0,
    blur_direction: str = "horizontal",
) -> list[Path]:
    """Create apparent high-speed videos by temporal subsampling.

    If keep_fps=True, output FPS is unchanged, so motion appears stride times faster.
    If keep_fps=False, output FPS is divided by stride, preserving approximate playback speed.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"Video has no readable frames: {input_path}")

    outputs: list[Path] = []
    manifest = []
    for stride in strides:
        stride = int(stride)
        if stride < 1:
            raise ValueError("strides must be >= 1")
        out_path = out_dir / f"{input_path.stem}_speedx{stride}.mp4"
        out_fps = fps if keep_fps else max(1.0, fps / stride)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
        selected = list(range(0, len(frames), stride))
        for idx in selected:
            frame = apply_linear_motion_blur(frames[idx], motion_blur, blur_direction)
            writer.write(frame)
        writer.release()
        outputs.append(out_path)
        manifest.append(
            {
                "input": str(input_path),
                "output": str(out_path),
                "stride": stride,
                "input_fps": fps,
                "output_fps": out_fps,
                "input_frames": len(frames),
                "output_frames": len(selected),
                "motion_blur": motion_blur,
                "blur_direction": blur_direction,
            }
        )
    (out_dir / "speed_augmentation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs


def make_speed_augmented_video_dir(
    input_dir: str | Path,
    out_dir: str | Path,
    strides: list[int],
    pattern: str = "*.mp4",
    max_videos: int | None = None,
    keep_fps: bool = True,
    motion_blur: int = 0,
    blur_direction: str = "horizontal",
) -> list[Path]:
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    videos = sorted(input_dir.rglob(pattern))
    if max_videos is not None:
        videos = videos[: max(0, int(max_videos))]
    if not videos:
        raise FileNotFoundError(f"No videos found under {input_dir} with pattern {pattern}")
    outputs: list[Path] = []
    combined_manifest = []
    for video in videos:
        rel_parent = video.relative_to(input_dir).parent
        video_out = out_dir / rel_parent / video.stem
        made = make_speed_augmented_videos(video, video_out, strides, keep_fps, motion_blur, blur_direction)
        outputs.extend(made)
        manifest_path = video_out / "speed_augmentation_manifest.json"
        if manifest_path.exists():
            combined_manifest.extend(json.loads(manifest_path.read_text(encoding="utf-8")))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "speed_augmentation_manifest.json").write_text(json.dumps(combined_manifest, indent=2), encoding="utf-8")
    return outputs


def make_speed_augmented_frame_csv(
    annotations_csv: str | Path,
    out_csv: str | Path,
    strides: list[int],
    frame_index_column: str = "frame_index",
) -> Path:
    """Duplicate CSV rows with virtual speed tags by selecting every stride-th frame.

    This is useful when annotations already include per-frame boxes. It does not hallucinate
    boxes for dropped frames; it keeps boxes from selected real frames and adds an `aug_speed`
    column for downstream split/training scripts.
    """
    annotations_csv = Path(annotations_csv)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with annotations_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    if frame_index_column not in fieldnames:
        # Fallback: infer order from row order, which is weaker but still usable.
        for i, row in enumerate(rows):
            row[frame_index_column] = str(i)
        fieldnames.append(frame_index_column)
    out_fields = fieldnames + [c for c in ["aug_speed"] if c not in fieldnames]
    out_rows: list[dict[str, str]] = []
    for stride in strides:
        for row in rows:
            idx = int(float(row[frame_index_column]))
            if idx % int(stride) == 0:
                new_row = dict(row)
                new_row["aug_speed"] = f"speedx{stride}"
                out_rows.append(new_row)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_csv


def make_static_hover_sample(
    input_video: str | Path,
    out_video: str | Path,
    x: int | None = None,
    y: int | None = None,
    radius: int = 2,
    color: tuple[int, int, int] = (15, 15, 15),
    max_frames: int | None = None,
    freeze_background: bool = False,
    jitter_px: int = 0,
    seed: int = 7,
) -> Path:
    """Overlay a tiny fixed dot on a video to simulate a hovering/static drone."""
    input_video = Path(input_video)
    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x = int(width * 0.55) if x is None else int(x)
    y = int(height * 0.42) if y is None else int(y)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frozen_frame = None
    rng = np.random.default_rng(seed)
    boxes = []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if freeze_background:
            if frozen_frame is None:
                frozen_frame = frame.copy()
            frame = frozen_frame.copy()
        if jitter_px > 0:
            dx = int(rng.integers(-jitter_px, jitter_px + 1))
            dy = int(rng.integers(-jitter_px, jitter_px + 1))
        else:
            dx = 0
            dy = 0
        cx = int(np.clip(x + dx, radius, width - radius - 1))
        cy = int(np.clip(y + dy, radius, height - radius - 1))
        cv2.circle(frame, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)
        boxes.append(
            {
                "frame_id": n,
                "cx": cx,
                "cy": cy,
                "bbox_xyxy": [float(cx - radius), float(cy - radius), float(cx + radius), float(cy + radius)],
            }
        )
        writer.write(frame)
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    cap.release()
    writer.release()
    meta = {
        "input": str(input_video),
        "output": str(out_video),
        "x": x,
        "y": y,
        "radius": radius,
        "frames": n,
        "fps": fps,
        "freeze_background": freeze_background,
        "jitter_px": jitter_px,
        "seed": seed,
        "boxes": boxes,
    }
    out_video.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_video


def make_moving_target_sample(
    input_video: str | Path,
    out_video: str | Path,
    start_x: int | None = None,
    start_y: int | None = None,
    vx: float = 16.0,
    vy: float = 0.0,
    radius: int = 2,
    color: tuple[int, int, int] = (0, 0, 0),
    max_frames: int = 80,
    freeze_background: bool = True,
) -> Path:
    """Overlay a tiny moving target and write per-frame boxes for tracker tests."""
    input_video = Path(input_video)
    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_x = int(width * 0.2) if start_x is None else int(start_x)
    start_y = int(height * 0.45) if start_y is None else int(start_y)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    ok, first = cap.read()
    if not ok:
        cap.release()
        writer.release()
        raise ValueError(f"Video has no readable frames: {input_video}")
    boxes = []
    for frame_id in range(max_frames):
        if freeze_background:
            frame = first.copy()
        else:
            if frame_id == 0:
                frame = first.copy()
            else:
                ok, frame = cap.read()
                if not ok:
                    break
        cx = int(np.clip(start_x + vx * frame_id, radius, width - radius - 1))
        cy = int(np.clip(start_y + vy * frame_id, radius, height - radius - 1))
        cv2.circle(frame, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)
        writer.write(frame)
        boxes.append({"frame_id": frame_id, "cx": cx, "cy": cy, "bbox_xyxy": [float(cx - radius), float(cy - radius), float(cx + radius), float(cy + radius)]})
    cap.release()
    writer.release()
    meta = {
        "input": str(input_video),
        "output": str(out_video),
        "radius": radius,
        "frames": len(boxes),
        "fps": fps,
        "start_x": start_x,
        "start_y": start_y,
        "vx": vx,
        "vy": vy,
        "freeze_background": freeze_background,
        "boxes": boxes,
    }
    out_video.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_video


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def build_static_hover_crop_dataset(
    metadata_paths: list[str | Path],
    out_dir: str | Path,
    class_name: str = "drone",
    negative_class: str = "background",
    negative_per_positive: int = 3,
    crop_scale: float = 8.0,
    crop_size: int = 128,
    seed: int = 17,
) -> Path:
    """Build a crop-recognizer folder dataset from static-hover JSON metadata.

    The JSON files are produced by make_static_hover_sample and contain per-frame boxes.
    Positives are cropped around those boxes. Negatives are same-size random boxes with
    low IoU against the positive box in the same frame.
    """
    out_dir = Path(out_dir)
    pos_dir = out_dir / class_name
    neg_dir = out_dir / negative_class
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest: list[dict[str, object]] = []

    for meta_path_like in metadata_paths:
        meta_path = Path(meta_path_like)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        video_path = Path(meta["output"])
        if not video_path.is_absolute():
            video_path = meta_path.parent / video_path.name if not Path(meta["output"]).exists() else Path(meta["output"])
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video from metadata {meta_path}: {video_path}")
        boxes_by_frame = {int(b["frame_id"]): b["bbox_xyxy"] for b in meta.get("boxes", [])}
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id not in boxes_by_frame:
                frame_id += 1
                continue
            bbox = [float(v) for v in boxes_by_frame[frame_id]]
            stem = f"{meta_path.stem}_{frame_id:06d}"
            crop = crop_with_context(frame, tuple(bbox), scale=crop_scale, out_size=crop_size)
            pos_path = pos_dir / f"{stem}.jpg"
            cv2.imwrite(str(pos_path), crop)
            manifest.append({"path": str(pos_path), "class": class_name, "frame_id": frame_id, "bbox_xyxy": bbox, "source_video": str(video_path)})

            h, w = frame.shape[:2]
            bw, bh = max(2.0, bbox[2] - bbox[0]), max(2.0, bbox[3] - bbox[1])
            for neg_idx in range(max(0, int(negative_per_positive))):
                neg_bbox = None
                for _ in range(100):
                    cx = float(rng.integers(int(bw), max(int(bw) + 1, w - int(bw))))
                    cy = float(rng.integers(int(bh), max(int(bh) + 1, h - int(bh))))
                    candidate = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
                    if _bbox_iou(candidate, bbox) < 0.01:
                        neg_bbox = candidate
                        break
                if neg_bbox is None:
                    continue
                neg_crop = crop_with_context(frame, tuple(neg_bbox), scale=crop_scale, out_size=crop_size)
                neg_path = neg_dir / f"{stem}_neg{neg_idx:02d}.jpg"
                cv2.imwrite(str(neg_path), neg_crop)
                manifest.append({"path": str(neg_path), "class": negative_class, "frame_id": frame_id, "bbox_xyxy": neg_bbox, "source_video": str(video_path)})
            frame_id += 1
        cap.release()

    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(r) for r in manifest) + "\n", encoding="utf-8")
    return manifest_path


def _load_exclude_boxes(metadata_paths: list[str | Path]) -> dict[str, dict[int, list[list[float]]]]:
    by_video: dict[str, dict[int, list[list[float]]]] = {}
    for meta_path_like in metadata_paths:
        meta_path = Path(meta_path_like)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        video_keys = {str(Path(meta["output"])), str((meta_path.parent / Path(meta["output"]).name))}
        boxes = {}
        for b in meta.get("boxes", []):
            boxes.setdefault(int(b["frame_id"]), []).append([float(v) for v in b["bbox_xyxy"]])
        for key in video_keys:
            by_video[key] = boxes
    return by_video


def mine_motion_hard_negative_crops(
    video_paths: list[str | Path],
    out_dir: str | Path,
    exclude_metadata_paths: list[str | Path] | None = None,
    max_frames: int | None = 120,
    frame_stride: int = 1,
    max_crops_per_video: int = 300,
    min_area: int = 3,
    max_area: int = 5000,
    min_motion_score: float = 0.08,
    artifact_q_threshold: float = 0.66,
    crop_scale: float = 8.0,
    crop_size: int = 128,
    exclude_padding_px: float = 24.0,
) -> Path:
    """Mine motion-generated hard negatives from videos.

    Known target boxes from metadata are excluded. Remaining motion candidates are useful
    as background/alignment-artifact negatives for crop recognition.
    """
    out_dir = Path(out_dir)
    exclude_by_video = _load_exclude_boxes(exclude_metadata_paths or [])
    manifest: list[dict[str, object]] = []
    for video_like in video_paths:
        video = Path(video_like)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video}")
        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
        cap.release()
        excludes = exclude_by_video.get(str(video), exclude_by_video.get(str(video.resolve()), {}))
        saved = 0
        for frame_id in range(1, len(frames), max(1, int(frame_stride))):
            if saved >= max_crops_per_video:
                break
            motion = compute_multik_motion(frames[: frame_id + 1], frame_id, (1, 2, 4))
            q = float(motion["best_quality"])
            candidates = candidates_from_motion(motion["motion_map"], min_area=min_area, max_area=max_area)
            candidates.sort(key=lambda c: c.motion_score, reverse=True)
            for cand in candidates:
                if saved >= max_crops_per_video:
                    break
                bbox = [float(v) for v in cand.bbox_xyxy]
                if cand.motion_score < min_motion_score:
                    continue
                cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
                skip = False
                for ex in excludes.get(frame_id, []):
                    padded = [ex[0] - exclude_padding_px, ex[1] - exclude_padding_px, ex[2] + exclude_padding_px, ex[3] + exclude_padding_px]
                    if _bbox_iou(bbox, ex) > 0.05 or (padded[0] <= cx <= padded[2] and padded[1] <= cy <= padded[3]):
                        skip = True
                        break
                if skip:
                    continue
                class_name = "alignment_artifact" if q < artifact_q_threshold or cand.motion_score > 0.16 else "background"
                class_dir = out_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                crop = crop_with_context(frames[frame_id], tuple(bbox), scale=crop_scale, out_size=crop_size)
                out_path = class_dir / f"{video.stem}_{frame_id:06d}_{saved:04d}.jpg"
                cv2.imwrite(str(out_path), crop)
                manifest.append(
                    {
                        "path": str(out_path),
                        "class": class_name,
                        "frame_id": frame_id,
                        "bbox_xyxy": bbox,
                        "motion_score": cand.motion_score,
                        "alignment_quality": q,
                        "source_video": str(video),
                    }
                )
                saved += 1
    manifest_path = out_dir / "hard_negative_manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(r) for r in manifest) + "\n", encoding="utf-8")
    return manifest_path


def export_static_hover_frames_csv(
    metadata_paths: list[str | Path],
    out_dir: str | Path,
    frame_stride: int = 1,
    max_frames_per_video: int | None = None,
) -> Path:
    """Export synthetic static-hover videos into frame images plus box CSV."""
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "annotations.csv"
    rows: list[dict[str, object]] = []
    for meta_path_like in metadata_paths:
        meta_path = Path(meta_path_like)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        video = Path(meta["output"])
        if not video.exists():
            video = meta_path.parent / video.name
        boxes_by_frame = {int(b["frame_id"]): [float(v) for v in b["bbox_xyxy"]] for b in meta.get("boxes", [])}
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video from metadata {meta_path}: {video}")
        kept = 0
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id in boxes_by_frame and frame_id % max(1, int(frame_stride)) == 0:
                frame_path = frames_dir / f"{meta_path.stem}_{frame_id:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                x1, y1, x2, y2 = boxes_by_frame[frame_id]
                rows.append({"frame_path": str(frame_path), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "class": "object"})
                kept += 1
                if max_frames_per_video is not None and kept >= max_frames_per_video:
                    break
            frame_id += 1
        cap.release()
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "x1", "y1", "x2", "y2", "class"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def export_static_hover_feature_csv(
    metadata_paths: list[str | Path],
    out_dir: str | Path,
    frame_stride: int = 4,
    negative_per_positive: int = 1,
    seed: int = 37,
) -> Path:
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for meta_path_like in metadata_paths:
        meta_path = Path(meta_path_like)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        video = Path(meta["output"])
        if not video.exists():
            video = meta_path.parent / video.name
        boxes_by_frame = {int(b["frame_id"]): [float(v) for v in b["bbox_xyxy"]] for b in meta.get("boxes", [])}
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video from metadata {meta_path}: {video}")
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id in boxes_by_frame and frame_id % max(1, int(frame_stride)) == 0:
                frame_path = frames_dir / f"{meta_path.stem}_{frame_id:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                pos = boxes_by_frame[frame_id]
                rows.append({"frame_path": str(frame_path), "x1": pos[0], "y1": pos[1], "x2": pos[2], "y2": pos[3], "class": "drone"})
                h, w = frame.shape[:2]
                bw, bh = max(2.0, pos[2] - pos[0]), max(2.0, pos[3] - pos[1])
                for _ in range(max(0, int(negative_per_positive))):
                    neg = None
                    for _attempt in range(100):
                        cx = float(rng.integers(int(bw), max(int(bw) + 1, w - int(bw))))
                        cy = float(rng.integers(int(bh), max(int(bh) + 1, h - int(bh))))
                        candidate = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
                        if _bbox_iou(candidate, pos) < 0.01:
                            neg = candidate
                            break
                    if neg is not None:
                        rows.append({"frame_path": str(frame_path), "x1": neg[0], "y1": neg[1], "x2": neg[2], "y2": neg[3], "class": "background"})
            frame_id += 1
        cap.release()
    csv_path = out_dir / "feature_boxes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "x1", "y1", "x2", "y2", "class"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def export_hard_negative_feature_csv(
    manifest_paths: list[str | Path],
    out_dir: str | Path,
    max_samples_per_class: int = 120,
) -> Path:
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    frame_cache: dict[tuple[str, int], Path] = {}
    video_cache: dict[str, cv2.VideoCapture] = {}
    try:
        for manifest_like in manifest_paths:
            for line in Path(manifest_like).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                cls = str(row.get("class", "background"))
                if cls not in {"background", "alignment_artifact"}:
                    continue
                if counts.get(cls, 0) >= max_samples_per_class:
                    continue
                video = str(row["source_video"])
                frame_id = int(row["frame_id"])
                key = (video, frame_id)
                if key not in frame_cache:
                    cap = video_cache.get(video)
                    if cap is None:
                        cap = cv2.VideoCapture(video)
                        if not cap.isOpened():
                            continue
                        video_cache[video] = cap
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame_path = frames_dir / f"{Path(video).stem}_{frame_id:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_cache[key] = frame_path
                x1, y1, x2, y2 = [float(v) for v in row["bbox_xyxy"]]
                rows_out.append({"frame_path": str(frame_cache[key]), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "class": cls})
                counts[cls] = counts.get(cls, 0) + 1
    finally:
        for cap in video_cache.values():
            cap.release()
    csv_path = out_dir / "hard_negative_feature_boxes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "x1", "y1", "x2", "y2", "class"])
        writer.writeheader()
        writer.writerows(rows_out)
    return csv_path


def build_static_hover_temporal_dataset(
    metadata_paths: list[str | Path],
    out_dir: str | Path,
    t: int = 5,
    frame_stride: int = 5,
    negative_per_positive: int = 1,
    crop_scale: float = 8.0,
    crop_size: int = 96,
    seed: int = 29,
) -> Path:
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    manifest: list[dict[str, object]] = []
    for meta_path_like in metadata_paths:
        meta_path = Path(meta_path_like)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        video = Path(meta["output"])
        if not video.exists():
            video = meta_path.parent / video.name
        boxes = {int(b["frame_id"]): [float(v) for v in b["bbox_xyxy"]] for b in meta.get("boxes", [])}
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video from metadata {meta_path}: {video}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if len(frames) < t:
            continue
        h, w = frames[0].shape[:2]
        for center in range(t - 1, len(frames), max(1, int(frame_stride))):
            if center not in boxes:
                continue
            window_ids = list(range(center - t + 1, center + 1))
            pos_bbox = boxes[center]
            sample_id = f"{meta_path.stem}_{center:06d}"
            for cls, bbox, suffix in [("drone", pos_bbox, "")]:
                sample_dir = out_dir / cls / f"{sample_id}{suffix}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                for j, fid in enumerate(window_ids):
                    crop = crop_with_context(frames[fid], tuple(bbox), scale=crop_scale, out_size=crop_size)
                    cv2.imwrite(str(sample_dir / f"frame_{j:03d}.jpg"), crop)
                manifest.append({"path": str(sample_dir), "class": cls, "center_frame": center, "bbox_xyxy": bbox, "source_video": str(video)})
            bw, bh = max(2.0, pos_bbox[2] - pos_bbox[0]), max(2.0, pos_bbox[3] - pos_bbox[1])
            for neg_idx in range(max(0, int(negative_per_positive))):
                neg_bbox = None
                for _ in range(100):
                    cx = float(rng.integers(int(bw), max(int(bw) + 1, w - int(bw))))
                    cy = float(rng.integers(int(bh), max(int(bh) + 1, h - int(bh))))
                    candidate = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
                    if _bbox_iou(candidate, pos_bbox) < 0.01:
                        neg_bbox = candidate
                        break
                if neg_bbox is None:
                    continue
                sample_dir = out_dir / "background" / f"{sample_id}_neg{neg_idx:02d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                for j, fid in enumerate(window_ids):
                    crop = crop_with_context(frames[fid], tuple(neg_bbox), scale=crop_scale, out_size=crop_size)
                    cv2.imwrite(str(sample_dir / f"frame_{j:03d}.jpg"), crop)
                manifest.append({"path": str(sample_dir), "class": "background", "center_frame": center, "bbox_xyxy": neg_bbox, "source_video": str(video)})
    manifest_path = out_dir / "temporal_manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(r) for r in manifest) + "\n", encoding="utf-8")
    return manifest_path


def build_detector_proposal_stage_b_dataset(
    metadata_paths: list[str | Path],
    out_dir: str | Path,
    yolo_weights: str | Path,
    yolo_conf: float = 0.05,
    tile_size: int = 256,
    tile_stride: int = 128,
    frame_stride: int = 2,
    max_frames_per_video: int | None = 80,
    max_proposals_per_frame: int = 8,
    max_negatives_per_frame: int = 4,
    match_iou: float = 0.1,
    match_center_px: float = 24.0,
    crop_scale: float = 4.0,
    crop_size: int = 128,
    tube_t: int = 5,
    tube_size: int = 96,
) -> dict[str, str]:
    """Build Stage B datasets from detector proposals instead of oracle boxes.

    Outputs:
    - crop/<class>/*.jpg for CropRecognizer;
    - temporal/<class>/<sample>/frame_*.jpg for TemporalRecognizer;
    - feature/annotations.csv for FeatureRecognizer;
    - proposal_manifest.jsonl with detector and matching metadata.
    """
    out_dir = Path(out_dir)
    crop_root = out_dir / "crop"
    temporal_root = out_dir / "temporal"
    feature_dir = out_dir / "feature"
    feature_frames_dir = feature_dir / "frames"
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_frames_dir.mkdir(parents=True, exist_ok=True)
    for cls in ("drone", "background"):
        (crop_root / cls).mkdir(parents=True, exist_ok=True)
        (temporal_root / cls).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    counters = {"drone": 0, "background": 0}

    for meta_like in metadata_paths:
        meta_path = Path(meta_like)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        video = Path(meta["output"])
        if not video.exists():
            video = meta_path.parent / video.name
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video from metadata {meta_path}: {video}")
        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if max_frames_per_video is not None and len(frames) >= max_frames_per_video:
                break
        cap.release()
        boxes_by_frame = {int(b["frame_id"]): tuple(float(v) for v in b["bbox_xyxy"]) for b in meta.get("boxes", [])}

        for frame_id in range(0, len(frames), max(1, int(frame_stride))):
            if frame_id not in boxes_by_frame:
                continue
            frame = frames[frame_id]
            feature_frame_path = feature_frames_dir / f"{meta_path.stem}_{frame_id:06d}.jpg"
            if not feature_frame_path.exists():
                cv2.imwrite(str(feature_frame_path), frame)
            gt = boxes_by_frame[frame_id]
            proposals = candidates_from_yolo_tiled(
                frame,
                str(yolo_weights),
                tile_size=tile_size,
                stride=tile_stride,
                conf=yolo_conf,
            )
            proposals = nms_candidates(proposals, iou_threshold=0.35)[: max(1, int(max_proposals_per_frame))]
            matched_positive = False
            negatives_this_frame = 0
            for cand_idx, cand in enumerate(proposals):
                iou = bbox_iou(cand.bbox_xyxy, gt)
                dist = center_distance(cand.bbox_xyxy, gt)
                is_positive = iou >= match_iou or dist <= match_center_px
                if is_positive:
                    cls = "drone"
                    matched_positive = True
                else:
                    if negatives_this_frame >= max(0, int(max_negatives_per_frame)):
                        continue
                    cls = "background"
                    negatives_this_frame += 1

                sample_id = f"{meta_path.stem}_{frame_id:06d}_{cand_idx:03d}_{cls}_{counters[cls]:06d}"
                crop = crop_with_context(frame, cand.bbox_xyxy, scale=crop_scale, out_size=crop_size)
                crop_path = crop_root / cls / f"{sample_id}.jpg"
                cv2.imwrite(str(crop_path), crop)

                tube_dir = temporal_root / cls / sample_id
                tube_dir.mkdir(parents=True, exist_ok=True)
                start = max(0, frame_id - tube_t + 1)
                ids = list(range(start, frame_id + 1))
                if len(ids) < tube_t:
                    ids = [ids[0]] * (tube_t - len(ids)) + ids
                for j, fid in enumerate(ids[-tube_t:]):
                    tube_crop = crop_with_context(frames[fid], cand.bbox_xyxy, scale=crop_scale, out_size=tube_size)
                    cv2.imwrite(str(tube_dir / f"frame_{j:03d}.jpg"), tube_crop)

                feature_rows.append(
                    {
                        "frame_path": str(feature_frame_path),
                        "frame_id": frame_id,
                        "x1": cand.bbox_xyxy[0],
                        "y1": cand.bbox_xyxy[1],
                        "x2": cand.bbox_xyxy[2],
                        "y2": cand.bbox_xyxy[3],
                        "class": cls,
                    }
                )
                manifest.append(
                    {
                        "class": cls,
                        "crop_path": str(crop_path),
                        "temporal_path": str(tube_dir),
                        "source_video": str(video),
                        "metadata": str(meta_path),
                        "frame_id": frame_id,
                        "proposal_bbox_xyxy": list(cand.bbox_xyxy),
                        "gt_bbox_xyxy": list(gt),
                        "proposal_score": cand.objectness,
                        "proposal_source": cand.source,
                        "match_iou": iou,
                        "match_center_distance": dist,
                        "matched_positive": is_positive,
                    }
                )
                counters[cls] += 1

            if not matched_positive:
                # Keep an explicit missed-positive row for Stage A diagnostics.
                manifest.append(
                    {
                        "class": "missed_drone",
                        "source_video": str(video),
                        "metadata": str(meta_path),
                        "frame_id": frame_id,
                        "gt_bbox_xyxy": list(gt),
                        "num_proposals": len(proposals),
                        "matched_positive": False,
                    }
                )

    feature_csv = feature_dir / "annotations.csv"
    with feature_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "frame_id", "x1", "y1", "x2", "y2", "class"])
        writer.writeheader()
        writer.writerows(feature_rows)
    manifest_path = out_dir / "proposal_manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(r) for r in manifest) + "\n", encoding="utf-8")
    summary = {
        "crop_root": str(crop_root),
        "temporal_root": str(temporal_root),
        "feature_csv": str(feature_csv),
        "manifest": str(manifest_path),
        "num_crop_drone": str(counters["drone"]),
        "num_crop_background": str(counters["background"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def mine_hard_negative_temporal_dataset(
    hard_negative_manifest: str | Path,
    out_dir: str | Path,
    t: int = 5,
    crop_scale: float = 8.0,
    crop_size: int = 96,
    max_samples_per_class: int = 80,
) -> Path:
    out_dir = Path(out_dir)
    rows = [json.loads(line) for line in Path(hard_negative_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    counts: dict[str, int] = {}
    frames_cache: dict[str, list[np.ndarray]] = {}
    manifest: list[dict[str, object]] = []
    for row in rows:
        cls = str(row.get("class", "background"))
        if counts.get(cls, 0) >= max_samples_per_class:
            continue
        video = str(row["source_video"])
        frame_id = int(row["frame_id"])
        if frame_id < t - 1:
            continue
        if video not in frames_cache:
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                continue
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
            frames_cache[video] = frames
        frames = frames_cache[video]
        if frame_id >= len(frames):
            continue
        bbox = [float(v) for v in row["bbox_xyxy"]]
        idx = counts.get(cls, 0)
        sample_dir = out_dir / cls / f"{Path(video).stem}_{frame_id:06d}_{idx:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for j, fid in enumerate(range(frame_id - t + 1, frame_id + 1)):
            crop = crop_with_context(frames[fid], tuple(bbox), scale=crop_scale, out_size=crop_size)
            cv2.imwrite(str(sample_dir / f"frame_{j:03d}.jpg"), crop)
        counts[cls] = idx + 1
        manifest.append({"path": str(sample_dir), "class": cls, "frame_id": frame_id, "bbox_xyxy": bbox, "source_video": video})
    manifest_path = out_dir / "hard_negative_temporal_manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(r) for r in manifest) + "\n", encoding="utf-8")
    return manifest_path


def split_folder_dataset(
    input_dir: str | Path,
    out_dir: str | Path,
    val_fraction: float = 0.25,
    seed: int = 53,
) -> Path:
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    rng = np.random.default_rng(seed)
    manifest = []
    for cls_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        items = [p for p in cls_dir.iterdir() if p.is_dir() or p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        order = rng.permutation(len(items)).tolist() if items else []
        val_count = int(round(len(items) * val_fraction))
        for pos, item_idx in enumerate(order):
            item = items[item_idx]
            split = "val" if pos < val_count else "train"
            dst = out_dir / split / cls_dir.name / item.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
            manifest.append({"source": str(item), "dest": str(dst), "class": cls_dir.name, "split": split})
    manifest_path = out_dir / "split_manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(r) for r in manifest) + "\n", encoding="utf-8")
    return manifest_path


def split_feature_csv(csv_path: str | Path, out_dir: str | Path, val_fraction: float = 0.25, seed: int = 59) -> tuple[Path, Path]:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(row.get("class", "unknown"), []).append(row)
    rng = np.random.default_rng(seed)
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for cls_rows in by_class.values():
        order = rng.permutation(len(cls_rows)).tolist()
        val_count = int(round(len(cls_rows) * val_fraction))
        for pos, idx in enumerate(order):
            (val_rows if pos < val_count else train_rows).append(cls_rows[idx])
    fieldnames = ["frame_path", "x1", "y1", "x2", "y2", "class"]
    train_csv = out_dir / "train.csv"
    val_csv = out_dir / "val.csv"
    for path, split_rows in [(train_csv, train_rows), (val_csv, val_rows)]:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
    return train_csv, val_csv
