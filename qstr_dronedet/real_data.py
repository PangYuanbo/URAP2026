from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from qstr_dronedet.candidates.yolo_p2_train import build_class_agnostic_yolo_dataset, build_tiled_class_agnostic_yolo_dataset
from qstr_dronedet.candidates.merge import bbox_iou, center_distance, nms_candidates
from qstr_dronedet.candidates.yolo_wrapper import candidates_from_yolo_tiled
from qstr_dronedet.features.roi import crop_with_context


REAL_CLASSES = {
    "drone",
    "bird",
    "airplane",
    "insect",
    "ground_object",
    "alignment_artifact",
    "background",
    "unknown",
}

REAL_TAGS = {
    "static_hovering",
    "fast_target",
    "bad_alignment",
    "tiny",
    "hard_negative",
}


@dataclass(frozen=True)
class RealDatasetBuildResult:
    frame_annotations_csv: Path
    frames_dir: Path
    data_yaml: Path
    summary_json: Path


@dataclass(frozen=True)
class AntiUAVSubsetResult:
    annotations_csv: Path
    manifest_csv: Path
    extracted_root: Path
    summary_json: Path


@dataclass(frozen=True)
class RealStageBDatasetResult:
    crop_root: Path
    temporal_root: Path
    manifest_jsonl: Path
    summary_json: Path


@dataclass(frozen=True)
class RealProposalStageBDatasetResult:
    crop_root: Path
    temporal_root: Path
    feature_csv: Path
    manifest_jsonl: Path
    summary_json: Path


def ensure_real_data_layout(root: str | Path = "data/real") -> list[Path]:
    root = Path(root)
    dirs = [
        root / "raw_videos" / "static_hovering",
        root / "raw_videos" / "fast_target",
        root / "raw_videos" / "bad_alignment",
        root / "raw_videos" / "hard_negative",
        root / "annotations",
        root / "motion_debug",
        root / "frames",
        root / "crops",
        root / "yolo_candidate",
        root / "stage_b",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _anti_uav_tag(tags: list[str]) -> str:
    if "FM" in tags:
        return "fast_target"
    if "TC-HARD" in tags or "LR" in tags or "OV" in tags or "OC" in tags:
        return "tiny"
    return "static_hovering"


def _anti_uav_select_sequences(
    zipf: zipfile.ZipFile,
    split: str,
    modality: str,
    max_sequences: int,
    start_index: int = 0,
) -> list[str]:
    prefix = f"Anti-UAV300/{split}/"
    suffix = f"/{modality}.mp4"
    seqs = sorted(
        {
            name[len(prefix) : -len(suffix)]
            for name in zipf.namelist()
            if name.startswith(prefix) and name.endswith(suffix)
        }
    )
    start = max(0, int(start_index))
    return seqs[start : start + max(0, int(max_sequences))]


def export_anti_uav300_subset_from_zip(
    zip_path: str | Path,
    out: str | Path,
    split: str = "test",
    modality: str = "visible",
    max_sequences: int = 5,
    start_index: int = 0,
    frame_stride: int = 10,
    max_frames_per_sequence: int | None = 80,
    class_name: str = "drone",
) -> AntiUAVSubsetResult:
    """Extract a small Anti-UAV300 subset and convert boxes to the QSTR real CSV.

    Anti-UAV JSON boxes use [x, y, w, h]. The exported CSV uses QSTR's
    video_path,frame_id,x1,y1,x2,y2,class,tag format.
    """
    zip_path = Path(zip_path)
    out = Path(out)
    extracted_root = out / "raw_videos" / split / modality
    annotations_dir = out / "annotations"
    extracted_root.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    annotations_csv = annotations_dir / "qstr_real_boxes.csv"
    manifest_csv = annotations_dir / "recording_manifest.csv"
    summary_json = out / "summary.json"

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of train, val, test")
    if modality not in {"visible", "infrared"}:
        raise ValueError("modality must be visible or infrared")

    rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path, "r") as zipf:
        label_tags: dict[str, list[str]] = {}
        label_name = f"Anti-UAV300/label_new/{split}.json"
        if label_name in zipf.namelist():
            label_tags = json.loads(zipf.read(label_name).decode("utf-8"))
        seqs = _anti_uav_select_sequences(zipf, split, modality, max_sequences, start_index=start_index)
        for seq in seqs:
            base = f"Anti-UAV300/{split}/{seq}"
            video_entry = f"{base}/{modality}.mp4"
            ann_entry = f"{base}/{modality}.json"
            video_out = extracted_root / seq / f"{modality}.mp4"
            ann_out = extracted_root / seq / f"{modality}.json"
            video_out.parent.mkdir(parents=True, exist_ok=True)
            if not video_out.exists():
                video_out.write_bytes(zipf.read(video_entry))
            if not ann_out.exists():
                ann_out.write_bytes(zipf.read(ann_entry))

            ann = json.loads(ann_out.read_text(encoding="utf-8"))
            exists = ann.get("exist", [])
            boxes = ann.get("gt_rect", [])
            tags = label_tags.get(seq, [])
            tag = _anti_uav_tag(tags)
            kept = 0
            for frame_id, box in enumerate(boxes):
                if frame_id >= len(exists) or not exists[frame_id]:
                    continue
                if frame_id % max(1, int(frame_stride)) != 0:
                    continue
                if max_frames_per_sequence is not None and kept >= max_frames_per_sequence:
                    break
                if not box or len(box) < 4:
                    continue
                x, y, w, h = (float(v) for v in box[:4])
                if w <= 0 or h <= 0:
                    continue
                rows.append(
                    {
                        "video_path": str(video_out),
                        "frame_id": frame_id,
                        "x1": x,
                        "y1": y,
                        "x2": x + w,
                        "y2": y + h,
                        "class": class_name,
                        "tag": tag,
                    }
                )
                kept += 1
            manifest_rows.append(
                {
                    "video_path": str(video_out),
                    "scenario": tag,
                    "camera_motion": "unknown",
                    "target_motion": "uav",
                    "notes": f"Anti-UAV300 {split}/{seq} {modality}; tags={','.join(tags)}; sampled_boxes={kept}",
                }
            )

    with annotations_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writeheader()
        writer.writerows(rows)
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "scenario", "camera_motion", "target_motion", "notes"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "zip_path": str(zip_path),
        "split": split,
        "modality": modality,
        "max_sequences": max_sequences,
        "start_index": start_index,
        "frame_stride": frame_stride,
        "max_frames_per_sequence": max_frames_per_sequence,
        "num_sequences": len(manifest_rows),
        "num_boxes": len(rows),
        "annotations_csv": str(annotations_csv),
        "manifest_csv": str(manifest_csv),
        "extracted_root": str(extracted_root),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return AntiUAVSubsetResult(annotations_csv, manifest_csv, extracted_root, summary_json)


def _resolve_video_path(video_path: str, video_root: Path | None) -> Path:
    path = Path(video_path)
    if not path.is_absolute() and video_root is not None:
        path = video_root / path
    return path


def _frame_file_name(video: Path, frame_id: int) -> str:
    digest = hashlib.sha1(str(video).encode("utf-8")).hexdigest()[:8]
    return f"{video.stem}_{digest}_{frame_id:06d}.jpg"


def _read_real_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"}
    columns = set(rows[0].keys()) if rows else set()
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    return rows


def _bbox_iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-6, aa + bb - inter)


def _sample_negative_box(
    width: int,
    height: int,
    pos: list[float],
    rng: np.random.Generator,
    scale_jitter: float = 1.6,
) -> list[float] | None:
    bw = max(8.0, pos[2] - pos[0])
    bh = max(8.0, pos[3] - pos[1])
    for _ in range(100):
        sw = min(float(width), bw * float(rng.uniform(1.0, scale_jitter)))
        sh = min(float(height), bh * float(rng.uniform(1.0, scale_jitter)))
        cx = float(rng.uniform(sw / 2.0, max(sw / 2.0 + 1.0, width - sw / 2.0)))
        cy = float(rng.uniform(sh / 2.0, max(sh / 2.0 + 1.0, height - sh / 2.0)))
        box = [cx - sw / 2.0, cy - sh / 2.0, cx + sw / 2.0, cy + sh / 2.0]
        if _bbox_iou_xyxy(box, pos) < 0.01:
            return box
    return None


def build_real_stage_b_datasets(
    annotations_csv: str | Path,
    out: str | Path,
    negative_per_positive: int = 1,
    crop_scale: float = 4.0,
    crop_size: int = 128,
    tube_t: int = 5,
    tube_size: int = 96,
    seed: int = 71,
    max_samples: int | None = None,
) -> RealStageBDatasetResult:
    """Build crop and temporal recognizer folder datasets from real-video CSV boxes."""
    annotations_csv = Path(annotations_csv)
    out = Path(out)
    crop_root = out / "crop"
    temporal_root = out / "temporal"
    for cls in ("drone", "background"):
        (crop_root / cls).mkdir(parents=True, exist_ok=True)
        (temporal_root / cls).mkdir(parents=True, exist_ok=True)
    rows = _read_real_rows(annotations_csv)
    if max_samples is not None:
        rows = rows[: max(0, int(max_samples))]
    rng = np.random.default_rng(seed)
    video_cache: dict[str, cv2.VideoCapture] = {}
    frame_cache: dict[tuple[str, int], np.ndarray] = {}
    manifest: list[dict[str, object]] = []
    counts = {"drone": 0, "background": 0}

    def read_frame(video: str, frame_id: int) -> np.ndarray | None:
        key = (video, frame_id)
        if key in frame_cache:
            return frame_cache[key]
        cap = video_cache.get(video)
        if cap is None:
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                return None
            video_cache[video] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            return None
        frame_cache[key] = frame
        if len(frame_cache) > 64:
            frame_cache.pop(next(iter(frame_cache)))
        return frame

    def write_sample(cls: str, video: str, frame_id: int, bbox: list[float], frame: np.ndarray, suffix: str) -> None:
        idx = counts[cls]
        sample_id = f"{Path(video).parent.name}_{frame_id:06d}_{suffix}_{idx:06d}"
        crop = crop_with_context(frame, bbox, scale=crop_scale, out_size=crop_size)
        crop_path = crop_root / cls / f"{sample_id}.jpg"
        cv2.imwrite(str(crop_path), crop)
        tube_dir = temporal_root / cls / sample_id
        tube_dir.mkdir(parents=True, exist_ok=True)
        ids = list(range(max(0, frame_id - tube_t + 1), frame_id + 1))
        if len(ids) < tube_t:
            ids = [ids[0]] * (tube_t - len(ids)) + ids
        for j, fid in enumerate(ids[-tube_t:]):
            tube_frame = read_frame(video, fid)
            if tube_frame is None:
                tube_frame = frame
            tube_crop = crop_with_context(tube_frame, bbox, scale=crop_scale, out_size=tube_size)
            cv2.imwrite(str(tube_dir / f"frame_{j:03d}.jpg"), tube_crop)
        counts[cls] = idx + 1
        manifest.append(
            {
                "class": cls,
                "crop_path": str(crop_path),
                "temporal_path": str(tube_dir),
                "source_video": video,
                "frame_id": frame_id,
                "bbox_xyxy": bbox,
            }
        )

    for row in rows:
        video = str(_resolve_video_path(row["video_path"], None))
        frame_id = int(float(row["frame_id"]))
        frame = read_frame(video, frame_id)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        pos = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
        write_sample("drone", video, frame_id, pos, frame, "pos")
        for neg_idx in range(max(0, int(negative_per_positive))):
            neg = _sample_negative_box(w, h, pos, rng)
            if neg is not None:
                write_sample("background", video, frame_id, neg, frame, f"neg{neg_idx:02d}")

    for cap in video_cache.values():
        cap.release()
    manifest_path = out / "stage_b_manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(r) for r in manifest) + "\n", encoding="utf-8")
    summary = {
        "annotations_csv": str(annotations_csv),
        "crop_root": str(crop_root),
        "temporal_root": str(temporal_root),
        "manifest": str(manifest_path),
        "counts": counts,
        "num_manifest_rows": len(manifest),
    }
    summary_json = out / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return RealStageBDatasetResult(crop_root, temporal_root, manifest_path, summary_json)


def build_real_detector_proposal_stage_b_dataset(
    annotations_csv: str | Path,
    out: str | Path,
    yolo_weights: str | Path,
    yolo_conf: float = 0.05,
    fallback_yolo_weights: str | Path | None = None,
    fallback_yolo_conf: float = 0.15,
    tile_size: int = 256,
    tile_stride: int = 128,
    max_samples: int | None = None,
    max_proposals_per_frame: int = 8,
    max_fallback_proposals_per_frame: int = 4,
    max_negatives_per_frame: int = 4,
    match_iou: float = 0.1,
    match_center_px: float = 24.0,
    artifact_score_threshold: float = 0.25,
    high_score_fp_threshold: float = 0.5,
    non_drone_label_mode: str = "multiclass_artifact",
    hard_positive_max_size_px: float = 24.0,
    hard_positive_max_score: float = 0.5,
    hard_positive_repeat: int = 1,
    crop_scale: float = 2.0,
    crop_size: int = 128,
    tube_t: int = 5,
    tube_size: int = 96,
    device: str | int | None = "0",
) -> RealProposalStageBDatasetResult:
    """Build Stage B datasets from detector proposals on real-video annotations."""
    if non_drone_label_mode not in {"multiclass_artifact", "binary_buckets"}:
        raise ValueError("non_drone_label_mode must be 'multiclass_artifact' or 'binary_buckets'")
    annotations_csv = Path(annotations_csv)
    out = Path(out)
    crop_root = out / "crop"
    temporal_root = out / "temporal"
    feature_dir = out / "feature"
    feature_frames_dir = feature_dir / "frames"
    feature_frames_dir.mkdir(parents=True, exist_ok=True)
    folder_classes = ("drone", "background", "alignment_artifact") if non_drone_label_mode == "multiclass_artifact" else ("drone", "background")
    for cls in folder_classes:
        (crop_root / cls).mkdir(parents=True, exist_ok=True)
        (temporal_root / cls).mkdir(parents=True, exist_ok=True)

    rows = _read_real_rows(annotations_csv)
    if max_samples is not None:
        rows = rows[: max(0, int(max_samples))]

    video_cache: dict[str, cv2.VideoCapture] = {}
    frame_cache: dict[tuple[str, int], np.ndarray] = {}
    manifest: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    counters = {"drone": 0, "background": 0, "alignment_artifact": 0, "missed_drone": 0}
    diagnostic_bucket_counts = {"drone": 0, "hard_tiny_positive": 0, "easy_background": 0, "high_score_detector_fp": 0, "alignment_artifact": 0}

    def read_frame(video: str, frame_id: int) -> np.ndarray | None:
        key = (video, frame_id)
        if key in frame_cache:
            return frame_cache[key]
        cap = video_cache.get(video)
        if cap is None:
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                return None
            video_cache[video] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            return None
        frame_cache[key] = frame
        if len(frame_cache) > 96:
            frame_cache.pop(next(iter(frame_cache)))
        return frame

    def write_proposal_sample(
        cls: str,
        diagnostic_bucket: str,
        video: str,
        frame_id: int,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        gt: tuple[float, float, float, float],
        cand_idx: int,
        score: float,
        source: str,
        hard_positive: bool = False,
    ) -> None:
        train_cls = "background" if non_drone_label_mode == "binary_buckets" and cls != "drone" else cls
        sample_id = f"{Path(video).parent.name}_{frame_id:06d}_{cand_idx:03d}_{diagnostic_bucket}_{counters[train_cls]:06d}"
        crop = crop_with_context(frame, bbox, scale=crop_scale, out_size=crop_size)
        crop_path = crop_root / train_cls / f"{sample_id}.jpg"
        cv2.imwrite(str(crop_path), crop)
        tube_dir = temporal_root / train_cls / sample_id
        tube_dir.mkdir(parents=True, exist_ok=True)
        ids = list(range(max(0, frame_id - tube_t + 1), frame_id + 1))
        if len(ids) < tube_t:
            ids = [ids[0]] * (tube_t - len(ids)) + ids
        for j, fid in enumerate(ids[-tube_t:]):
            tube_frame = read_frame(video, fid)
            if tube_frame is None:
                tube_frame = frame
            tube_crop = crop_with_context(tube_frame, bbox, scale=crop_scale, out_size=tube_size)
            cv2.imwrite(str(tube_dir / f"frame_{j:03d}.jpg"), tube_crop)

        feature_frame_path = feature_frames_dir / f"{Path(video).parent.name}_{frame_id:06d}.jpg"
        if not feature_frame_path.exists():
            cv2.imwrite(str(feature_frame_path), frame)
        feature_rows.append(
            {
                "frame_path": str(feature_frame_path),
                "frame_id": frame_id,
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
                "class": train_cls,
                "diagnostic_bucket": diagnostic_bucket,
                "proposal_source": source,
                "proposal_score": score,
            }
        )
        iou = bbox_iou(bbox, gt)
        dist = center_distance(bbox, gt)
        manifest.append(
            {
                "class": train_cls,
                "diagnostic_bucket": diagnostic_bucket,
                "crop_path": str(crop_path),
                "temporal_path": str(tube_dir),
                "source_video": video,
                "frame_id": frame_id,
                "proposal_bbox_xyxy": list(bbox),
                "gt_bbox_xyxy": list(gt),
                "proposal_score": score,
                "proposal_source": source,
                "match_iou": iou,
                "match_center_distance": dist,
                "matched_positive": train_cls == "drone",
                "hard_positive": hard_positive,
            }
        )
        counters[train_cls] += 1
        if diagnostic_bucket in diagnostic_bucket_counts:
            diagnostic_bucket_counts[diagnostic_bucket] += 1

    for row in rows:
        video = str(_resolve_video_path(row["video_path"], None))
        frame_id = int(float(row["frame_id"]))
        frame = read_frame(video, frame_id)
        if frame is None:
            continue
        gt = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
        proposals = candidates_from_yolo_tiled(
            frame,
            str(yolo_weights),
            tile_size=tile_size,
            stride=tile_stride,
            conf=yolo_conf,
            device=device,
            max_det=max(100, int(max_proposals_per_frame) * 4),
        )
        proposals = nms_candidates(proposals, iou_threshold=0.35)[: max(1, int(max_proposals_per_frame))]
        if fallback_yolo_weights is not None:
            fallback_proposals = candidates_from_yolo_tiled(
                frame,
                str(fallback_yolo_weights),
                tile_size=tile_size,
                stride=tile_stride,
                conf=fallback_yolo_conf,
                device=device,
                max_det=max(100, int(max_fallback_proposals_per_frame) * 4),
            )
            fallback_proposals = nms_candidates(fallback_proposals, iou_threshold=0.35)[: max(0, int(max_fallback_proposals_per_frame))]
            for cand in fallback_proposals:
                cand.extra["base_source"] = cand.source
                cand.source = f"{cand.source}_fallback"
            proposals = nms_candidates(proposals + fallback_proposals, iou_threshold=0.35)[: max(1, int(max_proposals_per_frame) + int(max_fallback_proposals_per_frame))]
        matched_positive = False
        negatives_this_frame = 0
        for cand_idx, cand in enumerate(proposals):
            iou = bbox_iou(cand.bbox_xyxy, gt)
            dist = center_distance(cand.bbox_xyxy, gt)
            if iou >= match_iou or dist <= match_center_px:
                cls = "drone"
                bw = max(0.0, cand.bbox_xyxy[2] - cand.bbox_xyxy[0])
                bh = max(0.0, cand.bbox_xyxy[3] - cand.bbox_xyxy[1])
                is_hard_positive = (
                    max(bw, bh) <= hard_positive_max_size_px
                    and (cand.objectness <= hard_positive_max_score or "fallback" in cand.source)
                )
                diagnostic_bucket = "hard_tiny_positive" if is_hard_positive else "drone"
                matched_positive = True
            else:
                is_hard_positive = False
                if negatives_this_frame >= max(0, int(max_negatives_per_frame)):
                    continue
                if "fallback" in cand.source and cand.objectness >= artifact_score_threshold:
                    cls = "alignment_artifact"
                    diagnostic_bucket = "alignment_artifact"
                elif cand.objectness >= high_score_fp_threshold:
                    cls = "background"
                    diagnostic_bucket = "high_score_detector_fp"
                else:
                    cls = "background"
                    diagnostic_bucket = "easy_background"
                negatives_this_frame += 1
            repeats = max(1, int(hard_positive_repeat)) if is_hard_positive else 1
            for repeat_idx in range(repeats):
                write_proposal_sample(
                    cls,
                    diagnostic_bucket,
                    video,
                    frame_id,
                    frame,
                    cand.bbox_xyxy,
                    gt,
                    cand_idx * 100 + repeat_idx,
                    cand.objectness,
                    cand.source,
                    hard_positive=is_hard_positive,
                )
        if not matched_positive:
            counters["missed_drone"] += 1
            manifest.append(
                {
                    "class": "missed_drone",
                    "source_video": video,
                    "frame_id": frame_id,
                    "gt_bbox_xyxy": list(gt),
                    "num_proposals": len(proposals),
                    "matched_positive": False,
                }
            )

    for cap in video_cache.values():
        cap.release()
    feature_csv = feature_dir / "annotations.csv"
    with feature_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "frame_id", "x1", "y1", "x2", "y2", "class", "diagnostic_bucket", "proposal_source", "proposal_score"])
        writer.writeheader()
        writer.writerows(feature_rows)
    manifest_path = out / "proposal_manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(r) for r in manifest) + "\n", encoding="utf-8")
    summary = {
        "annotations_csv": str(annotations_csv),
        "yolo_weights": str(yolo_weights),
        "fallback_yolo_weights": str(fallback_yolo_weights) if fallback_yolo_weights is not None else None,
        "non_drone_label_mode": non_drone_label_mode,
        "artifact_score_threshold": artifact_score_threshold,
        "high_score_fp_threshold": high_score_fp_threshold,
        "hard_positive_max_size_px": hard_positive_max_size_px,
        "hard_positive_max_score": hard_positive_max_score,
        "hard_positive_repeat": hard_positive_repeat,
        "crop_root": str(crop_root),
        "temporal_root": str(temporal_root),
        "feature_csv": str(feature_csv),
        "manifest": str(manifest_path),
        "counts": counters,
        "diagnostic_bucket_counts": diagnostic_bucket_counts,
        "num_feature_rows": len(feature_rows),
        "num_manifest_rows": len(manifest),
    }
    summary_json = out / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return RealProposalStageBDatasetResult(crop_root, temporal_root, feature_csv, manifest_path, summary_json)


def extract_real_annotated_frames(
    annotations_csv: str | Path,
    frames_dir: str | Path,
    out_csv: str | Path,
    video_root: str | Path | None = None,
    strict_labels: bool = True,
) -> Path:
    """Convert real-video box annotations into frame-image annotations."""
    annotations_csv = Path(annotations_csv)
    frames_dir = Path(frames_dir)
    out_csv = Path(out_csv)
    video_root_p = Path(video_root) if video_root else None
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_real_rows(annotations_csv)
    grouped: dict[tuple[Path, int], list[dict[str, str]]] = {}
    for row in rows:
        cls = row["class"].strip()
        tag = row["tag"].strip()
        if strict_labels and cls not in REAL_CLASSES:
            raise ValueError(f"Unsupported class '{cls}'; expected one of {sorted(REAL_CLASSES)}")
        if strict_labels and tag not in REAL_TAGS:
            raise ValueError(f"Unsupported tag '{tag}'; expected one of {sorted(REAL_TAGS)}")
        video = _resolve_video_path(row["video_path"], video_root_p)
        frame_id = int(float(row["frame_id"]))
        grouped.setdefault((video, frame_id), []).append(row)

    out_rows: list[dict[str, str]] = []
    for (video, frame_id), frame_rows in sorted(grouped.items(), key=lambda item: (str(item[0][0]), item[0][1])):
        if not video.exists():
            raise FileNotFoundError(f"Video not found: {video}")
        frame_path = frames_dir / _frame_file_name(video, frame_id)
        if not frame_path.exists():
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video: {video}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise ValueError(f"Could not read frame {frame_id} from {video}")
            cv2.imwrite(str(frame_path), frame)
        for row in frame_rows:
            out_rows.append(
                {
                    "frame_path": frame_path.name,
                    "x1": row["x1"],
                    "y1": row["y1"],
                    "x2": row["x2"],
                    "y2": row["y2"],
                    "class": row["class"].strip(),
                    "tag": row["tag"].strip(),
                    "source_video": str(video),
                    "frame_id": str(frame_id),
                }
            )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_path", "x1", "y1", "x2", "y2", "class", "tag", "source_video", "frame_id"],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    return out_csv


def build_real_yolo_candidate_dataset(
    annotations_csv: str | Path,
    out: str | Path,
    video_root: str | Path | None = None,
    tiled: bool = True,
    tile_size: int = 256,
    positives_per_box: int = 2,
    negatives_per_image: int = 2,
    val_fraction: float = 0.2,
    seed: int = 11,
    min_box_px: float = 8.0,
    negative_pad_px: float = 32.0,
    strict_labels: bool = True,
) -> RealDatasetBuildResult:
    out = Path(out)
    frames_dir = out / "frames"
    frame_csv = out / "frame_annotations.csv"
    extract_real_annotated_frames(annotations_csv, frames_dir, frame_csv, video_root=video_root, strict_labels=strict_labels)
    if tiled:
        data_yaml = build_tiled_class_agnostic_yolo_dataset(
            frame_csv,
            out / "yolo_tiled",
            images_root=frames_dir,
            tile_size=tile_size,
            positives_per_box=positives_per_box,
            negatives_per_image=negatives_per_image,
            val_fraction=val_fraction,
            seed=seed,
            min_box_px=min_box_px,
            negative_pad_px=negative_pad_px,
        )
    else:
        data_yaml = build_class_agnostic_yolo_dataset(
            frame_csv,
            out / "yolo_full",
            images_root=frames_dir,
            val_fraction=val_fraction,
            seed=seed,
            min_box_px=min_box_px,
        )

    with frame_csv.open("r", encoding="utf-8") as f:
        frame_rows = list(csv.DictReader(f))
    summary = {
        "input_annotations": str(annotations_csv),
        "frames_dir": str(frames_dir),
        "frame_annotations_csv": str(frame_csv),
        "data_yaml": str(data_yaml),
        "tiled": tiled,
        "num_boxes": len(frame_rows),
        "num_frames": len({r["frame_path"] for r in frame_rows}),
        "classes": {c: sum(1 for r in frame_rows if r["class"] == c) for c in sorted({r["class"] for r in frame_rows})},
        "tags": {t: sum(1 for r in frame_rows if r["tag"] == t) for t in sorted({r["tag"] for r in frame_rows})},
    }
    summary_json = out / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return RealDatasetBuildResult(frame_annotations_csv=frame_csv, frames_dir=frames_dir, data_yaml=data_yaml, summary_json=summary_json)
