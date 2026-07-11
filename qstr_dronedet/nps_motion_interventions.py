from __future__ import annotations

import json
import hashlib
import math
import os
import pickle
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


INTERVENTIONS = ("original", "slow_0p5", "fast_2x", "accelerate_g2", "decelerate_g2")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _files_equivalent(left: Path, right: Path) -> bool:
    if not left.exists() or not right.exists() or left.stat().st_size != right.stat().st_size:
        return False
    if os.path.samefile(left, right):
        return True
    left_hash = hashlib.sha256()
    right_hash = hashlib.sha256()
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                break
            left_hash.update(left_chunk)
            right_hash.update(right_chunk)
    return left_hash.digest() == right_hash.digest()


@dataclass(frozen=True)
class FlowQuality:
    median_error: float
    bad_ratio: float
    valid: bool


@dataclass(frozen=True)
class LabelInterpolation:
    boxes: list[list[float]]
    valid: bool
    reason: str | None


def parse_yolo_labels(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    boxes: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        row = [float(value) for value in parts[:5]]
        row[0] = int(row[0])
        boxes.append(row)
    return boxes


def write_yolo_labels(path: Path, boxes: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box in boxes:
        class_id, cx, cy, width, height = box[:5]
        cx = min(max(float(cx), 0.0), 1.0)
        cy = min(max(float(cy), 0.0), 1.0)
        width = min(max(float(width), 0.0), 1.0)
        height = min(max(float(height), 0.0), 1.0)
        lines.append(f"{int(class_id)} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def yolo_iou(left: Sequence[float], right: Sequence[float]) -> float:
    def corners(box: Sequence[float]) -> tuple[float, float, float, float]:
        _, cx, cy, width, height = box[:5]
        return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2

    ax1, ay1, ax2, ay2 = corners(left)
    bx1, by1, bx2, by2 = corners(right)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def interpolate_boxes(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    alpha: float,
    max_cost: float = 0.75,
    ambiguity_margin: float = 0.04,
) -> LabelInterpolation:
    if len(left) != len(right):
        return LabelInterpolation([], False, "target_count_changed")
    if not left:
        return LabelInterpolation([], True, None)
    cost = np.full((len(left), len(right)), 1e6, dtype=np.float32)
    for row_index, left_box in enumerate(left):
        for column_index, right_box in enumerate(right):
            if int(left_box[0]) != int(right_box[0]):
                continue
            center_distance = math.hypot(float(left_box[1]) - float(right_box[1]), float(left_box[2]) - float(right_box[2]))
            size_distance = abs(math.log(max(float(left_box[3]), 1e-6) / max(float(right_box[3]), 1e-6)))
            size_distance += abs(math.log(max(float(left_box[4]), 1e-6) / max(float(right_box[4]), 1e-6)))
            cost[row_index, column_index] = 0.55 * center_distance + 0.25 * min(size_distance, 2.0) + 0.20 * (1.0 - yolo_iou(left_box, right_box))
    rows, columns = linear_sum_assignment(cost)
    selected = cost[rows, columns]
    if np.any(selected >= max_cost):
        return LabelInterpolation([], False, "match_cost_too_high")
    if len(left) > 1:
        for row_index, column_index in zip(rows, columns):
            alternatives = np.delete(cost[row_index], column_index)
            if alternatives.size and float(alternatives.min() - cost[row_index, column_index]) < ambiguity_margin:
                return LabelInterpolation([], False, "ambiguous_target_match")
    output: list[list[float]] = []
    for row_index, column_index in zip(rows, columns):
        left_box = left[row_index]
        right_box = right[column_index]
        output.append(
            [
                int(left_box[0]),
                *[
                    min(max(float(left_box[index]) + (float(right_box[index]) - float(left_box[index])) * alpha, 0.0), 1.0)
                    for index in range(1, 5)
                ],
            ]
        )
    output.sort(key=lambda box: (int(box[0]), float(box[1]), float(box[2])))
    return LabelInterpolation(output, True, None)


def make_time_map(intervention: str, frame_count: int) -> np.ndarray:
    if frame_count < 1:
        return np.zeros((0,), dtype=np.float64)
    if frame_count == 1:
        return np.zeros((1,), dtype=np.float64)
    if intervention == "original":
        return np.arange(frame_count, dtype=np.float64)
    if intervention == "slow_0p5":
        return np.arange(frame_count * 2 - 1, dtype=np.float64) / 2.0
    if intervention == "fast_2x":
        values = np.arange(0, frame_count, 2, dtype=np.float64)
        if values[-1] != frame_count - 1:
            values = np.append(values, frame_count - 1)
        return values
    normalized = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    if intervention == "accelerate_g2":
        return normalized**2 * (frame_count - 1)
    if intervention == "decelerate_g2":
        return (1.0 - (1.0 - normalized) ** 2) * (frame_count - 1)
    raise ValueError(f"Unknown intervention: {intervention}")


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


class DISFrameInterpolator:
    def __init__(self, scale: float = 0.5, median_threshold: float = 3.0, bad_ratio_threshold: float = 0.25):
        if not hasattr(cv2, "DISOpticalFlow_create"):
            raise RuntimeError("OpenCV DIS optical flow is unavailable")
        self.scale = float(scale)
        self.median_threshold = float(median_threshold)
        self.bad_ratio_threshold = float(bad_ratio_threshold)
        self.engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def _flow(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        if self.scale != 1.0:
            size = (max(16, int(source_gray.shape[1] * self.scale)), max(16, int(source_gray.shape[0] * self.scale)))
            source_gray = cv2.resize(source_gray, size, interpolation=cv2.INTER_AREA)
            target_gray = cv2.resize(target_gray, size, interpolation=cv2.INTER_AREA)
        flow = self.engine.calc(source_gray, target_gray, None)
        if self.scale != 1.0:
            flow = cv2.resize(flow, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
            flow /= self.scale
        return flow.astype(np.float32)

    @staticmethod
    def _remap(image: np.ndarray, flow: np.ndarray, fraction: float) -> np.ndarray:
        height, width = image.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        map_x = grid_x - flow[..., 0] * float(fraction)
        map_y = grid_y - flow[..., 1] * float(fraction)
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    @staticmethod
    def _quality(forward: np.ndarray, backward: np.ndarray, median_threshold: float, bad_ratio_threshold: float) -> FlowQuality:
        height, width = forward.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        sample_x = grid_x + forward[..., 0]
        sample_y = grid_y + forward[..., 1]
        backward_at_forward = cv2.remap(backward, sample_x, sample_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        error = np.linalg.norm(forward + backward_at_forward, axis=2)
        magnitude = np.linalg.norm(forward, axis=2) + np.linalg.norm(backward_at_forward, axis=2)
        threshold = 1.0 + 0.05 * magnitude
        median_error = float(np.median(error))
        bad_ratio = float(np.mean(error > threshold))
        return FlowQuality(median_error, bad_ratio, median_error <= median_threshold and bad_ratio <= bad_ratio_threshold)

    def interpolate(self, left: np.ndarray, right: np.ndarray, alpha: float) -> tuple[np.ndarray, FlowQuality]:
        forward = self._flow(left, right)
        backward = self._flow(right, left)
        quality = self._quality(forward, backward, self.median_threshold, self.bad_ratio_threshold)
        left_warped = self._remap(left, forward, alpha)
        right_warped = self._remap(right, backward, 1.0 - alpha)
        return cv2.addWeighted(left_warped, 1.0 - alpha, right_warped, alpha, 0.0), quality


def source_label_path(labels_dir: Path, clip_name: str, source_frame_id: int) -> Path:
    return labels_dir / f"{clip_name}_{source_frame_id - 1:05d}.txt"


def output_paths(intervention_root: Path, split: str, clip_name: str, output_frame_id: int) -> dict[str, Path]:
    image_name = f"{clip_name}_{output_frame_id:05d}.png"
    return {
        "tvd_image": intervention_root / "TransVisDrone" / "AllFrames" / split / image_name,
        "tvd_label": intervention_root / "TransVisDrone" / "NPSvisdroneStyle" / split / "labels" / f"{clip_name}_{output_frame_id - 1:05d}.txt",
        "yolomg_image": intervention_root / "YOLOMG" / "images" / split / image_name,
        "yolomg_motion": intervention_root / "YOLOMG" / "images2" / split / image_name,
        "yolomg_label": intervention_root / "YOLOMG" / "labels" / split / image_name.replace(".png", ".txt"),
    }


def _remove_partial_clip(intervention_root: Path, split: str, clip_name: str) -> None:
    directories = (
        intervention_root / "TransVisDrone" / "AllFrames" / split,
        intervention_root / "TransVisDrone" / "NPSvisdroneStyle" / split / "labels",
        intervention_root / "YOLOMG" / "images" / split,
        intervention_root / "YOLOMG" / "images2" / split,
        intervention_root / "YOLOMG" / "labels" / split,
    )
    for directory in directories:
        if directory.exists():
            for path in directory.glob(f"{clip_name}_*"):
                path.unlink()


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def _write_audit(intervention_root: Path, split: str, clip_name: str, records: Sequence[dict], seed: int) -> None:
    if not records:
        return
    motion_scores = [0.0]
    for previous, current in zip(records, records[1:]):
        motion_scores.append(abs(float(current["source_position"]) - float(previous["source_position"])))
    random_index = random.Random(seed + int(clip_name.split("_")[-1])).randrange(len(records))
    indices = sorted({0, len(records) - 1, int(np.argmax(motion_scores)), random_index})
    panels = []
    for index in indices:
        record = records[index]
        image = _read_image(Path(record["output_image"]))
        labels = parse_yolo_labels(Path(record["output_label"]))
        height, width = image.shape[:2]
        for _, cx, cy, box_width, box_height in labels:
            x1 = int((cx - box_width / 2) * width)
            y1 = int((cy - box_height / 2) * height)
            x2 = int((cx + box_width / 2) * width)
            y2 = int((cy + box_height / 2) * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        title = f"out={record['output_frame_id']} src={record['source_position']:.2f} synth={record['synthetic']}"
        cv2.putText(image, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        target_width = 640
        scale = target_width / image.shape[1]
        panels.append(cv2.resize(image, (target_width, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA))
    audit = np.vstack(panels)
    audit_path = intervention_root / "audit" / split / f"{clip_name}.jpg"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(audit_path), audit)


def build_clip(
    source_frames_dir: Path,
    source_labels_dir: Path,
    intervention_root: Path,
    split: str,
    clip_name: str,
    intervention: str,
    interpolator: DISFrameInterpolator,
    motion_threshold: int = 16,
    max_frames: int | None = None,
    seed: int = 59,
) -> dict:
    source_frames = sorted(path for path in source_frames_dir.glob(f"{clip_name}_*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if max_frames is not None:
        source_frames = source_frames[: max(0, int(max_frames))]
    if not source_frames:
        raise FileNotFoundError(f"No source frames found for {clip_name} in {source_frames_dir}")
    marker = intervention_root / "clips" / split / f"{clip_name}.complete.json"
    if marker.exists():
        summary = json.loads(marker.read_text(encoding="utf-8"))
        if int(summary.get("source_frames", -1)) == len(source_frames) and summary.get("intervention") == intervention:
            return summary
    _remove_partial_clip(intervention_root, split, clip_name)
    mapping = make_time_map(intervention, len(source_frames))
    records: list[dict] = []
    fallback_count = 0
    synthetic_count = 0
    previous_output: np.ndarray | None = None
    for output_index, source_position in enumerate(mapping, start=1):
        left_index = min(max(int(math.floor(float(source_position) + 1e-9)), 0), len(source_frames) - 1)
        right_index = min(max(int(math.ceil(float(source_position) - 1e-9)), 0), len(source_frames) - 1)
        alpha = float(source_position - left_index) if right_index != left_index else 0.0
        paths = output_paths(intervention_root, split, clip_name, output_index)
        left_source = source_frames[left_index]
        right_source = source_frames[right_index]
        left_labels = parse_yolo_labels(source_label_path(source_labels_dir, clip_name, left_index + 1))
        right_labels = parse_yolo_labels(source_label_path(source_labels_dir, clip_name, right_index + 1))
        fallback_reason = None
        label_mode = "anchor"
        flow_quality = None
        synthetic = right_index != left_index and alpha > 1e-8
        if not synthetic:
            storage_mode = link_or_copy(left_source, paths["tvd_image"])
            output_image = _read_image(left_source)
            output_labels = left_labels
        else:
            label_result = interpolate_boxes(left_labels, right_labels, alpha)
            output_image, flow_quality = interpolator.interpolate(_read_image(left_source), _read_image(right_source), alpha)
            if not flow_quality.valid:
                fallback_reason = "flow_inconsistent"
            elif not label_result.valid:
                fallback_reason = label_result.reason
            if fallback_reason:
                nearest_index = left_index if alpha < 0.5 else right_index
                nearest_source = source_frames[nearest_index]
                storage_mode = link_or_copy(nearest_source, paths["tvd_image"])
                output_image = _read_image(nearest_source)
                output_labels = parse_yolo_labels(source_label_path(source_labels_dir, clip_name, nearest_index + 1))
                label_mode = "nearest_anchor_fallback"
                fallback_count += 1
            else:
                paths["tvd_image"].parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(paths["tvd_image"]), output_image):
                    raise RuntimeError(f"Failed to write interpolated frame: {paths['tvd_image']}")
                storage_mode = "synthesized"
                output_labels = label_result.boxes
                label_mode = "linear_matched"
                synthetic_count += 1
        write_yolo_labels(paths["tvd_label"], output_labels)
        link_or_copy(paths["tvd_image"], paths["yolomg_image"])
        write_yolo_labels(paths["yolomg_label"], output_labels)
        if previous_output is None:
            motion = np.zeros_like(output_image)
        else:
            motion = cv2.absdiff(output_image, previous_output)
            if motion_threshold > 0:
                gray = cv2.cvtColor(motion, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, motion_threshold, 255, cv2.THRESH_BINARY)
                motion = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        paths["yolomg_motion"].parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(paths["yolomg_motion"]), motion):
            raise RuntimeError(f"Failed to write motion image: {paths['yolomg_motion']}")
        previous_output = output_image
        records.append(
            {
                "intervention": intervention,
                "split": split,
                "clip": clip_name,
                "output_frame_id": output_index,
                "source_position": float(source_position + 1.0),
                "source_left_frame_id": left_index + 1,
                "source_right_frame_id": right_index + 1,
                "alpha": alpha,
                "synthetic": synthetic and fallback_reason is None,
                "storage_mode": storage_mode,
                "label_mode": label_mode,
                "fallback_reason": fallback_reason,
                "flow_quality": asdict(flow_quality) if flow_quality else None,
                "source_left_image": str(left_source),
                "source_right_image": str(right_source),
                "source_left_label": str(source_label_path(source_labels_dir, clip_name, left_index + 1)),
                "source_right_label": str(source_label_path(source_labels_dir, clip_name, right_index + 1)),
                "output_image": str(paths["tvd_image"]),
                "output_label": str(paths["tvd_label"]),
                "yolomg_image": str(paths["yolomg_image"]),
                "yolomg_motion": str(paths["yolomg_motion"]),
                "yolomg_label": str(paths["yolomg_label"]),
            }
        )
    manifest_path = intervention_root / "manifests" / split / f"{clip_name}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    _write_audit(intervention_root, split, clip_name, records, seed)
    summary = {
        "intervention": intervention,
        "split": split,
        "clip": clip_name,
        "source_frames": len(source_frames),
        "output_frames": len(records),
        "synthetic_frames": synthetic_count,
        "fallback_frames": fallback_count,
        "fallback_rate": fallback_count / max(1, synthetic_count + fallback_count),
        "last_output": str(records[-1]["output_image"]),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def discover_clips(frames_dir: Path) -> list[str]:
    clips = {"_".join(path.stem.split("_")[:2]) for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    return sorted(clips, key=lambda value: int(value.split("_")[-1]))


def write_dataset_metadata(intervention_root: Path, intervention: str, split_lengths: dict[str, dict[int, int]]) -> None:
    tvd_root = intervention_root / "TransVisDrone"
    for split, lengths in split_lengths.items():
        video_dir = tvd_root / "Videos" / split
        video_dir.mkdir(parents=True, exist_ok=True)
        with (video_dir / "video_length_dict.pkl").open("wb") as handle:
            pickle.dump(lengths, handle)
    available = sorted(split_lengths)
    fallback = "test" if "test" in available else available[0]
    (intervention_root / f"{intervention}_tvd.yaml").write_text(
        "\n".join(
            [
                f"path: {tvd_root / 'AllFrames'}",
                f"train: {'train' if 'train' in available else fallback}",
                f"val: {'val' if 'val' in available else fallback}",
                f"test: {'test' if 'test' in available else fallback}",
                f"inference: {'test' if 'test' in available else fallback}",
                f"annotation_path: {tvd_root / 'NPSvisdroneStyle'}",
                f"annotation_train: {'train' if 'train' in available else fallback}/labels",
                f"annotation_val: {'val' if 'val' in available else fallback}/labels",
                f"annotation_test: {'test' if 'test' in available else fallback}/labels",
                f"video_root_path: {tvd_root / 'Videos'}",
                f"video_root_path_train: {'train' if 'train' in available else fallback}",
                f"video_root_path_val: {'val' if 'val' in available else fallback}",
                f"video_root_path_test: {'test' if 'test' in available else fallback}",
                f"video_root_path_inference: {'test' if 'test' in available else fallback}",
                "nc: 1",
                "names: ['drone']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    yolomg_root = intervention_root / "YOLOMG"
    lists: dict[str, tuple[Path, Path]] = {}
    for split in available:
        images = sorted((yolomg_root / "images" / split).glob("*.png"))
        image_list = yolomg_root / f"{split}.txt"
        motion_list = yolomg_root / f"{split}2.txt"
        image_list.write_text("".join(str(path) + "\n" for path in images), encoding="utf-8")
        motion_list.write_text("".join(str(yolomg_root / "images2" / split / path.name) + "\n" for path in images), encoding="utf-8")
        lists[split] = (image_list, motion_list)
    fallback_pair = lists[fallback]
    lines = []
    for split in ("train", "val", "test"):
        image_list, motion_list = lists.get(split, fallback_pair)
        lines.extend((f"{split}: {image_list}", f"{split}2: {motion_list}"))
    lines.extend(("nc: 1", "names: ['drone']", ""))
    (intervention_root / f"{intervention}_yolomg.yaml").write_text("\n".join(lines), encoding="utf-8")


def _validate_output_image(intervention_root: Path, split: str, image: Path) -> list[str]:
    errors = []
    clip_name, frame_text = image.stem.rsplit("_", 1)
    frame_id = int(frame_text)
    paths = output_paths(intervention_root, split, clip_name, frame_id)
    if cv2.imread(str(image), cv2.IMREAD_COLOR) is None:
        errors.append(f"unreadable_image:{image}")
    for required in (paths["tvd_label"], paths["yolomg_image"], paths["yolomg_motion"], paths["yolomg_label"]):
        if not required.exists():
            errors.append(f"missing:{required}")
    if frame_id == 1 and paths["yolomg_motion"].exists():
        motion = cv2.imread(str(paths["yolomg_motion"]), cv2.IMREAD_GRAYSCALE)
        if motion is None or np.any(motion):
            errors.append(f"nonzero_first_motion:{paths['yolomg_motion']}")
    return errors


def validate_intervention(intervention_root: Path, intervention: str, splits: Iterable[str], workers: int = 16) -> dict:
    errors: list[str] = []
    split_summaries: dict[str, dict] = {}
    total_frames = 0
    total_labels = 0
    total_fallbacks = 0
    total_synthetic_candidates = 0
    per_clip_fallback: dict[str, float] = {}
    original_equivalence_checked = 0
    original_equivalence_failures = 0
    progress_path = intervention_root / "integrity_progress.json"
    split_names = list(splits)
    expected_frames = sum(len(list((intervention_root / "TransVisDrone" / "AllFrames" / split).glob("*.png"))) for split in split_names)
    checked_frames = 0
    progress_path.write_text(json.dumps({"checked": 0, "total": expected_frames, "split": None}, indent=2), encoding="utf-8")
    for split in split_names:
        frames_dir = intervention_root / "TransVisDrone" / "AllFrames" / split
        labels_dir = intervention_root / "TransVisDrone" / "NPSvisdroneStyle" / split / "labels"
        images = sorted(frames_dir.glob("*.png"))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for image, image_errors in zip(images, executor.map(lambda item: _validate_output_image(intervention_root, split, item), images)):
                errors.extend(image_errors)
                checked_frames += 1
                if checked_frames % 500 == 0 or checked_frames == expected_frames:
                    progress_path.write_text(json.dumps({"checked": checked_frames, "total": expected_frames, "split": split, "last_image": str(image)}, indent=2), encoding="utf-8")
        markers = sorted((intervention_root / "clips" / split).glob("*.complete.json"))
        summaries = [json.loads(path.read_text(encoding="utf-8")) for path in markers]
        for summary in summaries:
            per_clip_fallback[f"{split}/{summary['clip']}"] = float(summary["fallback_rate"])
            total_fallbacks += int(summary["fallback_frames"])
            total_synthetic_candidates += int(summary["synthetic_frames"]) + int(summary["fallback_frames"])
        if intervention == "original":
            for manifest_path in sorted((intervention_root / "manifests" / split).glob("*.jsonl")):
                for raw in manifest_path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(raw)
                    original_equivalence_checked += 1
                    source_image = Path(record["source_left_image"])
                    output_image = Path(record["output_image"])
                    same_image = _files_equivalent(source_image, output_image)
                    source_labels = parse_yolo_labels(Path(record["source_left_label"]))
                    output_labels = parse_yolo_labels(Path(record["output_label"]))
                    if not same_image or source_labels != output_labels:
                        original_equivalence_failures += 1
                        errors.append(f"original_mismatch:{output_image}")
        total_frames += len(images)
        label_count = len(list(labels_dir.glob("*.txt")))
        total_labels += label_count
        split_summaries[split] = {"frames": len(images), "labels": label_count, "clips": len(markers)}
    fallback_rate = total_fallbacks / max(1, total_synthetic_candidates)
    result = {
        "intervention": intervention,
        "valid": not errors,
        "errors": errors[:200],
        "total_frames": total_frames,
        "total_labels": total_labels,
        "fallback_frames": total_fallbacks,
        "synthetic_candidates": total_synthetic_candidates,
        "fallback_rate": fallback_rate,
        "per_clip_fallback_rate": per_clip_fallback,
        "sensitivity_only": fallback_rate > 0.10 or any(rate > 0.20 for rate in per_clip_fallback.values()),
        "original_equivalence_checked": original_equivalence_checked,
        "original_equivalence_failures": original_equivalence_failures,
        "original_source_equivalent": intervention != "original" or (original_equivalence_checked > 0 and original_equivalence_failures == 0),
        "splits": split_summaries,
    }
    (intervention_root / "integrity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
