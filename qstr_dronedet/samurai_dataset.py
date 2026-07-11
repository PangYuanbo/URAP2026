"""Convert frame-level UAV boxes into single-object SAMURAI sequences."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class BoxObservation:
    sequence: str
    frame_id: int
    box: tuple[float, float, float, float]
    video_path: str

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def size(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (max(1.0, x2 - x1), max(1.0, y2 - y1))


@dataclass
class AssociatedTrack:
    source_sequence: str
    local_id: int
    observations: dict[int, BoxObservation] = field(default_factory=dict)

    @property
    def first_frame(self) -> int:
        return min(self.observations)

    @property
    def last_frame(self) -> int:
        return max(self.observations)

    @property
    def visible_frames(self) -> int:
        return len(self.observations)

    @property
    def span_frames(self) -> int:
        return self.last_frame - self.first_frame + 1

    @property
    def name(self) -> str:
        return f"{self.source_sequence}__track_{self.local_id:03d}"

    def add(self, observation: BoxObservation) -> None:
        self.observations[observation.frame_id] = observation

    def predict_center(self, frame_id: int) -> tuple[float, float]:
        frames = sorted(self.observations)
        last = self.observations[frames[-1]]
        cx, cy = last.center
        if len(frames) < 2:
            return cx, cy
        previous = self.observations[frames[-2]]
        dt = max(1, last.frame_id - previous.frame_id)
        vx = (cx - previous.center[0]) / dt
        vy = (cy - previous.center[1]) / dt
        horizon = frame_id - last.frame_id
        return cx + vx * horizon, cy + vy * horizon


def load_box_csv(path: str | Path) -> list[BoxObservation]:
    observations: list[BoxObservation] = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"seq", "frame_id", "x1", "y1", "x2", "y2", "video_path"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing GT columns: {sorted(missing)}")
        for row in reader:
            box = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            observations.append(BoxObservation(row["seq"], int(row["frame_id"]), box, row["video_path"]))
    return observations


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _match_cost(track: AssociatedTrack, observation: BoxObservation) -> float | None:
    latest = track.observations[track.last_frame]
    px, py = track.predict_center(observation.frame_id)
    ox, oy = observation.center
    distance = math.hypot(ox - px, oy - py)
    lw, lh = latest.size
    ow, oh = observation.size
    scale = max(lw, lh, ow, oh, 4.0)
    normalized_distance = distance / scale
    size_ratio = max(ow / lw, lw / ow, oh / lh, lh / oh)
    overlap = _iou(latest.box, observation.box)
    if normalized_distance > 8.0 or size_ratio > 3.0:
        return None
    return normalized_distance + 0.35 * math.log(size_ratio) + 0.25 * (1.0 - overlap)


def associate_tracks(observations: Iterable[BoxObservation], *, max_gap: int = 2) -> list[AssociatedTrack]:
    grouped: dict[str, dict[int, list[BoxObservation]]] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        grouped[observation.sequence][observation.frame_id].append(observation)
    result: list[AssociatedTrack] = []
    for sequence in sorted(grouped):
        active: list[AssociatedTrack] = []
        completed: list[AssociatedTrack] = []
        next_id = 1
        for frame_id in sorted(grouped[sequence]):
            still_active: list[AssociatedTrack] = []
            for track in active:
                (still_active if frame_id - track.last_frame <= max_gap + 1 else completed).append(track)
            active = still_active
            frame_observations = sorted(grouped[sequence][frame_id], key=lambda item: item.center)
            candidates: list[tuple[float, int, int]] = []
            for track_index, track in enumerate(active):
                for observation_index, observation in enumerate(frame_observations):
                    cost = _match_cost(track, observation)
                    if cost is not None:
                        candidates.append((cost, track_index, observation_index))
            used_tracks: set[int] = set()
            used_observations: set[int] = set()
            for _, track_index, observation_index in sorted(candidates):
                if track_index in used_tracks or observation_index in used_observations:
                    continue
                active[track_index].add(frame_observations[observation_index])
                used_tracks.add(track_index)
                used_observations.add(observation_index)
            for observation_index, observation in enumerate(frame_observations):
                if observation_index not in used_observations:
                    track = AssociatedTrack(sequence, next_id)
                    next_id += 1
                    track.add(observation)
                    active.append(track)
        completed.extend(active)
        result.extend(completed)
    return sorted(result, key=lambda item: (item.source_sequence, item.first_frame, item.local_id))


def select_tracks(tracks: Iterable[AssociatedTrack], *, min_visible_frames: int = 8, min_visibility: float = 0.5, max_sequences: int | None = None) -> list[AssociatedTrack]:
    selected = [track for track in tracks if track.visible_frames >= min_visible_frames and track.visible_frames / track.span_frames >= min_visibility]
    return selected if max_sequences is None else selected[:max_sequences]


def _source_frame_path(frames_root: Path, observation: BoxObservation, frame_id: int) -> Path:
    observed_name = Path(observation.video_path).name
    prefix = observed_name.rsplit("_", 1)[0]
    suffix = Path(observed_name).suffix or ".png"
    return frames_root / f"{prefix}_{frame_id:05d}{suffix}"


def _place_image(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "jpeg":
        with Image.open(source) as image:
            image.convert("RGB").save(destination, quality=95)
    else:
        raise ValueError(f"Unsupported image mode: {mode}")
    return mode


def _xywh(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return box[0], box[1], box[2] - box[0], box[3] - box[1]


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def export_samurai_dataset(tracks: Iterable[AssociatedTrack], *, frames_root: str | Path, output_root: str | Path, split: str, image_mode: str = "hardlink", write_vos: bool = True, progress_path: str | Path | None = None) -> dict[str, object]:
    frames_root, output_root = Path(frames_root), Path(output_root)
    tracks = list(tracks)
    progress_path = Path(progress_path) if progress_path else output_root / "progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress = {"status": "running", "done_sequences": 0, "total_sequences": len(tracks), "done_frames": 0, "last_update": datetime.now(timezone.utc).isoformat()}
    _write_json_atomic(progress_path, progress)
    lasot_root = output_root / "lasot" / "uav"
    vos_images_root = output_root / "vos" / "JPEGImages"
    vos_masks_root = output_root / "vos" / "Annotations"
    source_rows: list[dict[str, object]] = []
    sequence_names: list[str] = []
    total_frames = visible_frames = 0
    for track_index, track in enumerate(tracks, 1):
        sequence_names.append(track.name)
        sequence_root = lasot_root / track.name
        image_root = sequence_root / "img"
        image_root.mkdir(parents=True, exist_ok=True)
        if write_vos:
            (vos_masks_root / track.name).mkdir(parents=True, exist_ok=True)
        anchor = track.observations[track.first_frame]
        gt_lines: list[str] = []
        full_occlusion: list[str] = []
        out_of_view: list[str] = []
        for sequence_index, frame_id in enumerate(range(track.first_frame, track.last_frame + 1), 1):
            observation = track.observations.get(frame_id)
            source = _source_frame_path(frames_root, observation or anchor, frame_id)
            if not source.is_file():
                raise FileNotFoundError(f"Missing source frame: {source}")
            output_name = f"{sequence_index:08d}.jpg"
            link_type = _place_image(source, image_root / output_name, image_mode)
            if write_vos:
                _place_image(source, vos_images_root / track.name / output_name, image_mode)
            box = observation.box if observation else None
            if box is None:
                gt_lines.append("0,0,0,0")
                full_occlusion.append("1")
                out_of_view.append("1")
            else:
                visible_frames += 1
                gt_lines.append(",".join(f"{value:.3f}" for value in _xywh(box)))
                full_occlusion.append("0")
                out_of_view.append("0")
            if write_vos:
                with Image.open(source) as image:
                    mask = Image.new("L", image.size, 0)
                if box is not None:
                    ImageDraw.Draw(mask).rectangle((max(0, math.floor(box[0])), max(0, math.floor(box[1])), max(0, math.ceil(box[2]) - 1), max(0, math.ceil(box[3]) - 1)), fill=1)
                mask.save(vos_masks_root / track.name / f"{sequence_index:08d}.png")
            source_rows.append({"derived_sequence": track.name, "sequence_index": sequence_index, "source_sequence": track.source_sequence, "source_frame_id": frame_id, "source_path": str(source), "visible": int(observation is not None), "image_storage": link_type})
            total_frames += 1
        (sequence_root / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="ascii")
        (sequence_root / "full_occlusion.txt").write_text(",".join(full_occlusion) + "\n", encoding="ascii")
        (sequence_root / "out_of_view.txt").write_text(",".join(out_of_view) + "\n", encoding="ascii")
        progress.update({"done_sequences": track_index, "done_frames": total_frames, "last_completed_sequence": track.name, "last_update": datetime.now(timezone.utc).isoformat()})
        _write_json_atomic(progress_path, progress)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{split}_set.txt").write_text("\n".join(sequence_names) + "\n", encoding="ascii")
    with (output_root / "source_map.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(source_rows[0]) if source_rows else ["derived_sequence"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(source_rows)
    manifest = {"format_version": 1, "split": split, "sequence_count": len(sequence_names), "frame_count": total_frames, "visible_frame_count": visible_frames, "occluded_frame_count": total_frames - visible_frames, "image_mode": image_mode, "vos_masks": "weak rectangular masks generated from bounding boxes" if write_vos else None, "first_frame_prompt": "groundtruth row 1, xywh"}
    _write_json_atomic(output_root / "manifest.json", manifest)
    progress.update({"status": "completed", "last_update": datetime.now(timezone.utc).isoformat()})
    _write_json_atomic(progress_path, progress)
    return manifest


def validate_samurai_dataset(output_root: str | Path, *, split: str) -> dict[str, int]:
    output_root = Path(output_root)
    names = [line.strip() for line in (output_root / f"{split}_set.txt").read_text().splitlines() if line.strip()]
    frames = visible = 0
    for name in names:
        root = output_root / "lasot" / "uav" / name
        images = sorted((root / "img").glob("*.jpg"))
        gt = [line.strip() for line in (root / "groundtruth.txt").read_text().splitlines() if line.strip()]
        if not images or len(images) != len(gt):
            raise ValueError(f"Sequence length mismatch: {name}")
        first = [float(value) for value in gt[0].split(",")]
        if len(first) != 4 or first[2] <= 0 or first[3] <= 0:
            raise ValueError(f"First-frame prompt is invalid: {name}")
        with Image.open(images[0]) as image:
            image.verify()
        frames += len(images)
        visible += sum(1 for line in gt if not line.startswith("0,0,0,0"))
    return {"sequences": len(names), "frames": frames, "visible_frames": visible}
