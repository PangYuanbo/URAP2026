from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FrameRef:
    seq: str
    frame_id: int
    path: Path


def _parse_frame(path: Path) -> FrameRef | None:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] != "Clip":
        return None
    try:
        frame_id = int(parts[-1])
    except ValueError:
        return None
    return FrameRef(seq=f"{parts[0]}_{parts[1]}", frame_id=frame_id, path=path)


def load_gt_csv(paths: list[Path]) -> dict[tuple[str, int], list[list[float]]]:
    out: dict[tuple[str, int], list[list[float]]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                seq = str(row["seq"])
                frame_id = int(float(row["frame_id"]))
                box = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
                out.setdefault((seq, frame_id), []).append(box)
    return out


def xyxy_to_cxcywh_norm(box: list[float], image_w: int, image_h: int) -> list[float]:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) * 0.5) / max(1, image_w)
    cy = ((y1 + y2) * 0.5) / max(1, image_h)
    w = max(0.0, x2 - x1) / max(1, image_w)
    h = max(0.0, y2 - y1) / max(1, image_h)
    return [cx, cy, w, h]


def hflip_cxcywh_norm(boxes: list[list[float]]) -> list[list[float]]:
    return [[1.0 - box[0], box[1], box[2], box[3]] for box in boxes]


class NPSClipDataset(Dataset):
    """NPS video clips for the native video detector MVP.

    The model receives a causal clip ending at the current frame and predicts
    current-frame boxes plus optional future box chunks.
    """

    def __init__(
        self,
        frames_dir: str | Path,
        gt_csv: str | Path | list[str | Path],
        clip_len: int = 8,
        future_len: int = 4,
        image_size: int = 320,
        max_samples: int | None = None,
        stride: int = 1,
        cache_dir: str | Path | None = None,
        augment_hflip_prob: float = 0.0,
        augment_brightness: float = 0.0,
        augment_contrast: float = 0.0,
    ) -> None:
        self.frames_dir = Path(frames_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        gt_paths = [Path(gt_csv)] if isinstance(gt_csv, (str, Path)) else [Path(p) for p in gt_csv]
        self.gt = load_gt_csv(gt_paths)
        self.clip_len = int(clip_len)
        self.future_len = int(future_len)
        self.image_size = int(image_size)
        self.stride = max(1, int(stride))
        self.augment_hflip_prob = max(0.0, min(1.0, float(augment_hflip_prob)))
        self.augment_brightness = max(0.0, float(augment_brightness))
        self.augment_contrast = max(0.0, float(augment_contrast))
        refs = [_parse_frame(path) for path in sorted(self.frames_dir.glob("Clip_*_*.png"))]
        self.frames = [ref for ref in refs if ref is not None]
        by_seq: dict[str, list[FrameRef]] = {}
        for ref in self.frames:
            by_seq.setdefault(ref.seq, []).append(ref)
        self.by_seq = {seq: sorted(items, key=lambda item: item.frame_id) for seq, items in by_seq.items()}
        self.index: list[tuple[str, int]] = []
        for seq, items in self.by_seq.items():
            for idx in range(0, len(items), self.stride):
                self.index.append((seq, idx))
        if max_samples is not None and max_samples > 0:
            self.index = self.index[: int(max_samples)]

    def __len__(self) -> int:
        return len(self.index)

    def _read_image(self, path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{path.stem}.pt"
            if cache_path.exists():
                item = torch.load(cache_path, map_location="cpu")
                tensor_raw = item.get("tensor")
                if isinstance(tensor_raw, torch.Tensor) and tensor_raw.ndim == 3:
                    _, cached_h, cached_w = tensor_raw.shape
                    if int(cached_h) == self.image_size and int(cached_w) == self.image_size:
                        tensor = tensor_raw.to(dtype=torch.float32) / 255.0
                        image_size = item["image_size"]
                        return tensor, (int(image_size[0]), int(image_size[1]))
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, (orig_w, orig_h)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq, center_idx = self.index[idx]
        items = self.by_seq[seq]
        center = items[center_idx]
        clip_tensors = []
        history_boxes: list[list[list[float]]] = []
        history_frame_ids: list[int] = []
        orig_size = (0, 0)
        for offset in range(self.clip_len - 1, -1, -1):
            src_idx = max(0, center_idx - offset)
            ref = items[src_idx]
            tensor, orig_size = self._read_image(ref.path)
            clip_tensors.append(tensor)
            history_frame_ids.append(ref.frame_id)
            frame_w, frame_h = orig_size
            history_boxes.append([
                xyxy_to_cxcywh_norm(box, frame_w, frame_h)
                for box in self.gt.get((seq, ref.frame_id), [])
            ])
        image_w, image_h = orig_size
        current_boxes = [
            xyxy_to_cxcywh_norm(box, image_w, image_h)
            for box in self.gt.get((seq, center.frame_id), [])
        ]
        future_boxes: list[list[list[float]]] = []
        for step in range(self.future_len + 1):
            future_idx = min(len(items) - 1, center_idx + step)
            ref = items[future_idx]
            future_boxes.append([
                xyxy_to_cxcywh_norm(box, image_w, image_h)
                for box in self.gt.get((seq, ref.frame_id), [])
            ])
        if self.augment_hflip_prob > 0.0 and random.random() < self.augment_hflip_prob:
            clip_tensors = [torch.flip(tensor, dims=[2]) for tensor in clip_tensors]
            history_boxes = [hflip_cxcywh_norm(boxes) for boxes in history_boxes]
            current_boxes = hflip_cxcywh_norm(current_boxes)
            future_boxes = [hflip_cxcywh_norm(boxes) for boxes in future_boxes]
        clip = torch.stack(clip_tensors, dim=0)
        if self.augment_brightness > 0.0:
            factor = random.uniform(max(0.0, 1.0 - self.augment_brightness), 1.0 + self.augment_brightness)
            clip = clip * factor
        if self.augment_contrast > 0.0:
            factor = random.uniform(max(0.0, 1.0 - self.augment_contrast), 1.0 + self.augment_contrast)
            clip = (clip - 0.5) * factor + 0.5
        clip = clip.clamp(0.0, 1.0)
        return {
            "clip": clip,
            "boxes": torch.tensor(current_boxes, dtype=torch.float32),
            "history_boxes": [torch.tensor(boxes, dtype=torch.float32) for boxes in history_boxes],
            "history_frame_ids": history_frame_ids,
            "future_boxes": [torch.tensor(boxes, dtype=torch.float32) for boxes in future_boxes],
            "seq": seq,
            "frame_id": center.frame_id,
            "image_id": center.path.stem,
            "image_path": str(center.path),
            "image_size": torch.tensor([image_w, image_h], dtype=torch.float32),
        }


def collate_nps_clips(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "clip": torch.stack([item["clip"] for item in batch], dim=0),
        "boxes": [item["boxes"] for item in batch],
        "history_boxes": [item["history_boxes"] for item in batch],
        "history_frame_ids": [item["history_frame_ids"] for item in batch],
        "future_boxes": [item["future_boxes"] for item in batch],
        "seq": [item["seq"] for item in batch],
        "frame_id": [item["frame_id"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "image_size": torch.stack([item["image_size"] for item in batch], dim=0),
    }
