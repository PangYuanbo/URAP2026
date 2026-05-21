from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from qstr_dronedet.types import CLASSES


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class CropFolderDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 128) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []
        for idx, cls in enumerate(CLASSES):
            for p in (self.root / cls).glob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    self.samples.append((p, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1), torch.tensor(label, dtype=torch.long)


class TemporalFolderDataset(Dataset):
    def __init__(self, root: str | Path, T: int = 5, image_size: int = 96) -> None:
        self.root = Path(root)
        self.T = T
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []
        for idx, cls in enumerate(CLASSES):
            cls_dir = self.root / cls
            if cls_dir.exists():
                for p in cls_dir.iterdir():
                    if p.is_dir():
                        self.samples.append((p, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        d, label = self.samples[idx]
        files = sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])
        if not files:
            raise FileNotFoundError(d)
        if len(files) < self.T:
            files = [files[0]] * (self.T - len(files)) + files
        files = files[-self.T:]
        frames = []
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(p)
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(np.transpose(img, (2, 0, 1)))
        return torch.from_numpy(np.stack(frames, axis=0)), torch.tensor(label, dtype=torch.long)


class FrameBoxCSVDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_size: int = 640) -> None:
        import csv

        self.csv_path = Path(csv_path)
        self.image_size = image_size
        with self.csv_path.open("r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        row = self.rows[idx]
        path = Path(row["frame_path"])
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        h0, w0 = img.shape[:2]
        scale_x = self.image_size / max(1, w0)
        scale_y = self.image_size / max(1, h0)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        box = torch.tensor(
            [
                float(row["x1"]) * scale_x,
                float(row["y1"]) * scale_y,
                float(row["x2"]) * scale_x,
                float(row["y2"]) * scale_y,
            ],
            dtype=torch.float32,
        )
        cls = row.get("class", "unknown")
        label = CLASSES.index(cls) if cls in CLASSES else 0
        return torch.from_numpy(img).permute(2, 0, 1), box, torch.tensor(label, dtype=torch.long), (self.image_size, self.image_size)
