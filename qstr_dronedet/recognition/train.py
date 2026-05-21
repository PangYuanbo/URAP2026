from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from qstr_dronedet.recognition.crop_recognizer import CropRecognizer
from qstr_dronedet.recognition.dataset import CropFolderDataset, FrameBoxCSVDataset, TemporalFolderDataset
from qstr_dronedet.recognition.feature_recognizer import FeatureRecognitionModel
from qstr_dronedet.recognition.temporal_recognizer import TemporalRecognizer
from qstr_dronedet.types import CLASSES


DRONE_CLASS_INDEX = CLASSES.index("drone")
BACKGROUND_CLASS_INDEX = CLASSES.index("background")


def _mapped_label(label: int, target_mode: str) -> int:
    if target_mode == "drone_binary":
        return int(label != DRONE_CLASS_INDEX)
    return int(label)


def _class_weights(dataset, num_classes: int, target_mode: str = "multiclass") -> torch.Tensor | None:
    samples = getattr(dataset, "samples", None)
    if not samples:
        return None
    if target_mode == "drone_binary":
        num_classes = 2
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, label in samples:
        counts[_mapped_label(int(label), target_mode)] += 1.0
    present = counts > 0
    if int(present.sum().item()) <= 1:
        return None
    weights = torch.zeros_like(counts)
    weights[present] = counts[present].sum() / (counts[present] * present.sum())
    return weights


def _balanced_sampler(dataset, target_mode: str = "multiclass") -> WeightedRandomSampler | None:
    samples = getattr(dataset, "samples", None)
    if not samples:
        return None
    labels = [_mapped_label(int(label), target_mode) for _, label in samples]
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    if len(counts) <= 1:
        return None
    weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _train(
    model: torch.nn.Module,
    dataset,
    out: str | Path,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    balance: str = "sampler",
    target_mode: str = "multiclass",
) -> Path:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; expected class subfolders with images")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training_device={device}")
    model.to(device)
    sampler = _balanced_sampler(dataset, target_mode) if balance == "sampler" else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if target_mode not in {"multiclass", "drone_binary"}:
        raise ValueError("target_mode must be 'multiclass' or 'drone_binary'")
    weight_classes = 2 if target_mode == "drone_binary" else (model.net[-1].out_features if hasattr(model, "net") else 8)
    weights = _class_weights(dataset, weight_classes, target_mode) if balance == "class_weight" else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            if target_mode == "drone_binary":
                logits = logits[:, [DRONE_CLASS_INDEX, BACKGROUND_CLASS_INDEX]]
                y = (y != DRONE_CLASS_INDEX).long()
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())
        mean_loss = total_loss / max(1, total)
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        print(f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "history": history, "target_mode": target_mode}, out)
    return out


def train_crop_recognizer(data: str | Path, out: str | Path, epochs: int = 10, balance: str = "sampler", target_mode: str = "multiclass") -> Path:
    return _train(CropRecognizer(), CropFolderDataset(data), out, epochs=epochs, balance=balance, target_mode=target_mode)


def train_temporal_recognizer(data: str | Path, out: str | Path, epochs: int = 10, balance: str = "sampler", target_mode: str = "multiclass") -> Path:
    return _train(TemporalRecognizer(), TemporalFolderDataset(data), out, epochs=epochs, balance=balance, target_mode=target_mode)


def train_feature_recognizer(data: str | Path, out: str | Path, epochs: int = 10, target_mode: str = "multiclass") -> Path:
    dataset = FrameBoxCSVDataset(data)
    if len(dataset) == 0:
        raise ValueError("Feature ROI dataset is empty")
    if target_mode not in {"multiclass", "drone_binary"}:
        raise ValueError("target_mode must be 'multiclass' or 'drone_binary'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training_device={device}")
    model = FeatureRecognitionModel().to(device)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for x, boxes, y, _ in loader:
            x, boxes, y = x.to(device), boxes.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x, boxes)
            if target_mode == "drone_binary":
                logits = logits[:, [DRONE_CLASS_INDEX, BACKGROUND_CLASS_INDEX]]
                y = (y != DRONE_CLASS_INDEX).long()
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())
        mean_loss = total_loss / max(1, total)
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        print(f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "history": history, "target_mode": target_mode}, out)
    return out
