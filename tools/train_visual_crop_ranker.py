from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet34_Weights, resnet34

from eval_tvd_predictionsgt_pkl import load_predictionsgt, row_to_det, row_to_label


def iou(box_a: list[float], box_b: list[float]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def crop_context(image: Image.Image, bbox: list[float], scale: float) -> Image.Image:
    width = max(1.0, bbox[2] - bbox[0])
    height = max(1.0, bbox[3] - bbox[1])
    side = max(32.0, max(width, height) * scale)
    center_x = (bbox[0] + bbox[2]) * 0.5
    center_y = (bbox[1] + bbox[3]) * 0.5
    left = max(0, int(round(center_x - side * 0.5)))
    top = max(0, int(round(center_y - side * 0.5)))
    right = min(image.width, int(round(center_x + side * 0.5)))
    bottom = min(image.height, int(round(center_y + side * 0.5)))
    return image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))


class CropDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, list[float], float]], image_size: int, context_scale: float) -> None:
        self.samples = samples
        self.context_scale = context_scale
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, bbox, label = self.samples[index]
        with Image.open(path) as source:
            crop = crop_context(source.convert("RGB"), bbox, self.context_scale)
        return self.transform(crop), torch.tensor(label, dtype=torch.float32)


def build_samples(predictions: Path, frame_root: Path, negative_min_score: float, negative_ratio: float) -> tuple[list[tuple[Path, list[float], float]], dict[str, int]]:
    data = load_predictionsgt(predictions)
    positives: list[tuple[Path, list[float], float]] = []
    negatives: list[tuple[Path, list[float], float]] = []
    missing = 0
    for image_id, item in data.items():
        frame_path = frame_root / f"{image_id}.png"
        if not frame_path.is_file():
            missing += 1
            continue
        detections = [row_to_det(row) for row in item.get("detections", [])]
        labels = [row_to_label(row) for row in item.get("labels", [])]
        detections = [row for row in detections if row is not None]
        labels = [row for row in labels if row is not None]
        matches: set[int] = set()
        candidates: list[tuple[float, int, int]] = []
        for det_index, detection in enumerate(detections):
            for label_index, label in enumerate(labels):
                overlap = iou(detection[:4], label[1:5])
                if overlap >= 0.5:
                    candidates.append((overlap, det_index, label_index))
        used_labels: set[int] = set()
        for _overlap, det_index, label_index in sorted(candidates, reverse=True):
            if det_index in matches or label_index in used_labels:
                continue
            matches.add(det_index)
            used_labels.add(label_index)
        for det_index, detection in enumerate(detections):
            sample = (frame_path, [float(value) for value in detection[:4]], 1.0 if det_index in matches else 0.0)
            if det_index in matches:
                positives.append(sample)
            elif float(detection[4]) >= negative_min_score:
                negatives.append(sample)
    random.Random(2026).shuffle(negatives)
    negatives = negatives[: int(round(len(positives) * negative_ratio))]
    samples = positives + negatives
    random.Random(2026).shuffle(samples)
    return samples, {"positives": len(positives), "negatives": len(negatives), "missing_frames": missing, "images": len(data)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--context-scale", type=float, default=4.0)
    parser.add_argument("--negative-min-score", type=float, default=0.005)
    parser.add_argument("--negative-ratio", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    torch.manual_seed(2026)
    np.random.seed(2026)
    samples, counts = build_samples(args.predictionsgt_pkl, args.frame_root, args.negative_min_score, args.negative_ratio)
    dataset = CropDataset(samples, args.image_size, args.context_scale)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.to(device)
    positive_count = max(1, counts["positives"])
    pos_weight = torch.tensor([counts["negatives"] / positive_count], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        record = {"epoch": float(epoch), "loss": float(np.mean(losses))}
        history.append(record)
        print(json.dumps({"kind": "visual_crop_train_progress", "epoch": epoch, "epochs": args.epochs, "loss": record["loss"], "device": str(device)}), flush=True)

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "image_size": args.image_size, "context_scale": args.context_scale}, args.out_model)
    summary = {"device": str(device), **counts, "samples": len(samples), "history": history, "model": str(args.out_model)}
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
