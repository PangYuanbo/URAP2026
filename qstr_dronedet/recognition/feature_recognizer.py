from __future__ import annotations

import torch
from torch import nn

from qstr_dronedet.features.backbone import SimpleBackbone
from qstr_dronedet.features.roi import roi_align_multiscale
from qstr_dronedet.types import CLASSES


class FeatureRecognizer(nn.Module):
    def __init__(self, in_channels: int = 48, num_classes: int = len(CLASSES)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureRecognitionModel(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES), backbone_width: int = 24) -> None:
        super().__init__()
        self.backbone = SimpleBackbone(width=backbone_width)
        self.head = FeatureRecognizer(in_channels=backbone_width * 2, num_classes=num_classes)

    def forward(self, images: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        # MVP path: one ROI per image. The ROI fallback currently assumes a single
        # image feature map, so iterate for clarity and correctness.
        outs = []
        for img, box in zip(images, boxes):
            feats = self.backbone(img.unsqueeze(0))
            roi = roi_align_multiscale(feats, box.unsqueeze(0), image_size=tuple(img.shape[-2:]), output_size=7)
            outs.append(self.head(roi)[0])
        return torch.stack(outs, dim=0)
