from __future__ import annotations

import torch
from torch import nn

from qstr_dronedet.types import CLASSES


class CropRecognizer(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES), width: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

