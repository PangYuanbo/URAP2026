from __future__ import annotations

import torch
from torch import nn

from qstr_dronedet.types import CLASSES


class _FrameEncoder(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalRecognizer(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES), width: int = 16, hidden: int = 48) -> None:
        super().__init__()
        self.encoder = _FrameEncoder(width)
        self.gru = nn.GRU(width * 2, hidden, batch_first=True)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        emb = self.encoder(x.reshape(b * t, c, h, w)).reshape(b, t, -1)
        out, _ = self.gru(emb)
        return self.head(out[:, -1])

