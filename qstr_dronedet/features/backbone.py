from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleBackbone(nn.Module):
    """Small CPU-friendly backbone returning P2/P3/P4 feature maps."""

    def __init__(self, width: int = 24) -> None:
        super().__init__()
        self.stem = _ConvBlock(3, width, stride=2)
        self.p2 = _ConvBlock(width, width * 2, stride=2)
        self.p3 = _ConvBlock(width * 2, width * 4, stride=2)
        self.p4 = _ConvBlock(width * 4, width * 8, stride=2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        p2 = self.p2(x)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        return {"p2": p2, "p3": p3, "p4": p4}


def preprocess_tensor(frame_bgr) -> torch.Tensor:
    import cv2
    import numpy as np

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

