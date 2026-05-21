from __future__ import annotations

import torch
from torch import nn

from qstr_dronedet.types import CLASSES


class MLPFusion(nn.Module):
    """Optional trainable fusion head; rule_fusion is the default MVP path."""

    def __init__(self, in_dim: int = len(CLASSES) * 3 + 5, hidden: int = 64, num_classes: int = len(CLASSES)) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

