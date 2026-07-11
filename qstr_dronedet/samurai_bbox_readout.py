"""Frozen-feature bbox readout used for the SAMURAI output-head ablation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def xywh_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    output = np.asarray(boxes, dtype=np.float32).copy()
    output[..., 0] += output[..., 2] / 2
    output[..., 1] += output[..., 3] / 2
    return output


def cxcywh_to_xywh(boxes: np.ndarray) -> np.ndarray:
    output = np.asarray(boxes, dtype=np.float32).copy()
    output[..., 0] -= output[..., 2] / 2
    output[..., 1] -= output[..., 3] / 2
    return output


def encode_delta(previous_xywh: np.ndarray, target_xywh: np.ndarray) -> np.ndarray:
    previous = xywh_to_cxcywh(previous_xywh)
    target = xywh_to_cxcywh(target_xywh)
    width = np.maximum(previous[..., 2], 1.0)
    height = np.maximum(previous[..., 3], 1.0)
    return np.stack(
        (
            (target[..., 0] - previous[..., 0]) / width,
            (target[..., 1] - previous[..., 1]) / height,
            np.log(np.maximum(target[..., 2], 1.0) / width),
            np.log(np.maximum(target[..., 3], 1.0) / height),
        ),
        axis=-1,
    ).astype(np.float32)


def decode_delta(previous_xywh: np.ndarray, delta: np.ndarray, image_wh: np.ndarray) -> np.ndarray:
    previous = xywh_to_cxcywh(np.asarray(previous_xywh, dtype=np.float32))
    delta = np.asarray(delta, dtype=np.float32)
    image_wh = np.asarray(image_wh, dtype=np.float32)
    width = np.maximum(previous[..., 2], 1.0)
    height = np.maximum(previous[..., 3], 1.0)
    center_x = previous[..., 0] + np.clip(delta[..., 0], -4.0, 4.0) * width
    center_y = previous[..., 1] + np.clip(delta[..., 1], -4.0, 4.0) * height
    target_width = width * np.exp(np.clip(delta[..., 2], -3.0, 3.0))
    target_height = height * np.exp(np.clip(delta[..., 3], -3.0, 3.0))
    result = cxcywh_to_xywh(np.stack((center_x, center_y, target_width, target_height), axis=-1))
    result[..., 0] = np.clip(result[..., 0], 0.0, np.maximum(image_wh[..., 0] - 1.0, 0.0))
    result[..., 1] = np.clip(result[..., 1], 0.0, np.maximum(image_wh[..., 1] - 1.0, 0.0))
    result[..., 2] = np.clip(result[..., 2], 1.0, np.maximum(image_wh[..., 0] - result[..., 0], 1.0))
    result[..., 3] = np.clip(result[..., 3], 1.0, np.maximum(image_wh[..., 1] - result[..., 1], 1.0))
    return result.astype(np.float32)


def normalized_previous_box(previous_xywh: np.ndarray, image_wh: np.ndarray) -> np.ndarray:
    center = xywh_to_cxcywh(previous_xywh)
    image_wh = np.maximum(np.asarray(image_wh, dtype=np.float32), 1.0)
    return np.stack(
        (
            center[..., 0] / image_wh[..., 0],
            center[..., 1] / image_wh[..., 1],
            np.log(np.maximum(center[..., 2] / image_wh[..., 0], 1e-6)),
            np.log(np.maximum(center[..., 3] / image_wh[..., 1], 1e-6)),
        ),
        axis=-1,
    ).astype(np.float32)


class BBoxReadout(nn.Module):
    def __init__(self, pointer_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(pointer_dim + 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
        )

    def forward(self, object_pointer: torch.Tensor, previous_box: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((object_pointer, previous_box), dim=-1))


@dataclass(frozen=True)
class TrackingMetrics:
    frames: int
    mean_iou: float
    success_auc: float
    success_50: float
    precision_20: float


def tracking_metrics(predictions: np.ndarray, targets: np.ndarray) -> TrackingMetrics:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    visible = (targets[:, 2] > 0) & (targets[:, 3] > 0)
    predictions, targets = predictions[visible], targets[visible]
    ax2, ay2 = predictions[:, 0] + predictions[:, 2], predictions[:, 1] + predictions[:, 3]
    bx2, by2 = targets[:, 0] + targets[:, 2], targets[:, 1] + targets[:, 3]
    intersection = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(predictions[:, 0], targets[:, 0])) * np.maximum(
        0, np.minimum(ay2, by2) - np.maximum(predictions[:, 1], targets[:, 1])
    )
    union = predictions[:, 2] * predictions[:, 3] + targets[:, 2] * targets[:, 3] - intersection
    ious = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    errors = np.hypot(
        predictions[:, 0] + predictions[:, 2] / 2 - targets[:, 0] - targets[:, 2] / 2,
        predictions[:, 1] + predictions[:, 3] / 2 - targets[:, 1] - targets[:, 3] / 2,
    )
    thresholds = np.linspace(0.0, 1.0, 21)
    return TrackingMetrics(
        frames=len(targets),
        mean_iou=float(ious.mean()),
        success_auc=float(np.mean([(ious >= threshold).mean() for threshold in thresholds])),
        success_50=float((ious >= 0.5).mean()),
        precision_20=float((errors <= 20.0).mean()),
    )
