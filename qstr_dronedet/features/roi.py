from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def crop_with_context(frame: np.ndarray, bbox: tuple[float, float, float, float], scale: float = 4.0, out_size: int = 128) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    side = max(bw, bh) * scale
    nx1, ny1 = int(np.floor(cx - side / 2)), int(np.floor(cy - side / 2))
    nx2, ny2 = int(np.ceil(cx + side / 2)), int(np.ceil(cy + side / 2))
    nx1, ny1 = max(0, nx1), max(0, ny1)
    nx2, ny2 = min(w, nx2), min(h, ny2)
    if nx2 <= nx1 or ny2 <= ny1:
        crop = np.zeros((out_size, out_size, frame.shape[2] if frame.ndim == 3 else 1), dtype=frame.dtype)
    else:
        crop = frame[ny1:ny2, nx1:nx2]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def extract_temporal_tube(frame_buffer: list[np.ndarray], bbox: tuple[float, float, float, float], T: int = 5, scale: float = 4.0, out_size: int = 96) -> np.ndarray:
    if not frame_buffer:
        return np.zeros((T, 3, out_size, out_size), dtype=np.float32)
    frames = frame_buffer[-T:]
    if len(frames) < T:
        frames = [frames[0]] * (T - len(frames)) + frames
    crops = []
    for frame in frames:
        crop = crop_with_context(frame, bbox, scale=scale, out_size=out_size)
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crops.append(np.transpose(rgb, (2, 0, 1)))
    return np.stack(crops, axis=0).astype(np.float32)


def roi_align_multiscale(features: dict[str, torch.Tensor], boxes: torch.Tensor, image_size: tuple[int, int], output_size: int = 7) -> torch.Tensor:
    if not features:
        raise ValueError("features cannot be empty")
    level = "p2" if "p2" in features else sorted(features.keys())[0]
    feat = features[level]
    h_img, w_img = image_size
    _, _, hf, wf = feat.shape
    try:
        from torchvision.ops import roi_align  # type: ignore

        if boxes.numel() == 0:
            return torch.empty((0, feat.shape[1], output_size, output_size), device=feat.device)
        batch_idx = torch.zeros((boxes.shape[0], 1), dtype=boxes.dtype, device=boxes.device)
        rois = torch.cat([batch_idx, boxes], dim=1)
        return roi_align(feat, rois, output_size=(output_size, output_size), spatial_scale=wf / float(w_img), aligned=True)
    except Exception:
        outs = []
        for b in boxes.to(feat.device).float():
            x1, y1, x2, y2 = b
            fx1 = int(torch.clamp(torch.floor(x1 / max(1, w_img) * wf), 0, wf - 1).item())
            fy1 = int(torch.clamp(torch.floor(y1 / max(1, h_img) * hf), 0, hf - 1).item())
            fx2 = int(torch.clamp(torch.ceil(x2 / max(1, w_img) * wf), fx1 + 1, wf).item())
            fy2 = int(torch.clamp(torch.ceil(y2 / max(1, h_img) * hf), fy1 + 1, hf).item())
            crop = feat[:, :, fy1:fy2, fx1:fx2]
            outs.append(F.interpolate(crop, size=(output_size, output_size), mode="bilinear", align_corners=False)[0])
        return torch.stack(outs, dim=0) if outs else torch.empty((0, feat.shape[1], output_size, output_size), device=feat.device)

