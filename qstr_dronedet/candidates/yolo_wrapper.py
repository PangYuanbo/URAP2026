from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

from qstr_dronedet.types import DetectionCandidate


@lru_cache(maxsize=4)
def _load_yolo(weight_path: str):
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"ultralytics is not installed; YOLO candidates disabled ({exc})") from exc
    return YOLO(weight_path)


def candidates_from_yolo(frame_bgr: np.ndarray, weight_path: str | None = None, class_agnostic: bool = True, conf: float = 0.05) -> list[DetectionCandidate]:
    if not weight_path:
        return []
    try:
        model = _load_yolo(str(weight_path))
    except RuntimeError as exc:
        warnings.warn(str(exc), RuntimeWarning)
        return []
    results = model.predict(frame_bgr, conf=conf, agnostic_nms=class_agnostic, verbose=False)
    out: list[DetectionCandidate] = []
    for res in results:
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy.detach().cpu().numpy().reshape(-1).astype(float)
            score = float(box.conf.detach().cpu().numpy().reshape(-1)[0])
            cls = int(box.cls.detach().cpu().numpy().reshape(-1)[0]) if getattr(box, "cls", None) is not None else -1
            out.append(DetectionCandidate(tuple(xyxy.tolist()), score, "yolo", extra={"class_id": cls}))
    return out


def _tile_origins(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    origins = list(range(0, max(1, length - tile_size + 1), max(1, stride)))
    last = length - tile_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def candidates_from_yolo_tiled(
    frame_bgr: np.ndarray,
    weight_path: str | None = None,
    tile_size: int = 256,
    stride: int = 192,
    class_agnostic: bool = True,
    conf: float = 0.05,
    device: str | int | None = None,
    max_det: int = 300,
) -> list[DetectionCandidate]:
    if not weight_path:
        return []
    try:
        model = _load_yolo(str(weight_path))
    except RuntimeError as exc:
        warnings.warn(str(exc), RuntimeWarning)
        return []
    h, w = frame_bgr.shape[:2]
    tile_size = int(tile_size)
    stride = int(stride)
    xs = _tile_origins(w, tile_size, stride)
    ys = _tile_origins(h, tile_size, stride)
    tiles = []
    origins: list[tuple[int, int]] = []
    for oy in ys:
        for ox in xs:
            tile = frame_bgr[oy : min(h, oy + tile_size), ox : min(w, ox + tile_size)]
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=frame_bgr.dtype)
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded
            tiles.append(tile)
            origins.append((ox, oy))
    if not tiles:
        return []
    results = model.predict(tiles, conf=conf, imgsz=tile_size, agnostic_nms=class_agnostic, verbose=False, device=device, max_det=max_det)
    out: list[DetectionCandidate] = []
    for res, (ox, oy) in zip(results, origins):
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy.detach().cpu().numpy().reshape(-1).astype(float)
            xyxy[[0, 2]] += ox
            xyxy[[1, 3]] += oy
            xyxy[0] = max(0.0, min(float(w), xyxy[0]))
            xyxy[2] = max(0.0, min(float(w), xyxy[2]))
            xyxy[1] = max(0.0, min(float(h), xyxy[1]))
            xyxy[3] = max(0.0, min(float(h), xyxy[3]))
            score = float(box.conf.detach().cpu().numpy().reshape(-1)[0])
            cls = int(box.cls.detach().cpu().numpy().reshape(-1)[0]) if getattr(box, "cls", None) is not None else -1
            out.append(DetectionCandidate(tuple(xyxy.tolist()), score, "yolo_tile", extra={"class_id": cls, "tile_origin_xy": [ox, oy]}))
    return out
