#!/usr/bin/env python3
"""Load an official SeqTrack checkpoint and run one synthetic tracking step."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqtrack-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def synthetic_frame(offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
    rng = np.random.default_rng(20260624)
    image = rng.integers(25, 45, size=(720, 1280, 3), dtype=np.uint8)
    x1, y1 = 620 + offset_x, 350 + offset_y
    image[y1 : y1 + 14, x1 : x1 + 28] = (220, 220, 220)
    image[y1 + 5 : y1 + 9, x1 - 8 : x1 + 36] = (180, 180, 180)
    return image


def main() -> int:
    args = parse_args()
    seqtrack_root = args.seqtrack_root.resolve()
    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("SeqTrack's official tracker requires CUDA")
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError("checkpoint or config is missing")

    # Upstream torch.load omits weights_only; the trusted checkpoint has metadata.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, str(seqtrack_root))
    from lib.config.seqtrack.config import cfg, update_config_from_file
    from lib.test.tracker.seqtrack import SEQTRACK
    from lib.test.utils import TrackerParams

    update_config_from_file(str(config_path))
    # A complete tracking checkpoint is loaded strictly below. Avoid downloading
    # encoder initialization weights that would be immediately overwritten.
    cfg.MODEL.ENCODER.PRETRAIN_TYPE = "scratch"
    params = TrackerParams()
    params.cfg = cfg
    params.checkpoint = str(checkpoint)
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.debug = 0
    params.save_all_boxes = False

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tracker = SEQTRACK(params, "ata")
    load_seconds = time.perf_counter() - started
    init_bbox = [620.0, 350.0, 28.0, 14.0]
    torch.cuda.synchronize()
    started = time.perf_counter()
    tracker.initialize(synthetic_frame(), {"init_bbox": init_bbox})
    prediction = tracker.track(synthetic_frame(3, -2))
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started

    parameter_count = sum(p.numel() for p in tracker.network.parameters())
    result = {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "parameter_count": parameter_count,
        "parameter_count_millions": parameter_count / 1_000_000,
        "template_size": int(cfg.TEST.TEMPLATE_SIZE),
        "search_size": int(cfg.TEST.SEARCH_SIZE),
        "encoder_build_pretrain_type": str(cfg.MODEL.ENCODER.PRETRAIN_TYPE),
        "load_seconds": load_seconds,
        "one_step_seconds": inference_seconds,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "initial_bbox_xywh": init_bbox,
        "predicted_bbox_xywh": [float(v) for v in prediction["target_bbox"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
