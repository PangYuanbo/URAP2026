#!/usr/bin/env python3
"""Adapt SeqTrack-B384 positional embeddings to ATA's 192/384 crops."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqtrack-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, str(args.seqtrack_root.resolve()))
    from lib.config.seqtrack.config import cfg, update_config_from_file
    from lib.models.seqtrack import build_seqtrack

    update_config_from_file(str(args.target_config.resolve()))
    cfg.MODEL.ENCODER.PRETRAIN_TYPE = "scratch"
    target_model = build_seqtrack(cfg)
    target_state = target_model.state_dict()
    source_checkpoint = torch.load(args.source.resolve(), map_location="cpu")
    source_state = dict(source_checkpoint["net"])

    key = "encoder.body.pos_embed"
    source_pos = source_state[key]
    target_pos = target_state[key]
    search_tokens = (cfg.DATA.SEARCH.SIZE // cfg.MODEL.ENCODER.STRIDE) ** 2
    source_template_tokens = source_pos.shape[1] - search_tokens
    target_template_tokens = target_pos.shape[1] - search_tokens
    source_side = math.isqrt(source_template_tokens)
    target_side = math.isqrt(target_template_tokens)
    if source_side**2 != source_template_tokens or target_side**2 != target_template_tokens:
        raise ValueError("Template position embeddings are not square grids")
    search_pos = source_pos[:, :search_tokens, :]
    template_pos = source_pos[:, search_tokens:, :]
    template_2d = template_pos.reshape(1, source_side, source_side, source_pos.shape[2]).permute(0, 3, 1, 2)
    resized = functional.interpolate(template_2d, size=(target_side, target_side), mode="bicubic", align_corners=True)
    resized = resized.permute(0, 2, 3, 1).reshape(1, target_template_tokens, source_pos.shape[2])
    source_state[key] = torch.cat((search_pos, resized), dim=1)
    target_model.load_state_dict(source_state, strict=True)

    output = dict(source_checkpoint)
    output["net"] = source_state
    output["ata_adaptation"] = {
        "source": str(args.source.resolve()),
        "target_config": str(args.target_config.resolve()),
        "method": "preserve search grid; bicubic-interpolate template grid",
        "source_position_shape": list(source_pos.shape),
        "target_position_shape": list(source_state[key].shape),
        "strict_load_verified": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(json.dumps(output["ata_adaptation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
