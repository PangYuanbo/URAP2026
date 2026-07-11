#!/usr/bin/env python3
"""Train a bbox-only readout on frozen SAM2/SAMURAI object-pointer features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qstr_dronedet.samurai_bbox_readout import BBoxReadout, encode_delta, normalized_previous_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260625)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.features)
    pointer = np.asarray(data["object_pointer"], dtype=np.float32)
    target = np.asarray(data["target_xywh"], dtype=np.float32)
    previous = np.asarray(data["previous_xywh"], dtype=np.float32)
    image_wh = np.asarray(data["image_wh"], dtype=np.float32)
    frame_index = np.asarray(data["frame_index"])
    valid = (frame_index > 0) & (target[:, 2] > 0) & (target[:, 3] > 0) & (previous[:, 2] > 0) & (previous[:, 3] > 0)
    pointer, target, previous, image_wh = pointer[valid], target[valid], previous[valid], image_wh[valid]
    pointer_mean = pointer.mean(axis=0)
    pointer_std = np.maximum(pointer.std(axis=0), 1e-5)
    pointer = (pointer - pointer_mean) / pointer_std
    previous_features = normalized_previous_box(previous, image_wh)
    delta = encode_delta(previous, target)

    device = torch.device(args.device)
    tensors = tuple(torch.from_numpy(value).to(device) for value in (pointer, previous_features, delta))
    model = BBoxReadout(pointer_dim=pointer.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(pointer), generator=generator)
        losses = []
        model.train()
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size].to(device)
            prediction = model(tensors[0][indices], tensors[1][indices])
            loss = functional.smooth_l1_loss(prediction, tensors[2][indices], beta=0.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(row)
        print(json.dumps(row), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "pointer_mean": torch.from_numpy(pointer_mean),
            "pointer_std": torch.from_numpy(pointer_std),
            "pointer_dim": pointer.shape[1],
            "training_rows": len(pointer),
            "epochs": args.epochs,
            "history": history,
            "source_features": str(args.features.resolve()),
        },
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
