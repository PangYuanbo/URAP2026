#!/usr/bin/env python3
"""Verify that a SAM2 training checkpoint reached the requested final phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-final-epoch", type=int, required=True)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    epoch = int(payload.get("epoch", -1))
    model = payload.get("model")
    if epoch < args.expected_final_epoch:
        raise RuntimeError(
            f"Checkpoint only reached epoch/phase {epoch}; "
            f"expected at least {args.expected_final_epoch}"
        )
    if not isinstance(model, dict) or not model:
        raise RuntimeError("Checkpoint has no non-empty model state")
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "epoch": epoch,
        "expected_final_epoch": args.expected_final_epoch,
        "model_tensors": len(model),
        "verified": True,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
