from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-fraction", type=float, required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if not 0.0 < args.memory_fraction < 1.0:
        raise ValueError("memory fraction must be between zero and one")

    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=0)
    repo_root = Path(__file__).resolve().parents[1]
    sam2_root = repo_root / "third_party" / "samurai" / "sam2"
    train_script = sam2_root / "training" / "train.py"
    sys.path.insert(0, str(sam2_root))
    sys.argv = [
        str(train_script),
        "-c",
        args.config,
        "--use-cluster",
        "0",
        "--num-gpus",
        "1",
    ]
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
