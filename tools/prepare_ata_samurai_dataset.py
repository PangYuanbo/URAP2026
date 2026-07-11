#!/usr/bin/env python
"""Materialize ATA in the layout consumed by the existing SAMURAI evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qstr_dronedet.ata_benchmark import materialize_samurai_layout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--image-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    manifest = materialize_samurai_layout(
        args.source_root, args.output_root, split=args.split, image_mode=args.image_mode
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
