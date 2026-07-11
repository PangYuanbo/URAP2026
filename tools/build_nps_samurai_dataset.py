#!/usr/bin/env python
"""Build a first-frame-prompt SAMURAI dataset from NPS box annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qstr_dronedet.samurai_dataset import (  # noqa: E402
    associate_tracks,
    export_samurai_dataset,
    load_box_csv,
    select_tracks,
    validate_samurai_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--min-visible-frames", type=int, default=8)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--image-mode", choices=("hardlink", "copy", "jpeg"), default="hardlink")
    parser.add_argument("--skip-vos", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    observations = load_box_csv(args.gt_csv)
    associated = associate_tracks(observations, max_gap=args.max_gap)
    selected = select_tracks(
        associated,
        min_visible_frames=args.min_visible_frames,
        min_visibility=args.min_visibility,
        max_sequences=args.max_sequences,
    )
    if not selected:
        raise RuntimeError("No tracks passed the selection filters")
    manifest = export_samurai_dataset(
        selected,
        frames_root=args.frames_root,
        output_root=args.output_root,
        split=args.split,
        image_mode=args.image_mode,
        write_vos=not args.skip_vos,
    )
    validation = validate_samurai_dataset(args.output_root, split=args.split)
    report = {
        "input_observations": len(observations),
        "associated_tracks": len(associated),
        "selected_tracks": len(selected),
        "manifest": manifest,
        "validation": validation,
    }
    (args.output_root / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
