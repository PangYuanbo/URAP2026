from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLIP_RE = re.compile(r"^(Clip_[^_]+)_\d+")


def clip_id_from_frame(path: Path) -> str | None:
    match = CLIP_RE.match(path.stem)
    if match:
        return match.group(1)
    parts = path.stem.split("_")
    if len(parts) >= 3 and parts[0] == "Clip":
        return "_".join(parts[:2])
    return None


def _link_or_copy(src: Path, dst: Path, mode: str) -> str:
    if dst.exists():
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"
    if mode == "symlink":
        os.symlink(src, dst)
        return "symlinked"
    try:
        os.link(src, dst)
        return "hardlinked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def prepare_flight_dirs(frames_dir: Path, out_dir: Path, mode: str = "hardlink") -> dict[str, Any]:
    frames_dir = frames_dir.resolve()
    out_dir = out_dir.resolve()
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames_dir not found: {frames_dir}")

    summary: dict[str, Any] = {
        "frames_dir": str(frames_dir),
        "out_dir": str(out_dir),
        "mode": mode,
        "clips": {},
        "unmatched": [],
        "actions": {"exists": 0, "hardlinked": 0, "symlinked": 0, "copied": 0},
    }

    image_paths = sorted(
        p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    for src in image_paths:
        clip_id = clip_id_from_frame(src)
        if not clip_id:
            summary["unmatched"].append(src.name)
            continue
        dst = out_dir / clip_id / src.name
        action = _link_or_copy(src, dst, mode=mode)
        summary["actions"][action] += 1
        clip_info = summary["clips"].setdefault(clip_id, {"frames": 0})
        clip_info["frames"] += 1

    summary["num_input_images"] = len(image_paths)
    summary["num_clips"] = len(summary["clips"])
    summary["num_unmatched"] = len(summary["unmatched"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group flat NPS frames (Clip_XXX_YYYY.png) into AICrowd flight folders for seg_test.py."
    )
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_flight_dirs(args.frames_dir, args.out_dir, mode=args.mode)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("frames=" + str(summary["num_input_images"]))
    print("clips=" + str(summary["num_clips"]))
    print("out_dir=" + summary["out_dir"])
    if summary["num_unmatched"]:
        print("unmatched=" + str(summary["num_unmatched"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
