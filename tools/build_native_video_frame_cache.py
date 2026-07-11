from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def atomic_torch_save(obj: object, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_json(path: Path, obj: object) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build resized frame cache for native video detector training/export.")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(args.frames_dir.glob("Clip_*_*.png"))
    if args.max_frames > 0:
        frame_paths = frame_paths[: args.max_frames]
    written = 0
    skipped = 0
    for idx, path in enumerate(frame_paths, start=1):
        out_path = args.cache_dir / f"{path.stem}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        img = Image.open(path).convert("RGB")
        image_size = img.size
        img = img.resize((args.image_size, args.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8).copy()
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        atomic_torch_save(
            {
                "tensor": tensor,
                "image_size": image_size,
                "resized_size": (args.image_size, args.image_size),
                "source": str(path),
            },
            out_path,
        )
        written += 1
        if idx == 1 or idx % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "kind": "native_video_frame_cache_progress",
                        "idx": idx,
                        "total": len(frame_paths),
                        "written": written,
                        "skipped": skipped,
                        "last": str(out_path),
                    }
                ),
                flush=True,
            )
    summary = {
        "kind": "native_video_frame_cache_done",
        "frames_dir": str(args.frames_dir.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "image_size": args.image_size,
        "resized_size": [args.image_size, args.image_size],
        "total": len(frame_paths),
        "written": written,
        "skipped": skipped,
    }
    atomic_write_json(args.cache_dir / "cache_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
