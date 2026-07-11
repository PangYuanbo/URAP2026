#!/usr/bin/env python
"""Download one public Google Drive file with resumable output."""

from __future__ import annotations

import argparse
from pathlib import Path

import gdown


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=args.file_id, output=str(args.output), resume=True, quiet=False)
    if result is None or not args.output.is_file():
        raise RuntimeError(f"Google Drive download failed: {args.file_id}")
    print(f"completed={args.output}")
    print(f"bytes={args.output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
