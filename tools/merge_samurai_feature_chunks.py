#!/usr/bin/env python3
"""Merge deduplicated per-sequence feature chunks from parallel extraction shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sequences", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks: dict[int, Path] = {}
    for root in args.chunk_root:
        for path in (root / "feature_chunks").glob("*.npz"):
            data = np.load(path)
            ids = np.unique(data["sequence_id"])
            if len(ids) != 1:
                raise ValueError(f"Expected one sequence id in {path}, got {ids}")
            sequence_id = int(ids[0])
            previous = chunks.get(sequence_id)
            if previous is not None:
                old = np.load(previous)
                if len(old["frame_index"]) != len(data["frame_index"]):
                    raise ValueError(f"Conflicting chunks for sequence {sequence_id}: {previous} and {path}")
            chunks[sequence_id] = path
    expected_ids = set(range(args.expected_sequences))
    if set(chunks) != expected_ids:
        missing = sorted(expected_ids - set(chunks))
        raise ValueError(f"Feature chunks incomplete: {len(chunks)}/{args.expected_sequences}; missing={missing[:20]}")
    arrays: dict[str, list[np.ndarray]] = {}
    for sequence_id in sorted(chunks):
        data = np.load(chunks[sequence_id])
        for key in data.files:
            arrays.setdefault(key, []).append(data[key])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{key: np.concatenate(values, axis=0) for key, values in arrays.items()})
    print({"sequences": len(chunks), "frames": int(sum(len(values) for values in arrays["frame_index"])), "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
