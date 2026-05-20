from __future__ import annotations

import argparse
from pathlib import Path
import pickle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    args = ap.parse_args()

    pkl_path = args.split_root / args.split / "video_length_dict.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Split file not found: {pkl_path}")

    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {pkl_path}, got {type(data)}")

    ids = sorted(int(k) for k in data.keys())
    print(" ".join(str(x) for x in ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
