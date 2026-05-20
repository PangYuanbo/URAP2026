"""
Merge per-flight AOT/Airborne Object Tracking result.json files into a single result.json
compatible with Amazon's airborne metrics scripts (aotcore.metrics.run_airborne_metrics).

Expected input layout:
  results_dir/
    <flight_id>/result.json
    <flight_id>/result.json
    ...

Output:
  results_dir/result.json (default) containing a single JSON list with all entries concatenated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_item(item: Any, src: Path) -> None:
    if not isinstance(item, dict):
        raise TypeError(f"{src}: expected list[dict], got {type(item)}")
    if "img_name" not in item or "detections" not in item:
        raise ValueError(f"{src}: missing required keys in item: {item.keys()}")
    if not isinstance(item["detections"], list):
        raise TypeError(f"{src}: 'detections' must be a list")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="Folder containing per-flight subfolders with result.json")
    ap.add_argument("--out", default=None, help="Output merged json path (default: <results-dir>/result.json)")
    ap.add_argument("--per-flight-filename", default="result.json", help="Filename inside each flight folder")
    ap.add_argument("--sort", action="store_true", help="Sort merged entries by img_name for stable diffs")
    ap.add_argument("--min-flights", type=int, default=0, help="Fail if fewer than this many flights are found")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if args.out is None:
        out_path = results_dir / "result.json"
    else:
        out_path = Path(args.out).resolve()

    if not results_dir.is_dir():
        raise SystemExit(f"--results-dir does not exist or is not a directory: {results_dir}")

    per_flight = []
    for child in sorted(results_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        cand = child / args.per_flight_filename
        if cand.is_file():
            per_flight.append(cand)

    if len(per_flight) < args.min_flights:
        raise SystemExit(
            f"Found only {len(per_flight)} flight result files under {results_dir}, "
            f"but --min-flights={args.min_flights}"
        )

    merged: list[dict[str, Any]] = []
    for p in per_flight:
        data = _load_json(p)
        if not isinstance(data, list):
            raise TypeError(f"{p}: expected a JSON list, got {type(data)}")
        for item in data:
            _validate_item(item, p)
            merged.append(item)

    if args.sort:
        merged.sort(key=lambda d: str(d.get("img_name", "")))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f)
    os.replace(tmp_path, out_path)

    print(f"Merged {len(per_flight)} flights into {out_path} ({len(merged)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

