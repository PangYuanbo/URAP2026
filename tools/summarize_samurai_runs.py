#!/usr/bin/env python
"""Summarize completed SAMURAI prediction CSVs with tiny-target metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(run_root: Path) -> dict[str, float | int | str]:
    rows = []
    for path in sorted((run_root / "predictions").glob("*.csv")):
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(row for row in csv.DictReader(handle) if row["visible"] == "1")
    if not rows:
        raise RuntimeError(f"No visible prediction rows found in {run_root}")
    ious = np.asarray([float(row["iou"]) for row in rows], dtype=np.float64)
    errors = np.asarray([float(row["center_error"]) for row in rows], dtype=np.float64)
    thresholds = np.linspace(0.0, 1.0, 21)
    return {
        "run": run_root.name,
        "sequences": len(list((run_root / "predictions").glob("*.csv"))),
        "visible_frames": len(rows),
        "mean_iou": float(ious.mean()),
        "success_auc": float(np.mean([(ious >= threshold).mean() for threshold in thresholds])),
        "success_50": float((ious >= 0.5).mean()),
        "precision_5": float((errors <= 5.0).mean()),
        "precision_10": float((errors <= 10.0).mean()),
        "precision_20": float((errors <= 20.0).mean()),
        "lost_iou_0": float((ious == 0.0).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = [summarize(path) for path in args.run_roots]
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
