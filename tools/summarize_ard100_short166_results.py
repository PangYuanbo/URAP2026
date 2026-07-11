#!/usr/bin/env python3
"""Summarize ARD100 short166 SAM2/SAMURAI results against NPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RUNS = (
    "image_box_zero_shot",
    "sam2_video_zero_shot",
    "samurai_zero_shot",
    "sam2_video_finetuned",
    "samurai_finetuned",
)
NPS = {
    "image_box_zero_shot": {"success_auc": 0.0917096801427669, "mean_iou": 0.05010184869921416, "success_50": 0.05814024576633857, "precision_20": 0.08809683927370544},
    "sam2_video_zero_shot": {"success_auc": 0.595417136103081, "mean_iou": 0.5975346156084901, "success_50": 0.7810111878706364, "precision_20": 0.8980864461698356},
    "samurai_zero_shot": {"success_auc": 0.6014026323374003, "mean_iou": 0.6037971754296045, "success_50": 0.7907317967842514, "precision_20": 0.8975362230237819},
    "sam2_video_finetuned": {"success_auc": 0.6765270147919779, "mean_iou": 0.6840025354426031, "success_50": 0.874915938130464, "precision_20": 0.9427156569053005},
    "samurai_finetuned": {"success_auc": 0.6572430035779061, "mean_iou": 0.6634502476838998, "success_50": 0.8603655927126002, "precision_20": 0.9225408082166656},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = {}
    for run in RUNS:
        path = args.results_root / "eval" / run / "canonical" / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        metrics = json.loads(path.read_text(encoding="utf-8"))
        ard = {key: float(metrics[key]) for key in ("success_auc", "mean_iou", "success_50", "precision_20")}
        rows[run] = {
            "ard100_short166": ard,
            "nps_short_tracks": NPS[run],
            "delta_ard100_minus_nps": {key: ard[key] - NPS[run][key] for key in ard},
            "sequences": int(metrics["sequences"]),
            "frames": int(metrics["frames"]),
            "visible_frames": int(metrics["visible_frames"]),
        }
    payload = {"protocol": "max 166 frames, first-frame GT only, no later correction", "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
