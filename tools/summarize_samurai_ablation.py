#!/usr/bin/env python3
"""Summarize the controlled NPS SAMURAI ablation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROWS = {
    "image_box_zero_shot": "ablation_image_box_zero_shot_test_v1/metrics.json",
    "sam2_video_zero_shot": "ablation_sam2_video_zero_shot_test_v1/metrics.json",
    "samurai_zero_shot": "zero_shot_base_plus_test_v1/metrics.json",
    "sam2_video_finetuned1": "ablation_sam2_video_finetuned1_test_v1/metrics.json",
    "samurai_finetuned1": "finetuned_stage1_base_plus_test_v1/metrics.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path(r"U:\URAP_runs\samurai"))
    parser.add_argument("--bbox-metrics", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260625)
    return parser.parse_args()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def paired_bootstrap(left: dict, right: dict, *, repeats: int, seed: int) -> dict[str, float] | None:
    left_rows = {row["sequence"]: row for row in left.get("sequence_results", [])}
    right_rows = {row["sequence"]: row for row in right.get("sequence_results", [])}
    names = sorted(set(left_rows) & set(right_rows))
    if not names:
        return None
    deltas = np.asarray([left_rows[name]["success_auc"] - right_rows[name]["success_auc"] for name in names])
    rng = np.random.default_rng(seed)
    samples = deltas[rng.integers(0, len(deltas), size=(repeats, len(deltas)))].mean(axis=1)
    return {
        "sequence_count": len(names),
        "mean_sequence_auc_delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def paired_bbox_bootstrap(
    report: dict, bbox_key: str, *, repeats: int, seed: int
) -> dict[str, float | int] | None:
    rows = report.get("sequence_results", [])
    if not rows:
        return None
    deltas = np.asarray(
        [row["mask_to_box_reference"]["success_auc"] - row[bbox_key]["success_auc"] for row in rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = deltas[rng.integers(0, len(deltas), size=(repeats, len(deltas)))].mean(axis=1)
    return {
        "sequence_count": len(rows),
        "mean_sequence_auc_delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "mask_wins": int(np.sum(deltas > 1e-12)),
        "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
        "bbox_wins": int(np.sum(deltas < -1e-12)),
    }


def metric_row(report: dict) -> dict[str, float | int]:
    return {
        key: report[key]
        for key in ("sequences", "frames", "visible_frames", "mean_iou", "success_auc", "success_50", "precision_20")
        if key in report
    }


def main() -> int:
    args = parse_args()
    reports = {name: load(args.run_root / relative) for name, relative in ROWS.items()}
    rows = {name: metric_row(report) for name, report in reports.items() if report is not None}
    comparisons = {}
    pairs = {
        "video_memory_zero_shot": ("sam2_video_zero_shot", "image_box_zero_shot"),
        "samurai_motion_zero_shot": ("samurai_zero_shot", "sam2_video_zero_shot"),
        "nps_finetune_sam2": ("sam2_video_finetuned1", "sam2_video_zero_shot"),
        "nps_finetune_samurai": ("samurai_finetuned1", "samurai_zero_shot"),
        "samurai_motion_finetuned1": ("samurai_finetuned1", "sam2_video_finetuned1"),
    }
    for name, (left_name, right_name) in pairs.items():
        left, right = reports.get(left_name), reports.get(right_name)
        if left is None or right is None:
            comparisons[name] = {"status": "pending"}
            continue
        comparisons[name] = {
            "status": "complete",
            "success_auc_delta": left["success_auc"] - right["success_auc"],
            "mean_iou_delta": left["mean_iou"] - right["mean_iou"],
            "success_50_delta": left["success_50"] - right["success_50"],
            "precision_20_delta": left["precision_20"] - right["precision_20"],
            "paired_sequence_bootstrap": paired_bootstrap(left, right, repeats=args.bootstrap, seed=args.seed),
        }

    bbox = load(args.bbox_metrics) if args.bbox_metrics else None
    if bbox is not None:
        mask = bbox["mask_to_box_reference"]
        bbox_autoregressive = bbox["bbox_readout"]
        bbox_mask_conditioned = bbox["bbox_readout_mask_conditioned"]
        bbox_gt_conditioned = bbox["bbox_readout_gt_conditioned"]
        comparisons["mask_vs_bbox_readout"] = {
            "status": "complete",
            "primary_comparison": "mask_vs_bbox_mask_conditioned",
            "mask": mask,
            "bbox_autoregressive": bbox_autoregressive,
            "bbox_mask_conditioned": bbox_mask_conditioned,
            "bbox_gt_conditioned": bbox_gt_conditioned,
            "head_only_success_auc_delta": mask["success_auc"] - bbox_mask_conditioned["success_auc"],
            "head_only_mean_iou_delta": mask["mean_iou"] - bbox_mask_conditioned["mean_iou"],
            "head_only_success_50_delta": mask["success_50"] - bbox_mask_conditioned["success_50"],
            "head_only_precision_20_delta": mask["precision_20"] - bbox_mask_conditioned["precision_20"],
            "head_only_paired_sequence_bootstrap": paired_bbox_bootstrap(
                bbox, "bbox_readout_mask_conditioned", repeats=args.bootstrap, seed=args.seed
            ),
            "autoregressive_success_auc_delta": mask["success_auc"] - bbox_autoregressive["success_auc"],
            "autoregressive_paired_sequence_bootstrap": paired_bbox_bootstrap(
                bbox, "bbox_readout", repeats=args.bootstrap, seed=args.seed + 1
            ),
            "gt_conditioned_success_auc_delta": mask["success_auc"] - bbox_gt_conditioned["success_auc"],
            "gt_conditioned_paired_sequence_bootstrap": paired_bbox_bootstrap(
                bbox, "bbox_readout_gt_conditioned", repeats=args.bootstrap, seed=args.seed + 2
            ),
            "success_auc_delta": mask["success_auc"] - bbox_mask_conditioned["success_auc"],
        }
    else:
        comparisons["mask_vs_bbox_readout"] = {"status": "pending"}

    result = {"rows": rows, "comparisons": comparisons}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# SAMURAI NPS Ablation", "", "| Row | AUC | mIoU | IoU>=0.5 | P@20 |", "|---|---:|---:|---:|---:|"]
    for name in ROWS:
        row = rows.get(name)
        if row is None:
            lines.append(f"| {name} | pending | pending | pending | pending |")
        else:
            lines.append(f"| {name} | {row['success_auc']:.4f} | {row['mean_iou']:.4f} | {row['success_50']:.4f} | {row['precision_20']:.4f} |")
    lines.extend(("", "## Contributions", ""))
    for name, comparison in comparisons.items():
        if comparison["status"] == "pending":
            lines.append(f"- `{name}`: pending")
        else:
            lines.append(f"- `{name}`: AUC delta {comparison['success_auc_delta']:+.4f}")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
