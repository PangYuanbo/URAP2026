from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qstr_dronedet.nps_motion_interventions import INTERVENTIONS, parse_yolo_labels, yolo_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize NPS motion intervention robustness with a unified metric implementation.")
    parser.add_argument("--dataset-root", type=Path, default=Path(r"U:\URAP_datasets\TransVisDrone\NPS_interventions\motion_v1"))
    parser.add_argument("--eval-root", type=Path, default=Path(r"C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_robustness\model_evals"))
    parser.add_argument("--out-dir", type=Path, default=Path(r"C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_robustness\report"))
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=59)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_predictions(path: Path, confidence: float) -> list[list[float]]:
    predictions = []
    if not path.exists():
        return predictions
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        confidence_value = float(parts[5]) if len(parts) > 5 else 1.0
        if confidence_value >= confidence:
            predictions.append([int(float(parts[0])), *[float(value) for value in parts[1:5]], confidence_value])
    return predictions


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    recall_curve = np.concatenate(([0.0], recall, [1.0]))
    precision_curve = np.concatenate(([1.0], precision, [0.0]))
    precision_curve = np.flip(np.maximum.accumulate(np.flip(precision_curve)))
    samples = np.linspace(0.0, 1.0, 101)
    interpolated = np.interp(samples, recall_curve, precision_curve)
    return float(np.sum((interpolated[:-1] + interpolated[1:]) * np.diff(samples) * 0.5))


def evaluate_frames(frames: Sequence[dict], iou_thresholds: Iterable[float] = np.arange(0.5, 0.96, 0.05)) -> dict:
    total_gt = sum(len(frame["gt"]) for frame in frames)
    total_predictions = sum(len(frame["pred"]) for frame in frames)
    ap_values = []
    counts_at_50 = None
    for threshold in iou_thresholds:
        ranked = []
        for frame_index, frame in enumerate(frames):
            ranked.extend((float(prediction[5]), frame_index, prediction) for prediction in frame["pred"])
        ranked.sort(key=lambda item: item[0], reverse=True)
        matched = {frame_index: set() for frame_index in range(len(frames))}
        true_positive = []
        false_positive = []
        for _, frame_index, prediction in ranked:
            best_index = -1
            best_iou = 0.0
            for gt_index, gt_box in enumerate(frames[frame_index]["gt"]):
                if gt_index in matched[frame_index] or int(gt_box[0]) != int(prediction[0]):
                    continue
                overlap = yolo_iou(gt_box, prediction)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = gt_index
            is_match = best_index >= 0 and best_iou >= threshold
            if is_match:
                matched[frame_index].add(best_index)
            true_positive.append(1 if is_match else 0)
            false_positive.append(0 if is_match else 1)
        tp_curve = np.cumsum(true_positive, dtype=np.float64)
        fp_curve = np.cumsum(false_positive, dtype=np.float64)
        recall = tp_curve / max(total_gt, 1)
        precision = tp_curve / np.maximum(tp_curve + fp_curve, 1.0)
        ap_values.append(average_precision(recall, precision) if total_gt else 0.0)
        if abs(float(threshold) - 0.5) < 1e-8:
            tp = int(tp_curve[-1]) if tp_curve.size else 0
            fp = int(fp_curve[-1]) if fp_curve.size else 0
            counts_at_50 = (tp, fp, total_gt - tp)
    tp, fp, fn = counts_at_50 or (0, total_predictions, total_gt)
    return {
        "frames": len(frames),
        "gt": total_gt,
        "predictions": total_predictions,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "map50": ap_values[0] if ap_values else 0.0,
        "map5095": float(np.mean(ap_values)) if ap_values else 0.0,
    }


def frame_rows(dataset_root: Path, eval_root: Path, model: str, intervention: str, confidence: float) -> list[dict]:
    prediction_dir = eval_root / model / intervention / "labels"
    rows = []
    for manifest_path in sorted((dataset_root / intervention / "manifests" / "test").glob("*.jsonl")):
        for record in load_manifest(manifest_path):
            image_stem = Path(record["output_image"]).stem
            rows.append(
                {
                    "clip": record["clip"],
                    "frame_id": int(record["output_frame_id"]),
                    "source_position": float(record["source_position"]),
                    "synthetic": bool(record["synthetic"]),
                    "fallback": bool(record["fallback_reason"]),
                    "gt": parse_yolo_labels(Path(record["output_label"])),
                    "pred": load_predictions(prediction_dir / f"{image_stem}.txt", confidence),
                }
            )
    return rows


def per_frame_counts(frame: dict, threshold: float = 0.5) -> tuple[int, int, int]:
    predictions = sorted(frame["pred"], key=lambda box: float(box[5]), reverse=True)
    matched = set()
    tp = 0
    for prediction in predictions:
        candidates = [(yolo_iou(gt, prediction), index) for index, gt in enumerate(frame["gt"]) if index not in matched and int(gt[0]) == int(prediction[0])]
        if candidates:
            overlap, index = max(candidates)
            if overlap >= threshold:
                matched.add(index)
                tp += 1
    return tp, len(predictions) - tp, len(frame["gt"]) - tp


def bootstrap_drop(original: dict[str, float], intervention: dict[str, float], samples: int, seed: int) -> tuple[float, float, float]:
    clips = sorted(set(original) & set(intervention))
    if not clips:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    drops = []
    for _ in range(samples):
        selected = rng.choice(clips, size=len(clips), replace=True)
        baseline = float(np.mean([original[clip] for clip in selected]))
        changed = float(np.mean([intervention[clip] for clip in selected]))
        drops.append((baseline - changed) / max(baseline, 1e-12))
    point = (float(np.mean(list(original.values()))) - float(np.mean(list(intervention.values())))) / max(float(np.mean(list(original.values()))), 1e-12)
    return point, float(np.percentile(drops, 2.5)), float(np.percentile(drops, 97.5))


def matched_original_frames(original_frames: Sequence[dict], changed_frames: Sequence[dict]) -> list[dict]:
    original_by_source = {(frame["clip"], int(round(frame["source_position"]))): frame for frame in original_frames}
    matched = []
    for changed in changed_frames:
        source_frame_id = int(round(changed["source_position"]))
        original = original_by_source.get((changed["clip"], source_frame_id))
        if original is not None:
            matched.append(original)
    return matched


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_retention(rows: Sequence[dict], out_path: Path) -> None:
    order = list(INTERVENTIONS)
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for model in sorted({row["model"] for row in rows}):
        selected = {row["intervention"]: row for row in rows if row["model"] == model and row["slice"] == "all"}
        if "original" not in selected:
            continue
        baseline = selected["original"]["map50"]
        values = [selected[name]["map50"] / max(baseline, 1e-12) if name in selected else np.nan for name in order]
        axis.plot(order, values, marker="o", linewidth=2, label=model)
    axis.axhline(0.9, color="red", linestyle="--", linewidth=1, label="10% drop")
    axis.set_ylabel("mAP@0.5 retention")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_worst_timeline(frames: Sequence[dict], title: str, out_path: Path) -> None:
    values = []
    for frame in frames:
        tp, fp, fn = per_frame_counts(frame)
        values.append(1.0 if fn == 0 and tp > 0 else (0.5 if tp > 0 else 0.0))
    fig, axis = plt.subplots(figsize=(12, 2.5))
    axis.plot([frame["frame_id"] for frame in frames], values, linewidth=1.2)
    axis.set_ylim(-0.05, 1.05)
    axis.set_title(title)
    axis.set_xlabel("output frame")
    axis.set_ylabel("success")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = [path.name for path in args.eval_root.iterdir() if path.is_dir()]
    summary_rows = []
    clip_rows = []
    per_frame_rows = []
    clip_metric_lookup: dict[tuple[str, str], dict[str, float]] = {}
    frames_lookup: dict[tuple[str, str], list[dict]] = {}
    for model in sorted(models):
        for intervention in INTERVENTIONS:
            run_dir = args.eval_root / model / intervention
            if not (run_dir / "complete.json").exists():
                continue
            frames = frame_rows(args.dataset_root, args.eval_root, model, intervention, args.confidence)
            frames_lookup[(model, intervention)] = frames
            for frame in frames:
                tp, fp, fn = per_frame_counts(frame)
                per_frame_rows.append(
                    {
                        "model": model,
                        "intervention": intervention,
                        "clip": frame["clip"],
                        "frame_id": frame["frame_id"],
                        "source_position": frame["source_position"],
                        "synthetic": frame["synthetic"],
                        "fallback": frame["fallback"],
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }
                )
            slices = {"all": frames, "anchor": [frame for frame in frames if not frame["synthetic"]], "synthetic": [frame for frame in frames if frame["synthetic"]]}
            for slice_name, selected in slices.items():
                if not selected:
                    continue
                metrics = evaluate_frames(selected)
                summary_rows.append({"model": model, "role": "cross_domain_control" if model == "yolomg_ard100" else "primary", "intervention": intervention, "slice": slice_name, **metrics})
            by_clip = defaultdict(list)
            for frame in frames:
                by_clip[frame["clip"]].append(frame)
            clip_metric_lookup[(model, intervention)] = {}
            for clip, selected in sorted(by_clip.items()):
                metrics = evaluate_frames(selected)
                clip_metric_lookup[(model, intervention)][clip] = metrics["map50"]
                clip_rows.append({"model": model, "intervention": intervention, "clip": clip, **metrics})
    baseline_by_model = {row["model"]: row["map50"] for row in summary_rows if row["intervention"] == "original" and row["slice"] == "all"}
    for row in summary_rows:
        row["map50_retention"] = row["map50"] / max(baseline_by_model.get(row["model"], 0.0), 1e-12)
    comparisons = []
    for model in sorted({row["model"] for row in summary_rows}):
        original_frames = frames_lookup.get((model, "original"), [])
        for intervention in INTERVENTIONS[1:]:
            changed = clip_metric_lookup.get((model, intervention), {})
            if not changed:
                continue
            matched_frames = matched_original_frames(original_frames, frames_lookup[(model, intervention)])
            matched_by_clip = defaultdict(list)
            for frame in matched_frames:
                matched_by_clip[frame["clip"]].append(frame)
            matched_metrics = {clip: evaluate_frames(frames)["map50"] for clip, frames in matched_by_clip.items()}
            drop, ci_low, ci_high = bootstrap_drop(matched_metrics, changed, args.bootstrap_samples, args.seed)
            common_clip_count = len(set(matched_metrics) & set(changed))
            claim_eligible = model != "yolomg_ard100" and common_clip_count >= 5
            comparisons.append(
                {
                    "model": model,
                    "intervention": intervention,
                    "common_clip_count": common_clip_count,
                    "claim_eligible": claim_eligible,
                    "matched_original_map50": evaluate_frames(matched_frames)["map50"],
                    "intervention_map50": evaluate_frames(frames_lookup[(model, intervention)])["map50"],
                    "relative_map50_drop": drop,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "supports_high_motion_limitation": claim_eligible and intervention in {"fast_2x", "accelerate_g2"} and drop > 0.10 and ci_low > 0.0,
                }
            )
    for comparison in comparisons:
        model, intervention = comparison["model"], comparison["intervention"]
        candidates = [row for row in clip_rows if row["model"] == model and row["intervention"] == intervention]
        if candidates:
            worst = min(candidates, key=lambda row: row["map50"])
            frames = [frame for frame in frames_lookup[(model, intervention)] if frame["clip"] == worst["clip"]]
            plot_worst_timeline(frames, f"{model} / {intervention} / {worst['clip']}", args.out_dir / f"timeline_{model}_{intervention}_{worst['clip']}.png")
    write_csv(args.out_dir / "summary.csv", summary_rows)
    write_csv(args.out_dir / "metrics_by_clip.csv", clip_rows)
    write_csv(args.out_dir / "metrics_by_frame.csv", per_frame_rows)
    write_csv(args.out_dir / "paired_bootstrap.csv", comparisons)
    plot_retention(summary_rows, args.out_dir / "map50_retention.png")
    integrity = {}
    for intervention in INTERVENTIONS:
        path = args.dataset_root / intervention / "integrity.json"
        if path.exists():
            integrity[intervention] = json.loads(path.read_text(encoding="utf-8"))
    interpretation_allowed = bool(integrity.get("original", {}).get("original_source_equivalent", False))
    if not interpretation_allowed:
        for comparison in comparisons:
            comparison["supports_high_motion_limitation"] = False
            comparison["interpretation_blocked_reason"] = "original_copy_failed_source_equivalence_gate"
    result = {"interpretation_allowed": interpretation_allowed, "summary": summary_rows, "paired_bootstrap": comparisons, "integrity": integrity}
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"models": models, "summary_rows": len(summary_rows), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
