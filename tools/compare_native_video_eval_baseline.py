from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_BASELINE = {
    "precision": 0.9161701278,
    "recall": 0.9013069500,
    "map50": 0.9384170538,
    "map5095": 0.4685363007,
    "f1": 0.9080775537,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare native video detector eval metrics against TransVisDrone baseline.")
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--method-name", default="NativeVideoMVP")
    parser.add_argument("--baseline-name", default="TransVisDrone")
    parser.add_argument("--primary-metric", default="map50", choices=["precision", "recall", "map50", "map5095", "f1"])
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--baseline-precision", type=float, default=DEFAULT_BASELINE["precision"])
    parser.add_argument("--baseline-recall", type=float, default=DEFAULT_BASELINE["recall"])
    parser.add_argument("--baseline-map50", type=float, default=DEFAULT_BASELINE["map50"])
    parser.add_argument("--baseline-map5095", type=float, default=DEFAULT_BASELINE["map5095"])
    parser.add_argument("--baseline-f1", type=float, default=DEFAULT_BASELINE["f1"])
    parser.add_argument("--require-full-split", action="store_true")
    args = parser.parse_args()

    eval_data = json.loads(args.eval_json.read_text(encoding="utf-8-sig"))
    full_split = eval_data.get("full_split")
    max_samples = eval_data.get("max_samples")
    baseline = {
        "precision": float(args.baseline_precision),
        "recall": float(args.baseline_recall),
        "map50": float(args.baseline_map50),
        "map5095": float(args.baseline_map5095),
        "f1": float(args.baseline_f1),
    }
    metrics = {}
    for key, baseline_value in baseline.items():
        value = float(eval_data.get(key, 0.0))
        delta = value - baseline_value
        metrics[key] = {
            "method": value,
            "baseline": baseline_value,
            "delta": delta,
            "beat": delta >= args.min_delta,
        }

    primary = metrics[args.primary_metric]
    full_split_required_failed = args.require_full_split and full_split is not True
    status = "beat_baseline" if primary["beat"] else "below_baseline"
    if full_split_required_failed:
        status = "not_full_split"
    result = {
        "method_name": args.method_name,
        "baseline_name": args.baseline_name,
        "eval_json": str(args.eval_json.resolve()),
        "primary_metric": args.primary_metric,
        "min_delta": args.min_delta,
        "require_full_split": bool(args.require_full_split),
        "full_split": full_split,
        "max_samples": max_samples,
        "status": status,
        "primary": primary,
        "metrics": metrics,
        "eval_counts": {
            "images": int(eval_data.get("images", 0)),
            "labels": int(eval_data.get("labels", 0)),
            "detections": int(eval_data.get("detections", 0)),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if primary["beat"] and not full_split_required_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
