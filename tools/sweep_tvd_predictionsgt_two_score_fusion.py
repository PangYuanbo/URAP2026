from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

from rescore_li_tetc_diagnostics_from_tracklets import load_tracklet_scores
from sweep_tvd_predictionsgt_action_rescore import evaluate_data, image_key, parse_csv_floats
from sweep_tvd_predictionsgt_score_fusion import load_predictionsgt, load_row_scores


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _logit(value: float) -> float:
    value = min(1.0 - 1e-6, max(1e-6, float(value)))
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def fuse(raw: float, meta: float, row: float, alpha: float, beta: float, mode: str) -> float:
    raw = _clip01(raw)
    meta = _clip01(meta)
    row = _clip01(row)
    if mode == "logit-3mix":
        raw_w = max(0.0, 1.0 - alpha - beta)
        return _clip01(_sigmoid(raw_w * _logit(raw) + alpha * _logit(meta) + beta * _logit(row)))
    if mode == "meta-logit-row-geom":
        base = _clip01(_sigmoid((1.0 - alpha) * _logit(raw) + alpha * _logit(meta)))
        return _clip01(math.exp((1.0 - beta) * math.log(max(base, 1e-9)) + beta * math.log(max(row, 1e-9))))
    if mode == "meta-logit-row-suppress":
        base = _clip01(_sigmoid((1.0 - alpha) * _logit(raw) + alpha * _logit(meta)))
        return _clip01(base * (1.0 - beta * (1.0 - row)))
    if mode == "meta-logit-row-boost":
        base = _clip01(_sigmoid((1.0 - alpha) * _logit(raw) + alpha * _logit(meta)))
        return _clip01(base + beta * row * (1.0 - base))
    raise ValueError(f"unknown mode: {mode}")


def clone_with_scores(
    data: dict[str, Any],
    meta_scores: dict[tuple[str, int, int], float],
    row_scores: dict[tuple[str, int, int], float],
    mode: str,
    alpha: float,
    beta: float,
    missing_score_behavior: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for image_id, item in data.items():
        new_item = {"labels": item.get("labels", []), "detections": []}
        for pred_index, row in enumerate(item.get("detections", [])):
            key = image_key(str(image_id), pred_index)
            meta_score = meta_scores.get(key)
            row_score = row_scores.get(key)
            if meta_score is None or row_score is None:
                if missing_score_behavior == "drop":
                    continue
                new_item["detections"].append(row)
                continue
            new_row = dict(row)
            raw = float(new_row.get("score", 0.0))
            new_row["raw_score"] = raw
            new_row["fusion_meta_score"] = float(meta_score)
            new_row["fusion_row_score"] = float(row_score)
            new_row["score"] = fuse(raw, float(meta_score), float(row_score), alpha, beta, mode)
            new_item["detections"].append(new_row)
        out[image_id] = new_item
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep two-score fusion for TransVisDrone predictionsgt pkl.")
    parser.add_argument("--tvd-root", type=Path, default=Path("papers/TransVisDrone"))
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--meta-tracklet-jsonl", type=Path, required=True)
    parser.add_argument("--meta-score-field", required=True)
    parser.add_argument("--row-tracklet-jsonl", type=Path, required=True)
    parser.add_argument("--row-score-field", required=True)
    parser.add_argument("--modes", nargs="*", default=["logit-3mix", "meta-logit-row-geom", "meta-logit-row-suppress"])
    parser.add_argument("--alphas", default="0.05 0.06 0.07 0.0775 0.085")
    parser.add_argument("--betas", default="0.005 0.01 0.02 0.05")
    parser.add_argument("--missing-score-behaviors", nargs="*", default=["keep"])
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--write-best-pkl", type=Path, default=None)
    args = parser.parse_args()

    data = load_predictionsgt(args.predictionsgt_pkl.resolve())
    meta_scores, meta_summary = load_tracklet_scores(args.meta_tracklet_jsonl.resolve(), args.meta_score_field, min_tracklet_rows=1)
    row_scores, row_summary = load_row_scores(args.row_tracklet_jsonl.resolve(), args.row_score_field, min_tracklet_rows=1)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for missing in args.missing_score_behaviors:
        for mode in args.modes:
            for alpha in parse_csv_floats(args.alphas):
                for beta in parse_csv_floats(args.betas):
                    fused = clone_with_scores(data, meta_scores, row_scores, mode, alpha, beta, missing)
                    metrics = evaluate_data(fused, args.tvd_root, args.out_json.parent)
                    row = {"mode": mode, "alpha": alpha, "beta": beta, "missing_score_behavior": missing, **metrics}
                    rows.append(row)
                    if best is None or float(row["map50"]) > float(best["map50"]):
                        best = row
    summary = {
        "predictionsgt_pkl": str(args.predictionsgt_pkl.resolve()),
        "meta_tracklet_jsonl": str(args.meta_tracklet_jsonl.resolve()),
        "meta_score_field": args.meta_score_field,
        "row_tracklet_jsonl": str(args.row_tracklet_jsonl.resolve()),
        "row_score_field": args.row_score_field,
        "meta_summary": meta_summary,
        "row_summary": row_summary,
        "best": best,
        "top": sorted(rows, key=lambda row: (-float(row["map50"]), -float(row["recall"])))[:20],
        "rows": rows,
    }
    if args.write_best_pkl is not None and best is not None:
        best_data = clone_with_scores(
            data,
            meta_scores,
            row_scores,
            str(best["mode"]),
            float(best["alpha"]),
            float(best["beta"]),
            str(best["missing_score_behavior"]),
        )
        args.write_best_pkl.parent.mkdir(parents=True, exist_ok=True)
        with args.write_best_pkl.open("wb") as f:
            pickle.dump(best_data, f)
        summary["best_pkl"] = str(args.write_best_pkl.resolve())
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["best", "top"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
