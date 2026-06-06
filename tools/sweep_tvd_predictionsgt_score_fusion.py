from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

from eval_tvd_predictionsgt_pkl import load_predictionsgt
from rescore_li_tetc_diagnostics_from_tracklets import load_tracklet_scores
from sweep_tvd_predictionsgt_action_rescore import evaluate_data, image_key, parse_csv_floats


def load_row_scores(tracklet_jsonl: Path, score_field: str, min_tracklet_rows: int) -> tuple[dict[tuple[str, int, int], float], dict[str, Any]]:
    scores: dict[tuple[str, int, int], float] = {}
    values: list[float] = []
    total_tracklets = 0
    skipped_short = 0
    missing_score_rows = 0
    rows_scored = 0
    with tracklet_jsonl.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            total_tracklets += 1
            item = json.loads(line)
            meta = dict(item.get("meta") or {})
            rows = [dict(row) for row in item.get("rows") or []]
            if len(rows) < min_tracklet_rows:
                skipped_short += 1
                continue
            for row in rows:
                raw_score = row.get(score_field)
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    missing_score_rows += 1
                    continue
                seq = str(row.get("seq") or meta.get("seq") or "")
                frame_id = row.get("frame_id")
                pred_index = row.get("prediction_index")
                if not seq or frame_id is None or pred_index is None:
                    continue
                scores[(seq, int(float(frame_id)), int(float(pred_index)))] = score
                values.append(score)
                rows_scored += 1
    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "score_field": score_field,
        "score_grain": "row",
        "total_tracklets": total_tracklets,
        "skipped_short_tracklets": skipped_short,
        "missing_score_rows": missing_score_rows,
        "scored_prediction_rows": rows_scored,
        "mean_score": sum(values) / len(values) if values else None,
    }
    return scores, summary


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


def fuse_score(raw_score: float, aux_score: float, alpha: float, mode: str) -> float:
    raw = _clip01(raw_score)
    aux = _clip01(aux_score)
    if mode == "replace":
        return aux
    if mode == "linear-mix":
        return _clip01((1.0 - alpha) * raw + alpha * aux)
    if mode == "logit-add":
        return _clip01(_sigmoid(_logit(raw) + alpha * _logit(aux)))
    if mode == "logit-mix":
        return _clip01(_sigmoid((1.0 - alpha) * _logit(raw) + alpha * _logit(aux)))
    if mode == "geom-mix":
        raw = max(raw, 1e-9)
        aux = max(aux, 1e-9)
        return _clip01(math.exp((1.0 - alpha) * math.log(raw) + alpha * math.log(aux)))
    if mode == "fp-suppress":
        return _clip01(raw * (1.0 - alpha * (1.0 - aux)))
    if mode == "tp-boost":
        return _clip01(raw + alpha * aux * (1.0 - raw))
    raise ValueError(f"unknown mode: {mode}")


def clone_with_fused_scores(
    data: dict[str, Any],
    score_map: dict[tuple[str, int, int], float],
    mode: str,
    alpha: float,
    missing_score_behavior: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for image_id, item in data.items():
        new_item = {"labels": item.get("labels", []), "detections": []}
        for pred_index, row in enumerate(item.get("detections", [])):
            aux_score = score_map.get(image_key(str(image_id), pred_index))
            if aux_score is None:
                if missing_score_behavior == "drop":
                    continue
                new_item["detections"].append(row)
                continue
            new_row = dict(row)
            raw_score = float(new_row.get("score", 0.0))
            new_row["raw_score"] = raw_score
            new_row["fusion_aux_score"] = float(aux_score)
            new_row["score"] = fuse_score(raw_score, float(aux_score), alpha, mode)
            new_item["detections"].append(new_row)
        out[image_id] = new_item
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep direct score fusion for TransVisDrone predictionsgt pkl.")
    parser.add_argument("--tvd-root", type=Path, default=Path("papers/TransVisDrone"))
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--tracklet-jsonl", type=Path, required=True)
    parser.add_argument("--score-field", default="meta_score_noleak")
    parser.add_argument("--per-row-score", action="store_true", help="Read score-field from each row instead of one score per tracklet")
    parser.add_argument("--min-tracklet-rows", type=int, default=1)
    parser.add_argument("--modes", nargs="*", default=["linear-mix", "logit-mix", "fp-suppress", "tp-boost"])
    parser.add_argument("--alphas", default="0.001 0.002 0.005 0.01 0.02 0.05 0.10 0.20")
    parser.add_argument("--missing-score-behaviors", nargs="*", default=["keep"])
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--write-best-pkl", type=Path, default=None)
    args = parser.parse_args()

    data = load_predictionsgt(args.predictionsgt_pkl.resolve())
    if args.per_row_score:
        score_map, score_summary = load_row_scores(args.tracklet_jsonl.resolve(), args.score_field, int(args.min_tracklet_rows))
    else:
        score_map, score_summary = load_tracklet_scores(args.tracklet_jsonl.resolve(), args.score_field, min_tracklet_rows=int(args.min_tracklet_rows))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    out_dir = args.out_json.parent
    for missing in args.missing_score_behaviors:
        for mode in args.modes:
            for alpha in parse_csv_floats(args.alphas):
                fused = clone_with_fused_scores(data, score_map, mode, alpha, missing)
                metrics = evaluate_data(fused, args.tvd_root, out_dir)
                row = {
                    "mode": mode,
                    "alpha": alpha,
                    "missing_score_behavior": missing,
                    **metrics,
                }
                rows.append(row)
                if best is None or float(row["map50"]) > float(best["map50"]):
                    best = row

    summary = {
        "predictionsgt_pkl": str(args.predictionsgt_pkl.resolve()),
        "tracklet_jsonl": str(args.tracklet_jsonl.resolve()),
        "score_field": args.score_field,
        "score_summary": score_summary,
        "best": best,
        "top": sorted(rows, key=lambda row: (-float(row["map50"]), -float(row["recall"])))[:20],
        "rows": rows,
    }
    if args.write_best_pkl is not None and best is not None:
        best_data = clone_with_fused_scores(
            data,
            score_map,
            str(best["mode"]),
            float(best["alpha"]),
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
