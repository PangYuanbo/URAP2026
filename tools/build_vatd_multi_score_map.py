from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path


def logit(value: float) -> float:
    value = min(1.0 - 1e-6, max(1e-6, float(value)))
    return math.log(value / (1.0 - value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def mix(a: float, b: float, weight: float) -> float:
    return sigmoid((1.0 - weight) * logit(a) + weight * logit(b))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--aot-jsonl", type=Path, required=True)
    parser.add_argument("--xgb-jsonl", type=Path, required=True)
    parser.add_argument("--out-pkl", type=Path, required=True)
    args = parser.parse_args()
    fields = [
        "val_row", "val_track_mean", "val_track_max", "val_row_mean_geom", "val_row_max_geom",
        "val_track_max_logit25", "val_track_max_logit50", "val_track_max_logit75",
        "val_aot_logit10", "val_aot_logit25", "val_aot_logit50",
        "val_xgb_logit10", "val_xgb_logit25", "val_xgb_logit50",
        "val_aot_xgb_logit",
    ]
    scores: dict[tuple[str, int, int], list[float]] = {}
    tracklets = 0
    rows = 0
    with args.val_jsonl.open("r", encoding="utf-8-sig") as fv, args.aot_jsonl.open("r", encoding="utf-8-sig") as fa, args.xgb_jsonl.open("r", encoding="utf-8-sig") as fx:
        for val_line, aot_line, xgb_line in zip(fv, fa, fx, strict=True):
            if not val_line.strip():
                continue
            val_item = json.loads(val_line)
            aot_item = json.loads(aot_line)
            xgb_item = json.loads(xgb_line)
            val_rows = val_item.get("rows") or []
            aot_rows = aot_item.get("rows") or []
            xgb_rows = xgb_item.get("rows") or []
            if not (len(val_rows) == len(aot_rows) == len(xgb_rows)):
                raise RuntimeError("row-count mismatch")
            val_values = [float(row["official_val_rank_score"]) for row in val_rows]
            if not val_values:
                continue
            track_mean = sum(val_values) / len(val_values)
            track_max = max(val_values)
            for val_row, aot_row, xgb_row, val_score in zip(val_rows, aot_rows, xgb_rows, val_values, strict=True):
                identity = (str(val_row.get("seq")), int(float(val_row.get("frame_id"))), int(float(val_row.get("prediction_index"))))
                aot_identity = (str(aot_row.get("seq")), int(float(aot_row.get("frame_id"))), int(float(aot_row.get("prediction_index"))))
                xgb_identity = (str(xgb_row.get("seq")), int(float(xgb_row.get("frame_id"))), int(float(xgb_row.get("prediction_index"))))
                if identity != aot_identity or identity != xgb_identity:
                    raise RuntimeError(f"identity mismatch: {identity} {aot_identity} {xgb_identity}")
                aot_score = float(aot_row["aot_rank_score"])
                xgb_score = float(xgb_row["xgb_rank_score"])
                values = [
                    val_score,
                    track_mean,
                    track_max,
                    math.sqrt(max(1e-12, val_score * track_mean)),
                    math.sqrt(max(1e-12, val_score * track_max)),
                    mix(val_score, track_max, 0.25),
                    mix(val_score, track_max, 0.50),
                    mix(val_score, track_max, 0.75),
                    mix(val_score, aot_score, 0.10),
                    mix(val_score, aot_score, 0.25),
                    mix(val_score, aot_score, 0.50),
                    mix(val_score, xgb_score, 0.10),
                    mix(val_score, xgb_score, 0.25),
                    mix(val_score, xgb_score, 0.50),
                    sigmoid(0.70 * logit(val_score) + 0.15 * logit(aot_score) + 0.15 * logit(xgb_score)),
                ]
                scores[identity] = values
                rows += 1
            tracklets += 1
            if tracklets % 10000 == 0:
                print(json.dumps({"kind": "score_map_progress", "tracklets": tracklets, "rows": rows}), flush=True)
    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pkl.open("wb") as handle:
        pickle.dump({"fields": fields, "scores": scores, "tracklets": tracklets, "rows": rows}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps({"kind": "score_map_done", "fields": fields, "tracklets": tracklets, "rows": rows, "out": str(args.out_pkl)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
