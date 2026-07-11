from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path


def clip_probability(value: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, float(value)))


def logit(value: float) -> float:
    value = clip_probability(value)
    return math.log(value / (1.0 - value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def identity(row: dict[str, object]) -> tuple[str, int, int]:
    return (
        str(row.get("seq")),
        int(float(row.get("frame_id", 0))),
        int(float(row.get("prediction_index", 0))),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--strong-jsonl", type=Path, required=True)
    parser.add_argument("--hard-jsonl", type=Path, required=True)
    parser.add_argument("--xgb-jsonl", type=Path, required=True)
    parser.add_argument("--out-pkl", type=Path, required=True)
    args = parser.parse_args()

    fields = [
        "val",
        "val_strong_logit10", "val_strong_logit25", "val_strong_logit50",
        "val_hard_logit10", "val_hard_logit25", "val_hard_logit50",
        "val_xgb_logit10", "val_xgb_logit25",
        "val_strong_hard_logit",
        "val_strong_hard_xgb_logit",
        "val_strong_hard_mean",
    ]
    scores: dict[tuple[str, int, int], list[float]] = {}
    rows = 0
    tracklets = 0
    paths = [args.val_jsonl, args.strong_jsonl, args.hard_jsonl, args.xgb_jsonl]
    handles = [path.open("r", encoding="utf-8-sig") for path in paths]
    try:
        for lines in zip(*handles, strict=True):
            items = [json.loads(line) for line in lines]
            row_sets = [item.get("rows") or [] for item in items]
            if len({len(row_set) for row_set in row_sets}) != 1:
                raise RuntimeError("row-count mismatch")
            for row_group in zip(*row_sets, strict=True):
                identities = [identity(row) for row in row_group]
                if len(set(identities)) != 1:
                    raise RuntimeError(f"identity mismatch: {identities}")
                val = float(row_group[0]["official_val_rank_score"])
                strong = float(row_group[1]["strong_val_rank_score"])
                hard = float(row_group[2]["hardpair_rank_score"])
                xgb = float(row_group[3]["xgb_rank_score"])
                logits = [logit(value) for value in (val, strong, hard, xgb)]
                values = [
                    val,
                    sigmoid(0.90 * logits[0] + 0.10 * logits[1]),
                    sigmoid(0.75 * logits[0] + 0.25 * logits[1]),
                    sigmoid(0.50 * logits[0] + 0.50 * logits[1]),
                    sigmoid(0.90 * logits[0] + 0.10 * logits[2]),
                    sigmoid(0.75 * logits[0] + 0.25 * logits[2]),
                    sigmoid(0.50 * logits[0] + 0.50 * logits[2]),
                    sigmoid(0.90 * logits[0] + 0.10 * logits[3]),
                    sigmoid(0.75 * logits[0] + 0.25 * logits[3]),
                    sigmoid(0.70 * logits[0] + 0.15 * logits[1] + 0.15 * logits[2]),
                    sigmoid(0.70 * logits[0] + 0.10 * logits[1] + 0.10 * logits[2] + 0.10 * logits[3]),
                    0.70 * val + 0.15 * strong + 0.15 * hard,
                ]
                scores[identities[0]] = values
                rows += 1
            tracklets += 1
    finally:
        for handle in handles:
            handle.close()

    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pkl.open("wb") as handle:
        pickle.dump({"fields": fields, "scores": scores, "tracklets": tracklets, "rows": rows}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps({"fields": fields, "tracklets": tracklets, "rows": rows, "output": str(args.out_pkl)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
