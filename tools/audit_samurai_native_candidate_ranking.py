from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval_tvd_predictionsgt_pkl import load_predictionsgt
from sweep_tvd_predictionsgt_action_rescore import image_key
from sweep_tvd_predictionsgt_score_fusion import load_row_scores


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def iou(left: list[float], right: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in left]
    bx1, by1, bx2, by2 = [float(value) for value in right]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / max(1e-9, union)


def logit(value: float) -> float:
    value = min(1.0 - 1e-6, max(1e-6, value))
    return math.log(value / (1.0 - value))


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def fused(raw: float, auxiliary: float, alpha: float) -> float:
    return sigmoid((1.0 - alpha) * logit(raw) + alpha * logit(auxiliary))


def candidate_correct(candidate: dict[str, Any], labels: list[dict[str, Any]], threshold: float) -> bool:
    candidate_box = candidate.get("bbox")
    if not isinstance(candidate_box, list) or len(candidate_box) != 4:
        return False
    candidate_class = int(candidate.get("category_id", 0))
    return any(
        int(label.get("category_id", 0)) == candidate_class
        and isinstance(label.get("bbox"), list)
        and len(label["bbox"]) == 4
        and iou(candidate_box, label["bbox"]) >= threshold
        for label in labels
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit native SAMURAI candidate reranking fixes and breaks.")
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--samurai-jsonl", type=Path, required=True)
    parser.add_argument("--score-field", default="samurai_native_score")
    parser.add_argument("--alphas", default="0 0.01 0.02 0.04 0.06 0.08 0.10 0.15 0.20 0.30 0.40 0.50")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    predictions = load_predictionsgt(args.predictionsgt_pkl)
    scores, score_summary = load_row_scores(args.samurai_jsonl, args.score_field, 1)
    alphas = [float(value) for value in args.alphas.replace(",", " ").split()]
    totals = {alpha: defaultdict(int) for alpha in alphas}
    sequence_totals = {alpha: defaultdict(lambda: defaultdict(int)) for alpha in alphas}
    examples = {alpha: {"fixes": [], "breaks": []} for alpha in alphas}
    covered_frames = 0

    for image_id, item in predictions.items():
        detections = list(item.get("detections") or [])
        if not detections:
            continue
        sequence, frame_id, _ = image_key(str(image_id), 0)
        available = [(index, scores.get((sequence, frame_id, index))) for index in range(len(detections))]
        if not any(score is not None for _, score in available):
            continue
        labels = list(item.get("labels") or [])
        if not labels:
            continue
        covered_frames += 1
        raw_index = max(range(len(detections)), key=lambda index: finite(detections[index].get("score")))
        raw_correct = candidate_correct(detections[raw_index], labels, args.iou_threshold)
        for alpha in alphas:
            selected = max(
                range(len(detections)),
                key=lambda index: fused(
                    finite(detections[index].get("score")),
                    finite(scores.get((sequence, frame_id, index))),
                    alpha,
                ),
            )
            selected_correct = candidate_correct(detections[selected], labels, args.iou_threshold)
            bucket = totals[alpha]
            seq_bucket = sequence_totals[alpha][sequence]
            for target in (bucket, seq_bucket):
                target["frames"] += 1
                target["raw_correct"] += int(raw_correct)
                target["selected_correct"] += int(selected_correct)
                target["fixes"] += int((not raw_correct) and selected_correct)
                target["breaks"] += int(raw_correct and (not selected_correct))
            kind = "fixes" if (not raw_correct) and selected_correct else "breaks" if raw_correct and (not selected_correct) else None
            if kind and len(examples[alpha][kind]) < 30:
                examples[alpha][kind].append({
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "raw_index": raw_index,
                    "selected_index": selected,
                    "raw_score": finite(detections[raw_index].get("score")),
                    "selected_raw_score": finite(detections[selected].get("score")),
                    "raw_top_samurai_score": finite(scores.get((sequence, frame_id, raw_index))),
                    "selected_samurai_score": finite(scores.get((sequence, frame_id, selected))),
                    "raw_margin": finite(detections[raw_index].get("score")) - finite(detections[selected].get("score")),
                })

    rows = []
    for alpha in alphas:
        row = dict(totals[alpha])
        frames = max(1, row.get("frames", 0))
        row.update({
            "alpha": alpha,
            "raw_top1_accuracy": row.get("raw_correct", 0) / frames,
            "selected_top1_accuracy": row.get("selected_correct", 0) / frames,
            "net_fixes": row.get("fixes", 0) - row.get("breaks", 0),
            "sequences": {name: dict(values) for name, values in sequence_totals[alpha].items()},
            "examples": examples[alpha],
        })
        rows.append(row)
    best = max(rows, key=lambda row: (row["selected_top1_accuracy"], row["net_fixes"])) if rows else None
    output = {
        "predictionsgt_pkl": str(args.predictionsgt_pkl.resolve()),
        "samurai_jsonl": str(args.samurai_jsonl.resolve()),
        "score_summary": score_summary,
        "covered_frames": covered_frames,
        "best": best,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"covered_frames": covered_frames, "best": best}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
