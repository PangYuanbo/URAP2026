from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite YOLOMG prediction label confidences from scored Route-B tracklets.")
    parser.add_argument("--images-list", type=Path, required=True)
    parser.add_argument("--pred-label-dir", type=Path, required=True)
    parser.add_argument("--tracklet-jsonl", type=Path, required=True)
    parser.add_argument("--out-label-dir", type=Path, required=True)
    parser.add_argument("--score-field", default="video_action_model_fusion_score")
    parser.add_argument("--invert-score", action="store_true", help="Use 1-score before applying center/beta.")
    parser.add_argument("--center", type=float, default=0.20)
    parser.add_argument("--beta", type=float, default=0.40)
    parser.add_argument("--mode", choices=["additive", "suppress-only", "boost-only"], default="additive")
    parser.add_argument("--missing-score-behavior", choices=["keep", "drop"], default="keep")
    parser.add_argument("--min-tracklet-rows", type=int, default=1)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=1.0)
    return parser.parse_args()


def read_image_list(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def adjusted_score(raw_score: float, action_score: float, center: float, beta: float, mode: str, clip_min: float, clip_max: float) -> float:
    delta = action_score - center
    if mode == "suppress-only":
        value = raw_score - beta * max(0.0, -delta)
    elif mode == "boost-only":
        value = raw_score + beta * max(0.0, delta)
    else:
        value = raw_score + beta * delta
    return min(float(clip_max), max(float(clip_min), float(value)))


def load_scores(tracklet_jsonl: Path, score_field: str, min_tracklet_rows: int) -> tuple[dict[tuple[str, int], float], dict[str, Any]]:
    score_by_prediction: dict[tuple[str, int], float] = {}
    total_tracklets = 0
    scored_tracklets = 0
    skipped_short = 0
    missing_score = 0
    rows_scored = 0
    values: list[float] = []
    with tracklet_jsonl.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            total_tracklets += 1
            item = json.loads(line)
            meta = dict(item.get("meta") or {})
            rows = [dict(row) for row in (item.get("rows") or [])]
            if len(rows) < min_tracklet_rows:
                skipped_short += 1
                continue
            raw_score = meta.get(score_field)
            if raw_score is None and rows:
                raw_score = rows[0].get(score_field)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                missing_score += 1
                continue
            scored_tracklets += 1
            values.append(score)
            for row in rows:
                image_path = row.get("image_path")
                pred_index = row.get("prediction_index")
                if image_path is None or pred_index is None:
                    continue
                key = (Path(str(image_path)).stem, int(float(pred_index)))
                score_by_prediction[key] = score
                rows_scored += 1
    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "score_field": score_field,
        "total_tracklets": total_tracklets,
        "scored_tracklets": scored_tracklets,
        "skipped_short_tracklets": skipped_short,
        "missing_score_tracklets": missing_score,
        "scored_prediction_rows": rows_scored,
        "mean_score": sum(values) / len(values) if values else None,
    }
    return score_by_prediction, summary


def main() -> None:
    args = parse_args()
    if args.clip_min > args.clip_max:
        raise ValueError("--clip-min must be <= --clip-max")
    images = read_image_list(args.images_list)
    score_by_prediction, score_summary = load_scores(args.tracklet_jsonl, args.score_field, args.min_tracklet_rows)
    args.out_label_dir.mkdir(parents=True, exist_ok=True)

    images_seen = 0
    label_files_seen = 0
    prediction_rows = 0
    rows_scored = 0
    rows_missing_score = 0
    rows_written = 0
    rows_dropped = 0
    for image_path in images:
        images_seen += 1
        in_path = args.pred_label_dir / f"{image_path.stem}.txt"
        out_path = args.out_label_dir / f"{image_path.stem}.txt"
        if not in_path.exists():
            continue
        label_files_seen += 1
        out_lines: list[str] = []
        for pred_index, line in enumerate(in_path.read_text(encoding="utf-8-sig").splitlines()):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            prediction_rows += 1
            raw_conf = float(parts[5]) if len(parts) >= 6 else 1.0
            score = score_by_prediction.get((image_path.stem, pred_index))
            if score is None:
                rows_missing_score += 1
                if args.missing_score_behavior == "drop":
                    rows_dropped += 1
                    continue
                new_conf = raw_conf
            else:
                rows_scored += 1
                action_score = 1.0 - score if args.invert_score else score
                new_conf = adjusted_score(raw_conf, action_score, args.center, args.beta, args.mode, args.clip_min, args.clip_max)
            out_parts = parts[:5] + [f"{new_conf:.8f}"]
            out_lines.append(" ".join(out_parts))
            rows_written += 1
        out_path.write_text(("\n".join(out_lines) + "\n") if out_lines else "", encoding="utf-8")

    summary = {
        "images_list": str(args.images_list),
        "pred_label_dir": str(args.pred_label_dir),
        "out_label_dir": str(args.out_label_dir),
        "center": args.center,
        "beta": args.beta,
        "mode": args.mode,
        "missing_score_behavior": args.missing_score_behavior,
        "images_seen": images_seen,
        "label_files_seen": label_files_seen,
        "prediction_rows": prediction_rows,
        "rows_scored": rows_scored,
        "rows_missing_score": rows_missing_score,
        "rows_written": rows_written,
        "rows_dropped": rows_dropped,
        "score_summary": score_summary,
        "invert_score": bool(args.invert_score),
    }
    summary_path = args.out_label_dir.parent / f"{args.out_label_dir.name}_rescore_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
