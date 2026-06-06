from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def adjusted_score(raw_score: float, action_score: float, center: float, beta: float, mode: str, clip_min: float, clip_max: float) -> float:
    delta = action_score - center
    if mode == "suppress-only":
        value = raw_score - beta * max(0.0, -delta)
    elif mode == "boost-only":
        value = raw_score + beta * max(0.0, delta)
    else:
        value = raw_score + beta * delta
    return min(float(clip_max), max(float(clip_min), float(value)))


def load_tracklet_scores(tracklet_jsonl: Path, score_field: str, min_tracklet_rows: int) -> tuple[dict[tuple[str, int, int], float], dict[str, Any]]:
    scores: dict[tuple[str, int, int], float] = {}
    values: list[float] = []
    total_tracklets = 0
    scored_tracklets = 0
    skipped_short = 0
    missing_score = 0
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
                seq = str(row.get("seq") or meta.get("seq") or "")
                frame_id = row.get("frame_id")
                pred_index = row.get("prediction_index")
                if not seq or frame_id is None or pred_index is None:
                    continue
                scores[(seq, int(float(frame_id)), int(float(pred_index)))] = score
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
    return scores, summary


def iter_diag_files(run_root: Path, profile: str, diagnostics_name: str) -> list[Path]:
    profile_root = run_root / profile
    if not profile_root.exists():
        raise FileNotFoundError(profile_root)
    return sorted(path / diagnostics_name for path in profile_root.iterdir() if path.is_dir() and (path / diagnostics_name).exists())


def main() -> int:
    parser = argparse.ArgumentParser(description="Rescore Li-TETC Route-B diagnostics from scored action tracklets.")
    parser.add_argument("--in-run-root", type=Path, required=True)
    parser.add_argument("--out-run-root", type=Path, required=True)
    parser.add_argument("--tracklet-jsonl", type=Path, required=True)
    parser.add_argument("--profile", default="hard_recovery")
    parser.add_argument("--diagnostics-name", default="diagnostics_raw.jsonl")
    parser.add_argument("--score-field", default="vatd_score")
    parser.add_argument("--center", type=float, default=0.20)
    parser.add_argument("--beta", type=float, default=0.40)
    parser.add_argument("--mode", choices=["additive", "suppress-only", "boost-only"], default="additive")
    parser.add_argument("--missing-score-behavior", choices=["keep", "drop"], default="keep")
    parser.add_argument("--min-tracklet-rows", type=int, default=1)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=1.0)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    if args.clip_min > args.clip_max:
        raise ValueError("--clip-min must be <= --clip-max")
    score_map, score_summary = load_tracklet_scores(args.tracklet_jsonl, args.score_field, int(args.min_tracklet_rows))
    in_root = args.in_run_root.resolve()
    out_root = args.out_run_root.resolve()
    profile_out = out_root / args.profile
    profile_out.mkdir(parents=True, exist_ok=True)

    diagnostics_files = 0
    rows_seen = 0
    rows_scored = 0
    rows_missing_score = 0
    rows_written = 0
    rows_dropped = 0
    for in_diag in iter_diag_files(in_root, args.profile, args.diagnostics_name):
        diagnostics_files += 1
        seq = in_diag.parent.name
        out_seq_dir = profile_out / seq
        out_seq_dir.mkdir(parents=True, exist_ok=True)
        out_diag = out_seq_dir / args.diagnostics_name
        out_lines: list[str] = []
        with in_diag.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows_seen += 1
                frame_id = int(float(row.get("frame_id", 0) or 0))
                pred_index = row.get("prediction_index")
                score = None
                if pred_index is not None:
                    score = score_map.get((str(row.get("seq") or seq), frame_id, int(float(pred_index))))
                raw_conf = float(max(row.get("objectness", 0.0), row.get("final_drone_score", 0.0), row.get("score", 0.0)))
                if score is None:
                    rows_missing_score += 1
                    if args.missing_score_behavior == "drop":
                        rows_dropped += 1
                        continue
                    new_conf = raw_conf
                else:
                    rows_scored += 1
                    new_conf = adjusted_score(raw_conf, float(score), float(args.center), float(args.beta), args.mode, float(args.clip_min), float(args.clip_max))
                    row[args.score_field] = float(score)
                row["objectness_raw"] = raw_conf
                row["final_drone_score_raw"] = raw_conf
                row["objectness"] = float(new_conf)
                row["final_drone_score"] = float(new_conf)
                row["score"] = float(new_conf)
                row["li_tetc_action_rescore"] = {
                    "score_field": args.score_field,
                    "center": float(args.center),
                    "beta": float(args.beta),
                    "mode": args.mode,
                    "missing_score_behavior": args.missing_score_behavior,
                }
                out_lines.append(json.dumps(row, ensure_ascii=False))
                rows_written += 1
        out_diag.write_text(("\n".join(out_lines) + "\n") if out_lines else "", encoding="utf-8")

    summary = {
        "in_run_root": str(in_root),
        "out_run_root": str(out_root),
        "profile": args.profile,
        "diagnostics_name": args.diagnostics_name,
        "score_field": args.score_field,
        "center": float(args.center),
        "beta": float(args.beta),
        "mode": args.mode,
        "missing_score_behavior": args.missing_score_behavior,
        "diagnostics_files": diagnostics_files,
        "rows_seen": rows_seen,
        "rows_scored": rows_scored,
        "rows_missing_score": rows_missing_score,
        "rows_written": rows_written,
        "rows_dropped": rows_dropped,
        "score_summary": score_summary,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
