from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval_li_tetc_proposal_diagnostics import evaluate, load_diagnostics
from eval_tvd_coco_pkl_on_li_tetc import Box, load_gt
from rescore_li_tetc_diagnostics_from_tracklets import adjusted_score, load_tracklet_scores


def parse_csv_floats(text: str) -> list[float]:
    return [float(part) for part in text.replace(",", " ").split() if part.strip()]


def rescore_preds(
    preds: dict[tuple[int, int], list[tuple[float, Box]]],
    run_root: Path,
    profile: str,
    diagnostics_name: str,
    score_map: dict[tuple[str, int, int], float],
    center: float,
    beta: float,
    mode: str,
    missing_score_behavior: str,
) -> dict[tuple[int, int], list[tuple[float, Box]]]:
    out: dict[tuple[int, int], list[tuple[float, Box]]] = {}
    profile_root = run_root / profile
    for seq_dir in sorted(path for path in profile_root.glob("Clip_*") if path.is_dir()):
        try:
            video_id = int(seq_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        diag = seq_dir / diagnostics_name
        if not diag.exists():
            continue
        with diag.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                bbox = row.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                frame_id = int(float(row.get("frame_id", 0) or 0))
                pred_index = row.get("prediction_index")
                action_score = None
                if pred_index is not None:
                    action_score = score_map.get((str(row.get("seq") or seq_dir.name), frame_id, int(float(pred_index))))
                raw_score = float(max(row.get("objectness", 0.0), row.get("final_drone_score", 0.0), row.get("score", 0.0)))
                if action_score is None:
                    if missing_score_behavior == "drop":
                        continue
                    score = raw_score
                else:
                    score = adjusted_score(raw_score, float(action_score), center, beta, mode, 0.0, 1.0)
                x1, y1, x2, y2 = [float(v) for v in bbox]
                if x2 > x1 and y2 > y1:
                    out.setdefault((video_id, frame_id), []).append((score, Box(x1=x1, y1=y1, x2=x2, y2=y2)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep Li-TETC action-rescore parameters and select same-FP recall.")
    parser.add_argument("--repo-root", type=Path, default=Path("Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tracklet-jsonl", type=Path, required=True)
    parser.add_argument("--profile", default="hard_recovery")
    parser.add_argument("--diagnostics-name", default="diagnostics_raw.jsonl")
    parser.add_argument("--score-field", default="vatd_score")
    parser.add_argument("--videos", type=int, nargs="*", default=list(range(41, 51)))
    parser.add_argument("--match-pt-pipeline-sampling", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--empty-stride", type=int, default=10)
    parser.add_argument("--centers", default="0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50")
    parser.add_argument("--betas", default="0.10 0.20 0.30 0.40 0.60 0.80")
    parser.add_argument("--modes", nargs="*", default=["additive", "suppress-only", "boost-only"])
    parser.add_argument("--thresholds", default="0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 0.22 0.24 0.26 0.28 0.30")
    parser.add_argument("--fp-limit", type=int, default=17762)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    gt = load_gt(repo_root, args.videos)
    if args.match_pt_pipeline_sampling:
        frame_stride = max(1, int(args.frame_stride))
        empty_stride = max(1, int(args.empty_stride))
        gt = {
            key: boxes
            for key, boxes in gt.items()
            if ((key[1] - 1) % frame_stride == 0) and (boxes or (key[1] % empty_stride == 0))
        }
    score_map, score_summary = load_tracklet_scores(args.tracklet_jsonl.resolve(), args.score_field, min_tracklet_rows=1)
    centers = parse_csv_floats(args.centers)
    betas = parse_csv_floats(args.betas)
    thresholds = parse_csv_floats(args.thresholds)

    rows: list[dict[str, Any]] = []
    best_under_fp: dict[str, Any] | None = None
    for mode in args.modes:
        for center in centers:
            for beta in betas:
                rescored = rescore_preds(
                    {},
                    args.run_root.resolve(),
                    args.profile,
                    args.diagnostics_name,
                    score_map,
                    center,
                    beta,
                    mode,
                    missing_score_behavior="keep",
                )
                metrics = evaluate(gt, rescored, thresholds, iou_thr=0.5)
                for threshold, metric in metrics.items():
                    row = {
                        "mode": mode,
                        "center": center,
                        "beta": beta,
                        "threshold": float(threshold),
                        **metric,
                    }
                    rows.append(row)
                    if int(metric["fp"]) <= int(args.fp_limit):
                        if best_under_fp is None or float(metric["recall"]) > float(best_under_fp["recall"]):
                            best_under_fp = row

    summary = {
        "repo_root": str(repo_root),
        "run_root": str(args.run_root.resolve()),
        "tracklet_jsonl": str(args.tracklet_jsonl.resolve()),
        "score_field": args.score_field,
        "score_summary": score_summary,
        "fp_limit": int(args.fp_limit),
        "thresholds": thresholds,
        "centers": centers,
        "betas": betas,
        "modes": args.modes,
        "best_under_fp": best_under_fp,
        "top_under_fp": sorted([row for row in rows if int(row["fp"]) <= int(args.fp_limit)], key=lambda row: (-float(row["recall"]), int(row["fp"])))[:20],
        "top_recall": sorted(rows, key=lambda row: (-float(row["recall"]), int(row["fp"])))[:20],
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["fp_limit", "best_under_fp", "top_under_fp"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
