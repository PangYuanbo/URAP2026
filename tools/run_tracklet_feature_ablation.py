from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


GROUPS = [
    "detector_confidence",
    "bbox_geometry",
    "temporal_continuity",
    "background_fp",
    "action_motion",
    "all_except_detector_confidence",
    "all_except_bbox_geometry",
    "all_except_temporal_continuity",
    "all_except_background_fp",
    "all_except_action_motion",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("RUN " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.0425)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--groups", nargs="*", default=GROUPS)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    train_tracklets = repo / "artifacts/nps_sota_research/tvd_nps_trainval_tracklets/tracklets_with_vatd_scores_nps_trainval_nocrop.jsonl"
    test_tracklets = repo / "artifacts/nps_sota_research/tvd_nps_test_tracklets_v2/tracklets_with_vatd_scores_nps_traintrain_nocrop.jsonl"
    row_tracklets = repo / "artifacts/nps_sota_research/tvd_nps_test_tracklets_v2/tracklets_with_row_score_unique_hardneg005_trainval_nps.jsonl"
    predictionsgt = repo / "papers/TransVisDrone/runs/val/NPS_URAP_D/nps_test_best_aug_bs8_half/predictionsgt/predictionsgt_split_0.pkl"
    status_path = out_dir / "status.json"
    summary_path = out_dir / "summary.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    results: list[dict[str, Any]] = []
    write_status(
        status_path,
        {
            "state": "running",
            "started": started,
            "done": 0,
            "total": len(args.groups),
            "current_group": None,
            "results": results,
        },
    )

    for index, group in enumerate(args.groups, start=1):
        group_dir = out_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        score_field = f"meta_ablate_{group}"
        scored_tracklets = group_dir / "test_tracklets_scored.jsonl"
        model_path = group_dir / "model.pt"
        train_summary = group_dir / "train_summary.json"
        fusion_json = group_dir / "fusion_fixed.json"
        best_pkl = group_dir / "fusion_fixed_best.pkl"
        write_status(
            status_path,
            {
                "state": "running",
                "started": started,
                "done": index - 1,
                "total": len(args.groups),
                "current_group": group,
                "results": results,
            },
        )

        run_cmd(
            [
                sys.executable,
                "tools/train_tracklet_meta_score_head.py",
                "--train-tracklets",
                str(train_tracklets),
                "--test-tracklets",
                str(test_tracklets),
                "--out-test-tracklets",
                str(scored_tracklets),
                "--out-model",
                str(model_path),
                "--out-summary",
                str(train_summary),
                "--score-field",
                score_field,
                "--epochs",
                str(args.epochs),
                "--feature-groups",
                group,
            ],
            repo,
        )
        run_cmd(
            [
                sys.executable,
                "tools/sweep_tvd_predictionsgt_two_score_fusion.py",
                "--predictionsgt-pkl",
                str(predictionsgt),
                "--meta-tracklet-jsonl",
                str(scored_tracklets),
                "--meta-score-field",
                score_field,
                "--row-tracklet-jsonl",
                str(row_tracklets),
                "--row-score-field",
                "row_score_unique_hardneg005_trainval",
                "--modes",
                "meta-logit-row-geom",
                "--alphas",
                str(args.alpha),
                "--betas",
                str(args.beta),
                "--out-json",
                str(fusion_json),
                "--write-best-pkl",
                str(best_pkl),
            ],
            repo,
        )
        train_data = read_json(train_summary)
        fusion_data = read_json(fusion_json)
        best = fusion_data["best"]
        result = {
            "group": group,
            "done_index": index,
            "num_features": train_data["num_features"],
            "features": train_data["features"],
            "missing_requested_features": train_data["missing_requested_features"],
            "map50": best["map50"],
            "map5095": best["map5095"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "train_summary": str(train_summary),
            "fusion_json": str(fusion_json),
        }
        results.append(result)
        write_status(
            status_path,
            {
                "state": "running",
                "started": started,
                "done": index,
                "total": len(args.groups),
                "current_group": None,
                "results": results,
            },
        )

    summary = {
        "state": "complete",
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "epochs": args.epochs,
        "alpha": args.alpha,
        "beta": args.beta,
        "protocol": "TransVisDrone/NPS train+val tracklets -> test predictionsgt, fixed two-score fusion params",
        "baseline_previous_best": {
            "map50": 0.9400926118963192,
            "map5095": 0.4693808442024173,
            "precision": 0.91415575853381,
            "recall": 0.9004519359960914,
            "f1": 0.9072521019889026,
        },
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_status(status_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
