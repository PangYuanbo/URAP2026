from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("RUN " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-windows-per-tracklet", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=0.0425)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    status_path = out_dir / "status.json"
    train_tracklets = repo / "artifacts/nps_sota_research/tvd_nps_trainval_tracklets/tracklets_with_vatd_scores_nps_trainval_nocrop.jsonl"
    test_tracklets = repo / "artifacts/nps_sota_research/tvd_nps_test_tracklets_v2/tracklets_with_vatd_scores_nps_traintrain_nocrop.jsonl"
    row_tracklets = repo / "artifacts/nps_sota_research/tvd_nps_test_tracklets_v2/tracklets_with_row_score_unique_hardneg005_trainval_nps.jsonl"
    predictionsgt = repo / "papers/TransVisDrone/runs/val/NPS_URAP_D/nps_test_best_aug_bs8_half/predictionsgt/predictionsgt_split_0.pkl"
    scored_tracklets = out_dir / "test_tracklets_scored.jsonl"
    model_path = out_dir / "two_branch_motion_action.pt"
    train_summary = out_dir / "train_summary.json"
    fusion_json = out_dir / "fusion_fixed.json"
    best_pkl = out_dir / "fusion_fixed_best.pkl"
    score_field = "two_branch_motion_action_score"
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    write_status(status_path, {"state": "running", "started": started, "step": "train", "done": 0, "total": 2})
    train_cmd = [
        sys.executable,
        "tools/train_two_branch_motion_action_scorer.py",
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
        "--batch-size",
        str(args.batch_size),
        "--past-len",
        "8",
        "--future-len",
        "2",
    ]
    if args.max_windows_per_tracklet is not None:
        train_cmd.extend(["--max-windows-per-tracklet", str(args.max_windows_per_tracklet)])
    run_cmd(train_cmd, repo)

    write_status(status_path, {"state": "running", "started": started, "step": "fusion_eval", "done": 1, "total": 2})
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
    summary = {
        "state": "complete",
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": "TransVisDrone/NPS train+val tracklets -> test predictionsgt, two-branch detector-evidence + motion-action model, fixed two-score fusion params",
        "train_summary": train_data,
        "fusion_best": fusion_data["best"],
        "fusion_json": str(fusion_json),
        "best_pkl": str(best_pkl),
        "baseline_previous_best": {
            "map50": 0.9400926118963192,
            "map5095": 0.4693808442024173,
            "precision": 0.91415575853381,
            "recall": 0.9004519359960914,
            "f1": 0.9072521019889026,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_status(status_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
