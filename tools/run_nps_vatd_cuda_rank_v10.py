from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
OUT = REPO / "tmp/nps_vatd_cuda_rank_v10"
RUNNER = REPO / "tmp/nps_vatd_cuda_rank_v10_runner"
PROGRESS = RUNNER / "progress.json"
PYTHON = Path(sys.executable)



def environment() -> dict[str, str]:
    return os.environ.copy()


def write_progress(stage: str, done: int, total: int = 2, **extra: object) -> None:
    RUNNER.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now(timezone.utc).astimezone().isoformat(), **extra}
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(args: list[str], stage: str, done: int) -> None:
    print(json.dumps({"kind": "nps_vatd_cuda_rank_command", "args": args}), flush=True)
    process = subprocess.Popen(args, cwd=REPO, env=environment())
    write_progress(stage, done, child_pid=process.pid, command=args)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, args)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run([
        str(PYTHON), "tools/train_detection_row_score_head.py",
        "--train-tracklets", "artifacts/nps_sota_research/tvd_nps_trainval_tracklets/tracklets_with_vatd_scores_nps_trainval_nocrop.jsonl",
        "--train-gt-csv", "artifacts/nps_sota_research/tvd_nps_train_route_b_v3/gt.csv", "artifacts/nps_sota_research/tvd_nps_val_route_b_v2/gt.csv",
        "--test-tracklets", "artifacts/nps_sota_research/tvd_nps_test_tracklets_v2/tracklets_with_vatd_scores_nps_traintrain_nocrop.jsonl",
        "--out-test-tracklets", str(OUT / "test_tracklets_scored.jsonl"), "--out-model", str(OUT / "model.pt"),
        "--out-summary", str(OUT / "train_summary.json"), "--score-field", "row_vatd_cuda_rank_score",
        "--iou-threshold", "0.5", "--negative-min-score", "0.005", "--label-policy", "unique-iou",
        "--epochs", "30", "--batch-size", "16384", "--hidden", "256", "--lr", "0.0005",
        "--pairwise-weight", "1.0", "--pairwise-pairs", "32768", "--model-kind", "unified-two-tower",
        "--tracklet-aux-weight", "0.25", "--feature-groups", "all",
    ], "train", 0)
    run([
        str(PYTHON), "tools/sweep_tvd_predictionsgt_two_score_fusion.py",
        "--predictionsgt-pkl", "papers/TransVisDrone/runs/val/NPS_URAP_D/nps_test_best_aug_bs8_half/predictionsgt/predictionsgt_split_0.pkl",
        "--meta-tracklet-jsonl", "artifacts/nps_sota_research/nps_vatd_gbdt_trainval_v1/test_tracklets_scored.jsonl",
        "--meta-score-field", "vatd_gbdt_score", "--row-tracklet-jsonl", str(OUT / "test_tracklets_scored.jsonl"),
        "--row-score-field", "row_vatd_cuda_rank_score", "--modes", "logit-3mix", "meta-logit-row-geom", "meta-logit-row-boost",
        "--alphas", "0.00 0.02 0.04 0.06", "--betas", "0.05 0.10 0.18 0.30",
        "--out-json", str(OUT / "fusion_sweep.json"), "--write-best-pkl", str(OUT / "best_predictionsgt.pkl"),
    ], "evaluate", 1)
    write_progress("done", 2)
    print(json.dumps({"kind": "nps_vatd_cuda_rank_done", "done": 2, "total": 2}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

