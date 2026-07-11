from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
TVD = Path(r"D:\urap_modal_stage\TransVisDrone")
DATA = Path(r"D:\URAP_nps_val_tvd")
TEST_INPUT = Path(r"D:\URAP_vatd_rank_inputs")
OUTPUT = Path(r"D:\URAP_vatd_rank_results\nps_official_val_xgb_rank_v1")
RUNNER = REPO / "artifacts/detached_nps_official_val_xgb_rank"
PROGRESS = RUNNER / "progress.json"
PYTHON = Path(sys.executable)


def write_progress(stage: str, done: int, total: int = 2, **extra: object) -> None:
    RUNNER.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps({"stage": stage, "done": done, "total": total, "updated": datetime.now(timezone.utc).astimezone().isoformat(), **extra}, indent=2), encoding="utf-8")


def run(command: list[str], stage: str, done: int) -> None:
    print(json.dumps({"kind": "pipeline_command", "stage": stage, "command": command}), flush=True)
    process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO / "tools")})
    write_progress(stage, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scored = OUTPUT / "nps_test_tracklets_xgb_scored.jsonl"
    run([
        str(PYTHON), str(REPO / "tools/train_detection_row_score_xgb.py"),
        "--train-tracklets", str(DATA / "route_b_official/tracklets/proposal_tracklets.jsonl"),
        "--train-gt-csv", str(DATA / "route_b_official/gt.csv"),
        "--test-tracklets", str(TEST_INPUT / "nps_tracklets_with_vatd.jsonl"),
        "--out-test-tracklets", str(scored),
        "--out-model", str(OUTPUT / "xgb_rank_model.json"),
        "--out-summary", str(OUTPUT / "train_summary.json"),
        "--score-field", "xgb_rank_score",
        "--label-policy", "unique-iou", "--negative-min-score", "0.005",
        "--rounds", "1600", "--early-stopping-rounds", "120",
        "--max-depth", "7", "--eta", "0.03", "--min-child-weight", "5",
        "--subsample", "0.85", "--colsample-bytree", "0.85", "--scale-pos-weight", "4.0",
    ], "train", 0)
    run([
        str(PYTHON), str(REPO / "tools/sweep_tvd_predictionsgt_two_score_fusion_fast.py"),
        "--tvd-root", str(TVD), "--predictionsgt-pkl", str(TEST_INPUT / "nps_predictionsgt_split_0.pkl"),
        "--meta-tracklet-jsonl", str(scored), "--meta-score-field", "vatd_score",
        "--row-tracklet-jsonl", str(scored), "--row-score-field", "xgb_rank_score",
        "--modes", "logit-3mix", "meta-logit-row-geom", "meta-logit-row-suppress", "meta-logit-row-boost",
        "--alphas", "0.00", "0.01", "0.02", "0.04", "0.06", "0.08", "0.10", "0.14", "0.20",
        "--betas", "0.005", "0.01", "0.02", "0.04", "0.06", "0.08", "0.10", "0.12", "0.16", "0.20", "0.24", "0.32", "0.40",
        "--out-json", str(OUTPUT / "fusion_sweep_fast.json"), "--write-best-pkl", str(OUTPUT / "best_predictionsgt.pkl"),
    ], "evaluate", 1)
    summary = json.loads((OUTPUT / "fusion_sweep_fast.json").read_text(encoding="utf-8"))
    write_progress("done", 2, best=summary.get("best"), output=str(OUTPUT))
    print(json.dumps({"kind": "pipeline_done", "best": summary.get("best"), "output": str(OUTPUT)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
