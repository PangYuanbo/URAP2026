from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(r"C:\Users\aaron\Desktop\URAP")
TVD = Path(r"D:\urap_modal_stage\TransVisDrone")
VAL = Path(r"D:\URAP_nps_val_tvd")
OUTPUT = Path(r"D:\URAP_vatd_rank_results\nps_val_samurai_cmc_v1")
RUNNER = REPO / "artifacts/detached_nps_val_samurai_cmc"
PROGRESS = RUNNER / "progress.json"
PYTHON = Path(sys.executable)


def progress(stage: str, done: int, total: int = 2, **extra: object) -> None:
    RUNNER.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps({"stage": stage, "done": done, "total": total, "updated": datetime.now(timezone.utc).astimezone().isoformat(), **extra}, indent=2), encoding="utf-8")


def run(command: list[str], stage: str, done: int) -> None:
    print(json.dumps({"kind": "pipeline_command", "stage": stage, "command": command}), flush=True)
    process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO)})
    progress(stage, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scored = OUTPUT / "val_tracklets_samurai_cmc.jsonl"
    run([
        str(PYTHON), str(REPO / "tools/score_tracklets_samurai_cmc.py"),
        "--input-tracklets", str(VAL / "route_b_official/tracklets/proposal_tracklets.jsonl"),
        "--frame-root", str(VAL / "AllFrames/val"),
        "--output-tracklets", str(scored),
        "--summary-json", str(OUTPUT / "score_summary.json"),
        "--progress-json", str(RUNNER / "score_progress.json"),
        "--homography-cache", str(OUTPUT / "homographies.pkl"),
        "--score-field", "samurai_cmc_score",
    ], "score", 0)
    run([
        str(PYTHON), str(REPO / "tools/sweep_tvd_predictionsgt_two_score_fusion_fast.py"),
        "--tvd-root", str(TVD),
        "--predictionsgt-pkl", str(VAL / "runs/nps_val_rank_source/predictionsgt/predictionsgt_split_0_official_labels.pkl"),
        "--meta-tracklet-jsonl", str(scored), "--meta-score-field", "samurai_cmc_track_score",
        "--row-tracklet-jsonl", str(scored), "--row-score-field", "samurai_cmc_score",
        "--modes", "logit-3mix", "meta-logit-row-geom", "meta-logit-row-suppress", "meta-logit-row-boost",
        "--alphas", "0.00", "0.01", "0.02", "0.04", "0.06", "0.08", "0.10", "0.14", "0.20",
        "--betas", "0.005", "0.01", "0.02", "0.04", "0.06", "0.08", "0.10", "0.12", "0.16", "0.20", "0.24", "0.32", "0.40",
        "--out-json", str(OUTPUT / "fusion_sweep.json"),
        "--write-best-pkl", str(OUTPUT / "best_predictionsgt.pkl"),
    ], "evaluate", 1)
    summary = json.loads((OUTPUT / "fusion_sweep.json").read_text(encoding="utf-8"))
    progress("done", 2, best=summary.get("best"), output=str(OUTPUT))
    print(json.dumps({"kind": "pipeline_done", "best": summary.get("best")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
