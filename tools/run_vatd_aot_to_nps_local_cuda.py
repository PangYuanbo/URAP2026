from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
INPUT = Path(r"D:\URAP_vatd_rank_inputs")
OUTPUT = Path(r"D:\URAP_vatd_rank_results\aot_to_nps_cuda_rank_v1")
RUNNER = REPO / "artifacts/detached_vatd_aot_to_nps_local_cuda"
PROGRESS = RUNNER / "progress.json"


def write_progress(stage: str, done: int, total: int = 2, **extra: object) -> None:
    RUNNER.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now(timezone.utc).astimezone().isoformat(), **extra}
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(command: list[str], stage: str, done: int) -> None:
    print(json.dumps({"kind": "command", "command": command}), flush=True)
    process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    write_progress(stage, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main() -> int:
    required = {
        "train_tracklets": INPUT / "aot_proposal_tracklets.jsonl",
        "train_gt": INPUT / "aot_gt.csv",
        "test_tracklets": INPUT / "nps_tracklets_with_vatd.jsonl",
        "predictions": INPUT / "nps_predictionsgt_split_0.pkl",
    }
    missing = [str(path) for path in required.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("missing inputs: " + ", ".join(missing))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scored = OUTPUT / "nps_tracklets_aot_rank_scored.jsonl"
    run([
        sys.executable, str(REPO / "tools/train_detection_row_score_head.py"),
        "--train-tracklets", str(required["train_tracklets"]),
        "--train-gt-csv", str(required["train_gt"]),
        "--test-tracklets", str(required["test_tracklets"]),
        "--out-test-tracklets", str(scored),
        "--out-model", str(OUTPUT / "aot_rank_model.pt"),
        "--out-summary", str(OUTPUT / "train_summary.json"),
        "--score-field", "aot_rank_score",
        "--iou-threshold", "0.5", "--negative-min-score", "0.005", "--label-policy", "unique-iou",
        "--epochs", "30", "--batch-size", "32768", "--hidden", "256", "--lr", "0.0005",
        "--pairwise-weight", "1.0", "--pairwise-pairs", "65536", "--model-kind", "unified-two-tower",
        "--tracklet-aux-weight", "0.25", "--feature-groups", "all",
    ], "train", 0)
    run([
        sys.executable, str(REPO / "tools/sweep_tvd_predictionsgt_two_score_fusion.py"),
        "--tvd-root", r"D:\urap_modal_stage\TransVisDrone",
        "--predictionsgt-pkl", str(required["predictions"]),
        "--meta-tracklet-jsonl", str(scored), "--meta-score-field", "vatd_score",
        "--row-tracklet-jsonl", str(scored), "--row-score-field", "aot_rank_score",
        "--modes", "logit-3mix", "meta-logit-row-geom", "meta-logit-row-suppress", "meta-logit-row-boost",
        "--alphas", "0.00 0.02 0.04 0.06 0.08 0.10 0.14",
        "--betas", "0.01 0.02 0.04 0.06 0.10 0.16 0.24",
        "--missing-score-behaviors", "keep",
        "--out-json", str(OUTPUT / "fusion_sweep.json"),
        "--write-best-pkl", str(OUTPUT / "best_predictionsgt.pkl"),
    ], "evaluate", 1)
    summary = json.loads((OUTPUT / "fusion_sweep.json").read_text(encoding="utf-8"))
    write_progress("done", 2, best=summary.get("best"), output=str(OUTPUT))
    print(json.dumps({"kind": "done", "best": summary.get("best"), "output": str(OUTPUT)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
