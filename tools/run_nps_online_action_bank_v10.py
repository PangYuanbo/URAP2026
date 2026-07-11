from __future__ import annotations
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
PYTHON = Path(sys.executable)
OUT = Path(r"D:\URAP_vatd_rank_results\nps_online_action_bank_v10")
RUN = REPO / "artifacts" / "detached_nps_online_action_bank_v10"
PROGRESS = RUN / "progress.json"
FPS = REPO / "data_templates" / "nps_sequence_fps.json"
TOTAL = 2


def report(stage: str, done: int, **extra) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": TOTAL, "updated": datetime.now(timezone.utc).astimezone().isoformat(), **extra}
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def execute(name: str, predictions: Path, frames: Path, homographies: Path, output: Path, summary: Path, done: int) -> None:
    command = [
        str(PYTHON), str(REPO / "tools" / "score_predictionsgt_online_action_bank.py"),
        "--predictionsgt-pkl", str(predictions),
        "--frame-root", str(frames),
        "--homography-cache", str(homographies),
        "--out-jsonl", str(output),
        "--out-summary", str(summary),
        "--sequence-fps-json", str(FPS),
        "--short-seconds", "1.0", "--long-seconds", "3.0",
        "--beam-size", "6", "--start-gate", "0.12", "--update-gate", "0.08", "--internal-alpha", "2.5",
    ]
    report(name, done, command=command)
    process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONUNBUFFERED": "1"})
    report(name, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    report(name + "_done", done + 1, output=str(output))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    execute(
        "train_action_bank", Path(r"D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl"),
        Path(r"U:\URAP_datasets\TransVisDrone\NPS\AllFrames\train"),
        Path(r"D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2\homographies\train.pkl"),
        OUT / "train_scores.jsonl", OUT / "train_summary.json", 0,
    )
    execute(
        "test_action_bank", Path(r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl"),
        Path(r"U:\URAP_datasets\TransVisDrone\NPS\AllFrames\test"),
        Path(r"D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2\homographies\test.pkl"),
        OUT / "test_scores.jsonl", OUT / "test_summary.json", 1,
    )
    report("done", TOTAL, output=str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
