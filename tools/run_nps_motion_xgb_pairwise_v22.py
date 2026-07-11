from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
PYTHON = Path(sys.executable)
RUN = REPO / "artifacts" / "detached_nps_motion_xgb_pairwise_v22"
PROGRESS = RUN / "progress.json"
OUT = Path(r"D:\URAP_vatd_rank_results\nps_motion_xgb_pairwise_v22")


def report(stage: str, done: int, **extra) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": 3, "updated": datetime.now(timezone.utc).astimezone().isoformat(), **extra}
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def execute(stage: str, done: int, command: list[str]) -> None:
    process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO) + os.pathsep + str(REPO / "tools"), "PYTHONUNBUFFERED": "1"})
    report(stage, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    execute("train_pairwise", 0, [
        str(PYTHON), str(REPO / "tools" / "train_action_bank_motion_xgb_ranker.py"),
        "--train-pkl", r"D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl",
        "--train-aux", r"D:\URAP_vatd_rank_results\nps_online_action_bank_v14\train_scores.jsonl",
        "--val-pkl", r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl",
        "--val-aux", r"D:\URAP_vatd_rank_results\nps_online_action_bank_v14\val_scores.jsonl",
        "--test-pkl", r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl",
        "--test-aux", r"D:\URAP_vatd_rank_results\nps_online_action_bank_v14\test_scores.jsonl",
        "--out-val", str(OUT / "val_scores.jsonl"), "--out-test", str(OUT / "test_scores.jsonl"),
        "--out-model", str(OUT / "model.ubj"), "--out-summary", str(OUT / "train_summary.json"),
        "--score-field", "xgb_pairwise_score",
    ])
    common = [str(PYTHON), str(REPO / "tools" / "sweep_tvd_predictionsgt_score_fusion.py"), "--tvd-root", r"D:\urap_modal_stage\TransVisDrone", "--per-row-score", "--score-field", "xgb_pairwise_score"]
    execute("select_val", 1, common + [
        "--predictionsgt-pkl", r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl",
        "--tracklet-jsonl", str(OUT / "val_scores.jsonl"),
        "--modes", "replace", "linear-mix", "logit-mix", "geom-mix", "fp-suppress", "tp-boost",
        "--alphas", "0.001 0.002 0.005 0.01 0.02 0.04 0.06 0.08 0.10 0.14 0.20 0.30 0.40 0.55 0.70 0.85 1.0",
        "--out-json", str(OUT / "val_sweep.json"),
    ])
    best = json.loads((OUT / "val_sweep.json").read_text(encoding="utf-8"))["best"]
    execute("fixed_test", 2, common + [
        "--predictionsgt-pkl", r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl",
        "--tracklet-jsonl", str(OUT / "test_scores.jsonl"), "--modes", best["mode"], "--alphas", str(best["alpha"]),
        "--out-json", str(OUT / "test_fixed.json"),
    ])
    test = json.loads((OUT / "test_fixed.json").read_text(encoding="utf-8"))["best"]
    summary = {"protocol": "train 1-36, validation 37-40, fixed test 41-50", "validation_best": best, "test_fixed": test, "target_map50": 0.97, "target_met": test["map50"] >= 0.97}
    (OUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", 3, summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
