from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
PYTHON = Path(sys.executable)
TVD = Path(r"D:\urap_modal_stage\TransVisDrone")
BASE = Path(r"D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2")
TRAIN_CMC = BASE / "train_tracklets_causal_cmc.jsonl"
VAL_ACTION = BASE / "val_tracklets_action_bank.jsonl"
TEST_ACTION = BASE / "test_tracklets_action_bank.jsonl"
ACTION_WEIGHTS = BASE / "action_bank_causal_cmc.pt"
TRAIN_GT = Path(r"D:\URAP_nps_train_tvd\route_b_official\gt.csv")
VAL_PKL = Path(r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl")
TEST_PKL = Path(r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl")
OUTPUT = Path(r"D:\URAP_vatd_rank_results\nps_action_bank_listwise_v3")
RUNNER = REPO / "artifacts/detached_nps_action_bank_listwise_v3"
PROGRESS = RUNNER / "progress.json"
MARKERS = RUNNER / "stage_markers"
TOTAL = 4


def write_progress(stage: str, done: int, **extra: object) -> None:
    RUNNER.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps({"stage": stage, "done": done, "total": TOTAL, "updated": datetime.now(timezone.utc).astimezone().isoformat(), **extra}, indent=2), encoding="utf-8")


def run(command: list[str], stage: str, done: int) -> None:
    marker = MARKERS / f"{done + 1:02d}_{stage}.json"
    if marker.is_file():
        write_progress(f"{stage}_already_done", done + 1, marker=str(marker))
        return
    print(json.dumps({"kind": "action_bank_listwise_command", "stage": stage, "command": command}), flush=True)
    process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO)})
    write_progress(stage, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    MARKERS.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"stage": stage, "completed": datetime.now(timezone.utc).astimezone().isoformat()}, indent=2), encoding="utf-8")
    write_progress(f"{stage}_done", done + 1, marker=str(marker))


def main() -> int:
    required = (TVD, TRAIN_CMC, VAL_ACTION, TEST_ACTION, ACTION_WEIGHTS, TRAIN_GT, VAL_PKL, TEST_PKL)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_action = OUTPUT / "train_tracklets_action_bank.jsonl"
    val_ranked = OUTPUT / "val_tracklets_listwise.jsonl"
    test_ranked = OUTPUT / "test_tracklets_listwise.jsonl"
    model = OUTPUT / "action_bank_listwise.pt"
    train_summary = OUTPUT / "listwise_train_summary.json"
    val_sweep = OUTPUT / "val_fusion_sweep.json"
    test_eval = OUTPUT / "test_fixed_fusion.json"

    run([
        str(PYTHON), str(REPO / "tools/score_action_bank.py"), "--tracklets", str(TRAIN_CMC),
        "--weights", str(ACTION_WEIGHTS), "--out", str(train_action), "--batch-size", "4096", "--chunk-tracklets", "5000", "--device", "cuda",
    ], "score_train_action_bank", 0)

    run([
        str(PYTHON), str(REPO / "tools/train_action_bank_listwise.py"),
        "--train-tracklets", str(train_action), "--train-gt-csv", str(TRAIN_GT),
        "--validation-tracklets", str(VAL_ACTION), "--out-validation-tracklets", str(val_ranked),
        "--test-tracklets", str(TEST_ACTION), "--out-test-tracklets", str(test_ranked),
        "--out-model", str(model), "--out-summary", str(train_summary),
        "--epochs", "12", "--frame-batch-size", "256", "--inference-batch-size", "8192", "--hidden", "192", "--lr", "0.0005",
    ], "train_listwise_selector", 1)

    run([
        str(PYTHON), str(REPO / "tools/sweep_tvd_predictionsgt_score_fusion.py"),
        "--tvd-root", str(TVD), "--predictionsgt-pkl", str(VAL_PKL), "--tracklet-jsonl", str(val_ranked),
        "--score-field", "action_bank_listwise_score", "--per-row-score",
        "--modes", "replace", "linear-mix", "logit-mix", "geom-mix", "fp-suppress", "tp-boost",
        "--alphas", "0.001 0.002 0.005 0.01 0.02 0.04 0.06 0.08 0.10 0.14 0.20 0.30 0.40 0.55 0.70 0.85 1.0",
        "--out-json", str(val_sweep), "--write-best-pkl", str(OUTPUT / "val_best_predictionsgt.pkl"),
    ], "select_on_validation", 2)

    best = json.loads(val_sweep.read_text(encoding="utf-8"))["best"]
    run([
        str(PYTHON), str(REPO / "tools/sweep_tvd_predictionsgt_score_fusion.py"),
        "--tvd-root", str(TVD), "--predictionsgt-pkl", str(TEST_PKL), "--tracklet-jsonl", str(test_ranked),
        "--score-field", "action_bank_listwise_score", "--per-row-score", "--modes", str(best["mode"]), "--alphas", str(best["alpha"]),
        "--out-json", str(test_eval), "--write-best-pkl", str(OUTPUT / "test_fixed_best_predictionsgt.pkl"),
    ], "evaluate_test_fixed", 3)

    test = json.loads(test_eval.read_text(encoding="utf-8"))["best"]
    summary = {
        "protocol": "train Clips1-36; select on Clips37-40; fixed test Clips41-50",
        "selector": "causal frame-listwise Action Bank + SAMURAI CMC",
        "validation_best": best,
        "test_fixed": test,
        "target_map50": 0.97,
        "target_met": float(test["map50"]) >= 0.97,
        "candidate_oracle_map50": 0.9920831168831169,
    }
    (OUTPUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_progress("done", TOTAL, summary=summary)
    print(json.dumps({"kind": "action_bank_listwise_official_done", **summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
