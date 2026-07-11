from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
TVD = Path(r"D:\urap_modal_stage\TransVisDrone")
TRAIN_TRACKLETS = Path(r"D:\URAP_nps_train_tvd\route_b_official\tracklets\proposal_tracklets.jsonl")
VAL_TRACKLETS = Path(r"D:\URAP_nps_val_tvd\route_b_official\tracklets\proposal_tracklets.jsonl")
VAL_PKL = Path(r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl")
TEST_TRACKLETS = Path(r"D:\URAP_vatd_rank_inputs\nps_tracklets_with_vatd.jsonl")
TEST_PKL = Path(r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl")
OUTPUT = Path(r"D:\URAP_vatd_rank_results\nps_action_bank_v1")
RUNNER = REPO / "artifacts/detached_nps_action_bank_v1"
PROGRESS = RUNNER / "progress.json"
PYTHON = Path(sys.executable)


def write_progress(stage: str, done: int, total: int = 5, **extra: object) -> None:
    RUNNER.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now(timezone.utc).astimezone().isoformat(), **extra}
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(command: list[str], stage: str, done: int) -> None:
    print(json.dumps({"kind": "nps_action_bank_command", "stage": stage, "command": command}), flush=True)
    process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO)})
    write_progress(stage, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def require_inputs() -> None:
    missing = [str(path) for path in (TVD, TRAIN_TRACKLETS, VAL_TRACKLETS, VAL_PKL, TEST_TRACKLETS, TEST_PKL) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing official Action Bank inputs: {missing}")


def main() -> int:
    require_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    weights = OUTPUT / "action_bank_train.pt"
    val_scored = OUTPUT / "val_tracklets_action_bank.jsonl"
    val_sweep = OUTPUT / "val_fusion_sweep.json"
    test_scored = OUTPUT / "test_tracklets_action_bank.jsonl"
    test_eval = OUTPUT / "test_fixed_fusion.json"
    fps_map = REPO / "data_templates/nps_sequence_fps.json"

    run([
        str(PYTHON), str(REPO / "tools/train_action_bank.py"),
        "--train-tracklets", str(TRAIN_TRACKLETS), "--out", str(weights),
        "--epochs", "8", "--batch-size", "1024", "--lr", "0.0003",
        "--short-tokens", "12", "--long-tokens", "18",
        "--sequence-fps-json", str(fps_map), "--cache-dir", str(OUTPUT / "train_action_bank_cache"), "--device", "cuda",
    ], "train_action_bank", 0)

    run([
        str(PYTHON), str(REPO / "tools/score_action_bank.py"),
        "--tracklets", str(VAL_TRACKLETS), "--weights", str(weights),
        "--out", str(val_scored), "--batch-size", "4096", "--device", "cuda",
    ], "score_validation", 1)

    run([
        str(PYTHON), str(REPO / "tools/sweep_tvd_predictionsgt_score_fusion.py"),
        "--tvd-root", str(TVD), "--predictionsgt-pkl", str(VAL_PKL),
        "--tracklet-jsonl", str(val_scored), "--score-field", "action_bank_learned_score", "--per-row-score",
        "--modes", "linear-mix", "logit-mix", "geom-mix", "fp-suppress", "tp-boost",
        "--alphas", "0.001 0.002 0.005 0.01 0.02 0.04 0.06 0.08 0.10 0.14 0.20 0.30 0.40",
        "--out-json", str(val_sweep), "--write-best-pkl", str(OUTPUT / "val_best_predictionsgt.pkl"),
    ], "select_on_validation", 2)

    run([
        str(PYTHON), str(REPO / "tools/score_action_bank.py"),
        "--tracklets", str(TEST_TRACKLETS), "--weights", str(weights),
        "--out", str(test_scored), "--batch-size", "4096", "--device", "cuda",
    ], "score_test", 3)

    validation = json.loads(val_sweep.read_text(encoding="utf-8"))
    best = validation["best"]
    run([
        str(PYTHON), str(REPO / "tools/sweep_tvd_predictionsgt_score_fusion.py"),
        "--tvd-root", str(TVD), "--predictionsgt-pkl", str(TEST_PKL),
        "--tracklet-jsonl", str(test_scored), "--score-field", "action_bank_learned_score", "--per-row-score",
        "--modes", str(best["mode"]), "--alphas", str(best["alpha"]),
        "--out-json", str(test_eval), "--write-best-pkl", str(OUTPUT / "test_fixed_best_predictionsgt.pkl"),
    ], "evaluate_test_fixed", 4)

    test = json.loads(test_eval.read_text(encoding="utf-8"))
    summary = {
        "protocol": "train Clips1-36; select fusion on Clips37-40; freeze and evaluate Clips41-50",
        "validation_best": best,
        "test_fixed": test["best"],
        "target_map50": 0.97,
        "target_met": float(test["best"]["map50"]) >= 0.97,
        "weights": str(weights),
        "val_scored": str(val_scored),
        "test_scored": str(test_scored),
    }
    (OUTPUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_progress("done", 5, summary=summary, output=str(OUTPUT))
    print(json.dumps({"kind": "nps_action_bank_done", **summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
