from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
PYTHON = Path(sys.executable)
RUN = REPO / "artifacts" / "detached_nps_action_memory_action_only_v84"
PROGRESS = RUN / "progress.json"
OUT = Path(r"D:\URAP_vatd_rank_results\nps_action_memory_action_only_v84")
TVD = Path(r"D:\urap_modal_stage\TransVisDrone")
TRAIN_PKL = Path(r"D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl")
VAL_PKL = Path(r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl")
TEST_PKL = Path(r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl")
ONLINE = Path(r"D:\URAP_vatd_rank_results\nps_online_action_bank_v14")
BASE = Path(r"D:\URAP_vatd_rank_results\action_chunk_causal_v38")
INCUMBENT = Path(r"D:\URAP_vatd_rank_results\action_chunk_causal_memory_v59")
TOTAL = 3


def report(stage: str, done: int, **extra) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "done": done,
        "total": TOTAL,
        "updated": datetime.now(timezone.utc).astimezone().isoformat(),
        **extra,
    }
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def execute(stage: str, done: int, command: list[str]) -> None:
    process = subprocess.Popen(
        command,
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO) + os.pathsep + str(REPO / "tools"), "PYTHONUNBUFFERED": "1"},
    )
    report(stage, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    report(stage + "_done", done + 1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    execute("train_cross_attention", 0, [
        str(PYTHON), str(REPO / "tools" / "train_action_bank_motion_token_listwise.py"),
        "--train-pkl", str(TRAIN_PKL), "--train-aux-tracklets", str(ONLINE / "train_scores.jsonl"),
        "--val-pkl", str(VAL_PKL), "--val-aux-tracklets", str(ONLINE / "val_scores.jsonl"),
        "--test-pkl", str(TEST_PKL), "--test-aux-tracklets", str(ONLINE / "test_scores.jsonl"),
        "--out-val-scores", str(OUT / "val_scores.jsonl"), "--out-test-scores", str(OUT / "test_scores.jsonl"),
        "--out-model", str(OUT / "model.pt"), "--out-summary", str(OUT / "train_summary.json"),
        "--score-field", "action_memory_action_only_score", "--cross-attention", "--action-only-query",
        "--epochs", "18", "--frame-batch-size", "128", "--inference-batch-size", "8192",
        "--hidden", "256", "--lr", "0.0003", "--attention-heads", "4", "--memory-layers", "1", "--write-loss-weight", "0.05",
    ])
    execute("select_guarded_residual", 1, [
        str(PYTHON), str(REPO / "tools" / "sweep_action_memory_cross_attention_residual.py"),
        "--tvd-root", str(TVD), "--predictionsgt-pkl", str(VAL_PKL),
        "--base-jsonl", str(BASE / "val_oof_scores.jsonl"), "--base-field", "action_chunk_causal_score",
        "--incumbent-jsonl", str(INCUMBENT / "val_oof_scores.jsonl"), "--incumbent-field", "action_chunk_causal_memory_score",
        "--cross-attention-jsonl", str(OUT / "val_scores.jsonl"), "--cross-attention-field", "action_memory_action_only_score",
        "--incumbent-cap", "0.5", "--incumbent-weight", "0.5",
        "--modes", "boost-only,symmetric", "--caps", ".05,.1,.25,.5", "--weights", "0,.1,.25,.5", "--alphas", ".2",
        "--out-json", str(OUT / "val_guarded_sweep.json"),
    ])
    execute("evaluate_fixed_test", 2, [
        str(PYTHON), str(REPO / "tools" / "sweep_action_memory_cross_attention_residual.py"),
        "--tvd-root", str(TVD), "--predictionsgt-pkl", str(TEST_PKL),
        "--base-jsonl", str(BASE / "test_scores.jsonl"), "--base-field", "action_chunk_causal_score",
        "--incumbent-jsonl", str(INCUMBENT / "test_scores.jsonl"), "--incumbent-field", "action_chunk_causal_memory_score",
        "--cross-attention-jsonl", str(OUT / "test_scores.jsonl"), "--cross-attention-field", "action_memory_action_only_score",
        "--incumbent-cap", "0.5", "--incumbent-weight", "0.5",
        "--fixed-config-json", str(OUT / "val_guarded_sweep.json"), "--out-json", str(OUT / "test_fixed.json"),
    ])
    validation = json.loads((OUT / "val_guarded_sweep.json").read_text(encoding="utf-8"))["best"]
    test = json.loads((OUT / "test_fixed.json").read_text(encoding="utf-8"))["best"]
    incumbent_test = 0.9488904815634672
    summary = {
        "protocol": "causal current-candidate Query to 1s/3s Action Memory K/V; validation-selected guarded residual; fixed test",
        "validation_selection": validation,
        "test_fixed": test,
        "incumbent_strict_causal_map50": incumbent_test,
        "champion_map50": max(incumbent_test, float(test["map50"])),
        "champion": "action_memory_action_only_v84" if float(test["map50"]) > incumbent_test else "incumbent_v60",
    }
    (OUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", TOTAL, summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
