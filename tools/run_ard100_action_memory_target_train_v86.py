from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO = Path(r"C:\Users\aaron\Desktop\URAP")
PYTHON = Path(sys.executable)
RUN = REPO / "artifacts" / "detached_ard100_action_memory_target_v86"
PROGRESS = RUN / "progress.json"
OUT = Path(r"D:\URAP_vatd_rank_results\ard100_action_memory_target_v86")
CANDIDATES = OUT / "yolomg_ard100_train_candidates_v86"
SOURCE = Path(r"D:\URAP_vatd_rank_results\ard100_yolomg_generalization_v2")
VAL_AUX = Path(r"D:\URAP_vatd_rank_results\ard100_action_memory_cross_attention_v83\val_aux.jsonl")
TEST_AUX = Path(r"D:\URAP_vatd_rank_results\ard100_action_memory_cross_attention_v83\test_aux.jsonl")
INIT_MODEL = Path(r"D:\URAP_vatd_rank_results\nps_action_memory_action_only_v84\model.pt")
TOTAL = 7


def report(stage: str, done: int, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": TOTAL, "updated": datetime.now().astimezone().isoformat(), **extra}
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def execute(stage: str, done: int, command: list[str]) -> None:
    process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO) + os.pathsep + str(REPO / "tools"), "PYTHONUNBUFFERED": "1"})
    report(stage, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    report(stage + "_done", done + 1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = CANDIDATES / "results.txt"
    while not results.is_file():
        report("waiting_for_train_candidates", 0, expected=str(results))
        time.sleep(30)
    train_pkl = OUT / "ard100_train_predictionsgt.pkl"
    train_aux = OUT / "train_aux.jsonl"
    execute("convert_train_candidates", 0, [
        str(PYTHON), str(REPO / "tools" / "convert_ard100_yolomg_predictionsgt.py"),
        "--image-list", r"D:\URAP_datasets\ARD100_YOLOMG\train.txt", "--prediction-label-root", str(CANDIDATES / "labels"),
        "--out-pkl", str(train_pkl), "--out-summary", str(OUT / "train_conversion_summary.json"),
    ])
    if train_aux.is_file() and (OUT / "train_aux_summary.json").is_file():
        report("build_train_camera_compensated_memory_reused", 2, artifact=str(train_aux))
    else:
        execute("build_train_camera_compensated_memory", 1, [
            str(PYTHON), str(REPO / "tools" / "score_predictionsgt_online_action_bank_motion_parallel.py"),
            "--predictionsgt-pkl", str(train_pkl), "--frame-root", r"D:\URAP_datasets\TransVisDrone\ARD100\AllFrames\train",
            "--homography-cache", str(OUT / "train_homographies.pkl"), "--out-jsonl", str(train_aux), "--out-summary", str(OUT / "train_aux_summary.json"),
            "--workers", "8", "--fps", "29.97002997", "--short-seconds", "1", "--long-seconds", "3", "--beam-size", "6",
            "--short-token-count", "8", "--long-token-count", "16", "--start-gate", "0.12", "--update-gate", "0.08", "--internal-alpha", "2.5",
        ])
    execute("finetune_ard100_action_memory", 2, [
        str(PYTHON), str(REPO / "tools" / "train_action_bank_motion_token_listwise.py"),
        "--train-pkl", str(train_pkl), "--train-aux-tracklets", str(train_aux),
        "--val-pkl", str(SOURCE / "ard100_yolomg_val_predictionsgt.pkl"), "--val-aux-tracklets", str(VAL_AUX),
        "--test-pkl", str(SOURCE / "ard100_yolomg_val_predictionsgt.pkl"), "--test-aux-tracklets", str(VAL_AUX),
        "--out-val-scores", str(OUT / "val_scores.jsonl"), "--out-test-scores", str(OUT / "val_scores_duplicate.jsonl"),
        "--out-model", str(OUT / "model.pt"), "--out-summary", str(OUT / "train_summary.json"),
        "--score-field", "ard100_target_action_memory_score", "--cross-attention", "--action-only-query", "--init-model", str(INIT_MODEL),
        "--epochs", "1", "--frame-batch-size", "128", "--inference-batch-size", "8192", "--hidden", "256", "--lr", "0.00002",
        "--attention-heads", "4", "--memory-layers", "1", "--write-loss-weight", "0.05",
        "--max-train-candidates-per-frame", "64", "--robust-pairwise",
    ])
    execute("score_fixed_test", 3, [
        str(PYTHON), str(REPO / "tools" / "score_action_memory_cross_attention.py"),
        "--predictionsgt-pkl", str(SOURCE / "ard100_yolomg_predictionsgt.pkl"), "--aux-jsonl", str(TEST_AUX), "--model", str(OUT / "model.pt"),
        "--out-scores", str(OUT / "test_scores.jsonl"), "--out-summary", str(OUT / "test_score_summary.json"), "--score-field", "ard100_target_action_memory_score",
    ])
    execute("select_validation_fusion", 4, [
        str(PYTHON), str(REPO / "tools" / "sweep_tvd_predictionsgt_score_fusion.py"),
        "--predictionsgt-pkl", str(SOURCE / "ard100_yolomg_val_predictionsgt.pkl"), "--tracklet-jsonl", str(OUT / "val_scores.jsonl"),
        "--score-field", "ard100_target_action_memory_score", "--per-row-score", "--modes", "linear-mix", "logit-mix", "geom-mix", "fp-suppress", "tp-boost",
        "--alphas", ".02,.05,.1,.2,.3,.4,.5,.7", "--missing-score-behaviors", "keep", "--out-json", str(OUT / "val_fusion.json"),
    ])
    best = json.loads((OUT / "val_fusion.json").read_text(encoding="utf-8"))["best"]
    execute("evaluate_fixed_test", 5, [
        str(PYTHON), str(REPO / "tools" / "sweep_tvd_predictionsgt_score_fusion.py"),
        "--predictionsgt-pkl", str(SOURCE / "ard100_yolomg_predictionsgt.pkl"), "--tracklet-jsonl", str(OUT / "test_scores.jsonl"),
        "--score-field", "ard100_target_action_memory_score", "--per-row-score", "--modes", str(best["mode"]),
        "--alphas", str(best["alpha"]), "--missing-score-behaviors", "keep", "--out-json", str(OUT / "test_fixed.json"),
    ])
    test = json.loads((OUT / "test_fixed.json").read_text(encoding="utf-8"))["best"]
    detector = json.loads((SOURCE / "detector_baseline.json").read_text(encoding="utf-8"))
    incumbent = json.loads((SOURCE / "action_bank_summary.json").read_text(encoding="utf-8"))["action_bank"]
    summary = {
        "protocol": "ARD100 train fine-tuning from NPS Action Memory; ARD100 val selection; one fixed test evaluation",
        "validation_selection": best, "target_trained": test, "detector_baseline": detector, "incumbent_action_bank": incumbent,
        "gain_vs_detector_points": 100.0 * (float(test["map50"]) - float(detector["map50"])),
        "gain_vs_incumbent_points": 100.0 * (float(test["map50"]) - float(incumbent["map50"])),
        "test_labels_used_for_training": False,
        "champion": "target_trained_v86" if float(test["map50"]) > float(incumbent["map50"]) else "incumbent_action_bank",
    }
    (OUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", TOTAL, summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
