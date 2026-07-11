from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
PYTHON = Path(sys.executable)
RUN = REPO / "artifacts" / "detached_nps_action_bank_native_gate_v10"
PROGRESS = RUN / "progress.json"
OUT = Path(r"D:\URAP_vatd_rank_results\nps_action_bank_native_gate_v10")
FPS = REPO / "data_templates" / "nps_sequence_fps.json"
TVD = Path(r"D:\urap_modal_stage\TransVisDrone")
TRAIN_PKL = Path(r"D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl")
VAL_PKL = Path(r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl")
TEST_PKL = Path(r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl")
ONLINE = Path(r"D:\URAP_vatd_rank_results\nps_online_action_bank_v12")
NATIVE_TRAIN = Path(r"D:\URAP_vatd_rank_results\nps_samurai_native_train_v9")
NATIVE_VAL = Path(r"D:\URAP_vatd_rank_results\nps_samurai_native_v7")
NATIVE_TEST = Path(r"D:\URAP_vatd_rank_results\nps_samurai_native_test_v10")
TOTAL = 5


def report(stage: str, done: int, **extra) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": TOTAL, "updated": datetime.now(timezone.utc).astimezone().isoformat(), **extra}
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def execute(stage: str, done: int, command: list[str]) -> None:
    report(stage, done, command=command)
    process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO) + os.pathsep + str(REPO / "tools"), "PYTHONUNBUFFERED": "1"})
    report(stage, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    report(stage + "_done", done + 1)


def wait_for_features() -> None:
    required = [
        NATIVE_TRAIN / "train_score_summary.json",
        ONLINE / "train_summary.json",
        ONLINE / "val_summary.json",
        ONLINE / "test_summary.json",
    ]
    while not all(path.is_file() for path in required):
        present = [str(path) for path in required if path.is_file()]
        missing = [str(path) for path in required if not path.is_file()]
        report("waiting_for_train_and_bank_features", 0, present=present, missing=missing)
        time.sleep(60)
    report("feature_inputs_ready", 1, inputs=[str(path) for path in required])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    NATIVE_TEST.mkdir(parents=True, exist_ok=True)
    wait_for_features()
    execute("score_native_test", 1, [
        str(PYTHON), str(REPO / "tools" / "score_predictionsgt_samurai_native.py"),
        "--predictionsgt-pkl", str(TEST_PKL), "--frame-root", r"U:\URAP_datasets\TransVisDrone\NPS\AllFrames\test",
        "--output-jsonl", str(NATIVE_TEST / "test_scores.jsonl"), "--output-summary", str(NATIVE_TEST / "test_score_summary.json"),
        "--frame-cache", r"U:\URAP_datasets\TransVisDrone\NPS\SAMURAI\native_test_frames",
        "--progress-json", str(RUN / "native_test_progress.json"), "--sequence-fps-json", str(FPS),
        "--start-gate", "0.55", "--reset-gate", "0.70", "--reset-iou", "0.05", "--object-gate", "0.20",
        "--reset-policy", "any", "--reset-patience", "1", "--disagreement-reset-gate", "0.70",
    ])
    execute("train_candidate_gate", 2, [
        str(PYTHON), str(REPO / "tools" / "train_action_bank_all_candidate_listwise.py"),
        "--train-pkl", str(TRAIN_PKL), "--train-aux-tracklets", str(ONLINE / "train_scores.jsonl"), "--train-native-tracklets", str(NATIVE_TRAIN / "train_scores.jsonl"),
        "--val-pkl", str(VAL_PKL), "--val-aux-tracklets", str(ONLINE / "val_scores.jsonl"), "--val-native-tracklets", str(NATIVE_VAL / "val_scores.jsonl"),
        "--test-pkl", str(TEST_PKL), "--test-aux-tracklets", str(ONLINE / "test_scores.jsonl"), "--test-native-tracklets", str(NATIVE_TEST / "test_scores.jsonl"),
        "--out-val-scores", str(OUT / "val_gate_scores.jsonl"), "--out-test-scores", str(OUT / "test_gate_scores.jsonl"),
        "--out-model", str(OUT / "model.pt"), "--out-summary", str(OUT / "train_summary.json"),
        "--score-field", "action_bank_native_gate_score", "--epochs", "18", "--frame-batch-size", "192", "--inference-batch-size", "16384", "--hidden", "256", "--lr", "0.0004",
    ])
    execute("select_validation_fusion", 3, [
        str(PYTHON), str(REPO / "tools" / "sweep_tvd_predictionsgt_score_fusion.py"),
        "--tvd-root", str(TVD), "--predictionsgt-pkl", str(VAL_PKL), "--tracklet-jsonl", str(OUT / "val_gate_scores.jsonl"),
        "--score-field", "action_bank_native_gate_score", "--per-row-score", "--modes", "replace", "linear-mix", "logit-mix", "geom-mix", "fp-suppress", "tp-boost",
        "--alphas", "0.001 0.002 0.005 0.01 0.02 0.04 0.06 0.08 0.10 0.14 0.20 0.30 0.40 0.55 0.70 0.85 1.0",
        "--out-json", str(OUT / "val_fusion_sweep.json"), "--write-best-pkl", str(OUT / "val_best.pkl"),
    ])
    best = json.loads((OUT / "val_fusion_sweep.json").read_text(encoding="utf-8"))["best"]
    execute("evaluate_official_test_fixed", 4, [
        str(PYTHON), str(REPO / "tools" / "sweep_tvd_predictionsgt_score_fusion.py"),
        "--tvd-root", str(TVD), "--predictionsgt-pkl", str(TEST_PKL), "--tracklet-jsonl", str(OUT / "test_gate_scores.jsonl"),
        "--score-field", "action_bank_native_gate_score", "--per-row-score", "--modes", str(best["mode"]), "--alphas", str(best["alpha"]),
        "--out-json", str(OUT / "test_fixed_fusion.json"), "--write-best-pkl", str(OUT / "test_fixed_best.pkl"),
    ])
    test = json.loads((OUT / "test_fixed_fusion.json").read_text(encoding="utf-8"))["best"]
    summary = {"protocol": "train Clips1-36; select Clips37-40; fixed test Clips41-50", "validation_best": best, "test_fixed": test, "target_map50": 0.97, "target_met": float(test["map50"]) >= 0.97, "future_supervision": "future 1-second consistency used only in training"}
    (OUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", TOTAL, summary=summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
