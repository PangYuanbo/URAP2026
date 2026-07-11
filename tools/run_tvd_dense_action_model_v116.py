from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\Users\aaron\Desktop\URAP")
FEATURE_RUN = ROOT / "artifacts" / "detached_tvd_dense_action_features_v115"
FEATURES = Path(r"D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\action_features_v115")
OUT = Path(r"D:\URAP_vatd_rank_results\tvd_dense_action_model_v116")
RUN = ROOT / "artifacts" / "detached_tvd_dense_action_model_v116"
TRAIN = Path(r"D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\official_train_dense\predictionsgt\predictionsgt_split_0_fixed_canvas.pkl")
VAL = Path(r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl")
TEST = Path(r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl")
FULL = Path(r"D:\URAP_vatd_rank_results\action_chunk_full_dev_v36")
NEIGHBOR = Path(r"D:\URAP_vatd_rank_results\action_chunk_neighbor_v44")
VATD_MAP50 = 0.93844


def report(stage: str, done: int, total: int = 4, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now().astimezone().isoformat(), **extra}
    (RUN / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def execute(stage: str, done: int, command: list[str]) -> None:
    report(stage, done, command=command)
    code = subprocess.call(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
    if code:
        raise RuntimeError(f"{stage} failed with exit code {code}")


def wait_for_features() -> None:
    progress_path = FEATURE_RUN / "progress.json"
    pid_path = FEATURE_RUN / "pid.txt"
    while True:
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            stage = progress.get("stage")
            if stage == "done":
                return
            if stage == "stopped_by_gate":
                raise RuntimeError(f"feature generation stopped by distribution gate: {progress}")
        if pid_path.exists():
            feature_pid = int(pid_path.read_text().strip())
            check = subprocess.run(["powershell", "-NoProfile", "-Command", f"if(Get-Process -Id {feature_pid} -ErrorAction SilentlyContinue){{exit 0}}else{{exit 1}}"], check=False)
            if check.returncode and progress_path.exists():
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                raise RuntimeError(f"feature process stopped before completion: {progress}")
        time.sleep(30)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report("wait_features", 0)
    wait_for_features()

    val_scores = OUT / "val_oof_scores.jsonl"
    test_scores = OUT / "test_scores.jsonl"
    field = "tvd_dense_action_score"
    execute(
        "train_dense_action_bank",
        1,
        [
            sys.executable,
            str(ROOT / "tools" / "train_action_chunk_neighbor_full.py"),
            "--train-pkl", str(TRAIN),
            "--train-forward", str(FEATURES / "train_forward.jsonl"),
            "--train-backward", str(FEATURES / "train_backward.jsonl"),
            "--train-neighbor", str(FEATURES / "train_neighbor.jsonl"),
            "--val-pkl", str(VAL),
            "--val-forward", str(FULL / "val_forward.jsonl"),
            "--val-backward", str(FULL / "val_backward.jsonl"),
            "--val-neighbor", str(NEIGHBOR / "val_neighbor_scores.jsonl"),
            "--test-pkl", str(TEST),
            "--test-forward", str(FULL / "test_forward.jsonl"),
            "--test-backward", str(FULL / "test_backward.jsonl"),
            "--test-neighbor", str(NEIGHBOR / "test_neighbor_scores.jsonl"),
            "--out-val-scores", str(val_scores),
            "--out-test-scores", str(test_scores),
            "--out-model-dir", str(OUT / "models"),
            "--out-summary", str(OUT / "model_summary.json"),
            "--score-field", field,
            "--sequence-size-json", str(ROOT / "data_templates" / "nps_sequence_sizes_actual.json"),
        ],
    )

    val_sweep = OUT / "val_sweep.json"
    execute(
        "select_fusion_on_val",
        2,
        [
            sys.executable,
            str(ROOT / "tools" / "sweep_tvd_predictionsgt_score_fusion.py"),
            "--tvd-root", r"D:\urap_modal_stage\TransVisDrone",
            "--predictionsgt-pkl", str(VAL),
            "--tracklet-jsonl", str(val_scores),
            "--per-row-score",
            "--score-field", field,
            "--modes", "geom-mix", "logit-mix", "fp-suppress", "replace",
            "--alphas", ".01,.02,.04,.06,.08,.1,.14,.2,.3,.4,.55,.7,1",
            "--out-json", str(val_sweep),
        ],
    )
    best = json.loads(val_sweep.read_text(encoding="utf-8"))["best"]

    test_fixed = OUT / "test_fixed.json"
    execute(
        "evaluate_fixed_test",
        3,
        [
            sys.executable,
            str(ROOT / "tools" / "sweep_tvd_predictionsgt_score_fusion.py"),
            "--tvd-root", r"D:\urap_modal_stage\TransVisDrone",
            "--predictionsgt-pkl", str(TEST),
            "--tracklet-jsonl", str(test_scores),
            "--per-row-score",
            "--score-field", field,
            "--modes", str(best["mode"]),
            "--alphas", str(best["alpha"]),
            "--out-json", str(test_fixed),
        ],
    )
    test = json.loads(test_fixed.read_text(encoding="utf-8"))["best"]
    gain = 100.0 * (float(test["map50"]) - VATD_MAP50)
    summary = {
        "protocol": "dense official-train candidates plus official validation OOF; fixed test evaluation",
        "val_best": best,
        "test": test,
        "vatd_map50": VATD_MAP50,
        "gain_over_vatd_points": gain,
        "target_3_to_5_met": 3.0 <= gain <= 5.0,
    }
    (OUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", 4, summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


