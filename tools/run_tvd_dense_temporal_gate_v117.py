from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\Users\aaron\Desktop\URAP")
UPSTREAM_RUN = ROOT / "artifacts" / "detached_tvd_dense_action_model_v116"
UPSTREAM = Path(r"D:\URAP_vatd_rank_results\tvd_dense_action_model_v116")
EXPERT = Path(r"D:\URAP_vatd_rank_results\action_chunk_multi_expert_v52")
OUT = Path(r"D:\URAP_vatd_rank_results\tvd_dense_temporal_gate_v117")
RUN = ROOT / "artifacts" / "detached_tvd_dense_temporal_gate_v117"
VAL = Path(r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl")
TEST = Path(r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl")
FPS = ROOT / "data_templates" / "nps_sequence_fps.json"
VATD_MAP50 = 0.93844


def report(stage: str, done: int, total: int = 3, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now().astimezone().isoformat(), **extra}
    (RUN / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def execute(stage: str, done: int, command: list[str]) -> None:
    report(stage, done, command=command)
    code = subprocess.call(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
    if code:
        raise RuntimeError(f"{stage} failed with exit code {code}")


def wait_upstream() -> None:
    progress_path = UPSTREAM_RUN / "progress.json"
    pid_path = UPSTREAM_RUN / "pid.txt"
    while True:
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("stage") == "done":
                return
        if pid_path.exists():
            worker_pid = int(pid_path.read_text().strip())
            check = subprocess.run(["powershell", "-NoProfile", "-Command", f"if(Get-Process -Id {worker_pid} -ErrorAction SilentlyContinue){{exit 0}}else{{exit 1}}"], check=False)
            if check.returncode:
                detail = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else None
                raise RuntimeError(f"V116 stopped before completion: {detail}")
        time.sleep(30)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report("wait_v116", 0)
    wait_upstream()
    common = [
        sys.executable,
        str(ROOT / "tools" / "sweep_action_chunk_temporal_multiplicity.py"),
        "--tvd-root", r"D:\urap_modal_stage\TransVisDrone",
        "--sequence-fps-json", str(FPS),
        "--base-field", "tvd_dense_action_score",
        "--expert-field", "action_chunk_multi_expert_score",
    ]
    val_sweep = OUT / "val_sweep.json"
    execute(
        "select_temporal_gate_on_validation",
        1,
        common + [
            "--predictionsgt-pkl", str(VAL),
            "--base-jsonl", str(UPSTREAM / "val_oof_scores.jsonl"),
            "--expert-jsonl", str(EXPERT / "val_expert_scores.jsonl"),
            "--out-json", str(val_sweep),
        ],
    )
    test_fixed = OUT / "test_fixed.json"
    execute(
        "evaluate_fixed_test",
        2,
        common + [
            "--predictionsgt-pkl", str(TEST),
            "--base-jsonl", str(UPSTREAM / "test_scores.jsonl"),
            "--expert-jsonl", str(EXPERT / "test_expert_scores.jsonl"),
            "--fixed-config-json", str(val_sweep),
            "--out-json", str(test_fixed),
        ],
    )
    validation = json.loads(val_sweep.read_text(encoding="utf-8"))["best"]
    test = json.loads(test_fixed.read_text(encoding="utf-8"))["best"]
    gain = 100.0 * (float(test["map50"]) - VATD_MAP50)
    summary = {
        "protocol": "dense train+validation Action Bank base with original multi-target temporal expert; validation selection; fixed test",
        "validation_selection": validation,
        "test_fixed": test,
        "vatd_map50": VATD_MAP50,
        "gain_over_vatd_points": gain,
        "target_3_to_5_met": 3.0 <= gain <= 5.0,
    }
    (OUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", 3, summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
