from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
RUN = REPO / "artifacts" / "detached_ard100_vatd_baseline_v1"
PROGRESS = RUN / "progress.json"
OUTPUT = Path(r"D:\URAP_vatd_rank_results\ard100_generalization_v1")
MERGED = OUTPUT / "ard100_test_predictionsgt.pkl"
FRAME_ROOT = Path(r"D:\URAP_datasets\TransVisDrone\ARD100\AllFrames\test")
VATD_WEIGHTS = REPO / "artifacts" / "ego_adaptive_vatd" / "nps_train_full_u_epoch1_noshuffle_20260606" / "ego_adaptive_vatd.pt"
ACTION_SUMMARY = OUTPUT / "official_generalization_summary.json"
ROUTE_ROOT = OUTPUT / "vatd_route_b"
TRACKLETS = OUTPUT / "vatd_tracklets" / "proposal_tracklets.jsonl"
VATD_SCORES = OUTPUT / "ard100_vatd_scores.jsonl"
ATTACHED = OUTPUT / "ard100_tracklets_with_vatd.jsonl"


def report(stage: str, done: int, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": 6, "updated": datetime.now().astimezone().isoformat(), **extra}
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def execute(stage: str, done: int, command: list[str]) -> None:
    process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO)})
    report(stage, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise RuntimeError(f"{stage} failed with exit code {code}")


def main() -> int:
    if not VATD_WEIGHTS.is_file():
        raise FileNotFoundError(VATD_WEIGHTS)
    while not MERGED.is_file() or MERGED.stat().st_size == 0:
        report("waiting_for_merged_predictions", 0, merged=str(MERGED))
        time.sleep(30)
    execute("export_route_b", 1, [sys.executable, str(REPO / "tools" / "export_tvd_predictionsgt_to_route_b.py"), "--predictionsgt-pkl", str(MERGED), "--out-run-root", str(ROUTE_ROOT / "run_root"), "--out-gt-csv", str(ROUTE_ROOT / "gt.csv"), "--out-summary", str(ROUTE_ROOT / "export_summary.json"), "--frame-root", str(FRAME_ROOT), "--image-width", "1920", "--image-height", "1080"] )
    execute("build_vatd_tracklets", 2, [sys.executable, "-m", "qstr_dronedet.cli", "build-proposal-tracklet-dataset", "--run-roots", str(ROUTE_ROOT / "run_root"), "--gt-csv", str(ROUTE_ROOT / "gt.csv"), "--out", str(OUTPUT / "vatd_tracklets"), "--profile", "hard_recovery", "--diagnostics-name", "diagnostics_raw.jsonl", "--max-gap", "3", "--base-radius", "18", "--radius-per-side", ".75", "--min-iou", ".05", "--min-score", "0", "--min-tracklet-rows", "3", "--iou-threshold", ".5", "--center-threshold", "24", "--hard-tiny-side", "24", "--hard-low-score", ".25"] )
    while not ACTION_SUMMARY.is_file():
        report("waiting_for_action_bank_gpu_release", 3, action_summary=str(ACTION_SUMMARY))
        time.sleep(30)
    execute("score_frozen_vatd", 3, [sys.executable, "-m", "qstr_dronedet.cli", "score-ego-adaptive-vatd-tracklets", "--tracklet-jsonl", str(TRACKLETS), "--weights", str(VATD_WEIGHTS), "--out", str(VATD_SCORES), "--frame-root", str(FRAME_ROOT), "--image-name-template", "{seq}_{frame_id_05d}.jpg", "--error-scale", ".02", "--min-tracklet-rows", "9", "--batch-size", "512", "--num-workers", "0", "--frame-cache-size", "256", "--fusion-mode", "motion_action"] )
    execute("attach_vatd_scores", 4, [sys.executable, "-m", "qstr_dronedet.cli", "attach-vatd-scores-to-tracklets", "--tracklet-jsonl", str(TRACKLETS), "--vatd-scores", str(VATD_SCORES), "--out", str(ATTACHED)] )
    execute("evaluate_strong_fixed_vatd", 5, [sys.executable, str(REPO / "tools" / "sweep_tvd_predictionsgt_action_rescore.py"), "--tvd-root", r"D:\urap_modal_stage\TransVisDrone", "--predictionsgt-pkl", str(MERGED), "--tracklet-jsonl", str(ATTACHED), "--score-field", "vatd_score", "--centers", ".01", "--betas", ".02", "--modes", "boost-only", "--missing-score-behaviors", "keep", "--out-json", str(OUTPUT / "vatd_strong_fixed.json")] )
    baseline = json.loads((OUTPUT / "detector_baseline.json").read_text(encoding="utf-8"))
    vatd = json.loads((OUTPUT / "vatd_strong_fixed.json").read_text(encoding="utf-8"))["best"]
    action_summary = json.loads(ACTION_SUMMARY.read_text(encoding="utf-8"))
    action = action_summary["action_bank_zero_shot"]
    summary = {
        "protocol": "ARD100 test; frozen NPS VATD checkpoint and frozen NPS fusion configuration; no ARD100 fitting",
        "detector_baseline": baseline,
        "vatd_baseline": vatd,
        "action_bank": action,
        "action_bank_gain_over_vatd": float(action["map50"]) - float(vatd["map50"]),
        "gain_over_vatd_points": 100.0 * (float(action["map50"]) - float(vatd["map50"])),
        "target_minimum_met": float(action["map50"]) - float(vatd["map50"]) >= .03,
    }
    (OUTPUT / "official_vatd_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", 6, summary=summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
