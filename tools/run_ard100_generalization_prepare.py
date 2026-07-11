from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
RUN = REPO / "artifacts" / "detached_ard100_generalization_prepare_v1"
PROGRESS = RUN / "progress.json"
OUTPUT = Path(r"D:\URAP_vatd_rank_results\ard100_generalization_v1")
CANDIDATES = OUTPUT / "tvd_nps_zero_shot_candidates" / "aotpredictions" / "predictions_split_0.pkl"
FRAME_ROOT = Path(r"D:\URAP_datasets\TransVisDrone\ARD100\AllFrames\test")
LABEL_ROOT = Path(r"D:\URAP_datasets\TransVisDrone\ARD100\Annotations\test")
FPS_JSON = REPO / "data_templates" / "ard100_sequence_fps.json"
SIZE_JSON = REPO / "data_templates" / "ard100_sequence_sizes.json"
MERGED = OUTPUT / "ard100_test_predictionsgt.pkl"
HOMOGRAPHY = OUTPUT / "ard100_test_homographies.pkl"


def report(stage: str, done: int, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": 4, "updated": datetime.now().astimezone().isoformat(), **extra}
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def execute(stage: str, done: int, command: list[str]) -> None:
    process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO)})
    report(stage, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise RuntimeError(f"{stage} failed with exit code {code}")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    while not CANDIDATES.is_file() or CANDIDATES.stat().st_size == 0:
        report("waiting_for_detector_candidates", 0, candidate_pkl=str(CANDIDATES))
        time.sleep(30)
    execute("merge_labels", 1, [sys.executable, str(REPO / "tools" / "merge_ard100_aot_predictions.py"), "--aot-pkl", str(CANDIDATES), "--frame-root", str(FRAME_ROOT), "--label-root", str(LABEL_ROOT), "--sequence-size-json", str(SIZE_JSON), "--out-pkl", str(MERGED), "--out-summary", str(OUTPUT / "merge_summary.json")])
    execute("evaluate_detector_baseline", 2, [sys.executable, str(REPO / "tools" / "eval_tvd_predictionsgt_pkl.py"), "--tvd-root", r"D:\urap_modal_stage\TransVisDrone", "--predictionsgt-pkl", str(MERGED), "--out-json", str(OUTPUT / "detector_baseline.json")])
    execute("precompute_camera_motion", 3, [sys.executable, str(REPO / "tools" / "precompute_nps_homographies.py"), "--frame-root", str(FRAME_ROOT), "--output-cache", str(HOMOGRAPHY), "--partial-dir", str(OUTPUT / "homography_parts"), "--progress-json", str(OUTPUT / "homography_progress.json"), "--workers", "6", "--max-size", "320"])
    report("done", 4, merged_pkl=str(MERGED), baseline_json=str(OUTPUT / "detector_baseline.json"), homography_cache=str(HOMOGRAPHY), fps_json=str(FPS_JSON), size_json=str(SIZE_JSON))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
