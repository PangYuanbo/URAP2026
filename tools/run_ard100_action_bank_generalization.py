from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
RUN = REPO / "artifacts" / "detached_ard100_action_bank_v1"
PROGRESS = RUN / "progress.json"
OUTPUT = Path(r"D:\URAP_vatd_rank_results\ard100_generalization_v1")
MERGED = OUTPUT / "ard100_test_predictionsgt.pkl"
BASELINE = OUTPUT / "detector_baseline.json"
HOMOGRAPHY = OUTPUT / "ard100_test_homographies.pkl"
FRAME_ROOT = Path(r"D:\URAP_datasets\TransVisDrone\ARD100\AllFrames\test")
FPS_JSON = REPO / "data_templates" / "ard100_sequence_fps.json"
V53_CONFIG = Path(r"D:\URAP_vatd_rank_results\action_chunk_temporal_gate_v53\val_sweep.json")
BASE_MODELS = Path(r"D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46\models")
EXPERT_MODELS = Path(r"D:\URAP_vatd_rank_results\action_chunk_multi_expert_v52\models")


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
    required = [FPS_JSON, V53_CONFIG, BASE_MODELS, EXPERT_MODELS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen generalization inputs: {missing}")
    while not all(path.is_file() and path.stat().st_size > 0 for path in (MERGED, BASELINE, HOMOGRAPHY)):
        report("waiting_for_preparation", 0, merged=MERGED.exists(), baseline=BASELINE.exists(), homography=HOMOGRAPHY.exists())
        time.sleep(30)

    forward = OUTPUT / "ard100_forward_model_features.jsonl"
    backward = OUTPUT / "ard100_backward_model_features.jsonl"
    neighbor = OUTPUT / "ard100_neighbor_features.jsonl"
    base_scores = OUTPUT / "ard100_v46_base_scores.jsonl"
    expert_scores = OUTPUT / "ard100_v52_expert_scores.jsonl"

    common = [sys.executable, str(REPO / "tools" / "score_predictionsgt_action_chunk_bank.py"), "--predictionsgt-pkl", str(MERGED), "--frame-root", str(FRAME_ROOT), "--homography-cache", str(HOMOGRAPHY), "--sequence-fps-json", str(FPS_JSON), "--compact-model-output"]
    execute("score_forward_action_bank", 1, common + ["--out-jsonl", str(forward), "--out-summary", str(OUTPUT / "forward_summary.json")])
    execute("score_backward_action_bank", 2, common + ["--reverse", "--out-jsonl", str(backward), "--out-summary", str(OUTPUT / "backward_summary.json")])
    execute("score_true_time_neighbors", 3, [sys.executable, str(REPO / "tools" / "score_action_chunk_neighbor_bank.py"), "--predictionsgt-pkl", str(MERGED), "--homography-cache", str(HOMOGRAPHY), "--sequence-fps-json", str(FPS_JSON), "--seconds", ".25,1,3", "--bidirectional", "--out-jsonl", str(neighbor), "--out-summary", str(OUTPUT / "neighbor_summary.json")])
    execute("apply_frozen_nps_ensembles", 4, [sys.executable, str(REPO / "tools" / "predict_action_chunk_pretrained_ensembles.py"), "--predictionsgt-pkl", str(MERGED), "--forward-jsonl", str(forward), "--backward-jsonl", str(backward), "--neighbor-jsonl", str(neighbor), "--base-model-dir", str(BASE_MODELS), "--expert-model-dir", str(EXPERT_MODELS), "--out-base-jsonl", str(base_scores), "--out-expert-jsonl", str(expert_scores), "--out-summary", str(OUTPUT / "frozen_ensemble_summary.json")])
    execute("evaluate_fixed_v53_gate", 5, [sys.executable, str(REPO / "tools" / "sweep_action_chunk_temporal_multiplicity.py"), "--tvd-root", r"D:\urap_modal_stage\TransVisDrone", "--predictionsgt-pkl", str(MERGED), "--base-jsonl", str(base_scores), "--base-field", "action_chunk_neighbor_score", "--expert-jsonl", str(expert_scores), "--expert-field", "action_chunk_multi_expert_score", "--sequence-fps-json", str(FPS_JSON), "--fixed-config-json", str(V53_CONFIG), "--out-json", str(OUTPUT / "action_bank_v53_zero_shot.json")])

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    action = json.loads((OUTPUT / "action_bank_v53_zero_shot.json").read_text(encoding="utf-8"))["best"]
    gain = float(action["map50"]) - float(baseline["map50"])
    summary = {
        "protocol": "ARD100 test; frozen NPS detector and frozen NPS V46/V52/V53 Action Bank; no ARD100 fitting or parameter selection",
        "detector_baseline": baseline,
        "action_bank_zero_shot": action,
        "absolute_map50_gain": gain,
        "percentage_point_gain": 100.0 * gain,
        "target_gain_points": [3.0, 5.0],
        "target_minimum_met": gain >= 0.03,
    }
    (OUTPUT / "official_generalization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", 6, summary=summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
