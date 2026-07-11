from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
PYTHON = Path(sys.executable)
EVAL_PYTHON = REPO / "papers" / "TransVisDrone" / ".venv" / "Scripts" / "python.exe"
RUN = REPO / "artifacts" / "detached_aot_action_memory_action_only_v85"
PROGRESS = RUN / "progress.json"
OUT = REPO / "artifacts" / "route_b_official" / "aot_action_memory_action_only_v85"
SOURCE = REPO / "artifacts" / "route_b_official" / "aot_clean_flow_v18_recovery_full_v57_final" / "aotpredictions" / "predictions_split_0.pkl"
MODEL = Path(r"D:\URAP_vatd_rank_results\nps_action_memory_action_only_v84\model.pt")
FRAMES = Path(r"D:\URAP_datasets\TransVisDrone\AOT_part1_yolo_fulltest\test")
DATASET = Path(r"D:\URAP_datasets\AOT\part1")
TOTAL = 5


def report(stage: str, done: int, **extra) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": TOTAL, "updated": datetime.now().astimezone().isoformat(), **extra}
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def execute(stage: str, done: int, command: list[str], cwd: Path = REPO) -> None:
    process = subprocess.Popen(command, cwd=cwd, env={**os.environ, "PYTHONPATH": str(REPO) + os.pathsep + str(REPO / "tools"), "PYTHONUNBUFFERED": "1"})
    report(stage, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    report(stage + "_done", done + 1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    predictionsgt = OUT / "v57_predictionsgt.pkl"
    auxiliary = OUT / "action_memory_aux.jsonl"
    scores = OUT / "action_memory_scores.jsonl"
    rescored = OUT / "rescored" / "aotpredictions"
    evaluation = OUT / "official_eval"
    execute("convert_v57", 0, [str(PYTHON), str(REPO / "tools" / "convert_aot_prediction_list_to_predictionsgt.py"), "--input", str(SOURCE), "--out", str(predictionsgt)])
    execute("build_camera_compensated_action_memory", 1, [
        str(PYTHON), str(REPO / "tools" / "score_predictionsgt_online_action_bank_motion_parallel.py"),
        "--predictionsgt-pkl", str(predictionsgt), "--frame-root", str(FRAMES), "--homography-cache", str(OUT / "homographies.pkl"),
        "--out-jsonl", str(auxiliary), "--out-summary", str(OUT / "action_memory_aux_summary.json"),
        "--workers", "8",
        "--fps", "10", "--short-seconds", "1", "--long-seconds", "3", "--beam-size", "6",
        "--short-token-count", "8", "--long-token-count", "16", "--start-gate", "0.12", "--update-gate", "0.08", "--internal-alpha", "2.5",
    ])
    execute("score_frozen_action_only_cross_attention", 2, [
        str(PYTHON), str(REPO / "tools" / "score_action_memory_cross_attention.py"),
        "--predictionsgt-pkl", str(predictionsgt), "--aux-jsonl", str(auxiliary), "--model", str(MODEL),
        "--out-scores", str(scores), "--out-summary", str(OUT / "score_summary.json"), "--score-field", "action_memory_action_only_score", "--batch-size", "8192",
    ])
    execute("apply_fixed_zero_shot_residual", 3, [
        str(PYTHON), str(REPO / "tools" / "apply_action_memory_scores_to_aot.py"),
        "--input", str(SOURCE), "--scores", str(scores), "--score-field", "action_memory_action_only_score",
        "--mode", "symmetric", "--cap", "0.25", "--weight", "0.5", "--out-dir", str(rescored),
    ])
    execute("official_aot_evaluation", 4, [
        str(EVAL_PYTHON), ".\\evaluate_aot.py", "--results_folder", str(rescored), "--evaluation_folder", str(evaluation),
        "--detection_threshold", "0.2", "--dataset-path", str(DATASET),
    ], REPO / "papers" / "TransVisDrone")
    summaries = sorted(evaluation.rglob("summary_far_*_min_intruder_fl_dr_0p5_in_win_30.json"))
    if not summaries:
        raise RuntimeError("official AOT evaluation produced no summary")
    result = json.loads(summaries[-1].read_text(encoding="utf-8"))
    incumbent_afdr = 0.8999138980434073
    summary = {
        "protocol": "frozen NPS Action-only Cross-Attention; AOT fps=10; camera compensated 1s/3s memory; fixed zero-shot residual",
        "cross_attention": {"afdr": result["fl_dr_in_range"], "fppi": result["fppi"], "far": result["far"]},
        "incumbent_v57": {"afdr": incumbent_afdr, "fppi": 0.24073023217464268, "far": 78.48837209302326},
        "champion": "cross_attention_v85" if float(result["fl_dr_in_range"]) > incumbent_afdr else "incumbent_v57",
        "architecture_modified_for_aot": False,
        "summary_path": str(summaries[-1]),
    }
    (OUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", TOTAL, summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
