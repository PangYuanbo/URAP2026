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
RUN = REPO / "artifacts" / "detached_aot_action_memory_target_v86"
PROGRESS = RUN / "progress.json"
OUT = REPO / "artifacts" / "route_b_official" / "aot_action_memory_target_v86"
V85 = REPO / "artifacts" / "route_b_official" / "aot_action_memory_action_only_v85"
SOURCE = REPO / "artifacts" / "route_b_official" / "aot_clean_flow_v18_recovery_full_v57_final" / "aotpredictions" / "predictions_split_0.pkl"
PART0 = REPO / "artifacts" / "route_b_official" / "aot_action_chunk_transfer_v1" / "validation_source" / "predictions_split_0.pkl"
LABELS = Path(r"D:\URAP_datasets\TransVisDrone\AOT_part1_yolo_fulltest\test\part0\labels")
INIT_MODEL = Path(r"D:\URAP_vatd_rank_results\nps_action_memory_action_only_v84\model.pt")
DATASET = Path(r"D:\URAP_datasets\AOT\part1")
TOTAL = 7


def report(stage: str, done: int, **extra: object) -> None:
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
    corrected_predictionsgt = OUT / "full_corrected_predictionsgt.pkl"
    corrected_aux = OUT / "full_corrected_aux.jsonl"
    execute("convert_center_xywh_correctly", 0, [
        str(PYTHON), str(REPO / "tools" / "convert_aot_prediction_list_to_predictionsgt.py"), "--input", str(SOURCE), "--out", str(corrected_predictionsgt),
    ])
    execute("build_corrected_camera_compensated_memory", 1, [
        str(PYTHON), str(REPO / "tools" / "score_predictionsgt_online_action_bank_motion_parallel.py"),
        "--predictionsgt-pkl", str(corrected_predictionsgt), "--frame-root", str(Path(r"D:\URAP_datasets\TransVisDrone\AOT_part1_yolo_fulltest\test")),
        "--homography-cache", str(OUT / "homographies.pkl"), "--out-jsonl", str(corrected_aux), "--out-summary", str(OUT / "full_corrected_aux_summary.json"),
        "--workers", "8", "--fps", "10", "--short-seconds", "1", "--long-seconds", "3", "--beam-size", "6",
        "--short-token-count", "8", "--long-token-count", "16", "--start-gate", "0.12", "--update-gate", "0.08", "--internal-alpha", "2.5",
    ])
    execute("prepare_part0_train_val", 2, [
        str(PYTHON), str(REPO / "tools" / "prepare_aot_action_memory_target_split.py"),
        "--full-predictionsgt", str(corrected_predictionsgt), "--part0-source", str(PART0),
        "--label-root", str(LABELS), "--full-aux", str(corrected_aux), "--out-dir", str(OUT / "data"),
    ])
    execute("finetune_cross_action_memory", 3, [
        str(PYTHON), str(REPO / "tools" / "train_action_bank_motion_token_listwise.py"),
        "--train-pkl", str(OUT / "data" / "train_predictionsgt.pkl"), "--train-aux-tracklets", str(OUT / "data" / "train_aux.jsonl"),
        "--val-pkl", str(OUT / "data" / "val_predictionsgt.pkl"), "--val-aux-tracklets", str(OUT / "data" / "val_aux.jsonl"),
        "--test-pkl", str(corrected_predictionsgt), "--test-aux-tracklets", str(corrected_aux),
        "--out-val-scores", str(OUT / "val_scores.jsonl"), "--out-test-scores", str(OUT / "full_scores.jsonl"),
        "--out-model", str(OUT / "model.pt"), "--out-summary", str(OUT / "train_summary.json"),
        "--score-field", "aot_target_action_memory_score", "--cross-attention", "--action-only-query", "--init-model", str(INIT_MODEL),
        "--epochs", "24", "--frame-batch-size", "96", "--inference-batch-size", "8192", "--hidden", "256", "--lr", "0.00005",
        "--attention-heads", "4", "--memory-layers", "1", "--write-loss-weight", "0.05",
    ])
    execute("select_validation_fusion", 4, [
        str(PYTHON), str(REPO / "tools" / "sweep_tvd_predictionsgt_score_fusion.py"),
        "--predictionsgt-pkl", str(OUT / "data" / "val_predictionsgt.pkl"), "--tracklet-jsonl", str(OUT / "val_scores.jsonl"),
        "--score-field", "aot_target_action_memory_score", "--per-row-score", "--modes", "linear-mix", "logit-mix", "geom-mix", "fp-suppress", "tp-boost",
        "--alphas", ".02,.05,.1,.2,.3,.4,.5,.7", "--missing-score-behaviors", "keep", "--out-json", str(OUT / "val_fusion.json"),
    ])
    best = json.loads((OUT / "val_fusion.json").read_text(encoding="utf-8"))["best"]
    execute("apply_fixed_full_aot", 5, [
        str(PYTHON), str(REPO / "tools" / "apply_action_memory_scores_to_aot.py"), "--input", str(SOURCE),
        "--scores", str(OUT / "full_scores.jsonl"), "--score-field", "aot_target_action_memory_score",
        "--fusion-mode", str(best["mode"]), "--alpha", str(best["alpha"]), "--out-dir", str(OUT / "rescored" / "aotpredictions"),
    ])
    execute("official_full_aot_evaluation", 6, [
        str(EVAL_PYTHON), ".\\evaluate_aot.py", "--results_folder", str(OUT / "rescored" / "aotpredictions"),
        "--evaluation_folder", str(OUT / "official_eval"), "--detection_threshold", "0.2", "--dataset-path", str(DATASET),
    ], REPO / "papers" / "TransVisDrone")
    summaries = sorted((OUT / "official_eval").rglob("summary_far_*_min_intruder_fl_dr_0p5_in_win_30.json"))
    result = json.loads(summaries[-1].read_text(encoding="utf-8"))
    incumbent = {"afdr": 0.8999138980434073, "fppi": 0.24073023217464268, "far": 78.48837209302326}
    target = {"afdr": float(result["fl_dr_in_range"]), "fppi": float(result["fppi"]), "far": float(result["far"])}
    summary = {
        "protocol": "AOT part0 target-domain fine-tuning from NPS Action Memory; clip-disjoint train/validation; fixed full-AOT evaluation",
        "validation_selection": best, "target_trained": target, "incumbent_v57": incumbent,
        "afdr_gain_points": 100.0 * (target["afdr"] - incumbent["afdr"]),
        "champion": "target_trained_v86" if target["afdr"] > incumbent["afdr"] else "incumbent_v57",
        "full_test_labels_used_for_training": False,
    }
    (OUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", TOTAL, summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
