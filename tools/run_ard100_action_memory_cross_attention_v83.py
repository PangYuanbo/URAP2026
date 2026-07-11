from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
PYTHON = Path(sys.executable)
RUN = REPO / "artifacts" / "detached_ard100_action_memory_cross_attention_v83"
PROGRESS = RUN / "progress.json"
SOURCE = Path(r"D:\URAP_vatd_rank_results\ard100_yolomg_generalization_v2")
OUT = Path(r"D:\URAP_vatd_rank_results\ard100_action_memory_cross_attention_v83")
MODEL = Path(r"D:\URAP_vatd_rank_results\nps_action_memory_cross_attention_v82\model.pt")
VAL_PKL = SOURCE / "ard100_yolomg_val_predictionsgt.pkl"
TEST_PKL = SOURCE / "ard100_yolomg_predictionsgt.pkl"
VAL_FRAMES = Path(r"U:\URAP_datasets\TransVisDrone\ARD100\AllFrames\val")
TEST_FRAMES = Path(r"U:\URAP_datasets\TransVisDrone\ARD100\AllFrames\test")
VAL_H = SOURCE / "ard100_val_homographies.pkl"
TEST_H = Path(r"D:\URAP_vatd_rank_results\ard100_generalization_v1\ard100_test_homographies.pkl")
VAL_FPS = REPO / "data_templates" / "ard100_val_sequence_fps.json"
TEST_FPS = REPO / "data_templates" / "ard100_sequence_fps.json"
TVD = Path(r"D:\urap_modal_stage\TransVisDrone")
TOTAL = 6


def report(stage: str, done: int, **extra) -> None:
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


def online_command(pkl: Path, frames: Path, homographies: Path, fps: Path, scores: Path, summary: Path) -> list[str]:
    return [
        str(PYTHON), str(REPO / "tools" / "score_predictionsgt_online_action_bank_motion.py"),
        "--predictionsgt-pkl", str(pkl), "--frame-root", str(frames), "--homography-cache", str(homographies),
        "--out-jsonl", str(scores), "--out-summary", str(summary), "--sequence-fps-json", str(fps),
        "--short-seconds", "1.0", "--long-seconds", "3.0", "--beam-size", "6",
        "--short-token-count", "8", "--long-token-count", "16", "--start-gate", "0.12", "--update-gate", "0.08", "--internal-alpha", "2.5",
    ]


def frozen_command(pkl: Path, auxiliary: Path, scores: Path, summary: Path) -> list[str]:
    return [
        str(PYTHON), str(REPO / "tools" / "score_action_memory_cross_attention.py"),
        "--predictionsgt-pkl", str(pkl), "--aux-jsonl", str(auxiliary), "--model", str(MODEL),
        "--out-scores", str(scores), "--out-summary", str(summary), "--batch-size", "8192",
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    execute("build_val_action_memory", 0, online_command(VAL_PKL, VAL_FRAMES, VAL_H, VAL_FPS, OUT / "val_aux.jsonl", OUT / "val_aux_summary.json"))
    execute("build_test_action_memory", 1, online_command(TEST_PKL, TEST_FRAMES, TEST_H, TEST_FPS, OUT / "test_aux.jsonl", OUT / "test_aux_summary.json"))
    execute("score_val_frozen_model", 2, frozen_command(VAL_PKL, OUT / "val_aux.jsonl", OUT / "val_scores.jsonl", OUT / "val_score_summary.json"))
    execute("score_test_frozen_model", 3, frozen_command(TEST_PKL, OUT / "test_aux.jsonl", OUT / "test_scores.jsonl", OUT / "test_score_summary.json"))
    execute("select_val_fusion", 4, [
        str(PYTHON), str(REPO / "tools" / "sweep_tvd_predictionsgt_score_fusion.py"),
        "--tvd-root", str(TVD), "--predictionsgt-pkl", str(VAL_PKL), "--tracklet-jsonl", str(OUT / "val_scores.jsonl"),
        "--score-field", "action_memory_cross_attention_score", "--per-row-score",
        "--modes", "replace", "linear-mix", "logit-mix", "geom-mix", "fp-suppress", "tp-boost",
        "--alphas", "0 0.001 0.002 0.005 0.01 0.02 0.04 0.06 0.08 0.1 0.14 0.2 0.3 0.4 0.55 0.7 0.85 1.0",
        "--out-json", str(OUT / "val_fusion.json"),
    ])
    best = json.loads((OUT / "val_fusion.json").read_text(encoding="utf-8"))["best"]
    execute("evaluate_test_fixed", 5, [
        str(PYTHON), str(REPO / "tools" / "sweep_tvd_predictionsgt_score_fusion.py"),
        "--tvd-root", str(TVD), "--predictionsgt-pkl", str(TEST_PKL), "--tracklet-jsonl", str(OUT / "test_scores.jsonl"),
        "--score-field", "action_memory_cross_attention_score", "--per-row-score", "--modes", str(best["mode"]), "--alphas", str(best["alpha"]),
        "--out-json", str(OUT / "test_fixed.json"),
    ])
    detector = json.loads((SOURCE / "detector_baseline.json").read_text(encoding="utf-8"))
    incumbent = json.loads((SOURCE / "action_bank_summary.json").read_text(encoding="utf-8"))["action_bank"]
    cross_attention = json.loads((OUT / "test_fixed.json").read_text(encoding="utf-8"))["best"]
    champion = incumbent if float(incumbent["map50"]) >= float(cross_attention["map50"]) else cross_attention
    summary = {
        "protocol": "frozen NPS Action Memory Cross-Attention; ARD100 validation selects fusion only; fixed ARD100 test",
        "detector": detector,
        "incumbent_action_bank": incumbent,
        "cross_attention": cross_attention,
        "champion": champion,
        "champion_source": "incumbent_action_bank" if champion is incumbent else "cross_attention",
        "architecture_modified_for_ard100": False,
    }
    (OUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", TOTAL, summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
