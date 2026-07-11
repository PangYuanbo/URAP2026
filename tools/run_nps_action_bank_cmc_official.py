from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
PYTHON = Path(sys.executable)
TVD = Path(r"D:\urap_modal_stage\TransVisDrone")
TRAIN_TRACKLETS = Path(r"D:\URAP_nps_train_tvd\route_b_official\tracklets\proposal_tracklets.jsonl")
VAL_TRACKLETS = Path(r"D:\URAP_nps_val_tvd\route_b_official\tracklets\proposal_tracklets.jsonl")
VAL_PKL = Path(r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl")
TEST_TRACKLETS = Path(r"D:\URAP_vatd_rank_inputs\nps_tracklets_with_vatd.jsonl")
TEST_PKL = Path(r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl")
FRAMES = Path(r"U:\URAP_datasets\TransVisDrone\NPS\AllFrames")
OUTPUT = Path(r"D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2")
RUNNER = REPO / "artifacts/detached_nps_action_bank_cmc_v2"
PROGRESS = RUNNER / "progress.json"
MARKERS = RUNNER / "stage_markers"
TOTAL = 10


def write_progress(stage: str, done: int, **extra: object) -> None:
    RUNNER.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": TOTAL, "updated": datetime.now(timezone.utc).astimezone().isoformat(), **extra}
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(command: list[str], stage: str, done: int, *, child_progress: Path | None = None) -> None:
    marker = MARKERS / f"{done + 1:02d}_{stage}.json"
    if marker.is_file():
        print(json.dumps({"kind": "nps_action_bank_stage_resumed", "stage": stage, "marker": str(marker)}), flush=True)
        write_progress(f"{stage}_already_done", done + 1, marker=str(marker))
        return
    print(json.dumps({"kind": "nps_action_bank_command", "stage": stage, "command": command}), flush=True)
    process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO)})
    write_progress(stage, done, child_pid=process.pid, command=command, child_progress=str(child_progress) if child_progress else None)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    MARKERS.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"stage": stage, "completed": datetime.now(timezone.utc).astimezone().isoformat(), "command": command}, indent=2), encoding="utf-8")
    write_progress(f"{stage}_done", done + 1, marker=str(marker))


def require_inputs() -> None:
    required = (TVD, TRAIN_TRACKLETS, VAL_TRACKLETS, VAL_PKL, TEST_TRACKLETS, TEST_PKL, FRAMES / "train", FRAMES / "val", FRAMES / "test")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing official CMC Action Bank inputs: {missing}")


def cmc_command(source: Path, split: str, output: Path, cache: Path, summary: Path, progress: Path) -> list[str]:
    return [
        str(PYTHON), str(REPO / "tools/score_tracklets_samurai_cmc.py"),
        "--input-tracklets", str(source), "--frame-root", str(FRAMES / split),
        "--output-tracklets", str(output), "--summary-json", str(summary),
        "--progress-json", str(progress), "--homography-cache", str(cache),
        "--max-size", "320", "--causal-only", "--cache-save-every", "50000",
    ]


def main() -> int:
    require_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    homographies = OUTPUT / "homographies"
    homographies.mkdir(parents=True, exist_ok=True)
    train_cache = homographies / "train.pkl"
    val_cache = homographies / "val.pkl"
    test_cache = homographies / "test.pkl"
    existing_val_cache = Path(r"D:\URAP_vatd_rank_results\nps_val_samurai_cmc_v1\homographies.pkl")
    if not val_cache.exists() and existing_val_cache.exists():
        shutil.copy2(existing_val_cache, val_cache)

    train_cmc = OUTPUT / "train_tracklets_causal_cmc.jsonl"
    val_cmc = OUTPUT / "val_tracklets_causal_cmc.jsonl"
    test_cmc = OUTPUT / "test_tracklets_causal_cmc.jsonl"
    weights = OUTPUT / "action_bank_causal_cmc.pt"
    val_scored = OUTPUT / "val_tracklets_action_bank.jsonl"
    val_sweep = OUTPUT / "val_fusion_sweep.json"
    test_scored = OUTPUT / "test_tracklets_action_bank.jsonl"
    test_eval = OUTPUT / "test_fixed_fusion.json"
    fps_map = REPO / "data_templates/nps_sequence_fps.json"

    for done, split, cache in ((0, "train", train_cache), (1, "test", test_cache)):
        progress = RUNNER / f"precompute_{split}_progress.json"
        run([
            str(PYTHON), str(REPO / "tools/precompute_nps_homographies.py"),
            "--frame-root", str(FRAMES / split), "--output-cache", str(cache),
            "--partial-dir", str(homographies / f"{split}_parts"), "--progress-json", str(progress),
            "--max-size", "320", "--workers", "6",
        ], f"precompute_{split}_homographies", done, child_progress=progress)

    for done, split, source, output, cache in (
        (2, "train", TRAIN_TRACKLETS, train_cmc, train_cache),
        (3, "val", VAL_TRACKLETS, val_cmc, val_cache),
        (4, "test", TEST_TRACKLETS, test_cmc, test_cache),
    ):
        progress = RUNNER / f"annotate_{split}_progress.json"
        run(cmc_command(source, split, output, cache, OUTPUT / f"{split}_cmc_summary.json", progress), f"annotate_{split}_causal_cmc", done, child_progress=progress)

    run([
        str(PYTHON), str(REPO / "tools/train_action_bank.py"),
        "--train-tracklets", str(train_cmc), "--out", str(weights),
        "--epochs", "10", "--batch-size", "1024", "--lr", "0.0003",
        "--short-tokens", "12", "--long-tokens", "18", "--sequence-fps-json", str(fps_map),
        "--cache-dir", str(OUTPUT / "train_action_bank_cache"), "--device", "cuda",
    ], "train_action_bank_causal_cmc", 5)

    run([
        str(PYTHON), str(REPO / "tools/score_action_bank.py"), "--tracklets", str(val_cmc),
        "--weights", str(weights), "--out", str(val_scored), "--batch-size", "4096", "--device", "cuda",
    ], "score_validation", 6)

    run([
        str(PYTHON), str(REPO / "tools/sweep_tvd_predictionsgt_score_fusion.py"),
        "--tvd-root", str(TVD), "--predictionsgt-pkl", str(VAL_PKL), "--tracklet-jsonl", str(val_scored),
        "--score-field", "action_bank_learned_score", "--per-row-score", "--modes", "linear-mix", "logit-mix", "geom-mix", "fp-suppress", "tp-boost",
        "--alphas", "0.001 0.002 0.005 0.01 0.02 0.04 0.06 0.08 0.10 0.14 0.20 0.30 0.40",
        "--out-json", str(val_sweep), "--write-best-pkl", str(OUTPUT / "val_best_predictionsgt.pkl"),
    ], "select_on_validation", 7)

    run([
        str(PYTHON), str(REPO / "tools/score_action_bank.py"), "--tracklets", str(test_cmc),
        "--weights", str(weights), "--out", str(test_scored), "--batch-size", "4096", "--device", "cuda",
    ], "score_test", 8)

    validation = json.loads(val_sweep.read_text(encoding="utf-8"))
    best = validation["best"]
    run([
        str(PYTHON), str(REPO / "tools/sweep_tvd_predictionsgt_score_fusion.py"),
        "--tvd-root", str(TVD), "--predictionsgt-pkl", str(TEST_PKL), "--tracklet-jsonl", str(test_scored),
        "--score-field", "action_bank_learned_score", "--per-row-score", "--modes", str(best["mode"]), "--alphas", str(best["alpha"]),
        "--out-json", str(test_eval), "--write-best-pkl", str(OUTPUT / "test_fixed_best_predictionsgt.pkl"),
    ], "evaluate_test_fixed", 9)

    test = json.loads(test_eval.read_text(encoding="utf-8"))
    summary = {
        "protocol": "train Clips1-36; select fusion on Clips37-40; freeze and evaluate Clips41-50",
        "causal_inference": True,
        "camera_motion": "adjacent background homography; camera displacement subtracted before Action Bank velocity",
        "validation_best": best,
        "test_fixed": test["best"],
        "target_map50": 0.97,
        "target_met": float(test["best"]["map50"]) >= 0.97,
        "weights": str(weights),
    }
    (OUTPUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_progress("done", TOTAL, summary=summary, output=str(OUTPUT))
    print(json.dumps({"kind": "nps_action_bank_cmc_done", **summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
