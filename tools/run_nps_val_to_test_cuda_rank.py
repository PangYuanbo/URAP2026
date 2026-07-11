from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
TVD = Path(r"D:\urap_modal_stage\TransVisDrone")
DATA = Path(r"D:\URAP_nps_val_tvd")
TEST_INPUT = Path(r"D:\URAP_vatd_rank_inputs")
OUTPUT = Path(r"D:\URAP_vatd_rank_results\nps_val_to_test_cuda_rank_v1")
RUNNER = REPO / "artifacts/detached_nps_val_to_test_cuda_rank"
PROGRESS = RUNNER / "progress.json"
PYTHON = Path(sys.executable)


def write_progress(stage: str, done: int, total: int = 5, **extra: object) -> None:
    RUNNER.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps({"stage": stage, "done": done, "total": total, "updated": datetime.now(timezone.utc).astimezone().isoformat(), **extra}, indent=2), encoding="utf-8")


def run(command: list[str], cwd: Path, stage: str, done: int) -> None:
    print(json.dumps({"kind": "pipeline_command", "stage": stage, "command": command}), flush=True)
    process = subprocess.Popen(command, cwd=cwd, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO)})
    write_progress(stage, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main() -> int:
    frames = DATA / "AllFrames/val"
    labels = DATA / "NPSvisdroneStyle/val/labels"
    videos = DATA / "Videos"
    weights = DATA / "weights/best.pt"
    frame_count = sum(1 for _ in frames.glob("*.png")) if frames.is_dir() else 0
    label_count = sum(1 for _ in labels.glob("*.txt")) if labels.is_dir() else 0
    if frame_count != 5944 or label_count != 5944 or not weights.is_file() or weights.stat().st_size < 900_000_000:
        raise RuntimeError(f"inputs incomplete frames={frame_count} labels={label_count} weights={weights.stat().st_size if weights.exists() else 0}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    yaml_path = DATA / "NPS_URAP_D_val.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {str(DATA / 'AllFrames').replace(chr(92), '/')}",
            "train: val", "val: val", "test: val", "inference: val",
            f"annotation_path: {str(DATA / 'NPSvisdroneStyle').replace(chr(92), '/')}",
            "annotation_train: val/labels", "annotation_val: val/labels", "annotation_test: val/labels",
            f"video_root_path: {str(videos).replace(chr(92), '/')}",
            "video_root_path_train: val", "video_root_path_val: val", "video_root_path_test: val", "video_root_path_inference: val",
            "nc: 1", "names: ['drone']", "",
        ]), encoding="utf-8")
    inference_root = DATA / "runs"
    run([
        str(PYTHON), "val.py", "--data", str(yaml_path), "--weights", str(weights), "--task", "val",
        "--img", "1280", "--num-frames", "5", "--augment", "--save-json", "--save-json-gt",
        "--device", "0", "--batch-size", "8", "--half", "--project", str(inference_root),
        "--name", "nps_val_rank_source", "--exist-ok",
    ], TVD, "inference", 0)
    predictions = inference_root / "nps_val_rank_source/predictionsgt/predictionsgt_split_0.pkl"
    if not predictions.is_file():
        raise FileNotFoundError(predictions)
    route = DATA / "route_b"
    run([
        str(PYTHON), str(REPO / "tools/export_tvd_predictionsgt_to_route_b.py"),
        "--predictionsgt-pkl", str(predictions), "--out-run-root", str(route / "run"),
        "--out-gt-csv", str(route / "gt.csv"), "--out-summary", str(route / "export_summary.json"),
        "--frame-root", str(frames), "--profile", "hard_recovery", "--diagnostics-name", "diagnostics_raw.jsonl",
    ], REPO, "export", 1)
    tracklets = route / "tracklets"
    run([
        str(PYTHON), "-m", "qstr_dronedet.cli", "build-proposal-tracklet-dataset",
        "--run-roots", str(route / "run"), "--gt-csv", str(route / "gt.csv"), "--out", str(tracklets),
        "--profile", "hard_recovery", "--diagnostics-name", "diagnostics_raw.jsonl",
        "--max-gap", "3", "--base-radius", "18", "--radius-per-side", "0.75", "--min-iou", "0.05",
        "--min-score", "0.0", "--min-tracklet-rows", "3", "--iou-threshold", "0.5", "--center-threshold", "24",
    ], REPO, "tracklets", 2)
    train_tracklets = tracklets / "proposal_tracklets.jsonl"
    scored = OUTPUT / "nps_test_tracklets_val_rank_scored.jsonl"
    run([
        str(PYTHON), str(REPO / "tools/train_detection_row_score_head.py"),
        "--train-tracklets", str(train_tracklets), "--train-gt-csv", str(route / "gt.csv"),
        "--test-tracklets", str(TEST_INPUT / "nps_tracklets_with_vatd.jsonl"),
        "--out-test-tracklets", str(scored), "--out-model", str(OUTPUT / "val_rank_model.pt"),
        "--out-summary", str(OUTPUT / "train_summary.json"), "--score-field", "val_rank_score",
        "--iou-threshold", "0.5", "--negative-min-score", "0.005", "--label-policy", "unique-iou",
        "--epochs", "30", "--batch-size", "32768", "--hidden", "256", "--lr", "0.0005",
        "--pairwise-weight", "1.0", "--pairwise-pairs", "65536", "--model-kind", "unified-two-tower",
        "--tracklet-aux-weight", "0.25", "--feature-groups", "all",
    ], REPO, "train", 3)
    run([
        str(PYTHON), str(REPO / "tools/sweep_tvd_predictionsgt_two_score_fusion_fast.py"),
        "--tvd-root", str(TVD), "--predictionsgt-pkl", str(TEST_INPUT / "nps_predictionsgt_split_0.pkl"),
        "--meta-tracklet-jsonl", str(scored), "--meta-score-field", "vatd_score",
        "--row-tracklet-jsonl", str(scored), "--row-score-field", "val_rank_score",
        "--modes", "logit-3mix", "meta-logit-row-geom", "meta-logit-row-suppress", "meta-logit-row-boost",
        "--alphas", "0.00", "0.02", "0.04", "0.06", "0.08", "0.10", "0.14", "0.20",
        "--betas", "0.005", "0.01", "0.02", "0.04", "0.06", "0.10", "0.16", "0.24", "0.32",
        "--out-json", str(OUTPUT / "fusion_sweep_fast.json"), "--write-best-pkl", str(OUTPUT / "best_predictionsgt.pkl"),
    ], REPO, "evaluate", 4)
    summary = json.loads((OUTPUT / "fusion_sweep_fast.json").read_text(encoding="utf-8"))
    write_progress("done", 5, best=summary.get("best"), output=str(OUTPUT))
    print(json.dumps({"kind": "pipeline_done", "best": summary.get("best"), "output": str(OUTPUT)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
