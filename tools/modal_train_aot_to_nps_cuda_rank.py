from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "urap-vatd-aot-to-nps-rank-v1"
RUN_NAME = "aot_to_nps_cuda_rank_v1"

app = modal.App(APP_NAME)
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .pip_install("numpy==1.26.4", "matplotlib==3.9.2", "pandas==2.2.3", "pyyaml==6.0.2", "scipy==1.14.1", "tqdm==4.67.1", "seaborn==0.13.2")
    .add_local_file(ROOT / "tools/train_detection_row_score_head.py", "/workspace/tools/train_detection_row_score_head.py", copy=True)
    .add_local_file(ROOT / "tools/eval_tvd_predictionsgt_pkl.py", "/workspace/tools/eval_tvd_predictionsgt_pkl.py", copy=True)
    .add_local_file(ROOT / "tools/rescore_li_tetc_diagnostics_from_tracklets.py", "/workspace/tools/rescore_li_tetc_diagnostics_from_tracklets.py", copy=True)
    .add_local_file(ROOT / "tools/sweep_tvd_predictionsgt_action_rescore.py", "/workspace/tools/sweep_tvd_predictionsgt_action_rescore.py", copy=True)
    .add_local_file(ROOT / "tools/sweep_tvd_predictionsgt_score_fusion.py", "/workspace/tools/sweep_tvd_predictionsgt_score_fusion.py", copy=True)
    .add_local_file(ROOT / "tools/sweep_tvd_predictionsgt_two_score_fusion.py", "/workspace/tools/sweep_tvd_predictionsgt_two_score_fusion.py", copy=True)
    .add_local_dir(Path(r"D:\urap_modal_stage\TransVisDrone\utils"), "/workspace/tvd/utils", copy=True)
)
vatd = modal.Volume.from_name("vatd-artifacts")
results = modal.Volume.from_name("vatd-rank-results-v1", create_if_missing=True)


def run_command(command: list[str], cwd: str = "/workspace/tools") -> None:
    print(json.dumps({"kind": "command", "command": command, "time": time.time()}), flush=True)
    subprocess.run(command, cwd=cwd, env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=True)


@app.function(
    image=image,
    gpu="L40S",
    cpu=16,
    memory=65536,
    volumes={"/vatd": vatd, "/results": results},
    timeout=24 * 60 * 60,
)
def train_and_evaluate() -> dict:
    vatd.reload()
    results.reload()
    output = Path("/results") / RUN_NAME
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.json"
    progress.write_text(json.dumps({"stage": "train", "done": 0, "total": 2, "updated": time.time()}, indent=2), encoding="utf-8")
    results.commit()

    train_tracklets = Path("/vatd/aot_indep/saliency_tracklets/proposal_tracklets.jsonl")
    train_gt = Path("/vatd/aot_indep/gt.csv")
    test_tracklets = Path("/vatd/nps/vatd/tracklets_with_vatd.jsonl")
    predictions = Path("/vatd/nps/predictionsgt_split_0.pkl")
    for required in (train_tracklets, train_gt, test_tracklets, predictions):
        if not required.is_file():
            raise FileNotFoundError(required)

    scored = output / "nps_tracklets_aot_rank_scored.jsonl"
    run_command([
        sys.executable, "/workspace/tools/train_detection_row_score_head.py",
        "--train-tracklets", str(train_tracklets),
        "--train-gt-csv", str(train_gt),
        "--test-tracklets", str(test_tracklets),
        "--out-test-tracklets", str(scored),
        "--out-model", str(output / "aot_rank_model.pt"),
        "--out-summary", str(output / "train_summary.json"),
        "--score-field", "aot_rank_score",
        "--iou-threshold", "0.5",
        "--negative-min-score", "0.005",
        "--label-policy", "unique-iou",
        "--epochs", "30",
        "--batch-size", "32768",
        "--hidden", "256",
        "--lr", "0.0005",
        "--pairwise-weight", "1.0",
        "--pairwise-pairs", "65536",
        "--model-kind", "unified-two-tower",
        "--tracklet-aux-weight", "0.25",
        "--feature-groups", "all",
    ])
    progress.write_text(json.dumps({"stage": "evaluate", "done": 1, "total": 2, "updated": time.time()}, indent=2), encoding="utf-8")
    results.commit()

    run_command([
        sys.executable, "/workspace/tools/sweep_tvd_predictionsgt_two_score_fusion.py",
        "--tvd-root", "/workspace/tvd",
        "--predictionsgt-pkl", str(predictions),
        "--meta-tracklet-jsonl", str(scored),
        "--meta-score-field", "vatd_score",
        "--row-tracklet-jsonl", str(scored),
        "--row-score-field", "aot_rank_score",
        "--modes", "logit-3mix", "meta-logit-row-geom", "meta-logit-row-suppress", "meta-logit-row-boost",
        "--alphas", "0.00 0.02 0.04 0.06 0.08 0.10 0.14",
        "--betas", "0.01 0.02 0.04 0.06 0.10 0.16 0.24",
        "--missing-score-behaviors", "keep",
        "--out-json", str(output / "fusion_sweep.json"),
        "--write-best-pkl", str(output / "best_predictionsgt.pkl"),
    ])
    summary = json.loads((output / "fusion_sweep.json").read_text(encoding="utf-8"))
    complete = {
        "stage": "done",
        "done": 2,
        "total": 2,
        "updated": time.time(),
        "best": summary.get("best"),
        "output": str(output),
        "gpu": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }
    progress.write_text(json.dumps(complete, indent=2), encoding="utf-8")
    (output / "COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    results.commit()
    print(json.dumps(complete, indent=2), flush=True)
    return complete


@app.local_entrypoint()
def main() -> None:
    result = train_and_evaluate.remote()
    print(json.dumps(result, indent=2), flush=True)
