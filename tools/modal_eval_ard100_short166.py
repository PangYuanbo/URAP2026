from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
app = modal.App("urap-ard100-samurai-short166-eval-v1")
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("hydra-core==1.3.2", "iopath==0.1.10", "opencv-python-headless==4.10.0.84", "tqdm==4.67.1")
    .add_local_file(ROOT / "tools/eval_samurai_nps.py", "/workspace/eval_samurai_nps.py", copy=True)
    .add_local_file(ROOT / "tools/merge_samurai_eval_shards.py", "/workspace/merge_samurai_eval_shards.py", copy=True)
)
code = modal.Volume.from_name("urap-code-artifacts-v1")
weights = modal.Volume.from_name("urap-model-weights-v1")
dataset = modal.Volume.from_name("urap-ard100-samurai-short166-v1")
tvd = modal.Volume.from_name("urap-ard100-transvisdrone-links-v1")
results = modal.Volume.from_name("urap-ard100-samurai-short166-results-v1", create_if_missing=True)


def checkpoint_for(run: str) -> str:
    if run.endswith("finetuned"):
        return "/results/finetune_base_plus_ard100_short166_stage1/checkpoints/checkpoint.pt"
    return "/weights/SAMURAI/sam2.1_hiera_base_plus.pt"


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=49152,
    volumes={"/code": code, "/weights": weights, "/dataset": dataset, "/tvd": tvd, "/results": results},
    timeout=24 * 60 * 60,
)
def evaluate(run: str, shard_count: int, shard_index: int) -> dict:
    allowed = {"image_box_zero_shot", "sam2_video_zero_shot", "samurai_zero_shot", "sam2_video_finetuned", "samurai_finetuned"}
    if run not in allowed:
        raise ValueError(run)
    for volume in (code, weights, dataset, tvd, results):
        volume.reload()
    manifest = json.loads(Path("/dataset/ARD100_SAMURAI_SHORT166/test_v1/manifest.json").read_text(encoding="utf-8"))
    checkpoint = checkpoint_for(run)
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    sam2_root = Path("/code/third_party/samurai/sam2")
    config = "configs/samurai/sam2.1_hiera_b+.yaml" if run.startswith("samurai") else "configs/sam2.1/sam2.1_hiera_b+.yaml"
    mode = "image-box" if run.startswith("image_box") else "video"
    output_root = Path(f"/results/eval/{run}/shard_{shard_index:02d}_of_{shard_count:02d}")
    env = os.environ.copy()
    env.update(PYTHONPATH=str(sam2_root), PYTHONUTF8="1")
    command = [
        sys.executable, "/workspace/eval_samurai_nps.py",
        "--dataset-root", "/dataset/ARD100_SAMURAI_SHORT166/test_v1",
        "--split", "test",
        "--checkpoint", checkpoint,
        "--model-config", config,
        "--output-root", str(output_root),
        "--device", "cuda:0",
        "--dtype", "bfloat16",
        "--propagation-mode", mode,
        "--resume",
        "--async-loading-frames",
        "--sequence-shard-count", str(shard_count),
        "--sequence-shard-index", str(shard_index),
    ]
    subprocess.run(command, cwd=sam2_root, env=env, check=True)
    results.commit()
    metrics = json.loads((output_root / "metrics.json").read_text(encoding="utf-8"))
    print(json.dumps({"run": run, "shard": shard_index, "done": metrics["sequences"], "total": manifest["sequence_count"]}, indent=2), flush=True)
    return metrics


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    volumes={"/dataset": dataset, "/results": results},
    timeout=60 * 60,
)
def merge(run: str, shard_count: int) -> dict:
    dataset.reload()
    results.reload()
    expected = json.loads(Path("/dataset/ARD100_SAMURAI_SHORT166/test_v1/manifest.json").read_text(encoding="utf-8"))["sequence_count"]
    output_root = Path(f"/results/eval/{run}/canonical")
    command = [sys.executable, "/workspace/merge_samurai_eval_shards.py"]
    for shard in range(shard_count):
        command.extend(["--shard-root", f"/results/eval/{run}/shard_{shard:02d}_of_{shard_count:02d}"])
    command.extend(["--output-root", str(output_root), "--expected-sequences", str(expected)])
    subprocess.run(command, check=True)
    results.commit()
    return json.loads((output_root / "metrics.json").read_text(encoding="utf-8"))


@app.local_entrypoint()
def main(run: str, shards: int = 4) -> None:
    calls = [evaluate.spawn(run, shards, shard) for shard in range(shards)]
    for call in calls:
        call.get()
    print(json.dumps(merge.remote(run, shards), indent=2))
