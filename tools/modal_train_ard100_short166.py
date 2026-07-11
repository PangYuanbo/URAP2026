from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
app = modal.App("urap-ard100-samurai-short166-train-v1")
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "hydra-core==1.3.2", "iopath==0.1.10", "submitit==1.5.2", "fvcore==0.1.5.post20221221",
        "tensorboard==2.18.0", "opencv-python-headless==4.10.0.84", "tensordict==0.5.0",
    )
    .add_local_file(
        ROOT / "third_party/samurai/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_ARD100_short166_stage1.yaml",
        "/workspace/sam2.1_hiera_b+_ARD100_short166_stage1.yaml",
        copy=True,
    )
)
code = modal.Volume.from_name("urap-code-artifacts-v1")
weights = modal.Volume.from_name("urap-model-weights-v1")
dataset = modal.Volume.from_name("urap-ard100-samurai-short166-v1")
tvd = modal.Volume.from_name("urap-ard100-transvisdrone-links-v1")
results = modal.Volume.from_name("urap-ard100-samurai-short166-results-v1", create_if_missing=True)


def commit_results(stop: threading.Event) -> None:
    while not stop.wait(120):
        results.commit()
        print(json.dumps({"kind": "results_commit", "time": time.time()}), flush=True)


@app.function(
    image=image,
    gpu="L40S",
    cpu=12,
    memory=65536,
    volumes={"/code": code, "/weights": weights, "/dataset": dataset, "/tvd": tvd, "/results": results},
    timeout=24 * 60 * 60,
)
def train() -> dict:
    for volume in (code, weights, dataset, tvd, results):
        volume.reload()
    manifest = Path("/dataset/ARD100_SAMURAI_SHORT166/train_v1/manifest.json")
    if not manifest.is_file():
        raise FileNotFoundError("Short166 train dataset is not complete")
    work = Path("/tmp/sam2")
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree("/code/third_party/samurai/sam2", work, symlinks=True)
    config_target = work / "sam2/configs/sam2.1_training/sam2.1_hiera_b+_ARD100_short166_stage1.yaml"
    shutil.copy2("/workspace/sam2.1_hiera_b+_ARD100_short166_stage1.yaml", config_target)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(work), PYTHONUTF8="1", SAM2_BUILD_CUDA="0", HYDRA_FULL_ERROR="1")
    stop = threading.Event()
    committer = threading.Thread(target=commit_results, args=(stop,), daemon=True)
    committer.start()
    command = [
        sys.executable,
        str(work / "training/train.py"),
        "-c", "configs/sam2.1_training/sam2.1_hiera_b+_ARD100_short166_stage1.yaml",
        "--use-cluster", "0",
        "--num-gpus", "1",
    ]
    try:
        subprocess.run(command, cwd=work, env=env, check=True)
    finally:
        stop.set()
        committer.join(timeout=5)
        results.commit()
    checkpoint = Path("/results/finetune_base_plus_ard100_short166_stage1/checkpoints/checkpoint.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    complete = {
        "status": "completed",
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "dataset": json.loads(manifest.read_text(encoding="utf-8")),
    }
    marker = Path("/results/finetune_base_plus_ard100_short166_stage1/TRAIN_COMPLETE.json")
    marker.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    results.commit()
    print(json.dumps(complete, indent=2), flush=True)
    return complete


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=49152,
    volumes={"/code": code, "/weights": weights, "/dataset": dataset, "/tvd": tvd, "/results": results},
    timeout=2 * 60 * 60,
)
def smoke() -> dict:
    for volume in (code, weights, dataset, tvd, results):
        volume.reload()
    source_root = Path("/dataset/ARD100_SAMURAI_SHORT166/val_v1")
    names = [line.strip() for line in (source_root / "val_set.txt").read_text().splitlines() if line.strip()]
    if not names:
        raise RuntimeError("No val smoke tracklet exists")
    first_image = next((source_root / "lasot/uav" / names[0] / "img").glob("*.jpg"))
    if not first_image.is_file() or first_image.stat().st_size == 0:
        raise FileNotFoundError(f"Cross-volume frame link is unreadable: {first_image}")
    work = Path("/tmp/sam2-smoke")
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree("/code/third_party/samurai/sam2", work, symlinks=True)
    config_text = Path("/workspace/sam2.1_hiera_b+_ARD100_short166_stage1.yaml").read_text(encoding="utf-8")
    smoke_list = Path("/tmp/smoke_set.txt")
    smoke_list.write_text(names[0] + "\n", encoding="ascii")
    replacements = {
        "/dataset/ARD100_SAMURAI_SHORT166/train_v1/vos/JPEGImages": str(source_root / "vos/JPEGImages"),
        "/dataset/ARD100_SAMURAI_SHORT166/train_v1/vos/Annotations": str(source_root / "vos/Annotations"),
        "/dataset/ARD100_SAMURAI_SHORT166/train_v1/train_set.txt": str(smoke_list),
        "/dataset/ARD100_SAMURAI_SHORT166/train_v1/lasot/uav": str(source_root / "lasot/uav"),
        "phases_per_epoch: 16": "phases_per_epoch: 1",
        "save_dir: /results/finetune_base_plus_ard100_short166_stage1/checkpoints": "save_dir: /results/finetune_base_plus_ard100_short166_smoke/checkpoints",
        "save_freq: 4": "save_freq: 1",
        "experiment_log_dir: /results/finetune_base_plus_ard100_short166_stage1": "experiment_log_dir: /results/finetune_base_plus_ard100_short166_smoke",
    }
    for old, new in replacements.items():
        config_text = config_text.replace(old, new)
    config_target = work / "sam2/configs/sam2.1_training/sam2.1_hiera_b+_ARD100_short166_smoke.yaml"
    config_target.write_text(config_text, encoding="utf-8")
    env = os.environ.copy()
    env.update(PYTHONPATH=str(work), PYTHONUTF8="1", SAM2_BUILD_CUDA="0", HYDRA_FULL_ERROR="1")
    command = [sys.executable, str(work / "training/train.py"), "-c", "configs/sam2.1_training/sam2.1_hiera_b+_ARD100_short166_smoke.yaml", "--use-cluster", "0", "--num-gpus", "1"]
    subprocess.run(command, cwd=work, env=env, check=True)
    results.commit()
    checkpoint = Path("/results/finetune_base_plus_ard100_short166_smoke/checkpoints/checkpoint.pt")
    result = {"status": "completed", "sequence": names[0], "source_frames": len(list((source_root / "lasot/uav" / names[0] / "img").glob("*.jpg"))), "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size}
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(smoke_only: bool = False) -> None:
    result = smoke.remote() if smoke_only else train.remote()
    print(json.dumps(result, indent=2))
