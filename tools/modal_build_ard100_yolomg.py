from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal


app = modal.App("urap-ard100-yolomg-build-v1")
repo_root = Path(__file__).resolve().parents[1]
yolomg_root = repo_root / "URAP-UAV-to-UAV-Detection-and-Tracking" / "papers" / "YOLOMG"
builder_script = yolomg_root / "tools_data" / "build_ard100_yolomg_dataset.py"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("opencv-python-headless==4.10.0.84", "numpy==2.1.3", "pyyaml==6.0.2")
    .add_local_file(builder_script, remote_path="/opt/yolomg/tools_data/build_ard100_yolomg_dataset.py", copy=True)
)
raw_volume = modal.Volume.from_name("urap-ard100-raw-v1")
train_volume = modal.Volume.from_name("urap-ard100-yolomg-train-v1", create_if_missing=True)
eval_volume = modal.Volume.from_name("urap-ard100-yolomg-eval-v1", create_if_missing=True)


def rewrite_runtime_paths(root: Path, runtime_root: str) -> None:
    build_root = str(root)
    for list_path in root.glob("*.txt"):
        text = list_path.read_text(encoding="utf-8")
        list_path.write_text(text.replace(build_root, runtime_root), encoding="utf-8")
    yaml_path = root / "ARD100_mask32_local.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "train: /data_train/ARD100_YOLOMG/train.txt",
                "train2: /data_train/ARD100_YOLOMG/train2.txt",
                "val: /data_eval/ARD100_YOLOMG/val.txt",
                "val2: /data_eval/ARD100_YOLOMG/val2.txt",
                "test: /data_eval/ARD100_YOLOMG/test.txt",
                "test2: /data_eval/ARD100_YOLOMG/test2.txt",
                "nc: 1",
                "names: ['UAV']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "MOUNT_README.txt").write_text(
        "Mount urap-ard100-yolomg-train-v1 at /data_train and "
        "urap-ard100-yolomg-eval-v1 at /data_eval.\n",
        encoding="utf-8",
    )


def run_builder(include_splits: str, output_root: Path, runtime_root: str) -> dict:
    script = Path("/opt/yolomg/tools_data/build_ard100_yolomg_dataset.py")
    command = [
        "python",
        str(script),
        "--annotations-zip",
        "/raw/ARD100/annotations.zip",
        "--train-videos-dir",
        "/raw/ARD100/train_videos",
        "--test-videos-dir",
        "/raw/ARD100/test_videos",
        "--output-root",
        str(output_root),
        "--annotations-work-dir",
        "/tmp/ard100_annotations",
        "--include-splits",
        include_splits,
        "--progress-every",
        "500",
    ]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, check=True)
    rewrite_runtime_paths(output_root, runtime_root)
    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = {
            kind: len(list((output_root / kind / split).glob("*")))
            for kind in ("images", "images2", "labels")
        }
    marker = output_root / "BUILD_COMPLETE.json"
    result = {"complete": True, "include_splits": include_splits, "counts": counts}
    marker.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


@app.function(
    image=image,
    volumes={"/raw": raw_volume, "/output": train_volume},
    cpu=16,
    memory=32768,
    timeout=24 * 60 * 60,
)
def build_train() -> dict:
    raw_volume.reload()
    result = run_builder("train", Path("/output/ARD100_YOLOMG"), "/data_train/ARD100_YOLOMG")
    train_volume.commit()
    return result


@app.function(
    image=image,
    volumes={"/raw": raw_volume, "/output": eval_volume},
    cpu=16,
    memory=32768,
    timeout=24 * 60 * 60,
)
def build_eval() -> dict:
    raw_volume.reload()
    result = run_builder("val,test", Path("/output/ARD100_YOLOMG"), "/data_eval/ARD100_YOLOMG")
    eval_volume.commit()
    return result


@app.local_entrypoint()
def main() -> None:
    calls = {"train": build_train.spawn(), "eval": build_eval.spawn()}
    print(json.dumps({name: call.object_id for name, call in calls.items()}, indent=2), flush=True)
    results = {name: call.get() for name, call in calls.items()}
    print(json.dumps({"complete": True, "results": results}, indent=2), flush=True)
