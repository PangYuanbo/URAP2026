from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import modal


app = modal.App("urap-dataset-build-v1")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("opencv-python-headless==4.10.0.84", "tqdm==4.67.1")
    .add_local_file(
        "tools/prepare_transvisdrone_nps.py",
        remote_path="/opt/urap/tools/prepare_transvisdrone_nps.py",
        copy=True,
    )
)

nps_raw = modal.Volume.from_name("nps-dataset")
nps_formatted = modal.Volume.from_name("urap-nps-formatted-v1")

SPLITS = {
    "train": list(range(1, 37)),
    "val": list(range(37, 41)),
    "test": list(range(41, 51)),
}


def load_prepare_module():
    path = "/opt/urap/tools/prepare_transvisdrone_nps.py"
    spec = importlib.util.spec_from_file_location("prepare_transvisdrone_nps", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def annotation_path(annos_dir: Path, clip_id: int) -> Path:
    candidates = [
        annos_dir / f"Clip_{clip_id:03d}.txt",
        annos_dir / f"Clip_{clip_id}_gt.txt",
        annos_dir / f"Clip_{clip_id:03d}_gt.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No annotation found for Clip_{clip_id}: {candidates}")


@app.function(
    image=image,
    volumes={"/input": nps_raw, "/output": nps_formatted},
    cpu=8,
    memory=32768,
    timeout=86400,
)
def build_nps_split(split: str, png_compression: int = 3) -> dict:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    module = load_prepare_module()
    videos_dir = Path("/input/Videos")
    annos_dir = Path("/input/annotations/Video_Annotation")
    out_root = Path("/output/NPS")
    frames_dir = out_root / "AllFrames" / split
    labels_dir = out_root / "NPSvisdroneStyle" / split / "labels"
    video_meta_dir = out_root / "Videos" / split
    progress_path = out_root / f"build_progress_{split}.json"
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    video_meta_dir.mkdir(parents=True, exist_ok=True)

    clip_lengths: dict[int, int] = {}
    for index, clip_id in enumerate(SPLITS[split], start=1):
        video_path = videos_dir / f"Clip_{clip_id}.mov"
        frame_count, (image_width, image_height) = module.extract_frames_png(
            video_path=str(video_path),
            out_dir=str(frames_dir),
            clip_id=clip_id,
            png_compression=png_compression,
        )
        module.write_yolo_labels_for_clip(
            anno_path=str(annotation_path(annos_dir, clip_id)),
            labels_dir=str(labels_dir),
            clip_id=clip_id,
            img_w=image_width,
            img_h=image_height,
        )
        clip_lengths[clip_id] = int(frame_count)
        with (video_meta_dir / "video_length_dict.pkl").open("wb") as handle:
            pickle.dump(dict(clip_lengths), handle)
        progress = {
            "split": split,
            "done": index,
            "total": len(SPLITS[split]),
            "last_clip": f"Clip_{clip_id}",
            "frames": sum(clip_lengths.values()),
            "pid": os.getpid(),
        }
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        nps_formatted.commit()
        print(json.dumps(progress), flush=True)

    result = {
        "split": split,
        "clips": len(clip_lengths),
        "frames": sum(clip_lengths.values()),
        "complete": True,
    }
    (out_root / f"build_complete_{split}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    nps_formatted.commit()
    return result


@app.local_entrypoint()
def main(splits: str = "train,val,test", png_compression: int = 3) -> None:
    calls = []
    for split in [value.strip() for value in splits.split(",") if value.strip()]:
        call = build_nps_split.spawn(split, png_compression)
        calls.append((split, call))
    call_records = [{"split": split, "call_id": call.object_id} for split, call in calls]
    print(json.dumps({"app": app.name, "calls": call_records}, indent=2), flush=True)
    results = []
    for split, call in calls:
        results.append({"split": split, "result": call.get()})
    print(json.dumps({"complete": True, "results": results}, indent=2), flush=True)
