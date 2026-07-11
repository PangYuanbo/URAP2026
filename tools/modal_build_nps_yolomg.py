from __future__ import annotations

import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

app = modal.App("urap-nps-yolomg-build-v1")
image = (modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("opencv-python-headless==4.10.0.84", "numpy==2.1.3"))
source_volume = modal.Volume.from_name("urap-nps-formatted-v1")
output_volume = modal.Volume.from_name("urap-nps-yolomg-v1")
SPLITS = {"train": 36, "val": 4, "test": 10}
FRAME_RE = re.compile(r"^(Clip_\d+)_(\d+)$")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def frame_parts(frame: Path) -> tuple[str, int]:
    match = FRAME_RE.match(frame.stem)
    if match is None:
        raise ValueError(f"Unexpected NPS frame name: {frame.name}")
    return match.group(1), int(match.group(2))


def write_config(root: Path) -> None:
    mounted_root = Path("/data/NPS_YOLOMG")
    lines = []
    for split in SPLITS:
        lines.append(f"{split}: {mounted_root / f'{split}.txt'}")
        lines.append(f"{split}2: {mounted_root / f'{split}2.txt'}")
    lines.extend(["nc: 1", "names: ['UAV']", ""])
    (root / "NPS_yolomg.yaml").write_text("\n".join(lines), encoding="utf-8")
    (root / "MOUNT_README.txt").write_text(
        "Mount urap-nps-yolomg-v1 at /data when using NPS_yolomg.yaml.\n", encoding="utf-8")


def write_frame(source_frame: Path, source_labels: Path, output_images: Path,
                output_images2: Path, output_labels: Path, previous_image,
                threshold: int):
    import cv2
    import numpy as np

    frame_clip, frame_number = frame_parts(source_frame)
    destination_image = output_images / source_frame.name
    link_or_copy(source_frame, destination_image)
    current_image = cv2.imread(str(source_frame), cv2.IMREAD_COLOR)
    if current_image is None:
        raise RuntimeError(f"Unreadable source frame: {source_frame}")
    difference = cv2.absdiff(current_image, previous_image) if previous_image is not None else np.zeros_like(current_image)
    if threshold > 0:
        gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        difference = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    destination_image2 = output_images2 / source_frame.name
    if not cv2.imwrite(str(destination_image2), difference):
        raise RuntimeError(f"Failed writing motion mask: {destination_image2}")
    source_label = source_labels / f"{frame_clip}_{frame_number - 1:05d}.txt"
    destination_label = output_labels / f"{source_frame.stem}.txt"
    if source_label.exists():
        link_or_copy(source_label, destination_label)
    else:
        destination_label.write_text("", encoding="utf-8")
    return current_image


@app.function(
    image=image,
    volumes={"/source": source_volume, "/output": output_volume},
    cpu=4,
    memory=16384,
    timeout=86400,
)
def build_train_clip(clip_id: int, threshold: int = 16) -> dict:
    source_volume.reload()
    if not Path("/source/NPS/build_complete_train.json").exists():
        raise RuntimeError("Formatted NPS train split is not complete")
    clip_name = f"Clip_{clip_id}"
    source_frames = Path("/source/NPS/AllFrames/train")
    source_labels = Path("/source/NPS/NPSvisdroneStyle/train/labels")
    root = Path("/output/NPS_YOLOMG")
    output_images = root / "images/train"
    output_images2 = root / "images2/train"
    output_labels = root / "labels/train"
    for directory in (output_images, output_images2, output_labels):
        directory.mkdir(parents=True, exist_ok=True)
    frames = sorted(source_frames.glob(f"{clip_name}_*.png"), key=lambda path: path.name)
    if not frames:
        raise RuntimeError(f"No frames for {clip_name}")
    previous_image = None
    for index, source_frame in enumerate(frames, start=1):
        previous_image = write_frame(source_frame, source_labels, output_images, output_images2,
                                     output_labels, previous_image, threshold)
        if index % 500 == 0:
            output_volume.commit()
            print(json.dumps({"clip": clip_name, "done": index, "total": len(frames)}), flush=True)
    marker = root / "train_clips" / f"{clip_name}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"clip": clip_name, "frames": len(frames), "complete": True}, indent=2), encoding="utf-8")
    output_volume.commit()
    return {"clip": clip_name, "frames": len(frames)}


def build_train_clip_local(clip_id: int, threshold: int = 16) -> dict:
    clip_name = f"Clip_{clip_id}"
    source_frames = Path("/source/NPS/AllFrames/train")
    source_labels = Path("/source/NPS/NPSvisdroneStyle/train/labels")
    root = Path("/output/NPS_YOLOMG")
    marker = root / "train_clips" / f"{clip_name}.json"
    frames = sorted(source_frames.glob(f"{clip_name}_*.png"), key=lambda path: path.name)
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("complete") and int(saved.get("frames", -1)) == len(frames):
            return saved
    output_images = root / "images/train"
    output_images2 = root / "images2/train"
    output_labels = root / "labels/train"
    previous_image = None
    for source_frame in frames:
        previous_image = write_frame(source_frame, source_labels, output_images, output_images2,
                                     output_labels, previous_image, threshold)
    result = {"clip": clip_name, "frames": len(frames), "complete": True}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


@app.function(
    image=image,
    volumes={"/source": source_volume, "/output": output_volume},
    cpu=16,
    memory=65536,
    timeout=86400,
)
def build_train_parallel(threshold: int = 16, workers: int = 8) -> dict:
    source_volume.reload()
    if not Path("/source/NPS/build_complete_train.json").exists():
        raise RuntimeError("Formatted NPS train split is not complete")
    root = Path("/output/NPS_YOLOMG")
    for directory in (root / "images/train", root / "images2/train", root / "labels/train", root / "train_clips"):
        directory.mkdir(parents=True, exist_ok=True)
    pending = []
    for clip_id in range(1, 37):
        marker = root / "train_clips" / f"Clip_{clip_id}.json"
        if not marker.exists():
            pending.append(clip_id)
    completed = 36 - len(pending)
    for offset in range(0, len(pending), workers):
        batch = pending[offset:offset + workers]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(build_train_clip_local, clip_id, threshold): clip_id for clip_id in batch}
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(json.dumps({"done": completed, "total": 36, **result}), flush=True)
        progress = {"split": "train", "clips_done": completed, "clips_total": 36,
                    "last_batch": batch, "done": 0, "total": 51951}
        (root / "build_progress_train.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
        output_volume.commit()
    return finalize_train.local()


@app.function(image=image, volumes={"/output": output_volume}, cpu=2, memory=8192, timeout=7200)
def finalize_train() -> dict:
    root = Path("/output/NPS_YOLOMG")
    frames = sorted((root / "images/train").glob("*.png"), key=lambda path: path.name)
    masks = sorted((root / "images2/train").glob("*.png"), key=lambda path: path.name)
    labels = sorted((root / "labels/train").glob("*.txt"), key=lambda path: path.name)
    if len(frames) != 51951 or len(masks) != len(frames) or len(labels) != len(frames):
        raise RuntimeError(f"Train count mismatch images={len(frames)} images2={len(masks)} labels={len(labels)}")
    (root / "train.txt").write_text("".join(f"/data/NPS_YOLOMG/images/train/{path.name}\n" for path in frames), encoding="utf-8")
    (root / "train2.txt").write_text("".join(f"/data/NPS_YOLOMG/images2/train/{path.name}\n" for path in masks), encoding="utf-8")
    result = {"split": "train", "frames": len(frames), "clips": 36, "complete": True}
    (root / "build_progress_train.json").write_text(json.dumps({"split": "train", "done": len(frames), "total": len(frames), "clips_done": 36, "clips_total": 36, "last_frame": frames[-1].name}, indent=2), encoding="utf-8")
    (root / "build_complete_train.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_config(root)
    output_volume.commit()
    return result


@app.function(
    image=image,
    volumes={"/source": source_volume, "/output": output_volume},
    cpu=8,
    memory=32768,
    timeout=86400,
)
def build_split(split: str, threshold: int = 16) -> dict:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    completion = Path(f"/source/NPS/build_complete_{split}.json")
    for _ in range(8640):
        source_volume.reload()
        if completion.exists():
            break
        time.sleep(10)
    if not completion.exists():
        raise TimeoutError(f"Formatted NPS split did not become ready: {split}")

    source_frames = Path("/source/NPS/AllFrames") / split
    source_labels = Path("/source/NPS/NPSvisdroneStyle") / split / "labels"
    root = Path("/output/NPS_YOLOMG")
    output_images = root / "images" / split
    output_images2 = root / "images2" / split
    output_labels = root / "labels" / split
    for directory in (output_images, output_images2, output_labels):
        directory.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np

    frames = sorted(source_frames.glob("*.png"), key=lambda path: path.name)
    if not frames:
        raise RuntimeError(f"No source frames found: {source_frames}")
    list_path = root / f"{split}.txt"
    list2_path = root / f"{split}2.txt"
    progress_path = root / f"build_progress_{split}.json"
    previous_clip = None
    previous_image = None
    current_clip = None
    completed_clips = 0
    empty_labels = 0
    image_lines = []
    image2_lines = []

    for index, source_frame in enumerate(frames, start=1):
        frame_clip, frame_number = frame_parts(source_frame)
        if current_clip != frame_clip:
            current_clip = frame_clip
            completed_clips += 1
        destination_image = output_images / source_frame.name
        link_or_copy(source_frame, destination_image)
        current_image = cv2.imread(str(source_frame), cv2.IMREAD_COLOR)
        if current_image is None:
            raise RuntimeError(f"Unreadable source frame: {source_frame}")
        if previous_clip == frame_clip and previous_image is not None:
            difference = cv2.absdiff(current_image, previous_image)
        else:
            difference = np.zeros_like(current_image, dtype=np.uint8)
        if threshold > 0:
            gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            difference = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        destination_image2 = output_images2 / source_frame.name
        if not cv2.imwrite(str(destination_image2), difference):
            raise RuntimeError(f"Failed writing motion mask: {destination_image2}")

        source_label = source_labels / f"{frame_clip}_{frame_number - 1:05d}.txt"
        destination_label = output_labels / f"{source_frame.stem}.txt"
        if source_label.exists():
            link_or_copy(source_label, destination_label)
            if source_label.stat().st_size == 0:
                empty_labels += 1
        else:
            destination_label.write_text("", encoding="utf-8")
            empty_labels += 1
        image_lines.append(f"/data/NPS_YOLOMG/images/{split}/{source_frame.name}")
        image2_lines.append(f"/data/NPS_YOLOMG/images2/{split}/{source_frame.name}")
        previous_clip = frame_clip
        previous_image = current_image

        if index % 500 == 0 or index == len(frames):
            list_path.write_text("\n".join(image_lines) + "\n", encoding="utf-8")
            list2_path.write_text("\n".join(image2_lines) + "\n", encoding="utf-8")
            progress = {
                "split": split, "done": index, "total": len(frames),
                "clips_done": completed_clips, "clips_total": SPLITS[split],
                "last_clip": frame_clip, "last_frame": source_frame.name,
                "empty_labels": empty_labels, "pid": os.getpid(),
            }
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            write_config(root)
            output_volume.commit()
            print(json.dumps(progress), flush=True)

    result = {"split": split, "frames": len(frames), "clips": completed_clips,
              "empty_labels": empty_labels, "complete": True}
    (root / f"build_complete_{split}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    write_config(root)
    output_volume.commit()
    return result


@app.local_entrypoint()
def main(splits: str = "train,val,test", threshold: int = 16) -> None:
    calls = []
    for split in [value.strip() for value in splits.split(",") if value.strip()]:
        if split == "train":
            calls.append((split, build_train_parallel.spawn(threshold)))
        else:
            calls.append((split, build_split.spawn(split, threshold)))
    print(json.dumps({"app": app.name, "calls": [
        {"split": split, "call_id": call.object_id} for split, call in calls]}, indent=2), flush=True)
    results = [{"split": split, "result": call.get()} for split, call in calls]
    print(json.dumps({"complete": True, "results": results}, indent=2), flush=True)
