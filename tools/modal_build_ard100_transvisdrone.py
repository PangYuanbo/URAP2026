from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import modal


app = modal.App("urap-ard100-transvisdrone-build-v1")
image = modal.Image.debian_slim(python_version="3.11")
train_volume = modal.Volume.from_name("urap-ard100-yolomg-train-v1")
eval_volume = modal.Volume.from_name("urap-ard100-yolomg-eval-v1")
output_volume = modal.Volume.from_name("urap-ard100-transvisdrone-links-v1")


def link_file(runtime_source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination) and destination.is_symlink() and destination.readlink() == runtime_source:
        return
    destination.unlink(missing_ok=True)
    destination.symlink_to(runtime_source)


def parse_name(path: Path) -> tuple[int, int]:
    video_name, frame_text = path.stem.split("_", maxsplit=1)
    return int(video_name.removeprefix("phantom")), int(frame_text)


def convert_split(source_root: Path, runtime_root: Path, output_root: Path, split: str) -> dict:
    source_images = source_root / "images" / split
    source_labels = source_root / "labels" / split
    output_images = output_root / "AllFrames" / split
    output_labels = output_root / "Annotations" / split
    output_videos = output_root / "Videos" / split
    for directory in (output_images, output_labels, output_videos):
        directory.mkdir(parents=True, exist_ok=True)
    images = sorted(source_images.glob("*.jpg"), key=lambda path: path.name)
    lengths = {}
    for index, image in enumerate(images, start=1):
        clip_id, frame_id = parse_name(image)
        destination_stem = f"Clip_{clip_id}_{frame_id:05d}"
        link_file(runtime_root / "images" / split / image.name, output_images / f"{destination_stem}.jpg")
        label = source_labels / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        link_file(runtime_root / "labels" / split / label.name, output_labels / f"{destination_stem}.txt")
        lengths[clip_id] = max(lengths.get(clip_id, 0), frame_id)
        if index % 20000 == 0:
            print(json.dumps({"split": split, "done": index, "total": len(images), "last": image.name}), flush=True)
            output_volume.commit()
    with (output_videos / "video_length_dict.pkl").open("wb") as handle:
        pickle.dump(lengths, handle)
    result = {"split": split, "images": len(images), "labels": len(images), "videos": len(lengths)}
    (output_root / f"BUILD_COMPLETE_{split}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    output_volume.commit()
    return result


@app.function(
    image=image,
    volumes={"/train": train_volume, "/eval": eval_volume, "/output": output_volume},
    cpu=8,
    memory=16384,
    timeout=24 * 60 * 60,
)
def build() -> dict:
    train_volume.reload()
    eval_volume.reload()
    output_volume.reload()
    output_root = Path("/output/ARD100_TVD")
    results = [
        convert_split(Path("/train/ARD100_YOLOMG"), Path("/data_train/ARD100_YOLOMG"), output_root, "train"),
        convert_split(Path("/eval/ARD100_YOLOMG"), Path("/data_eval/ARD100_YOLOMG"), output_root, "val"),
        convert_split(Path("/eval/ARD100_YOLOMG"), Path("/data_eval/ARD100_YOLOMG"), output_root, "test"),
    ]
    mounted_root = Path("/data_tvd/ARD100_TVD")
    yaml_text = "\n".join(
        [
            f"path: {mounted_root}",
            f"train: {mounted_root / 'AllFrames/train'}",
            f"val: {mounted_root / 'AllFrames/val'}",
            f"test: {mounted_root / 'AllFrames/test'}",
            f"inference: {mounted_root / 'AllFrames/test'}",
            f"annotation_path: {mounted_root / 'Annotations'}",
            f"annotation_train: {mounted_root / 'Annotations/train'}",
            f"annotation_val: {mounted_root / 'Annotations/val'}",
            f"annotation_test: {mounted_root / 'Annotations/test'}",
            f"video_root_path: {mounted_root / 'Videos'}",
            f"video_root_path_train: {mounted_root / 'Videos/train'}",
            f"video_root_path_val: {mounted_root / 'Videos/val'}",
            f"video_root_path_test: {mounted_root / 'Videos/test'}",
            f"video_root_path_inference: {mounted_root / 'Videos/test'}",
            "nc: 1",
            "names: ['drone']",
            "",
        ]
    )
    (output_root / "ARD100_TVD.yaml").write_text(yaml_text, encoding="utf-8")
    summary = {"complete": True, "mount": "/data", "results": results}
    (output_root / "BUILD_COMPLETE.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "MOUNT_README.txt").write_text(
        "Mount urap-ard100-transvisdrone-links-v1 at /data_tvd, "
        "urap-ard100-yolomg-train-v1 at /data_train, and "
        "urap-ard100-yolomg-eval-v1 at /data_eval.\n",
        encoding="utf-8",
    )
    output_volume.commit()
    return summary


@app.local_entrypoint()
def main() -> None:
    call = build.spawn()
    print(json.dumps({"call_id": call.object_id}, indent=2), flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)
