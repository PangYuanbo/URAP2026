from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import modal


app = modal.App("urap-ard100-aot-ready-audit-v1")
repo_root = Path(__file__).resolve().parents[1]
aot_manifest = repo_root / "artifacts" / "aot_part1_local_manifest.json"
image = modal.Image.debian_slim(python_version="3.11").add_local_file(
    aot_manifest, remote_path="/opt/aot_part1_local_manifest.json", copy=True
)
volumes = {
    "/ard_train": modal.Volume.from_name("urap-ard100-yolomg-train-v1"),
    "/ard_eval": modal.Volume.from_name("urap-ard100-yolomg-eval-v1"),
    "/aot": modal.Volume.from_name("urap-aot-part1-raw-v1"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_files(root: Path) -> tuple[int, int]:
    count = 0
    total_bytes = 0
    for directory, _dirs, files in os.walk(root):
        for filename in files:
            path = Path(directory) / filename
            count += 1
            total_bytes += path.stat().st_size
    return count, total_bytes


def audit_ard_root(mount: Path, enabled: tuple[str, ...]) -> dict:
    root = mount / "ARD100_YOLOMG"
    marker = json.loads((root / "BUILD_COMPLETE.json").read_text(encoding="utf-8"))
    splits = {}
    failures = []
    for split in ("train", "val", "test"):
        names = {}
        for kind in ("images", "images2", "labels"):
            directory = root / kind / split
            files = sorted(path.name for path in directory.glob("*") if path.is_file()) if directory.exists() else []
            names[kind] = files
        images = names["images"]
        masks = names["images2"]
        labels = [Path(name).with_suffix(".jpg").name for name in names["labels"]]
        if split in enabled:
            if images != masks:
                failures.append(f"{split}: image/images2 names differ")
            if images != labels:
                failures.append(f"{split}: image/label names differ")
            if not images:
                failures.append(f"{split}: empty")
        elif images or masks or names["labels"]:
            failures.append(f"{split}: expected empty")
        list_images = [line for line in (root / f"{split}.txt").read_text(encoding="utf-8").splitlines() if line]
        list_masks = [line for line in (root / f"{split}2.txt").read_text(encoding="utf-8").splitlines() if line]
        if len(list_images) != len(images):
            failures.append(f"{split}: list/image count mismatch")
        if len(list_masks) != len(masks):
            failures.append(f"{split}: list/mask count mismatch")
        splits[split] = {kind: len(value) for kind, value in names.items()}
    return {"marker": marker, "splits": splits, "failures": failures}


@app.function(image=image, volumes=volumes, cpu=8, memory=16384, timeout=4 * 60 * 60)
def audit() -> dict:
    for volume in volumes.values():
        volume.reload()
    train = audit_ard_root(Path("/ard_train"), ("train",))
    evaluation = audit_ard_root(Path("/ard_eval"), ("val", "test"))

    manifest = json.loads(Path("/opt/aot_part1_local_manifest.json").read_text(encoding="utf-8"))
    aot_root = Path("/aot/AOT_part1")
    aot_marker = json.loads((aot_root / "SYNC_COMPLETE.json").read_text(encoding="utf-8"))
    aot_progress = json.loads((aot_root / "sync_progress.json").read_text(encoding="utf-8"))
    image_count, image_bytes = count_files(aot_root / "Images")
    groundtruth = aot_root / "ImageSets/groundtruth.json"
    aot_failures = []
    if image_count != manifest["image_count"]:
        aot_failures.append(f"image_count {image_count} != {manifest['image_count']}")
    if image_bytes != manifest["total_bytes"]:
        aot_failures.append(f"image_bytes {image_bytes} != {manifest['total_bytes']}")
    groundtruth_hash = sha256(groundtruth)
    if groundtruth_hash != manifest["groundtruth"]["sha256"]:
        aot_failures.append("groundtruth SHA256 mismatch")
    if aot_marker.get("complete") is not True or aot_progress.get("status") != "complete":
        aot_failures.append("completion markers invalid")

    result = {
        "complete": not train["failures"] and not evaluation["failures"] and not aot_failures,
        "ard100_train": train,
        "ard100_eval": evaluation,
        "aot": {
            "marker": aot_marker,
            "progress": aot_progress,
            "image_count": image_count,
            "image_bytes": image_bytes,
            "groundtruth_bytes": groundtruth.stat().st_size,
            "groundtruth_sha256": groundtruth_hash,
            "failures": aot_failures,
        },
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(audit.remote(), indent=2))
