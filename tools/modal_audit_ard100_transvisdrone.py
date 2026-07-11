from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import modal


app = modal.App("urap-ard100-transvisdrone-audit-v1")
image = modal.Image.debian_slim(python_version="3.11")
volumes = {
    "/data_tvd": modal.Volume.from_name("urap-ard100-transvisdrone-links-v1"),
    "/data_train": modal.Volume.from_name("urap-ard100-yolomg-train-v1"),
    "/data_eval": modal.Volume.from_name("urap-ard100-yolomg-eval-v1"),
}


@app.function(image=image, volumes=volumes, cpu=8, memory=16384, timeout=4 * 60 * 60)
def audit() -> dict:
    for volume in volumes.values():
        volume.reload()
    root = Path("/data_tvd/ARD100_TVD")
    marker = json.loads((root / "BUILD_COMPLETE.json").read_text(encoding="utf-8"))
    expected = {item["split"]: item for item in marker["results"]}
    failures = []
    splits = {}
    for split in ("train", "val", "test"):
        image_dir = root / "AllFrames" / split
        label_dir = root / "Annotations" / split
        images = sorted(path.name for path in image_dir.iterdir() if path.is_symlink())
        labels = sorted(path.name for path in label_dir.iterdir() if path.is_symlink())
        broken_images = [name for name in images if not (image_dir / name).exists()]
        broken_labels = [name for name in labels if not (label_dir / name).exists()]
        normalized_labels = [Path(name).with_suffix(".jpg").name for name in labels]
        if images != normalized_labels:
            failures.append(f"{split}: image/label names differ")
        if broken_images or broken_labels:
            failures.append(f"{split}: broken links images={len(broken_images)} labels={len(broken_labels)}")
        if len(images) != expected[split]["images"] or len(labels) != expected[split]["labels"]:
            failures.append(f"{split}: marker count mismatch")
        pkl_path = root / "Videos" / split / "video_length_dict.pkl"
        with pkl_path.open("rb") as handle:
            lengths = pickle.load(handle)
        if len(lengths) != expected[split]["videos"]:
            failures.append(f"{split}: video count mismatch")
        if any(not isinstance(key, int) or not isinstance(value, int) or value <= 0 for key, value in lengths.items()):
            failures.append(f"{split}: invalid video length dictionary")
        splits[split] = {
            "images": len(images),
            "labels": len(labels),
            "broken_images": len(broken_images),
            "broken_labels": len(broken_labels),
            "videos": len(lengths),
        }
    yaml_text = (root / "ARD100_TVD.yaml").read_text(encoding="utf-8")
    required_mounts = ["/data_tvd/ARD100_TVD"]
    for mount in required_mounts:
        if mount not in yaml_text:
            failures.append(f"YAML missing mount path: {mount}")
    result = {"complete": not failures, "marker": marker, "splits": splits, "failures": failures}
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(audit.remote(), indent=2))
