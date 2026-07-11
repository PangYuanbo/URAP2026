from __future__ import annotations

import json
import os
from pathlib import Path

import modal


app = modal.App("inspect-urap-assets")

VOLUMES = {
    "/mnt/nps-dataset": modal.Volume.from_name("nps-dataset"),
    "/mnt/aot-dataset": modal.Volume.from_name("aot-dataset"),
    "/mnt/tvd-aot-weights": modal.Volume.from_name("tvd-aot-weights"),
    "/mnt/vatd-artifacts": modal.Volume.from_name("vatd-artifacts"),
}


@app.function(volumes=VOLUMES, timeout=600)
def inspect() -> dict:
    roots = [Path(path) for path in VOLUMES]
    weight_suffixes = {".pt", ".pth", ".ckpt", ".onnx"}
    marker_suffixes = {".yaml", ".yml", ".pkl", ".csv", ".json"}
    result: dict[str, object] = {"volumes": {}, "weights": [], "markers": []}

    for root in roots:
        file_count = 0
        total_bytes = 0
        top_level = []
        if root.exists():
            for child in sorted(root.iterdir()):
                top_level.append({"name": child.name, "type": "dir" if child.is_dir() else "file"})
            for directory, _, filenames in os.walk(root):
                for filename in filenames:
                    path = Path(directory) / filename
                    try:
                        size = path.stat().st_size
                    except OSError:
                        continue
                    file_count += 1
                    total_bytes += size
                    suffix = path.suffix.lower()
                    record = {"path": str(path), "bytes": size}
                    if suffix in weight_suffixes:
                        result["weights"].append(record)
                    if suffix in marker_suffixes:
                        result["markers"].append(record)
        result["volumes"][root.name] = {
            "file_count": file_count,
            "total_bytes": total_bytes,
            "top_level": top_level,
        }

    result["weights"] = sorted(result["weights"], key=lambda item: item["path"])
    result["markers"] = sorted(result["markers"], key=lambda item: item["path"])
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(inspect.remote(), indent=2))
