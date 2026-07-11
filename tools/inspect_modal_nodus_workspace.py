from __future__ import annotations

import json
import os
from pathlib import Path

import modal


app = modal.App("inspect-urap-nodus-workspaces")

VOLUMES = {
    "/mnt/stateless": modal.Volume.from_name("nodus-stateless-lab-workspaces-hackmit25"),
    "/mnt/snapshot-home": modal.Volume.from_name("nodus-codex-snapshot-home-ybpang1"),
    "/mnt/snapshot-workspace": modal.Volume.from_name("nodus-codex-snapshot-workspace-ybpang1"),
    "/mnt/ws-b5": modal.Volume.from_name("nodus-ws-b5eeda12-7769-4557-90d4-f251533bba58"),
    "/mnt/ws-d8": modal.Volume.from_name("nodus-ws-d8da835a-9e2c-4cf7-ab76-33b223aafe5f"),
}

KEYWORDS = ("urap", "nps", "ard100", "yolomg", "transvisdrone", "best.pt", "last.pt")
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx"}


@app.function(volumes=VOLUMES, timeout=1200)
def inspect() -> dict:
    result: dict[str, object] = {"volumes": {}, "matches": [], "weights": []}
    for mount in VOLUMES:
        root = Path(mount)
        file_count = 0
        directory_count = 0
        top_level = []
        if root.exists():
            top_level = [child.name for child in sorted(root.iterdir())]
            for directory, directories, filenames in os.walk(root):
                directory_count += len(directories)
                for filename in filenames:
                    file_count += 1
                    path = Path(directory) / filename
                    lowered = str(path).lower()
                    if path.suffix.lower() in WEIGHT_SUFFIXES:
                        try:
                            size = path.stat().st_size
                        except OSError:
                            size = None
                        result["weights"].append({"path": str(path), "bytes": size})
                    if any(keyword in lowered for keyword in KEYWORDS):
                        result["matches"].append(str(path))
        result["volumes"][root.name] = {
            "file_count": file_count,
            "directory_count": directory_count,
            "top_level": top_level[:100],
        }
    result["weights"] = sorted(result["weights"], key=lambda item: item["path"])
    result["matches"] = sorted(set(result["matches"]))[:5000]
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(inspect.remote(), indent=2))
