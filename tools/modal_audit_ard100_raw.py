from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal


app = modal.App("urap-ard100-raw-audit-v1")
volume = modal.Volume.from_name("urap-ard100-raw-v1")
image = modal.Image.debian_slim(python_version="3.11")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(image=image, volumes={"/volume": volume}, timeout=60 * 60)
def audit() -> dict:
    volume.reload()
    root = Path("/volume/ARD100")
    train = sorted((root / "train_videos").glob("*.mp4"))
    test = sorted((root / "test_videos").glob("*.mp4"))
    annotations = root / "annotations.zip"
    extract_frames = root / "YOLOMG_extract_frames.py"
    required = [annotations, extract_frames]
    missing = [str(path) for path in required if not path.is_file()]
    result = {
        "complete": not missing and len(train) == 65 and len(test) == 35,
        "train_count": len(train),
        "test_count": len(test),
        "train_bytes": sum(path.stat().st_size for path in train),
        "test_bytes": sum(path.stat().st_size for path in test),
        "train_files": {path.name: path.stat().st_size for path in train},
        "test_files": {path.name: path.stat().st_size for path in test},
        "annotations_bytes": annotations.stat().st_size if annotations.exists() else None,
        "annotations_sha256": sha256(annotations) if annotations.exists() else None,
        "extract_frames_bytes": extract_frames.stat().st_size if extract_frames.exists() else None,
        "extract_frames_sha256": sha256(extract_frames) if extract_frames.exists() else None,
        "missing": missing,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(audit.remote(), indent=2))
