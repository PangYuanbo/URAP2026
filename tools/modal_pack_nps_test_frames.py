from __future__ import annotations

import json
import hashlib
import shutil
import tarfile
from pathlib import Path

import modal


app = modal.App("urap-nps-test-frames-pack-v1")
image = modal.Image.debian_slim(python_version="3.11")
source = modal.Volume.from_name("urap-nps-formatted-v1")
packs = modal.Volume.from_name("urap-nps-packs-v1", create_if_missing=True)


@app.function(image=image, volumes={"/source": source, "/packs": packs}, cpu=8, memory=32768, timeout=6 * 60 * 60)
def pack() -> dict[str, object]:
    source.reload()
    packs.reload()
    root = Path("/source/NPS/AllFrames/test")
    files = sorted(root.glob("*.png"))
    if len(files) < 12350:
        raise RuntimeError(f"Incomplete NPS test frames: {len(files)}")
    local_output = Path("/tmp/NPS_AllFrames_test.tar")
    with tarfile.open(local_output, "w") as archive:
        archive.add(root, arcname="test", recursive=True)
    chunk_bytes = 512 * 1024 * 1024
    parts = []
    with local_output.open("rb") as source_file:
        part_index = 0
        while True:
            payload = source_file.read(chunk_bytes)
            if not payload:
                break
            name = f"NPS_AllFrames_test.part{part_index:03d}"
            output = Path("/packs") / name
            output.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            part = {"name": name, "bytes": len(payload), "sha256": digest}
            parts.append(part)
            packs.commit()
            print(json.dumps({"kind": "pack_part", "done": len(parts), "part": part}), flush=True)
            part_index += 1
    result = {"status": "completed", "files": len(files), "archive_bytes": local_output.stat().st_size, "parts": parts}
    Path("/packs/NPS_AllFrames_test.parts.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    packs.commit()
    print(json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(pack.remote(), indent=2))
