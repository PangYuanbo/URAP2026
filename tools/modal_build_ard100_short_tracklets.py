from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal


app = modal.App("urap-ard100-samurai-short166-build-v1")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("numpy==1.26.4", "opencv-python-headless==4.10.0.84", "pillow==11.0.0")
    .add_local_dir(Path(__file__).resolve().parent, remote_path="/workspace/tools", copy=True)
)
raw = modal.Volume.from_name("urap-ard100-raw-v1")
tvd = modal.Volume.from_name("urap-ard100-transvisdrone-links-v1")
output = modal.Volume.from_name("urap-ard100-samurai-short166-v1", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/raw": raw, "/tvd": tvd, "/output": output},
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
)
def build(split: str, max_videos: int | None = None) -> dict:
    if split not in {"train", "val", "test"}:
        raise ValueError(split)
    for volume in (raw, tvd, output):
        volume.reload()
    command = [
        sys.executable,
        "/workspace/tools/build_ard100_short_tracklets.py",
        "--source-root", "/tvd/ARD100_TVD",
        "--raw-video-root", "/raw/ARD100",
        "--annotations-zip", "/raw/ARD100/annotations.zip",
        "--split", split,
        "--output-root", f"/output/ARD100_SAMURAI_SHORT166/{split}_v1",
        "--max-gap", "2",
        "--max-frames", "166",
        "--min-visible-frames", "8",
        "--min-visibility", "0.5",
        "--image-mode", "symlink",
        "--resume",
    ]
    if max_videos is not None:
        command.extend(["--max-videos", str(max_videos)])
    subprocess.run(command, check=True)
    output.commit()
    manifest_path = Path(f"/output/ARD100_SAMURAI_SHORT166/{split}_v1/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


@app.local_entrypoint()
def main(split: str = "all", max_videos: int = 0) -> None:
    selected = ("train", "val", "test") if split == "all" else (split,)
    limit = max_videos or None
    calls = [build.spawn(item, limit) for item in selected]
    results = [call.get() for call in calls]
    print(json.dumps(results, indent=2))
