from __future__ import annotations

import json
import tarfile
from pathlib import Path

import modal


app = modal.App("urap-ard100-samurai-short166-pack-v1")
image = modal.Image.debian_slim(python_version="3.11")
dataset = modal.Volume.from_name("urap-ard100-samurai-short166-v1")
tvd = modal.Volume.from_name("urap-ard100-transvisdrone-links-v1")
packs = modal.Volume.from_name("urap-ard100-samurai-short166-packs-v1", create_if_missing=True)
EXPECTED = {"train": 55, "val": 10, "test": 35}


@app.function(image=image, volumes={"/dataset": dataset, "/tvd": tvd, "/packs": packs}, cpu=8, memory=32768, timeout=24 * 60 * 60)
def pack(split: str) -> dict:
    if split not in EXPECTED:
        raise ValueError(split)
    for volume in (dataset, tvd, packs):
        volume.reload()
    root = Path(f"/dataset/ARD100_SAMURAI_SHORT166/{split}_v1")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest["source_video_count"]) != EXPECTED[split]:
        raise RuntimeError(f"Incomplete {split} dataset: {manifest['source_video_count']}/{EXPECTED[split]}")
    output = Path(f"/packs/ARD100_SAMURAI_SHORT166_{split}_v1.tar")
    def include(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if "/vos/JPEGImages" in tarinfo.name:
            return None
        return tarinfo
    with tarfile.open(output, "w", dereference=True) as archive:
        archive.add(root, arcname=f"{split}_v1", recursive=True, filter=include)
    result = {"status": "completed", "split": split, "archive": str(output), "archive_bytes": output.stat().st_size, "manifest": manifest}
    (Path("/packs") / f"PACK_COMPLETE_{split}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    packs.commit()
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(split: str = "all") -> None:
    selected = tuple(EXPECTED) if split == "all" else (split,)
    calls = [pack.spawn(item) for item in selected]
    print(json.dumps([call.get() for call in calls], indent=2))
