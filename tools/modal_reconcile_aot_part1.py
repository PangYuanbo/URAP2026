from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("urap-aot-part1-reconcile-v1")
repo_root = Path(__file__).resolve().parents[1]
local_manifest = repo_root / "artifacts" / "aot_part1_local_manifest.json"
image = modal.Image.debian_slim(python_version="3.11").add_local_file(
    local_manifest, remote_path="/opt/aot_part1_local_manifest.json", copy=True
)
volume = modal.Volume.from_name("urap-aot-part1-raw-v1")


@app.function(image=image, volumes={"/volume": volume}, timeout=4 * 60 * 60)
def reconcile(apply: bool = False) -> dict:
    volume.reload()
    manifest = json.loads(Path("/opt/aot_part1_local_manifest.json").read_text(encoding="utf-8"))
    expected = {}
    for flight in manifest["flights"]:
        for filename, size in flight["files"].items():
            expected[f"{flight['flight_id']}/{filename}"] = int(size)

    images_root = Path("/volume/AOT_part1/Images")
    actual = {}
    for flight_dir in images_root.iterdir():
        if not flight_dir.is_dir():
            actual[flight_dir.name] = flight_dir.stat().st_size
            continue
        for path in flight_dir.iterdir():
            if path.is_file():
                actual[f"{flight_dir.name}/{path.name}"] = path.stat().st_size

    unexpected = sorted(set(actual).difference(expected))
    missing = sorted(set(expected).difference(actual))
    mismatched = sorted(
        name for name in set(expected).intersection(actual) if expected[name] != actual[name]
    )
    removed = []
    if apply:
        if missing or mismatched:
            raise RuntimeError("Refusing cleanup while expected files are missing or size-mismatched")
        for name in unexpected:
            path = images_root / name
            path.unlink()
            removed.append(name)
        volume.commit()

    result = {
        "apply": apply,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "unexpected_count": len(unexpected),
        "unexpected_bytes": sum(actual[name] for name in unexpected),
        "unexpected": unexpected,
        "missing_count": len(missing),
        "missing": missing[:100],
        "mismatched_count": len(mismatched),
        "mismatched": mismatched[:100],
        "removed_count": len(removed),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(apply: bool = False) -> None:
    print(json.dumps(reconcile.remote(apply), indent=2))
