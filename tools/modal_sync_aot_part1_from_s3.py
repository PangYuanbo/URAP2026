from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal


app = modal.App("urap-aot-part1-s3-sync-v1")
repo_root = Path(__file__).resolve().parents[1]
local_manifest = repo_root / "artifacts" / "aot_part1_local_manifest.json"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boto3==1.39.4")
    .add_local_file(local_manifest, remote_path="/opt/aot_part1_local_manifest.json", copy=True)
)
output_volume = modal.Volume.from_name("urap-aot-part1-raw-v1", create_if_missing=True)
BUCKET = "airborne-obj-detection-challenge-training"
PREFIX = "part1/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(
    image=image,
    volumes={"/output": output_volume},
    cpu=16,
    memory=32768,
    timeout=24 * 60 * 60,
)
def sync(workers: int = 64, commit_every: int = 5000) -> dict:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    output_volume.reload()
    manifest = json.loads(Path("/opt/aot_part1_local_manifest.json").read_text(encoding="utf-8"))
    expected = {}
    for flight in manifest["flights"]:
        for filename, size in flight["files"].items():
            expected[f"{PREFIX}Images/{flight['flight_id']}/{filename}"] = int(size)
    expected[f"{PREFIX}ImageSets/groundtruth.json"] = int(manifest["groundtruth"]["bytes"])

    client = boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            max_pool_connections=max(workers * 2, 128),
            retries={"max_attempts": 10, "mode": "adaptive"},
        ),
    )
    paginator = client.get_paginator("list_objects_v2")
    remote = {}
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for item in page.get("Contents", []):
            if item["Key"] in expected:
                remote[item["Key"]] = int(item["Size"])
    missing = sorted(set(expected).difference(remote))
    unexpected_sizes = sorted(
        key for key, size in remote.items() if expected.get(key) != size
    )
    if missing or unexpected_sizes or len(remote) != len(expected):
        raise RuntimeError(
            json.dumps(
                {
                    "missing_count": len(missing),
                    "missing_sample": missing[:20],
                    "size_mismatch_count": len(unexpected_sizes),
                    "size_mismatch_sample": unexpected_sizes[:20],
                    "remote_count": len(remote),
                    "expected_count": len(expected),
                },
                indent=2,
            )
        )

    root = Path("/output/AOT_part1")
    progress_path = root / "sync_progress.json"
    marker_dir = root / "completed_flights"
    marker_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    counters = {"done": 0, "skipped": 0, "bytes": 0, "started": time.time()}
    keys = sorted(expected)

    def download(key: str) -> tuple[str, int, bool]:
        relative = key.removeprefix(PREFIX)
        destination = root / relative
        size = expected[key]
        if destination.is_file() and destination.stat().st_size == size:
            return key, size, True
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.unlink(missing_ok=True)
        client.download_file(BUCKET, key, str(temporary))
        if temporary.stat().st_size != size:
            raise RuntimeError(f"Downloaded size mismatch: {key}")
        os.replace(temporary, destination)
        return key, size, False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download, key): key for key in keys}
        for future in as_completed(futures):
            key, size, skipped = future.result()
            with lock:
                counters["done"] += 1
                counters["bytes"] += size
                counters["skipped"] += int(skipped)
                done = counters["done"]
                if done % 100 == 0 or done == len(keys):
                    elapsed = max(time.time() - counters["started"], 1.0)
                    progress = {
                        "status": "running",
                        "done": done,
                        "total": len(keys),
                        "bytes_done": counters["bytes"],
                        "bytes_total": sum(expected.values()),
                        "skipped": counters["skipped"],
                        "last_key": key,
                        "files_per_second": done / elapsed,
                        "mib_per_second": counters["bytes"] / elapsed / 1024 / 1024,
                        "pid": os.getpid(),
                        "updated": time.time(),
                    }
                    progress_path.parent.mkdir(parents=True, exist_ok=True)
                    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
                    print(json.dumps(progress), flush=True)
                if done % commit_every == 0:
                    output_volume.commit()

    groundtruth = root / "ImageSets" / "groundtruth.json"
    actual_groundtruth_hash = sha256(groundtruth)
    if actual_groundtruth_hash != manifest["groundtruth"]["sha256"]:
        raise RuntimeError("groundtruth.json SHA256 mismatch")
    result = {
        "complete": True,
        "flight_count": manifest["flight_count"],
        "image_count": manifest["image_count"],
        "file_count": len(keys),
        "total_bytes": sum(expected.values()),
        "groundtruth_sha256": actual_groundtruth_hash,
        "skipped": counters["skipped"],
    }
    (root / "SYNC_COMPLETE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    progress_path.write_text(
        json.dumps({"status": "complete", "done": len(keys), "total": len(keys), **result}, indent=2),
        encoding="utf-8",
    )
    output_volume.commit()
    return result


@app.local_entrypoint()
def main(workers: int = 64, commit_every: int = 5000) -> None:
    call = sync.spawn(workers, commit_every)
    print(json.dumps({"call_id": call.object_id}, indent=2), flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)
