from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = Path(r"D:\URAP_local_datasets\OTB100")
PARTS = Path(r"D:\URAP_local_datasets\OTB100_release_parts")
PROGRESS = REPO / "artifacts" / "detached_otb100_full_download" / "progress.json"
METADATA = REPO / "data_templates" / "otb100_sequences.json"
URLS = [
    "https://github.com/yuyma/OTB100_dataset_download/releases/download/OTB100_dataset_download1/OTB100_1.zip",
    "https://github.com/yuyma/OTB100_dataset_download/releases/download/OTB100_dataset_download2/OTB100_2.zip",
]
EXPECTED = [1288298972, 1527581196]


def report(stage: str, done: int, total: int, **extra) -> None:
    payload = {"stage": stage, "done": done, "total": total, **extra}
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def download(url: str, destination: Path, offset: int, grand_total: int) -> None:
    if destination.is_file() and destination.stat().st_size >= EXPECTED[int(destination.stem[-1]) - 1]:
        report("download", offset + destination.stat().st_size, grand_total, file=str(destination), reused=True)
        return
    temporary = destination.with_suffix(".zip.part")
    request = urllib.request.Request(url, headers={"User-Agent": "URAP-OTB100-Evaluator/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            done += len(chunk)
            report("download", offset + done, grand_total, file=str(temporary), file_done=done, file_total=total)
    temporary.replace(destination)


def relative_name(name: str) -> str:
    cleaned = name.replace("\\", "/").lstrip("/")
    while cleaned.startswith("OTB100/") or cleaned.startswith("OTB100_1/") or cleaned.startswith("OTB100_2/"):
        cleaned = cleaned.split("/", 1)[1]
    return cleaned


def extract(archives: list[Path]) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    member_total = sum(len(zipfile.ZipFile(path).infolist()) for path in archives)
    done = 0
    for archive in archives:
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                done += 1
                relative = relative_name(member.filename)
                if not relative or relative.startswith("__MACOSX/"):
                    continue
                destination = TARGET / relative
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as input_file, destination.open("wb") as output_file:
                        shutil.copyfileobj(input_file, output_file, length=8 * 1024 * 1024)
                if done == 1 or done % 2000 == 0 or done == member_total:
                    report("extract", done, member_total, archive=str(archive), target=str(TARGET))


def verify() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    ready, missing = [], []
    for name, attributes in metadata.items():
        base = attributes.get("base", name)
        groundtruth = attributes.get("groundtruth", "groundtruth_rect.txt")
        sequence = TARGET / base
        if sequence.is_dir() and (sequence / "img").is_dir() and (sequence / groundtruth).is_file():
            ready.append(name)
        else:
            missing.append(name)
    report("done", len(ready), len(metadata), target=str(TARGET), missing=missing)
    if missing:
        raise RuntimeError(f"OTB100 incomplete: {len(missing)} sequences missing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
        return 0
    PARTS.mkdir(parents=True, exist_ok=True)
    archives = [PARTS / "OTB100_1.zip", PARTS / "OTB100_2.zip"]
    grand_total = sum(EXPECTED)
    offset = 0
    for url, archive, expected in zip(URLS, archives, EXPECTED):
        download(url, archive, offset, grand_total)
        offset += expected
    extract(archives)
    verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
