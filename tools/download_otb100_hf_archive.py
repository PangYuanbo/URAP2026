from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://huggingface.co/huangyuyang11/otb100/resolve/main/OTB100-sample.zip?download=true"
REPO = Path(__file__).resolve().parents[1]
TARGET = Path(r"D:\URAP_local_datasets\OTB100")
ARCHIVE = Path(r"D:\URAP_local_datasets\OTB100-sample.zip")
PROGRESS = REPO / "artifacts" / "detached_otb100_hf_download" / "progress.json"
METADATA = REPO / "data_templates" / "otb100_sequences.json"


def report(stage: str, done: int, total: int, **extra) -> None:
    payload = {"stage": stage, "done": done, "total": total, **extra}
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def download() -> None:
    temporary = ARCHIVE.with_suffix(".zip.part")
    request = urllib.request.Request(URL, headers={"User-Agent": "URAP-OTB100-Evaluator/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        with temporary.open("wb") as target:
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                done += len(chunk)
                report("download", done, total, archive=str(temporary))
    temporary.replace(ARCHIVE)


def normalized_members(source: zipfile.ZipFile):
    names = [name for name in source.namelist() if name and not name.startswith("__MACOSX/")]
    common = names[0].split("/", 1)[0] if names and all("/" in name and name.split("/", 1)[0] == names[0].split("/", 1)[0] for name in names) else None
    for name in names:
        relative = name.split("/", 1)[1] if common else name
        if relative:
            yield name, relative


def extract() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as source:
        members = list(normalized_members(source))
        for index, (name, relative) in enumerate(members, start=1):
            destination = TARGET / relative
            if name.endswith("/"):
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(name) as input_file, destination.open("wb") as output_file:
                    shutil.copyfileobj(input_file, output_file, length=4 * 1024 * 1024)
            if index == 1 or index % 1000 == 0 or index == len(members):
                report("extract", index, len(members), archive=str(ARCHIVE), target=str(TARGET))


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
    report("done", len(ready), len(metadata), target=str(TARGET), archive=str(ARCHIVE), missing=missing)
    if missing:
        raise RuntimeError(f"OTB100 archive incomplete: {len(missing)} sequences missing")


def main() -> int:
    if not ARCHIVE.is_file() or ARCHIVE.stat().st_size < 250_000_000:
        download()
    extract()
    verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
