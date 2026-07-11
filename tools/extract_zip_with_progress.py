import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path


def write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    args = parser.parse_args()

    archive = args.archive.resolve()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        total_files = sum(not member.is_dir() for member in members)
        total_bytes = sum(member.file_size for member in members if not member.is_dir())
        completed_files = 0
        completed_bytes = 0
        for member in members:
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as input_stream, target.open("wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
            completed_files += 1
            completed_bytes += member.file_size
            write_progress(
                args.progress,
                {
                    "status": "running",
                    "archive": str(archive),
                    "destination": str(destination),
                    "done": completed_files,
                    "total": total_files,
                    "bytes_done": completed_bytes,
                    "bytes_total": total_bytes,
                    "last_completed_unit": member.filename,
                    "last_output_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_seconds": time.time() - started,
                },
            )

    write_progress(
        args.progress,
        {
            "status": "complete",
            "archive": str(archive),
            "destination": str(destination),
            "done": total_files,
            "total": total_files,
            "bytes_done": total_bytes,
            "bytes_total": total_bytes,
            "last_completed_unit": members[-1].filename if members else "",
            "last_output_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": time.time() - started,
        },
    )


if __name__ == "__main__":
    main()
