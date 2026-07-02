from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NPS_VIDEOS_URL = "https://engineering.purdue.edu/~bouman/UAV_Dataset/Videos.zip"


def parse_clip_ids(value: str) -> list[int]:
    clip_ids: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Bad clip range: {part}")
            clip_ids.update(range(start, end + 1))
        else:
            clip_ids.add(int(part))
    return sorted(clip_ids)


def head_content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "URAP2026-nps-download/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        value = response.headers.get("Content-Length")
    return int(value) if value else None


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024.0
    return str(value)


def download_zip(url: str, zip_path: Path, min_free_after: int) -> dict[str, Any]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    expected = head_content_length(url)
    current_size = zip_path.stat().st_size if zip_path.is_file() else 0
    missing_bytes = max(0, (expected or 0) - current_size)
    free_bytes = shutil.disk_usage(zip_path.parent).free
    if expected is not None and current_size >= expected:
        return {"status": "already_present", "expected_bytes": expected, "path": str(zip_path)}
    if expected is not None and free_bytes - missing_bytes < min_free_after:
        raise RuntimeError(
            "Not enough free disk for NPS Videos.zip: "
            f"free={_human_bytes(free_bytes)}, needed={_human_bytes(missing_bytes)}, "
            f"min_free_after={_human_bytes(min_free_after)}"
        )

    subprocess.run(
        ["curl", "-L", "--fail", "--continue-at", "-", "--output", str(zip_path), url],
        check=True,
    )
    actual = zip_path.stat().st_size
    if expected is not None and actual != expected:
        raise RuntimeError(f"Downloaded size mismatch for {zip_path}: expected {expected}, got {actual}")
    return {"status": "downloaded", "expected_bytes": expected, "actual_bytes": actual, "path": str(zip_path)}


def _ranges(total_bytes: int, workers: int) -> list[tuple[int, int]]:
    chunk = (total_bytes + workers - 1) // workers
    ranges = []
    start = 0
    while start < total_bytes:
        end = min(total_bytes - 1, start + chunk - 1)
        ranges.append((start, end))
        start = end + 1
    return ranges


def _download_range(url: str, part_path: Path, start: int, end: int) -> dict[str, Any]:
    expected = end - start + 1
    current = part_path.stat().st_size if part_path.is_file() else 0
    if current == expected:
        return {"path": str(part_path), "status": "already_present", "bytes": current}
    if current > expected:
        part_path.unlink()
        current = 0

    request_start = start + current
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "URAP2026-nps-download/1.0",
            "Range": f"bytes={request_start}-{end}",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f"Server did not honor Range for {part_path}: HTTP {response.status}")
        mode = "ab" if current else "wb"
        with part_path.open(mode) as out:
            shutil.copyfileobj(response, out)

    actual = part_path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"Part size mismatch for {part_path}: expected {expected}, got {actual}")
    return {"path": str(part_path), "status": "downloaded", "bytes": actual}


def download_zip_ranges(url: str, zip_path: Path, min_free_after: int, workers: int) -> dict[str, Any]:
    if workers < 2:
        return download_zip(url, zip_path, min_free_after=min_free_after)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    expected = head_content_length(url)
    if expected is None:
        raise RuntimeError("Cannot use range download because Content-Length is unknown.")
    if zip_path.is_file() and zip_path.stat().st_size == expected:
        return {"status": "already_present", "expected_bytes": expected, "path": str(zip_path), "workers": workers}

    free_bytes = shutil.disk_usage(zip_path.parent).free
    missing_bytes = expected - (zip_path.stat().st_size if zip_path.is_file() else 0)
    if free_bytes - max(0, missing_bytes) < min_free_after:
        raise RuntimeError(
            "Not enough free disk for NPS Videos.zip: "
            f"free={_human_bytes(free_bytes)}, needed={_human_bytes(max(0, missing_bytes))}, "
            f"min_free_after={_human_bytes(min_free_after)}"
        )

    ranges = _ranges(expected, workers)
    part_paths = [zip_path.with_name(f"{zip_path.name}.part{idx:02d}") for idx in range(len(ranges))]

    if zip_path.is_file() and zip_path.stat().st_size < expected and not part_paths[0].exists():
        prefix_size = zip_path.stat().st_size
        first_expected = ranges[0][1] - ranges[0][0] + 1
        if prefix_size <= first_expected:
            zip_path.rename(part_paths[0])

    reports = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_range, url, part_path, start, end): idx
            for idx, (part_path, (start, end)) in enumerate(zip(part_paths, ranges))
        }
        for future in as_completed(futures):
            report = future.result()
            report["part"] = futures[future]
            reports.append(report)
            print(f"part {report['part']} {report['status']} {report['bytes']} bytes", flush=True)

    tmp_path = zip_path.with_suffix(zip_path.suffix + ".tmp")
    with tmp_path.open("wb") as out:
        for part_path in part_paths:
            with part_path.open("rb") as part:
                shutil.copyfileobj(part, out)
    actual = tmp_path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"Concatenated size mismatch for {tmp_path}: expected {expected}, got {actual}")
    tmp_path.replace(zip_path)
    for part_path in part_paths:
        part_path.unlink(missing_ok=True)
    return {
        "status": "downloaded_ranges",
        "expected_bytes": expected,
        "actual_bytes": actual,
        "path": str(zip_path),
        "workers": workers,
        "parts": sorted(reports, key=lambda item: item["part"]),
    }


def extract_clips(zip_path: Path, out_dir: Path, clip_ids: list[int]) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f"Clip_{clip_id}.mov": clip_id for clip_id in clip_ids}
    extracted: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        members_by_name = {Path(info.filename).name: info for info in archive.infolist() if not info.is_dir()}
        missing = [name for name in wanted if name not in members_by_name]
        if missing:
            raise RuntimeError(f"Videos.zip does not contain expected clips: {missing}")
        for name, clip_id in wanted.items():
            info = members_by_name[name]
            out_path = out_dir / name
            if not out_path.is_file() or out_path.stat().st_size != info.file_size:
                with archive.open(info) as source, out_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
            extracted.append(
                {
                    "clip_id": clip_id,
                    "member": info.filename,
                    "path": str(out_path),
                    "bytes": out_path.stat().st_size,
                }
            )
    return sorted(extracted, key=lambda item: item["clip_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NPS Videos.zip and extract selected Clip_*.mov files.")
    parser.add_argument("--url", default=NPS_VIDEOS_URL)
    parser.add_argument("--zip", type=Path, default=ROOT / "datasets" / "NPS" / "raw" / "Videos.zip")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "datasets" / "NPS" / "raw" / "Videos")
    parser.add_argument("--clips", default="37-40", help="Comma/range list such as '37-40,45'.")
    parser.add_argument("--min-free-after-gib", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=1, help="Use HTTP Range download with this many workers when >1.")
    parser.add_argument("--skip-download", action="store_true", help="Only extract from an existing --zip path.")
    parser.add_argument("--json", type=Path, default=ROOT / "runs" / "window_accuracy" / "papers" / "nps_video_download.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clip_ids = parse_clip_ids(args.clips)
    min_free_after = int(args.min_free_after_gib * 1024**3)
    download = {"status": "skipped", "path": str(args.zip)}
    if not args.skip_download:
        download = download_zip_ranges(args.url, args.zip, min_free_after=min_free_after, workers=args.workers)
    if not args.zip.is_file():
        raise FileNotFoundError(f"NPS Videos.zip not found: {args.zip}")
    extracted = extract_clips(args.zip, args.out_dir, clip_ids)
    report = {
        "url": args.url,
        "zip": str(args.zip),
        "out_dir": str(args.out_dir),
        "clips": clip_ids,
        "download": download,
        "extracted": extracted,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"clips={clip_ids}")
    print(f"zip={args.zip}")
    print(f"out_dir={args.out_dir}")
    print(f"json={args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
