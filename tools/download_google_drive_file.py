from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DRIVE_UC_URL = "https://drive.google.com/uc?export=download&id={file_id}"


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return str(value)


def _parse_confirm_form(page: str) -> tuple[str, dict[str, str]]:
    action_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', page)
    if not action_match:
        raise ValueError("Could not find Google Drive download confirmation form.")
    action = html.unescape(action_match.group(1))
    fields: dict[str, str] = {}
    for name, value in re.findall(r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"', page):
        fields[html.unescape(name)] = html.unescape(value)
    if "confirm" not in fields:
        raise ValueError("Google Drive confirmation form did not include a confirm token.")
    return action, fields


def _warning_metadata(page: str) -> dict[str, str | None]:
    name_size = re.search(r'<span class="uc-name-size">.*?<a [^>]*>(?P<name>[^<]+)</a>\s*\((?P<size>[^)]+)\)', page)
    return {
        "name": html.unescape(name_size.group("name")) if name_size else None,
        "warning_size": html.unescape(name_size.group("size")) if name_size else None,
    }


def _request(opener: urllib.request.OpenerDirector, url: str) -> urllib.response.addinfourl:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 URAP2026-drive-download/1.0"})
    return opener.open(req, timeout=120)


def resolve_download(file_id: str) -> tuple[urllib.request.OpenerDirector, str, dict[str, Any]]:
    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    first_url = DRIVE_UC_URL.format(file_id=urllib.parse.quote(file_id))
    with _request(opener, first_url) as response:
        content_type = response.headers.get("Content-Type", "")
        content_length = int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None
        if "text/html" not in content_type:
            return opener, response.geturl(), {
                "file_id": file_id,
                "content_type": content_type,
                "content_length": content_length,
                "confirmed": False,
            }
        page = response.read().decode("utf-8", "replace")

    action, fields = _parse_confirm_form(page)
    meta = _warning_metadata(page)
    confirm_url = action + "?" + urllib.parse.urlencode(fields)
    meta.update(
        {
            "file_id": file_id,
            "confirmed": True,
            "confirm_url_host": urllib.parse.urlparse(confirm_url).netloc,
        }
    )
    return opener, confirm_url, meta


def download_drive_file(
    file_id: str,
    out_path: Path,
    min_free_after_bytes: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    opener, download_url, meta = resolve_download(file_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _request(opener, download_url) as response:
        content_type = response.headers.get("Content-Type", "")
        content_length = int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None
        meta["content_type"] = content_type
        meta["content_length"] = content_length
        meta["download_url_host"] = urllib.parse.urlparse(response.geturl()).netloc

        if out_path.is_file() and not overwrite:
            existing_size = out_path.stat().st_size
            if content_length is None or existing_size == content_length:
                return {
                    **meta,
                    "path": str(out_path),
                    "status": "exists",
                    "bytes": existing_size,
                }

        free = shutil.disk_usage(out_path.parent).free
        if content_length is not None and free - content_length < min_free_after_bytes:
            raise RuntimeError(
                f"Not enough free space for {out_path}: need {_human_bytes(content_length)}, "
                f"free {_human_bytes(free)}, min free after {_human_bytes(min_free_after_bytes)}"
            )

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        start = time.time()
        downloaded = 0
        next_report = 100 * 1024 * 1024
        with tmp_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"downloaded={_human_bytes(downloaded)} target={out_path}", flush=True)
                    next_report += 100 * 1024 * 1024
        if content_length is not None and downloaded != content_length:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Short download for {out_path}: expected {content_length}, got {downloaded}")
        tmp_path.replace(out_path)

    return {
        **meta,
        "path": str(out_path),
        "status": "downloaded",
        "bytes": out_path.stat().st_size,
        "elapsed_seconds": round(time.time() - start, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a public Google Drive file with large-file confirmation handling.")
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-free-after-gib", type=float, default=8.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    report = download_drive_file(
        file_id=args.file_id,
        out_path=out_path,
        min_free_after_bytes=int(args.min_free_after_gib * 1024**3),
        overwrite=args.overwrite,
    )
    if args.json:
        json_path = args.json if args.json.is_absolute() else ROOT / args.json
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"json={json_path}")
    print(f"status={report['status']} path={report['path']} bytes={report['bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
