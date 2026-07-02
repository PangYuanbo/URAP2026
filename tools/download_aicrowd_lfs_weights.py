from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT
    / "papers"
    / "AICrowd_AOT_Challenge_Winner"
    / "submission-v022"
    / "airborne-detection-starter-kit-submission-v022"
)
DEFAULT_INVENTORY = ROOT / "runs" / "window_accuracy" / "aicrowd_lfs_weight_inventory.json"
LFS_BATCH_URL = "https://gitlab.aicrowd.com/dmytro_poplavskiy/airborne-detection-starter-kit.git/info/lfs/objects/batch"


class LfsAuthError(RuntimeError):
    pass


def basic_auth_header(username: str, token: str) -> str:
    raw = f"{username}:{token}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def lfs_batch_payload(weights: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        {
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": item["oid_sha256"], "size": item["size_bytes"]} for item in weights],
        }
    ).encode("utf-8")


def _request_json(url: str, payload: bytes, auth_header: str | None) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json",
        "User-Agent": "git-lfs/3.0 URAP2026",
    }
    if auth_header:
        headers["Authorization"] = auth_header
    req = urllib.request.Request(url, data=payload, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # type: ignore[attr-defined]
        body = exc.read().decode("utf-8", "replace")
        if exc.code in {401, 403}:
            raise LfsAuthError(body.strip() or str(exc)) from exc
        raise RuntimeError(f"LFS batch failed with HTTP {exc.code}: {body[:500]}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "git-lfs/3.0 URAP2026"})
    with urllib.request.urlopen(req, timeout=120) as response, tmp_path.open("wb") as f:
        shutil.copyfileobj(response, f, length=1024 * 1024)
    tmp_path.replace(out_path)


def load_inventory(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def download_weights(
    inventory_path: Path,
    snapshot_dir: Path,
    token: str | None,
    username: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    inventory = load_inventory(inventory_path)
    weights = inventory.get("weights", [])
    auth_header = basic_auth_header(username, token) if token else None
    batch = _request_json(LFS_BATCH_URL, lfs_batch_payload(weights), auth_header)

    results = []
    by_oid = {item.get("oid"): item for item in batch.get("objects", [])}
    for item in weights:
        rel = item["path"]
        target = snapshot_dir / rel
        expected_sha = item["oid_sha256"]
        if target.is_file() and _sha256(target) == expected_sha:
            results.append({"path": rel, "status": "already_verified"})
            continue
        batch_item = by_oid.get(expected_sha)
        href = ((batch_item or {}).get("actions") or {}).get("download", {}).get("href")
        if not href:
            error = (batch_item or {}).get("error")
            results.append({"path": rel, "status": "missing_download_url", "error": error})
            continue
        if dry_run:
            results.append({"path": rel, "status": "would_download", "size_bytes": item["size_bytes"]})
            continue
        _download(href, target)
        got = _sha256(target)
        if got != expected_sha:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"{target}: sha256 mismatch, expected {expected_sha}, got {got}")
        results.append({"path": rel, "status": "downloaded", "size_bytes": item["size_bytes"]})

    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    return {"inventory": str(inventory_path), "snapshot_dir": str(snapshot_dir), "counts": counts, "weights": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify AICrowd winner Git LFS model weights.")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--token", default=os.environ.get("AICROWD_GITLAB_TOKEN") or os.environ.get("GITLAB_TOKEN"))
    parser.add_argument("--username", default=os.environ.get("AICROWD_GITLAB_USERNAME") or "oauth2")
    parser.add_argument("--dry-run", action="store_true", help="Resolve LFS URLs but do not download files.")
    parser.add_argument("--json", type=Path, default=ROOT / "runs" / "window_accuracy" / "aicrowd_lfs_weight_download_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = download_weights(args.inventory.resolve(), args.snapshot_dir.resolve(), args.token, args.username, dry_run=args.dry_run)
    except LfsAuthError as exc:
        report = {
            "inventory": str(args.inventory.resolve()),
            "snapshot_dir": str(args.snapshot_dir.resolve()),
            "counts": {"auth_required": 1},
            "error": str(exc),
            "hint": "Set AICROWD_GITLAB_TOKEN or GITLAB_TOKEN with access to the AIcrowd GitLab project, then rerun this script.",
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("auth_required: AICrowd Git LFS weights")
        print(f"json={args.json}")
        return 2

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("counts=" + json.dumps(report["counts"], sort_keys=True))
    print(f"json={args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
