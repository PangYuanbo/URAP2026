from __future__ import annotations

import argparse
import json
import re
import urllib.parse
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


def parse_lfs_pointer(text: str) -> dict[str, Any]:
    oid = re.search(r"^oid sha256:([0-9a-f]+)$", text, flags=re.MULTILINE)
    size = re.search(r"^size (\d+)$", text, flags=re.MULTILINE)
    return {
        "oid_sha256": oid.group(1) if oid else None,
        "size_bytes": int(size.group(1)) if size else None,
    }


def _read_url(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "URAP2026-paper-pull/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", "replace")


def build_inventory(snapshot_dir: Path) -> dict[str, Any]:
    marker = snapshot_dir / ".urap_snapshot.json"
    marker_data = json.loads(marker.read_text(encoding="utf-8"))
    ref = marker_data["ref"]
    skipped = marker_data.get("skipped_files", [])
    project_url = "https://gitlab.aicrowd.com/dmytro_poplavskiy/airborne-detection-starter-kit"

    weights = []
    for rel in skipped:
        if not str(rel).endswith((".pt", ".pth")):
            continue
        url = f"{project_url}/-/raw/{urllib.parse.quote(ref, safe='')}/{urllib.parse.quote(str(rel), safe='/')}"
        pointer = parse_lfs_pointer(_read_url(url))
        weights.append({"path": rel, **pointer, "raw_pointer_url": url})

    total_size = sum(item["size_bytes"] or 0 for item in weights)
    return {
        "snapshot_dir": str(snapshot_dir),
        "ref": ref,
        "commit": marker_data.get("commit"),
        "short_id": marker_data.get("short_id"),
        "weight_count": len(weights),
        "total_size_bytes": total_size,
        "total_size_gib": total_size / 1024**3,
        "weights": weights,
        "note": "These are Git LFS pointer records. The actual model binaries are still needed to run AICrowd winner inference.",
    }


def write_markdown(path: Path, inventory: dict[str, Any]) -> None:
    lines = [
        "# AICrowd Winner v022 Missing LFS Weights",
        "",
        f"- Snapshot: `{inventory['snapshot_dir']}`",
        f"- Ref: `{inventory['ref']}`",
        f"- Commit: `{inventory['commit']}`",
        f"- Missing model files: `{inventory['weight_count']}`",
        f"- Total size: `{inventory['total_size_gib']:.2f} GiB`",
        "",
        "| Size MiB | Path | SHA256 |",
        "| ---: | --- | --- |",
    ]
    for item in inventory["weights"]:
        size_mib = (item["size_bytes"] or 0) / 1024**2
        lines.append(f"| {size_mib:.1f} | `{item['path']}` | `{item['oid_sha256'] or '-'}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory missing Git LFS model weights for the AICrowd winner snapshot.")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--json", type=Path, default=ROOT / "runs" / "window_accuracy" / "aicrowd_lfs_weight_inventory.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "runs" / "window_accuracy" / "aicrowd_lfs_weight_inventory.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = build_inventory(args.snapshot_dir.resolve())
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    write_markdown(args.markdown, inventory)
    print(f"weights={inventory['weight_count']}")
    print(f"total_size_gib={inventory['total_size_gib']:.2f}")
    print(f"json={args.json}")
    print(f"markdown={args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
