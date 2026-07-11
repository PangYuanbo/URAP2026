#!/usr/bin/env python
"""Audit the public ATA release for reproducibility readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qstr_dronedet.ata_benchmark import audit_ata_release


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "third_party" / "ATA")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_ata_release(args.dataset_root)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["ready_for_tracking"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
