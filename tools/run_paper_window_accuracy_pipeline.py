from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_paper_window_accuracy_readiness import audit_manifest, write_markdown_report
from tools.build_paper_window_accuracy_smoke import build_smoke_fixture, write_top_index
from tools.pull_paper_repos import sync_all
from tools.run_paper_window_accuracy_batch import run_manifest


def _status_counts(items: list[dict[str, Any]], key: str = "status") -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get(key, "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _ready_names(audit: dict[str, Any]) -> set[str]:
    return {run["name"] for run in audit["runs"] if run.get("status") == "ready_to_run"}


def run_pipeline(
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
    out_root: str | Path | None = None,
    skip_pull: bool = False,
    include_auth_required: bool = False,
    audit_only: bool = False,
    smoke: bool = False,
    smoke_out_root: str | Path | None = None,
) -> dict[str, Any]:
    base = Path(base_dir).resolve() if base_dir else ROOT
    manifest = Path(manifest_path).resolve()
    report: dict[str, Any] = {
        "manifest": str(manifest),
        "base_dir": str(base),
        "out_root_override": str(Path(out_root).resolve()) if out_root else None,
        "steps": {},
    }

    if skip_pull:
        report["steps"]["pull"] = {"skipped": True}
    else:
        pull = sync_all(root=base, include_auth_required=include_auth_required)
        pull["counts"] = _status_counts(pull["repos"])
        report["steps"]["pull"] = pull

    before = audit_manifest(manifest, base_dir=base, out_root=out_root)
    report["steps"]["audit_before"] = before
    ready = _ready_names(before)

    if audit_only or not ready:
        report["steps"]["run_ready"] = {"skipped": True, "ready": sorted(ready)}
    else:
        run_ready = run_manifest(manifest_path=manifest, base_dir=base, out_root=out_root, only=ready)
        report["steps"]["run_ready"] = run_ready

    after = audit_manifest(manifest, base_dir=base, out_root=out_root)
    report["steps"]["audit_after"] = after
    report["final_counts"] = after["counts"]

    if smoke:
        smoke_root = Path(smoke_out_root).resolve() if smoke_out_root else (base / "runs" / "window_accuracy" / "smoke")
        smoke_manifest = build_smoke_fixture(smoke_root)
        smoke_summary = run_manifest(manifest_path=smoke_manifest, base_dir=smoke_root)
        index = write_top_index(smoke_root, smoke_summary)
        report["steps"]["smoke"] = {
            "manifest": str(smoke_manifest),
            "summary": smoke_summary,
            "index": str(index),
        }
    else:
        report["steps"]["smoke"] = {"skipped": True}

    return report


def _write_pipeline_markdown(path: Path, report: dict[str, Any]) -> None:
    after = report["steps"]["audit_after"]
    lines = [
        "# Paper Window Accuracy Pipeline Report",
        "",
        f"- Manifest: `{report['manifest']}`",
        f"- Base dir: `{report['base_dir']}`",
        f"- Final counts: `{json.dumps(after['counts'], sort_keys=True)}`",
        "",
        "## Final Readiness",
        "",
        "| Method | Run | Status | Missing | Curves |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in after["runs"]:
        missing = ", ".join(run["missing"]) if run["missing"] else "-"
        curves = "yes" if run["curves"]["complete"] else f"no ({run['curves']['svg_count']} svg)"
        lines.append(f"| {run['method']} | {run['name']} | {run['status']} | {missing} | {curves} |")
    if not report["steps"].get("smoke", {}).get("skipped"):
        lines.extend(["", f"- Smoke index: `{report['steps']['smoke']['index']}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync paper repos, audit readiness, and render +/-3s curves for ready paper runs."
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "data_templates" / "paper_window_accuracy_runs.example.json")
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--include-auth-required", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="Do not run ready curve jobs.")
    parser.add_argument("--smoke", action="store_true", help="Also rebuild the end-to-end smoke curves.")
    parser.add_argument("--smoke-out-root", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=ROOT / "runs" / "window_accuracy" / "papers" / "pipeline_report.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "runs" / "window_accuracy" / "papers" / "pipeline_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_pipeline(
        manifest_path=args.manifest,
        base_dir=args.base_dir,
        out_root=args.out_root,
        skip_pull=args.skip_pull,
        include_auth_required=args.include_auth_required,
        audit_only=args.audit_only,
        smoke=args.smoke,
        smoke_out_root=args.smoke_out_root,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_pipeline_markdown(args.markdown, report)
    write_markdown_report(args.markdown.with_name("readiness_audit.md"), report["steps"]["audit_after"])

    after = report["steps"]["audit_after"]
    print("final_counts=" + json.dumps(after["counts"], sort_keys=True))
    print(f"json={args.json}")
    print(f"markdown={args.markdown}")
    if not report["steps"].get("smoke", {}).get("skipped"):
        print(f"smoke_index={report['steps']['smoke']['index']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
