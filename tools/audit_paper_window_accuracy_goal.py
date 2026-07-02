from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_paper_window_accuracy_readiness import audit_manifest
from tools.run_paper_window_accuracy_batch import FRAME_FORMATS, WINDOW_FORMATS
from tools.write_paper_window_accuracy_gap_report import build_gap_report


REQUIRED_METHODS = {
    "YOLOMG",
    "TransVisDrone",
    "ESOD",
    "AICrowd_Winner_v022",
    "EDTC",
    "Li_TETC_NPS",
}


def _safe_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _gate(name: str, ok: bool, evidence: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "evidence": evidence}


def _summarize_complete_run(run: dict[str, Any]) -> dict[str, Any]:
    summary = _safe_json(run["curves"]["summary"])
    return {
        "name": run["name"],
        "out": run["out"],
        "videos": summary.get("videos") if summary else None,
        "frames": summary.get("frames") if summary else None,
        "plot_index": run["curves"]["plot_index"],
        "low_accuracy_segments_csv": run["curves"]["low_accuracy_segments_csv"],
    }


def build_goal_audit(
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
    out_root: str | Path | None = None,
) -> dict[str, Any]:
    base = Path(base_dir).resolve() if base_dir else ROOT
    audit = audit_manifest(manifest_path, base_dir=base, out_root=out_root)
    gap_report = build_gap_report(manifest_path, base_dir=base, out_root=out_root)
    gaps_by_name = {gap["name"]: gap for gap in gap_report["gaps"]}

    methods: dict[str, dict[str, Any]] = {}
    for run in audit["runs"]:
        method = run["method"]
        method_info = methods.setdefault(
            method,
            {
                "repo": run["repo"],
                "runs": 0,
                "complete_runs": [],
                "missing_runs": [],
                "ready_runs": [],
                "unsupported_formats": [],
                "generation_command_count": 0,
            },
        )
        method_info["runs"] += 1
        for fmt_key in ("gt_format", "pred_format"):
            fmt = run[fmt_key]
            if fmt not in WINDOW_FORMATS:
                method_info["unsupported_formats"].append({"run": run["name"], "key": fmt_key, "format": fmt})
        if run.get("frame_manifest_format") and run["frame_manifest_format"] not in FRAME_FORMATS:
            method_info["unsupported_formats"].append(
                {"run": run["name"], "key": "frame_manifest_format", "format": run["frame_manifest_format"]}
            )

        if run["status"] == "complete_curves":
            method_info["complete_runs"].append(_summarize_complete_run(run))
        elif run["status"] == "ready_to_run":
            method_info["ready_runs"].append({"name": run["name"], "next_command": run["next_command"]})
        else:
            gap = gaps_by_name.get(run["name"], {})
            commands = gap.get("generation_commands", [])
            method_info["generation_command_count"] += len(commands)
            method_info["missing_runs"].append(
                {
                    "name": run["name"],
                    "missing": run["missing"],
                    "next_command": run["next_command"],
                    "generation_commands": commands,
                }
            )

    present_methods = set(methods)
    out_root_path = Path(audit["out_root"])
    dashboard = out_root_path / "dashboard.html"
    batch_summary = out_root_path / "batch_summary.json"
    window_module = base / "qstr_dronedet" / "evaluation" / "window_accuracy.py"

    all_runs_complete = all(run["status"] == "complete_curves" for run in audit["runs"])
    all_missing_have_commands = all(
        bool(gaps_by_name.get(run["name"], {}).get("generation_commands"))
        for run in audit["runs"]
        if run["status"] != "complete_curves"
    )
    all_repos_present = all(
        any(r["method"] == method and r["repo"].get("present") for r in audit["runs"])
        for method in REQUIRED_METHODS
    )
    all_formats_supported = not any(method["unsupported_formats"] for method in methods.values())
    all_runs_have_frame_manifest = all(run.get("frame_manifest") for run in audit["runs"])
    complete_runs_have_outputs = all(
        run["curves"]["complete"] for run in audit["runs"] if run["status"] == "complete_curves"
    )

    gates = [
        _gate(
            "required_methods_in_manifest",
            REQUIRED_METHODS.issubset(present_methods),
            f"present={sorted(present_methods)} required={sorted(REQUIRED_METHODS)}",
        ),
        _gate("paper_repositories_present", all_repos_present, "all required methods have a git checkout or API snapshot"),
        _gate("window_accuracy_module_present", window_module.is_file(), str(window_module)),
        _gate("manifest_formats_supported", all_formats_supported, "all manifest gt/pred formats are accepted by the shared scorer"),
        _gate("frame_manifests_configured", all_runs_have_frame_manifest, "each manifest run declares center-frame source for every-frame curves"),
        _gate("complete_runs_have_curve_artifacts", complete_runs_have_outputs, "summary/csv/worst/segments/html/svg are required"),
        _gate("dashboard_present", dashboard.is_file(), str(dashboard)),
        _gate("batch_summary_present", batch_summary.is_file(), str(batch_summary)),
        _gate("missing_runs_have_generation_commands", all_missing_have_commands, "every incomplete run has concrete next commands"),
        _gate("all_manifest_runs_complete", all_runs_complete, json.dumps(audit["counts"], sort_keys=True)),
    ]

    return {
        "status": "complete" if all(gate["ok"] for gate in gates) else "incomplete",
        "manifest": audit["manifest"],
        "out_root": audit["out_root"],
        "counts": audit["counts"],
        "gap_count": gap_report["gap_count"],
        "gates": gates,
        "methods": methods,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Paper Window Accuracy Goal Audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Manifest: `{report['manifest']}`",
        f"- Output root: `{report['out_root']}`",
        f"- Counts: `{json.dumps(report['counts'], sort_keys=True)}`",
        "",
        "## Gates",
        "",
        "| Gate | OK | Evidence |",
        "| --- | --- | --- |",
    ]
    for gate in report["gates"]:
        lines.append(f"| {gate['name']} | {'yes' if gate['ok'] else 'no'} | `{gate['evidence']}` |")

    lines.extend(["", "## Methods", "", "| Method | Runs | Complete | Missing | Ready | Command count |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for method, info in sorted(report["methods"].items()):
        lines.append(
            f"| {method} | {info['runs']} | {len(info['complete_runs'])} | "
            f"{len(info['missing_runs'])} | {len(info['ready_runs'])} | {info['generation_command_count']} |"
        )

    lines.extend(["", "## Incomplete Runs", ""])
    any_missing = False
    for method, info in sorted(report["methods"].items()):
        for run in info["missing_runs"]:
            any_missing = True
            lines.append(f"### {method} / {run['name']}")
            lines.append("")
            lines.append(f"- Missing: `{', '.join(run['missing']) if run['missing'] else '-'}`")
            lines.append(f"- Batch command: `{run['next_command']}`")
            lines.append(f"- Generation commands: `{len(run['generation_commands'])}`")
            lines.append("")
    if not any_missing:
        lines.append("None.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit completion against the paper-repo +/-3s curve objective.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data_templates" / "paper_window_accuracy_runs.example.json")
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=ROOT / "runs" / "window_accuracy" / "papers" / "goal_audit.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "runs" / "window_accuracy" / "papers" / "goal_audit.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_goal_audit(args.manifest, base_dir=args.base_dir, out_root=args.out_root)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.markdown, report)
    print(f"status={report['status']}")
    print("counts=" + json.dumps(report["counts"], sort_keys=True))
    print("gates=" + json.dumps({gate["name"]: gate["ok"] for gate in report["gates"]}, sort_keys=True))
    print(f"json={args.json}")
    print(f"markdown={args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
