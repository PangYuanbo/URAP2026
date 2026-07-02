from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_paper_window_accuracy_batch import _has_placeholder, _read_json, _resolve_path


REPO_BY_METHOD = {
    "YOLOMG": "papers/YOLOMG",
    "TransVisDrone": "papers/TransVisDrone",
    "ESOD": "papers/ESOD",
    "EDTC": "papers/EDTC",
    "Li_TETC_NPS": "papers/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking",
    "AICrowd_Winner_v022": "papers/AICrowd_AOT_Challenge_Winner/submission-v022/airborne-detection-starter-kit-submission-v022",
}


def _safe_rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def _has_txt(path: Path) -> bool:
    return (path.is_file() and path.suffix == ".txt") or (path.is_dir() and any(path.rglob("*.txt")))


def _has_aot_json(path: Path) -> bool:
    if path.is_file() and path.name == "result.json":
        return True
    return path.is_dir() and ((path / "result.json").is_file() or any((child / "result.json").is_file() for child in path.iterdir() if child.is_dir()))


def _has_aot_groundtruth_json(path: Path) -> bool:
    if path.is_file() and path.name == "groundtruth.json":
        return True
    return path.is_dir() and ((path / "groundtruth.json").is_file() or (path / "ImageSets" / "groundtruth.json").is_file())


def _has_antiuav_json(path: Path) -> bool:
    return path.is_dir() and ((path / "IR_label.json").is_file() or (path / "list.txt").is_file() or any(path.glob("*/IR_label.json")))


def _valid_box_path(path: Path, fmt: str) -> bool:
    if fmt == "image-dir":
        return path.is_dir() and any(p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"} for p in path.rglob("*"))
    if fmt == "yolo-dir":
        return path.is_dir() and any(path.rglob("*.txt"))
    if fmt in {"li-tetc-txt", "xywh-file"}:
        return _has_txt(path)
    if fmt == "aot-json":
        return _has_aot_json(path)
    if fmt == "aot-gt-json":
        return _has_aot_groundtruth_json(path)
    if fmt == "antiuav-json":
        return _has_antiuav_json(path)
    if fmt in {"tvd-pkl-gt", "tvd-pkl-pred"}:
        return path.is_file() and path.suffix == ".pkl"
    if fmt in {"csv", "jsonl"}:
        return path.is_file()
    return path.exists()


def _input_ok(value: Any, path: Path | None, fmt: str) -> bool:
    return bool(path and not _has_placeholder(value) and _valid_box_path(path, fmt))


def _curve_state(out_dir: Path) -> dict[str, Any]:
    summary = out_dir / "summary.json"
    csv_path = out_dir / "per_frame_window_metrics.csv"
    worst_path = out_dir / "worst_windows.csv"
    segments_path = out_dir / "low_accuracy_segments.csv"
    plot_index = out_dir / "plots" / "index.html"
    svgs = sorted((out_dir / "plots").glob("*_window_metrics.svg")) if (out_dir / "plots").is_dir() else []
    return {
        "summary": str(summary),
        "csv": str(csv_path),
        "worst_windows_csv": str(worst_path),
        "low_accuracy_segments_csv": str(segments_path),
        "plot_index": str(plot_index),
        "svg_count": len(svgs),
        "complete": summary.is_file() and csv_path.is_file() and worst_path.is_file() and segments_path.is_file() and plot_index.is_file() and bool(svgs),
    }


def _repo_state(method: str, base_dir: Path) -> dict[str, Any]:
    rel = REPO_BY_METHOD.get(method)
    if rel is None:
        return {"known": False, "present": None, "path": None}
    path = (base_dir / rel).resolve()
    git_present = (path / ".git").is_dir()
    snapshot_present = (path / ".urap_snapshot.json").is_file()
    return {
        "known": True,
        "present": git_present or snapshot_present,
        "path": str(path),
        "kind": "git" if git_present else "api_snapshot" if snapshot_present else None,
        "auth_required": method == "AICrowd_Winner_v022" and not snapshot_present,
    }


def audit_manifest(manifest_path: str | Path, base_dir: str | Path | None = None, out_root: str | Path | None = None) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    base = Path(base_dir).resolve() if base_dir else ROOT
    data = _read_json(manifest)
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        raise TypeError("manifest defaults must be an object")
    runs = data.get("runs", [])
    if not isinstance(runs, list):
        raise TypeError("manifest runs must be a list")
    out_root_path = _resolve_path(str(out_root or data.get("out_root", "runs/window_accuracy/papers")), base)

    audited = []
    for raw in runs:
        if not isinstance(raw, dict):
            raise TypeError(f"run entries must be objects, got {type(raw)}")
        cfg = {**defaults, **raw}
        name = str(cfg.get("name") or cfg.get("method") or "paper_run")
        method = str(cfg.get("method") or name)
        gt_raw = cfg.get("gt")
        pred_raw = cfg.get("pred")
        frame_manifest_raw = cfg.get("frame_manifest")
        gt_format = str(cfg.get("gt_format", "csv"))
        pred_format = str(cfg.get("pred_format", "csv"))
        frame_manifest_format = str(cfg.get("frame_manifest_format", "image-dir"))
        gt_path = _resolve_path(gt_raw, base) if isinstance(gt_raw, str) else None
        pred_path = _resolve_path(pred_raw, base) if isinstance(pred_raw, str) else None
        frame_manifest_path = _resolve_path(frame_manifest_raw, base) if isinstance(frame_manifest_raw, str) else None
        out_dir = _resolve_path(cfg["out"], base) if cfg.get("out") else (out_root_path / name)

        repo = _repo_state(method, base)
        gt_ok = _input_ok(gt_raw, gt_path, gt_format)
        pred_ok = _input_ok(pred_raw, pred_path, pred_format)
        frame_manifest_ok = True
        if frame_manifest_raw is not None:
            frame_manifest_ok = _input_ok(frame_manifest_raw, frame_manifest_path, frame_manifest_format)
        curves = _curve_state(out_dir)
        missing = []
        if repo.get("present") is False:
            missing.append("repo")
        if not gt_ok:
            missing.append("gt")
        if not pred_ok:
            missing.append("pred")
        if not frame_manifest_ok:
            missing.append("frame_manifest")

        if curves["complete"]:
            status = "complete_curves"
        elif missing:
            status = "missing_inputs"
        else:
            status = "ready_to_run"
        if status != "complete_curves" and method == "AICrowd_Winner_v022" and repo.get("present") is False:
            status = "auth_required"

        audited.append(
            {
                "name": name,
                "method": method,
                "status": status,
                "missing": missing,
                "repo": repo,
                "gt_format": gt_format,
                "pred_format": pred_format,
                "gt": {"path": str(gt_path) if gt_path else None, "exists": gt_ok, "placeholder": _has_placeholder(gt_raw)},
                "pred": {"path": str(pred_path) if pred_path else None, "exists": pred_ok, "placeholder": _has_placeholder(pred_raw)},
                "frame_manifest_format": frame_manifest_format if frame_manifest_raw is not None else None,
                "frame_manifest": {
                    "path": str(frame_manifest_path) if frame_manifest_path else None,
                    "exists": frame_manifest_ok,
                    "placeholder": _has_placeholder(frame_manifest_raw),
                } if frame_manifest_raw is not None else None,
                "out": str(out_dir),
                "curves": curves,
                "next_command": f"python3 tools/run_paper_window_accuracy_batch.py --manifest {_safe_rel(manifest, base)} --only {name}",
            }
        )

    counts: dict[str, int] = {}
    for item in audited:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {
        "manifest": str(manifest),
        "base_dir": str(base),
        "out_root": str(out_root_path),
        "counts": counts,
        "runs": audited,
    }


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Paper Window Accuracy Readiness Audit",
        "",
        f"- Manifest: `{report['manifest']}`",
        f"- Output root: `{report['out_root']}`",
        "",
        "| Method | Run | Status | Missing | Curves | Next command |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in report["runs"]:
        missing = ", ".join(run["missing"]) if run["missing"] else "-"
        curves = "yes" if run["curves"]["complete"] else f"no ({run['curves']['svg_count']} svg)"
        cmd = run["next_command"]
        lines.append(
            f"| {run['method']} | {run['name']} | {run['status']} | {missing} | {curves} | `{cmd}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit which paper window-accuracy runs are ready or already plotted.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data_templates" / "paper_window_accuracy_runs.example.json")
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="Write JSON audit report.")
    parser.add_argument("--markdown", type=Path, default=None, help="Write Markdown audit report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_manifest(args.manifest, base_dir=args.base_dir, out_root=args.out_root)
    for run in report["runs"]:
        missing = f" missing={','.join(run['missing'])}" if run["missing"] else ""
        print(f"{run['status']}: {run['method']} / {run['name']}{missing}")
    print("counts=" + json.dumps(report["counts"], sort_keys=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown:
        write_markdown_report(args.markdown, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
