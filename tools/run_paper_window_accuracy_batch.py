from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qstr_dronedet.evaluation.window_accuracy import run_window_accuracy
from tools.build_window_accuracy_dashboard import build_dashboard


WINDOW_FORMATS = {
    "csv",
    "jsonl",
    "yolo-dir",
    "aot-json",
    "aot-gt-json",
    "xywh-file",
    "antiuav-json",
    "li-tetc-txt",
    "tvd-pkl-gt",
    "tvd-pkl-pred",
}
FRAME_FORMATS = WINDOW_FORMATS | {"image-dir"}
WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return data


def _split_labels(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        labels = [part.strip() for part in value.split(",") if part.strip()]
        return labels or None
    if isinstance(value, list):
        labels = [str(part).strip() for part in value if str(part).strip()]
        return labels or None
    raise TypeError(f"labels must be a comma string or list, got {type(value)}")


def _is_absoluteish(value: str) -> bool:
    return WINDOWS_ABS_RE.match(value) is not None or Path(value).expanduser().is_absolute()


def _resolve_path(value: Any, base_dir: Path) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"path must be a string, got {type(value)}")
    expanded = Path(value).expanduser()
    if _is_absoluteish(value):
        return expanded
    return (base_dir / expanded).resolve()


def _has_placeholder(value: Any) -> bool:
    return isinstance(value, str) and ("<" in value or ">" in value)


def _require_format(value: str, key: str) -> str:
    if value not in WINDOW_FORMATS:
        raise ValueError(f"{key}={value!r} is not one of {sorted(WINDOW_FORMATS)}")
    return value


def _rel_or_abs(path: str | Path, root: Path) -> str:
    p = Path(path)
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return str(p)


def write_batch_index(out_root: str | Path, runs: list[dict[str, Any]]) -> Path:
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    sections = []
    for run in runs:
        status = str(run.get("status", "unknown"))
        name = html.escape(str(run.get("name", "")))
        method = html.escape(str(run.get("method", "")))
        out = html.escape(str(run.get("out", "")))
        links = []
        for key, label in (
            ("plot_index", "plots"),
            ("summary", "summary"),
            ("csv", "per-frame CSV"),
            ("worst_windows_csv", "worst windows"),
            ("low_accuracy_segments_csv", "low-accuracy segments"),
        ):
            value = run.get(key)
            if value and Path(str(value)).exists():
                href = html.escape(_rel_or_abs(str(value), root))
                links.append(f'<a href="{href}">{label}</a>')
        if not links and run.get("missing"):
            links.append("missing: " + html.escape(", ".join(str(x) for x in run["missing"])))
        sections.append(
            "<tr>"
            f"<td>{method}</td><td>{name}</td><td>{html.escape(status)}</td>"
            f"<td>{html.escape(str(run.get('videos', '-')))}</td><td>{html.escape(str(run.get('frames', '-')))}</td>"
            f"<td>{' | '.join(links) if links else html.escape(out)}</td>"
            "</tr>"
        )
    index = root / "index.html"
    dashboard_link = ""
    if (root / "dashboard.html").is_file():
        dashboard_link = "<p><a href='dashboard.html'>Cross-run dashboard</a></p>"
    index.write_text(
        "<!doctype html><meta charset='utf-8'><title>Paper Window Accuracy</title>"
        "<h1>Paper Window Accuracy</h1>"
        f"{dashboard_link}"
        "<table border='1' cellspacing='0' cellpadding='6'>"
        "<thead><tr><th>Method</th><th>Run</th><th>Status</th><th>Videos</th><th>Frames</th><th>Artifacts</th></tr></thead>"
        f"<tbody>{''.join(sections)}</tbody></table>\n",
        encoding="utf-8",
    )
    return index


def _run_one(
    run: dict[str, Any],
    defaults: dict[str, Any],
    base_dir: Path,
    out_root: Path,
    skip_missing: bool,
    dry_run: bool,
) -> dict[str, Any]:
    cfg = {**defaults, **run}
    name = str(cfg.get("name") or cfg.get("method") or "paper_run")
    method = str(cfg.get("method") or name)
    if "gt" not in cfg or "pred" not in cfg:
        raise ValueError(f"{name}: each run needs 'gt' and 'pred'")
    if "fps" not in cfg:
        raise ValueError(f"{name}: each run needs fps, either in defaults or the run")

    gt_path = _resolve_path(cfg["gt"], base_dir)
    pred_path = _resolve_path(cfg["pred"], base_dir)
    out_dir = _resolve_path(cfg["out"], base_dir) if cfg.get("out") else (out_root / name)
    gt_format = _require_format(str(cfg.get("gt_format", "csv")), "gt_format")
    pred_format = _require_format(str(cfg.get("pred_format", "csv")), "pred_format")
    frame_manifest_path = _resolve_path(cfg["frame_manifest"], base_dir) if cfg.get("frame_manifest") else None
    frame_manifest_format = None
    if cfg.get("frame_manifest_format"):
        frame_manifest_format = str(cfg["frame_manifest_format"])
        if frame_manifest_format not in FRAME_FORMATS:
            raise ValueError(f"frame_manifest_format={frame_manifest_format!r} is not one of {sorted(FRAME_FORMATS)}")

    missing = []
    if _has_placeholder(cfg["gt"]) or not gt_path.exists():
        missing.append(str(gt_path))
    if _has_placeholder(cfg["pred"]) or not pred_path.exists():
        missing.append(str(pred_path))
    if frame_manifest_path is not None and (_has_placeholder(cfg.get("frame_manifest")) or not frame_manifest_path.exists()):
        missing.append(str(frame_manifest_path))

    report: dict[str, Any] = {
        "name": name,
        "method": method,
        "gt": str(gt_path),
        "pred": str(pred_path),
        "out": str(out_dir),
        "gt_format": gt_format,
        "pred_format": pred_format,
    }
    if frame_manifest_path is not None:
        report["frame_manifest"] = str(frame_manifest_path)
        report["frame_manifest_format"] = frame_manifest_format or "image-dir"
    if missing:
        report["missing"] = missing
        if skip_missing or dry_run:
            report["status"] = "skipped_missing" if skip_missing else "missing"
            return report
        raise FileNotFoundError(f"{name}: missing required path(s): {missing}")
    if dry_run:
        report["status"] = "ready"
        return report

    img_width = cfg.get("img_width")
    img_height = cfg.get("img_height")
    if (img_width is None) != (img_height is None):
        raise ValueError(f"{name}: img_width and img_height must be provided together")
    img_size = (float(img_width), float(img_height)) if img_width is not None else None

    summary = run_window_accuracy(
        gt=gt_path,
        pred=pred_path,
        out_dir=out_dir,
        fps=float(cfg["fps"]),
        gt_format=gt_format,
        pred_format=pred_format,
        window_seconds=float(cfg.get("window_seconds", 3.0)),
        iou_threshold=float(cfg.get("iou", 0.5)),
        score_threshold=float(cfg.get("score_threshold", 0.0)),
        segment_threshold=float(cfg.get("segment_threshold", 0.5)),
        gt_labels=_split_labels(cfg.get("gt_labels")),
        pred_labels=_split_labels(cfg.get("pred_labels")),
        gt_frame_offset=int(cfg.get("gt_frame_offset", 0)),
        pred_frame_offset=int(cfg.get("pred_frame_offset", 0)),
        frame_manifest=frame_manifest_path,
        frame_manifest_format=frame_manifest_format,
        frame_manifest_offset=int(cfg.get("frame_manifest_offset", 0)),
        img_size=img_size,
        sparse_centers=bool(cfg.get("sparse_centers", False)),
        extra_summary={"name": name, "method": method},
    )
    report.update(
        {
            "status": "complete",
            "frames": summary["frames"],
            "videos": summary["videos"],
            "summary": str(out_dir / "summary.json"),
            "csv": summary["csv"],
            "worst_windows_csv": summary["worst_windows_csv"],
            "low_accuracy_segments_csv": summary["low_accuracy_segments_csv"],
            "plot_index": summary["plot_index"],
        }
    )
    return report


def run_manifest(
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
    out_root: str | Path | None = None,
    skip_missing: bool = False,
    dry_run: bool = False,
    only: set[str] | None = None,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    data = _read_json(manifest)
    root = Path(base_dir).resolve() if base_dir else ROOT
    out_root_path = _resolve_path(str(out_root or data.get("out_root", "runs/window_accuracy/papers")), root)
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        raise TypeError("manifest defaults must be an object")
    runs = data.get("runs", [])
    if not isinstance(runs, list):
        raise TypeError("manifest runs must be a list")

    out_root_path.mkdir(parents=True, exist_ok=True)
    reports = []
    for raw in runs:
        if not isinstance(raw, dict):
            raise TypeError(f"run entries must be objects, got {type(raw)}")
        name = str(raw.get("name") or raw.get("method") or "")
        method = str(raw.get("method") or name)
        if only and name not in only and method not in only:
            continue
        reports.append(_run_one(raw, defaults, root, out_root_path, skip_missing=skip_missing, dry_run=dry_run))

    summary = {
        "manifest": str(manifest),
        "base_dir": str(root),
        "out_root": str(out_root_path),
        "dry_run": dry_run,
        "runs": reports,
    }
    summary["complete"] = sum(1 for run in reports if run.get("status") == "complete")
    summary["skipped_missing"] = sum(1 for run in reports if run.get("status") == "skipped_missing")
    summary["ready"] = sum(1 for run in reports if run.get("status") == "ready")
    index = write_batch_index(out_root_path, reports)
    batch_summary = out_root_path / "batch_summary.json"
    summary["index"] = str(index)
    batch_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["batch_summary"] = str(batch_summary)
    if not dry_run:
        dashboard = build_dashboard(summary)
        summary["dashboard"] = str(dashboard)
        index = write_batch_index(out_root_path, reports)
        summary["index"] = str(index)
        batch_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run +/-3s per-video accuracy curves for multiple paper repositories from a JSON manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=ROOT, help="Base directory for relative manifest paths.")
    parser.add_argument("--out-root", type=Path, default=None, help="Override manifest out_root.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip runs whose gt/pred paths do not exist.")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest paths without computing curves.")
    parser.add_argument("--only", action="append", default=[], help="Run only matching name/method. Can be repeated.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_manifest(
        manifest_path=args.manifest,
        base_dir=args.base_dir,
        out_root=args.out_root,
        skip_missing=args.skip_missing,
        dry_run=args.dry_run,
        only=set(args.only) if args.only else None,
    )
    for run in summary["runs"]:
        status = run.get("status")
        print(f"{status}: {run['name']} -> {run['out']}")
        if run.get("missing"):
            print(f"  missing: {', '.join(run['missing'])}")
        elif run.get("plot_index"):
            print(f"  plot_index: {run['plot_index']}")
    print(f"batch_summary={summary['batch_summary']}")
    if summary.get("dashboard"):
        print(f"dashboard={summary['dashboard']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
