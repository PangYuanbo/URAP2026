from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _rel(path: str | Path, root: Path) -> str:
    p = Path(path)
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return str(p)


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _read_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object")
    return data


def _run_summary_path(run: dict[str, Any]) -> Path | None:
    value = run.get("summary")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_file() else None


def _plot_path(run_out: Path, video: str) -> Path:
    return run_out / "plots" / f"{video}_window_metrics.svg"


def collect_dashboard_rows(batch_summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    video_rows: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    gallery_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for run in batch_summary.get("runs", []):
        if not isinstance(run, dict):
            continue
        if run.get("status") != "complete":
            missing_rows.append(
                {
                    "method": str(run.get("method") or ""),
                    "run": str(run.get("name") or ""),
                    "status": str(run.get("status") or ""),
                    "missing": run.get("missing") or [],
                    "gt": run.get("gt"),
                    "gt_format": run.get("gt_format"),
                    "pred": run.get("pred"),
                    "pred_format": run.get("pred_format"),
                    "out": run.get("out"),
                }
            )
            continue
        summary_path = _run_summary_path(run)
        if summary_path is None:
            continue
        summary = _read_json(summary_path)
        method = str(run.get("method") or summary.get("method") or "")
        name = str(run.get("name") or summary.get("name") or "")
        out_dir = Path(str(run.get("out") or summary_path.parent))
        by_video = summary.get("by_video", {})
        if isinstance(by_video, dict):
            for video, stats in by_video.items():
                if not isinstance(stats, dict):
                    continue
                plot = _plot_path(out_dir, str(video))
                row = {
                    "method": method,
                    "run": name,
                    "video": str(video),
                    "frames": stats.get("frames"),
                    "mean_accuracy": stats.get("mean_accuracy"),
                    "min_accuracy": stats.get("min_accuracy"),
                    "mean_recall": stats.get("mean_recall"),
                    "min_recall": stats.get("min_recall"),
                    "worst_frame": stats.get("worst_frame_by_accuracy"),
                    "plot": str(plot) if plot.is_file() else None,
                    "plot_index": run.get("plot_index"),
                }
                video_rows.append(row)
                if row["plot"]:
                    gallery_rows.append(row)

        worst_path = Path(str(run.get("worst_windows_csv") or out_dir / "worst_windows.csv"))
        if worst_path.is_file():
            with worst_path.open("r", encoding="utf-8-sig", newline="") as f:
                for raw in csv.DictReader(f):
                    raw.update({"method": method, "run": name, "plot_index": run.get("plot_index")})
                    plot = _plot_path(out_dir, str(raw.get("video") or ""))
                    raw["plot"] = str(plot) if plot.is_file() else None
                    worst_rows.append(raw)

        segments_path = Path(str(run.get("low_accuracy_segments_csv") or summary.get("low_accuracy_segments_csv") or out_dir / "low_accuracy_segments.csv"))
        if segments_path.is_file():
            with segments_path.open("r", encoding="utf-8-sig", newline="") as f:
                for raw in csv.DictReader(f):
                    raw.update({"method": method, "run": name, "plot_index": run.get("plot_index")})
                    plot = _plot_path(out_dir, str(raw.get("video") or ""))
                    raw["plot"] = str(plot) if plot.is_file() else None
                    segment_rows.append(raw)

    video_rows.sort(key=lambda row: (float(row["mean_accuracy"]) if row.get("mean_accuracy") is not None else 9.0, row["method"], row["run"], row["video"]))
    worst_rows.sort(key=lambda row: (float(row.get("accuracy") or 9.0), row.get("method", ""), row.get("run", ""), row.get("video", "")))
    segment_rows.sort(key=lambda row: (float(row.get("min_accuracy") or 9.0), float(row.get("mean_accuracy") or 9.0), row.get("method", ""), row.get("run", ""), row.get("video", "")))
    gallery_rows.sort(key=lambda row: (row["method"], row["run"], row["video"]))
    missing_rows.sort(key=lambda row: (row["method"], row["run"]))
    return video_rows, worst_rows, segment_rows, gallery_rows, missing_rows


def build_dashboard(batch_summary: dict[str, Any], out_path: str | Path | None = None, worst_limit: int = 60, segment_limit: int = 80) -> Path:
    out_root = Path(str(batch_summary["out_root"])).resolve()
    out = Path(out_path).resolve() if out_path else out_root / "dashboard.html"
    video_rows, worst_rows, segment_rows, gallery_rows, missing_rows = collect_dashboard_rows(batch_summary)

    methods = sorted({row["method"] for row in video_rows} | {row["method"] for row in missing_rows})
    complete_run_count = len({(row["method"], row["run"]) for row in video_rows})
    video_count = len(video_rows)
    gap_report = out_root / "gap_report.md"
    gap_link = f"<p><a href=\"{html.escape(_rel(gap_report, out_root))}\">Missing-input gap report</a></p>" if gap_report.is_file() else ""

    def link(path: str | None, label: str) -> str:
        if not path:
            return html.escape(label)
        return f'<a href="{html.escape(_rel(path, out_root))}">{html.escape(label)}</a>'

    worst_html = []
    for row in worst_rows[:worst_limit]:
        worst_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('method', '')))}</td>"
            f"<td>{html.escape(str(row.get('run', '')))}</td>"
            f"<td>{link(row.get('plot'), str(row.get('video', '')))}</td>"
            f"<td>{html.escape(str(row.get('frame_id', '-')))}</td>"
            f"<td>{html.escape(str(row.get('window_start_frame', '-')))}..{html.escape(str(row.get('window_end_frame', '-')))}</td>"
            f"<td>{_fmt(row.get('accuracy'))}</td>"
            f"<td>{_fmt(row.get('recall'))}</td>"
            f"<td>{html.escape(str(row.get('gt', '-')))}</td>"
            f"<td>{html.escape(str(row.get('pred', '-')))}</td>"
            f"<td>{html.escape(str(row.get('tp', '-')))}</td>"
            f"<td>{html.escape(str(row.get('fp', '-')))}</td>"
            f"<td>{html.escape(str(row.get('fn', '-')))}</td>"
            "</tr>"
        )

    video_html = []
    for row in video_rows:
        video_html.append(
            "<tr>"
            f"<td>{html.escape(row['method'])}</td>"
            f"<td>{html.escape(row['run'])}</td>"
            f"<td>{link(row.get('plot'), row['video'])}</td>"
            f"<td>{html.escape(str(row.get('frames', '-')))}</td>"
            f"<td>{_fmt(row.get('mean_accuracy'))}</td>"
            f"<td>{_fmt(row.get('min_accuracy'))}</td>"
            f"<td>{_fmt(row.get('mean_recall'))}</td>"
            f"<td>{_fmt(row.get('min_recall'))}</td>"
            f"<td>{html.escape(str(row.get('worst_frame', '-')))}</td>"
            "</tr>"
        )

    missing_html = []
    for row in missing_rows:
        missing = row.get("missing") or []
        missing_text = ", ".join(str(item) for item in missing) if isinstance(missing, list) else str(missing)
        missing_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('method', '')))}</td>"
            f"<td>{html.escape(str(row.get('run', '')))}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(missing_text or '-')}</td>"
            f"<td><code>{html.escape(str(row.get('gt', '-')))}</code><br><small>{html.escape(str(row.get('gt_format', '-')))}</small></td>"
            f"<td><code>{html.escape(str(row.get('pred', '-')))}</code><br><small>{html.escape(str(row.get('pred_format', '-')))}</small></td>"
            f"<td><code>{html.escape(str(row.get('out', '-')))}</code></td>"
            "</tr>"
        )

    segment_html = []
    for row in segment_rows[:segment_limit]:
        segment_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('method', '')))}</td>"
            f"<td>{html.escape(str(row.get('run', '')))}</td>"
            f"<td>{link(row.get('plot'), str(row.get('video', '')))}</td>"
            f"<td>{html.escape(str(row.get('start_frame', '-')))}..{html.escape(str(row.get('end_frame', '-')))}</td>"
            f"<td>{_fmt(row.get('start_time_sec'), 1)}..{_fmt(row.get('end_time_sec'), 1)}</td>"
            f"<td>{html.escape(str(row.get('center_frames', '-')))}</td>"
            f"<td>{html.escape(str(row.get('window_start_frame', '-')))}..{html.escape(str(row.get('window_end_frame', '-')))}</td>"
            f"<td>{_fmt(row.get('min_accuracy'))}</td>"
            f"<td>{_fmt(row.get('mean_accuracy'))}</td>"
            f"<td>{_fmt(row.get('mean_recall'))}</td>"
            f"<td>{html.escape(str(row.get('worst_frame', '-')))}</td>"
            f"<td>{html.escape(str(row.get('gt', '-')))}</td>"
            f"<td>{html.escape(str(row.get('pred', '-')))}</td>"
            f"<td>{html.escape(str(row.get('fp', '-')))}</td>"
            f"<td>{html.escape(str(row.get('fn', '-')))}</td>"
            "</tr>"
        )

    gallery_html = []
    for row in gallery_rows:
        gallery_html.append(
            "<figure>"
            f"<figcaption>{html.escape(row['method'])} / {html.escape(row['run'])} / {html.escape(row['video'])}</figcaption>"
            f"<a href=\"{html.escape(_rel(str(row['plot']), out_root))}\"><img src=\"{html.escape(_rel(str(row['plot']), out_root))}\" loading=\"lazy\" alt=\"window accuracy curve\"></a>"
            "</figure>"
        )
    missing_body = "".join(missing_html) if missing_html else '<tr><td colspan="7">No missing runs in this batch.</td></tr>'

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Paper Window Accuracy Dashboard</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;color:#111;}"
        "table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 28px;}"
        "th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top;}"
        "th{background:#f4f4f4;position:sticky;top:0;}"
        "code{white-space:normal;word-break:break-word;}"
        "small{color:#555;}"
        ".cards{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 20px;}"
        ".card{border:1px solid #ddd;padding:10px 12px;border-radius:6px;min-width:150px;}"
        ".card strong{display:block;font-size:22px;}"
        ".gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;}"
        "figure{margin:0;border:1px solid #ddd;padding:8px;border-radius:6px;}"
        "figcaption{font-size:13px;margin-bottom:6px;font-weight:600;}"
        "img{width:100%;height:auto;display:block;}"
        "</style></head><body>"
        "<h1>Paper Window Accuracy Dashboard</h1>"
        "<p>Each center frame is scored over a +/-3s window. Lower accuracy rows identify video segments for inspection.</p>"
        f"{gap_link}"
        "<div class='cards'>"
        f"<div class='card'><span>Methods</span><strong>{len(methods)}</strong></div>"
        f"<div class='card'><span>Complete runs</span><strong>{complete_run_count}</strong></div>"
        f"<div class='card'><span>Missing runs</span><strong>{len(missing_rows)}</strong></div>"
        f"<div class='card'><span>Video curves</span><strong>{video_count}</strong></div>"
        f"<div class='card'><span>Low-accuracy segments</span><strong>{len(segment_rows)}</strong></div>"
        f"<div class='card'><span>Worst rows shown</span><strong>{min(worst_limit, len(worst_rows))}</strong></div>"
        "</div>"
        "<h2>Missing / Skipped Runs</h2>"
        "<table><thead><tr><th>Method</th><th>Run</th><th>Status</th><th>Missing</th><th>GT</th><th>Pred</th><th>Output dir</th></tr></thead>"
        f"<tbody>{missing_body}</tbody></table>"
        "<h2>Continuous Low-Accuracy Segments</h2>"
        "<table><thead><tr><th>Method</th><th>Run</th><th>Video</th><th>Center frames</th><th>Time sec</th><th>Length</th><th>Window span</th><th>Min accuracy</th><th>Mean accuracy</th><th>Mean recall</th><th>Worst frame</th><th>GT</th><th>Pred</th><th>FP</th><th>FN</th></tr></thead>"
        f"<tbody>{''.join(segment_html)}</tbody></table>"
        "<h2>Worst +/-3s Windows</h2>"
        "<table><thead><tr><th>Method</th><th>Run</th><th>Video</th><th>Frame</th><th>Window</th><th>Accuracy</th><th>Recall</th><th>GT</th><th>Pred</th><th>TP</th><th>FP</th><th>FN</th></tr></thead>"
        f"<tbody>{''.join(worst_html)}</tbody></table>"
        "<h2>Per-Video Scorecards</h2>"
        "<table><thead><tr><th>Method</th><th>Run</th><th>Video</th><th>Frames</th><th>Mean accuracy</th><th>Min accuracy</th><th>Mean recall</th><th>Min recall</th><th>Worst frame</th></tr></thead>"
        f"<tbody>{''.join(video_html)}</tbody></table>"
        "<h2>Curve Gallery</h2>"
        f"<div class='gallery'>{''.join(gallery_html)}</div>"
        "</body></html>\n",
        encoding="utf-8",
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cross-run dashboard for paper +/-3s window accuracy outputs.")
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--worst-limit", type=int, default=60)
    parser.add_argument("--segment-limit", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_summary = _read_json(args.batch_summary)
    out = build_dashboard(batch_summary, out_path=args.out, worst_limit=args.worst_limit, segment_limit=args.segment_limit)
    print(f"dashboard={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
