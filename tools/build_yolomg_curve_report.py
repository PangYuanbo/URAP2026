from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path


SUMMARY_FIELDS = [
    "video",
    "frames",
    "duration_sec",
    "precision",
    "recall",
    "f1",
    "ap50",
    "matched_confidence_mean",
    "tp",
    "fp",
    "fn",
]


def rel(path: str | Path, base: Path) -> str:
    return os.path.relpath(Path(path), base).replace("\\", "/")


def fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_rows(manifest: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for video, item in sorted(manifest["videos"].items()):
        summary = item["summary"]
        rows.append(
            {
                "video": video,
                "frames": int(summary["frame_count"]),
                "duration_sec": float(item["duration_sec_evaluated"]),
                "precision": float(summary["precision"]),
                "recall": float(summary["recall"]),
                "f1": float(summary["f1"]),
                "ap50": float(summary["ap50"]),
                "matched_confidence_mean": float(summary["matched_confidence_mean"]),
                "tp": int(summary["tp"]),
                "fp": int(summary["fp"]),
                "fn": int(summary["fn"]),
            }
        )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_html(manifest: dict, rows: list[dict[str, object]], out_path: Path) -> str:
    base = out_path.parent
    total_frames = int(manifest["total_frames"])
    weighted_f1 = sum(float(r["f1"]) * int(r["frames"]) for r in rows) / max(total_frames, 1)
    weighted_ap50 = sum(float(r["ap50"]) * int(r["frames"]) for r in rows) / max(total_frames, 1)
    weighted_conf = sum(float(r["matched_confidence_mean"]) * int(r["frames"]) for r in rows) / max(total_frames, 1)
    weighted_recall = sum(float(r["recall"]) * int(r["frames"]) for r in rows) / max(total_frames, 1)

    strongest = sorted(rows, key=lambda r: float(r["ap50"]), reverse=True)[:5]
    weakest = sorted(rows, key=lambda r: float(r["ap50"]))[:5]

    def metric_cards() -> str:
        cards = [
            ("Clips", len(rows)),
            ("Frames", f"{total_frames:,}"),
            ("Weighted AP50", fmt(weighted_ap50)),
            ("Weighted F1", fmt(weighted_f1)),
            ("Weighted Recall", fmt(weighted_recall)),
            ("Matched Conf.", fmt(weighted_conf)),
        ]
        return "\n".join(
            f'<div class="metric"><span>{html.escape(k)}</span><strong>{html.escape(str(v))}</strong></div>'
            for k, v in cards
        )

    def table_body(items: list[dict[str, object]]) -> str:
        parts = []
        for r in items:
            parts.append(
                "<tr>"
                f"<td>{html.escape(str(r['video']))}</td>"
                f"<td>{int(r['frames']):,}</td>"
                f"<td>{fmt(r['duration_sec'], 1)}</td>"
                f"<td>{fmt(r['ap50'])}</td>"
                f"<td>{fmt(r['f1'])}</td>"
                f"<td>{fmt(r['recall'])}</td>"
                f"<td>{fmt(r['precision'])}</td>"
                f"<td>{fmt(r['matched_confidence_mean'])}</td>"
                "</tr>"
            )
        return "\n".join(parts)

    clip_cards = []
    for r in rows:
        video = str(r["video"])
        item = manifest["videos"][video]
        conf_png = rel(item["matched_confidence_plot_png"], base)
        ap_png = rel(item["window_ap50_plot_png"], base)
        per_frame = rel(item["per_frame_csv"], base)
        per_window = rel(item["per_window_csv"], base)
        clip_cards.append(
            f"""
      <section class="clip" id="{html.escape(video)}">
        <div class="clip-head">
          <h2>{html.escape(video)}</h2>
          <div class="numbers">
            <span>AP50 <b>{fmt(r['ap50'])}</b></span>
            <span>F1 <b>{fmt(r['f1'])}</b></span>
            <span>Recall <b>{fmt(r['recall'])}</b></span>
            <span>Precision <b>{fmt(r['precision'])}</b></span>
          </div>
        </div>
        <div class="plots">
          <a href="{html.escape(conf_png)}"><img src="{html.escape(conf_png)}" alt="{html.escape(video)} matched confidence curve"></a>
          <a href="{html.escape(ap_png)}"><img src="{html.escape(ap_png)}" alt="{html.escape(video)} AP50 curve"></a>
        </div>
        <p class="links"><a href="{html.escape(per_frame)}">per-frame CSV</a> · <a href="{html.escape(per_window)}">per-window CSV</a></p>
      </section>
"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLOMG ARD100 Accuracy Curves</title>
  <style>
    :root {{
      --bg: #0b0d10;
      --panel: #151922;
      --panel2: #10141b;
      --text: #edf1f7;
      --muted: #a8b0bb;
      --line: #2a3342;
      --accent: #5eb1ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", system-ui, sans-serif;
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel2);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; font-weight: 650; }}
    p {{ color: var(--muted); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      padding: 18px 32px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 3px; font-size: 22px; }}
    main {{ padding: 0 32px 40px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 24px;
      background: var(--panel);
      border: 1px solid var(--line);
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
      font-size: 13px;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    a {{ color: var(--accent); text-decoration: none; }}
    .twocol {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 20px;
    }}
    .clip {{
      margin-top: 18px;
      padding: 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .clip-head {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: baseline;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    h2 {{ margin: 0; font-size: 20px; }}
    .numbers {{ display: flex; gap: 14px; flex-wrap: wrap; color: var(--muted); font-size: 13px; }}
    .numbers b {{ color: var(--text); }}
    .plots {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 12px;
    }}
    img {{
      display: block;
      width: 100%;
      background: white;
      border-radius: 6px;
    }}
    .links {{ margin: 10px 0 0; font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>YOLOMG on ARD100 Test Set</h1>
    <p>Weights: {html.escape(manifest['weights'])}<br>
       Data: ARD100_YOLOMG test split, {total_frames:,} evaluated frames. Curves are produced from the existing YOLOMG timeline evaluation.</p>
  </header>
  <div class="metrics">{metric_cards()}</div>
  <main>
    <div class="twocol">
      <section>
        <h2>Strongest Clips by AP50</h2>
        <table><thead><tr><th>Video</th><th>Frames</th><th>Sec</th><th>AP50</th><th>F1</th><th>Recall</th><th>Precision</th><th>Matched Conf.</th></tr></thead><tbody>{table_body(strongest)}</tbody></table>
      </section>
      <section>
        <h2>Weakest Clips by AP50</h2>
        <table><thead><tr><th>Video</th><th>Frames</th><th>Sec</th><th>AP50</th><th>F1</th><th>Recall</th><th>Precision</th><th>Matched Conf.</th></tr></thead><tbody>{table_body(weakest)}</tbody></table>
      </section>
    </div>
    <h2>All Clip Curves</h2>
    {''.join(clip_cards)}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-html", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    rows = summary_rows(manifest)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    write_summary_csv(args.out_summary, rows)
    args.out_html.write_text(build_html(manifest, rows, args.out_html), encoding="utf-8")
    print(f"wrote {args.out_html}")
    print(f"wrote {args.out_summary}")
    print(f"clips={len(rows)} frames={manifest['total_frames']}")


if __name__ == "__main__":
    main()
