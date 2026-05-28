from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def load_rows(csv_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "frame": int(row["frame_index"]),
                    "time": float(row["timestamp_sec"]),
                    "tp": int(row["tp"]),
                    "fp": int(row["fp"]),
                    "fn": int(row["fn"]),
                    "gt": int(row["gt_count"]),
                    "pred": int(row["pred_count"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                    "hit": float(row["frame_correct"]),
                    "matched_conf": float(row["matched_confidence"]),
                    "conf_mean": float(row["confidence_mean"]),
                }
            )
    rows.sort(key=lambda r: int(r["frame"]))
    return rows


def make_html(payload: dict[str, object]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    title = html.escape(str(payload["title"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #080a0d;
      --panel: #121720;
      --panel2: #0e131b;
      --line: #293242;
      --grid: #253044;
      --text: #eef3f9;
      --muted: #9ba7b5;
      --good: #2dd47d;
      --bad: #ff5f57;
      --accent: #65b7ff;
      --cursor: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", system-ui, sans-serif;
    }}
    .app {{
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 16px;
      background: var(--panel2);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }}
    .controls {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    label {{
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    select, button {{
      background: #1a2230;
      color: var(--text);
      border: 1px solid #344156;
      border-radius: 6px;
      padding: 7px 9px;
      font: inherit;
    }}
    button {{ cursor: pointer; }}
    main {{
      display: grid;
      grid-template-columns: minmax(420px, 1.25fr) minmax(360px, 0.75fr);
      gap: 14px;
      padding: 14px;
      min-height: 0;
    }}
    .videoPanel, .chartPanel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 0;
      overflow: hidden;
    }}
    .videoPanel {{
      display: grid;
      grid-template-rows: 1fr auto;
    }}
    video {{
      width: 100%;
      height: 100%;
      min-height: 360px;
      background: black;
      object-fit: contain;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      border-top: 1px solid var(--line);
    }}
    .metric {{
      padding: 10px 12px;
      border-right: 1px solid var(--line);
      min-width: 0;
    }}
    .metric:last-child {{ border-right: 0; }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .metric strong {{
      display: block;
      margin-top: 3px;
      font-size: 18px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .chartPanel {{
      display: grid;
      grid-template-rows: auto 1fr auto;
    }}
    .chartHead {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }}
    .chartHead h2 {{
      margin: 0 0 5px;
      font-size: 17px;
      font-weight: 650;
    }}
    .chartHead p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      min-height: 360px;
      display: block;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      padding: 10px 14px;
      color: var(--muted);
      border-top: 1px solid var(--line);
      font-size: 12px;
    }}
    .dot {{
      display: inline-block;
      width: 9px;
      height: 9px;
      border-radius: 50%;
      margin-right: 5px;
    }}
    .good {{ background: var(--good); }}
    .bad {{ background: var(--bad); }}
    .blue {{ background: var(--accent); }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      .meta {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      video {{ min-height: 260px; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>{title}</h1>
      <div class="controls">
        <label>Clip <select id="clipSelect"></select></label>
        <label>Metric
          <select id="metricSelect">
            <option value="f1">F1</option>
            <option value="recall">Recall</option>
            <option value="precision">Precision</option>
            <option value="matched_conf">Matched confidence</option>
            <option value="hit">Frame hit</option>
          </select>
        </label>
        <label>Window
          <select id="windowSelect">
            <option value="1">±1s</option>
            <option value="2" selected>±2s</option>
            <option value="3">±3s</option>
            <option value="5">±5s</option>
          </select>
        </label>
        <button id="jumpBad">Next low</button>
        <button id="jumpGood">Next high</button>
      </div>
    </header>
    <main>
      <section class="videoPanel">
        <video id="video" controls preload="metadata"></video>
        <div class="meta">
          <div class="metric"><span>Frame</span><strong id="frameMetric">-</strong></div>
          <div class="metric"><span>Current</span><strong id="valueMetric">-</strong></div>
          <div class="metric"><span>TP / FP / FN</span><strong id="countMetric">-</strong></div>
          <div class="metric"><span>GT / Pred</span><strong id="detMetric">-</strong></div>
          <div class="metric"><span>Clip AP50</span><strong id="clipMetric">-</strong></div>
        </div>
      </section>
      <section class="chartPanel">
        <div class="chartHead">
          <h2>Accuracy Curve Around Playback Cursor</h2>
          <p id="clipMeta">-</p>
        </div>
        <canvas id="chart"></canvas>
        <div class="legend">
          <span><i class="dot blue"></i>selected metric</span>
          <span><i class="dot good"></i>frame hit</span>
          <span><i class="dot bad"></i>frame miss / low value</span>
          <span>white line = current playback position</span>
        </div>
      </section>
    </main>
  </div>
  <script>
    const DATA = {data};
    const video = document.getElementById('video');
    const clipSelect = document.getElementById('clipSelect');
    const metricSelect = document.getElementById('metricSelect');
    const windowSelect = document.getElementById('windowSelect');
    const chart = document.getElementById('chart');
    const ctx = chart.getContext('2d');
    const frameMetric = document.getElementById('frameMetric');
    const valueMetric = document.getElementById('valueMetric');
    const countMetric = document.getElementById('countMetric');
    const detMetric = document.getElementById('detMetric');
    const clipMetric = document.getElementById('clipMetric');
    const clipMeta = document.getElementById('clipMeta');

    for (const name of Object.keys(DATA.clips)) {{
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      clipSelect.appendChild(opt);
    }}

    function currentClip() {{ return DATA.clips[clipSelect.value]; }}
    function metricValue(row) {{ return Number(row[metricSelect.value] || 0); }}
    function frameAtTime(clip, t) {{ return Math.max(1, Math.round(t * clip.fps) + 1); }}
    function nearestRow(rows, frame) {{
      if (!rows.length) return null;
      let lo = 0, hi = rows.length - 1;
      while (lo < hi) {{
        const mid = Math.floor((lo + hi) / 2);
        if (rows[mid].frame < frame) lo = mid + 1; else hi = mid;
      }}
      const a = rows[lo];
      const b = rows[Math.max(0, lo - 1)];
      return Math.abs(a.frame - frame) < Math.abs(b.frame - frame) ? a : b;
    }}

    function resizeCanvas() {{
      const rect = chart.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(360, Math.floor(rect.width * dpr));
      const h = Math.max(300, Math.floor(rect.height * dpr));
      if (chart.width !== w || chart.height !== h) {{
        chart.width = w;
        chart.height = h;
      }}
    }}

    function drawChart() {{
      resizeCanvas();
      const clip = currentClip();
      const rows = clip.rows;
      const metric = metricSelect.value;
      const currentFrame = frameAtTime(clip, video.currentTime || 0);
      const halfSec = Number(windowSelect.value);
      const halfFrames = Math.max(1, Math.round(halfSec * clip.fps));
      const f0 = Math.max(1, currentFrame - halfFrames);
      const f1 = Math.min(clip.frames, currentFrame + halfFrames);
      const visible = rows.filter(r => r.frame >= f0 && r.frame <= f1);

      const w = chart.width, h = chart.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#0d1118';
      ctx.fillRect(0, 0, w, h);

      const pad = {{ left: 48, right: 18, top: 24, bottom: 38 }};
      const plotW = w - pad.left - pad.right;
      const plotH = h - pad.top - pad.bottom;

      ctx.strokeStyle = '#253044';
      ctx.lineWidth = 1;
      ctx.fillStyle = '#9ba7b5';
      ctx.font = `${{12 * (window.devicePixelRatio || 1)}}px Segoe UI`;
      for (let yv = 0; yv <= 1.001; yv += 0.25) {{
        const y = pad.top + (1 - yv) * plotH;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(w - pad.right, y);
        ctx.stroke();
        ctx.fillText(yv.toFixed(2), 8, y + 4);
      }}
      for (let s = -halfSec; s <= halfSec; s += 1) {{
        const x = pad.left + ((s + halfSec) / (2 * halfSec)) * plotW;
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, h - pad.bottom);
        ctx.stroke();
        ctx.fillText((s >= 0 ? '+' : '') + s + 's', x - 12, h - 14);
      }}

      function xFor(frame) {{ return pad.left + ((frame - f0) / Math.max(1, f1 - f0)) * plotW; }}
      function yFor(v) {{ return pad.top + (1 - Math.max(0, Math.min(1, v))) * plotH; }}

      if (visible.length) {{
        ctx.strokeStyle = '#65b7ff';
        ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
        ctx.beginPath();
        visible.forEach((r, i) => {{
          const x = xFor(r.frame);
          const y = yFor(metricValue(r));
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }});
        ctx.stroke();

        for (const r of visible) {{
          const value = metricValue(r);
          ctx.fillStyle = r.hit > 0 ? '#2dd47d' : (value < 0.5 ? '#ff5f57' : '#7c8798');
          ctx.beginPath();
          ctx.arc(xFor(r.frame), yFor(value), r.hit > 0 ? 3.6 : 2.6, 0, Math.PI * 2);
          ctx.fill();
        }}
      }}

      const cursorX = xFor(currentFrame);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
      ctx.beginPath();
      ctx.moveTo(cursorX, pad.top - 4);
      ctx.lineTo(cursorX, h - pad.bottom + 4);
      ctx.stroke();

      const cur = nearestRow(rows, currentFrame);
      if (cur) {{
        frameMetric.textContent = `${{currentFrame}} / sample ${{cur.frame}}`;
        valueMetric.textContent = `${{metric}}=${{metricValue(cur).toFixed(3)}}`;
        countMetric.textContent = `${{cur.tp}} / ${{cur.fp}} / ${{cur.fn}}`;
        detMetric.textContent = `${{cur.gt}} / ${{cur.pred}}`;
      }}
    }}

    function loadClip(name) {{
      const clip = DATA.clips[name];
      video.src = clip.video_url;
      clipMeta.textContent = `${{clip.fps.toFixed(2)}} FPS | ${{clip.frames}} frames | ${{clip.duration.toFixed(1)}}s | ${{clip.rows.length}} scored frames`;
      clipMetric.textContent = clip.ap50.toFixed(3);
      video.load();
      drawChart();
    }}

    function jumpTo(predicate) {{
      const clip = currentClip();
      const currentFrame = frameAtTime(clip, video.currentTime || 0);
      const next = clip.rows.find(r => r.frame > currentFrame && predicate(r));
      if (next) {{
        video.currentTime = Math.max(0, next.time);
        drawChart();
      }}
    }}

    document.getElementById('jumpBad').addEventListener('click', () => jumpTo(r => metricValue(r) < 0.35));
    document.getElementById('jumpGood').addEventListener('click', () => jumpTo(r => metricValue(r) > 0.85));
    clipSelect.addEventListener('change', () => loadClip(clipSelect.value));
    metricSelect.addEventListener('change', drawChart);
    windowSelect.addEventListener('change', drawChart);
    video.addEventListener('timeupdate', drawChart);
    video.addEventListener('seeked', drawChart);
    video.addEventListener('loadedmetadata', drawChart);
    window.addEventListener('resize', drawChart);

    loadClip(clipSelect.value);
  </script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="YOLOMG ARD100 Video Accuracy Player")
    p.add_argument("--include-videos", default="", help="Comma-separated video ids; default includes all.")
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    include = {v.strip() for v in args.include_videos.replace(";", ",").split(",") if v.strip()}
    clips: dict[str, object] = {}
    for name, info in sorted(manifest["videos"].items()):
        if include and name not in include:
            continue
        rows = load_rows(Path(info["per_frame_csv"]))
        clips[name] = {
            "fps": float(info["fps"]),
            "frames": int(info["frame_count_video"]),
            "duration": float(info["duration_sec_video"]),
            "video_url": f"/video/{name}",
            "ap50": float(info["summary"]["ap50"]),
            "summary": info["summary"],
            "rows": rows,
        }

    if not clips:
        raise SystemExit("no clips selected")
    payload = {
        "title": args.title,
        "weights": manifest["weights"],
        "total_frames": manifest["total_frames"],
        "clips": clips,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(make_html(payload), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"clips={len(clips)}")


if __name__ == "__main__":
    main()
