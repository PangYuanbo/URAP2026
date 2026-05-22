from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from urllib.parse import quote

import cv2


def asset_url(path: Path, html_path: Path) -> str:
    rel = os.path.relpath(path.resolve(), html_path.resolve().parent).replace("\\", "/")
    return quote(rel, safe="/:")


def load_rows(csv_path: Path) -> dict[str, list[dict]]:
    by_video: dict[str, list[dict]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            video = row["video"]
            item = {
                "frame": int(row["frame"]),
                "gt": int(row["gt_count"]),
                "roi": int(row["roi_count"]),
                "positive_roi": int(row["positive_roi_count"]),
                "proposal_hit": int(row["proposal_hit"]),
                "pred_count": int(row["pred_count"]),
                "max_pred_conf": float(row["max_pred_conf"]),
                "best_iou": float(row["best_iou"]),
                "pred_hit": int(row["pred_hit"]),
                "win": float(row["win_score"]),
                "status": row["status"],
                "motion": float(row["max_motion_score"]),
            }
            by_video.setdefault(video, []).append(item)
    for rows in by_video.values():
        rows.sort(key=lambda x: x["frame"])
    return by_video


def video_meta(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"exists": video_path.exists(), "fps": 30.0, "frames": 0, "width": 0, "height": 0}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"exists": True, "fps": fps, "frames": frames, "width": width, "height": height}


def normalize_clip_name(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("empty clip value")
    if value.lower().startswith("clip_"):
        return f"Clip_{int(value.split('_', 1)[1]):03d}"
    return f"Clip_{int(value):03d}"


def parse_clip_filter(value: str) -> set[str]:
    if not value.strip():
        return set()
    return {normalize_clip_name(part) for part in value.replace(";", ",").split(",") if part.strip()}


def make_html(data_json: str, title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0d10;
      --panel: #151922;
      --panel2: #10141b;
      --text: #edf1f7;
      --muted: #9aa6b2;
      --grid: #2a3342;
      --good: #26d07c;
      --bad: #ff5f57;
      --warn: #f2c94c;
      --line: #63b3ff;
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
      gap: 18px;
      padding: 14px 18px;
      border-bottom: 1px solid #263041;
      background: #10141b;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    select, button {{
      background: #202838;
      border: 1px solid #354159;
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 14px;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(520px, 1.5fr) minmax(460px, 1fr);
      gap: 0;
      min-height: 0;
    }}
    .videoPane, .chartPane {{
      min-width: 0;
      min-height: 0;
      padding: 16px;
    }}
    .videoPane {{
      background: #06080b;
      border-right: 1px solid #263041;
      display: grid;
      grid-template-rows: 1fr auto;
      gap: 12px;
    }}
    video {{
      width: 100%;
      height: 100%;
      max-height: calc(100vh - 160px);
      object-fit: contain;
      background: #000;
      border: 1px solid #1f2632;
    }}
    .metaGrid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid #273144;
      border-radius: 8px;
      padding: 10px;
      min-height: 70px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .metric .value {{
      font-size: 20px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .chartPane {{
      background: var(--panel2);
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 12px;
    }}
    .chartHeader {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
    }}
    .chartHeader h2 {{
      margin: 0;
      font-size: 16px;
    }}
    .chartHeader .sub {{
      color: var(--muted);
      font-size: 13px;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      min-height: 360px;
      background: #0b1018;
      border: 1px solid #273144;
      border-radius: 8px;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      color: var(--muted);
      font-size: 13px;
      flex-wrap: wrap;
    }}
    .dot {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 6px;
      vertical-align: -1px;
    }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      .videoPane {{ border-right: 0; border-bottom: 1px solid #263041; }}
      .metaGrid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>{title}</h1>
      <div class="controls">
        <label>Clip <select id="clipSelect"></select></label>
        <label>Window <select id="windowSelect">
          <option value="1">±1s</option>
          <option value="2" selected>±2s</option>
          <option value="3">±3s</option>
          <option value="5">±5s</option>
        </select></label>
        <button id="jumpMiss">Next miss</button>
        <button id="jumpHit">Next hit</button>
      </div>
    </header>
    <main>
      <section class="videoPane">
        <video id="video" controls preload="metadata"></video>
        <div class="metaGrid">
          <div class="metric"><div class="label">Current frame</div><div class="value" id="frameMetric">-</div></div>
          <div class="metric"><div class="label">Win score</div><div class="value" id="winMetric">-</div></div>
          <div class="metric"><div class="label">Status</div><div class="value" id="statusMetric">-</div></div>
          <div class="metric"><div class="label">GT / ROI</div><div class="value" id="roiMetric">-</div></div>
        </div>
      </section>
      <section class="chartPane">
        <div class="chartHeader">
          <h2>Frame Accuracy Around Playback Cursor</h2>
          <div class="sub" id="clipMeta">-</div>
        </div>
        <canvas id="chart"></canvas>
        <div class="legend">
          <span><span class="dot" style="background: var(--line)"></span>win_score</span>
          <span><span class="dot" style="background: var(--good)"></span>GT covered / hit</span>
          <span><span class="dot" style="background: var(--bad)"></span>GT miss</span>
          <span><span class="dot" style="background: var(--cursor)"></span>current video time</span>
        </div>
      </section>
    </main>
  </div>

  <script>
    const DATA = {data_json};
    const clipSelect = document.getElementById('clipSelect');
    const windowSelect = document.getElementById('windowSelect');
    const video = document.getElementById('video');
    const canvas = document.getElementById('chart');
    const ctx = canvas.getContext('2d');
    const frameMetric = document.getElementById('frameMetric');
    const winMetric = document.getElementById('winMetric');
    const statusMetric = document.getElementById('statusMetric');
    const roiMetric = document.getElementById('roiMetric');
    const clipMeta = document.getElementById('clipMeta');

    for (const clip of Object.keys(DATA.clips)) {{
      const opt = document.createElement('option');
      opt.value = clip;
      opt.textContent = clip;
      clipSelect.appendChild(opt);
    }}

    function currentClip() {{
      return DATA.clips[clipSelect.value];
    }}

    function frameAtTime(clip, time) {{
      return Math.round(time * clip.fps);
    }}

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
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }}

    function drawChart() {{
      resizeCanvas();
      const clip = currentClip();
      const rows = clip.rows;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const pad = {{ left: 46, right: 18, top: 22, bottom: 36 }};
      const plotW = Math.max(1, w - pad.left - pad.right);
      const plotH = Math.max(1, h - pad.top - pad.bottom);
      const currentFrame = frameAtTime(clip, video.currentTime || 0);
      const halfSec = Number(windowSelect.value);
      const f0 = currentFrame - halfSec * clip.fps;
      const f1 = currentFrame + halfSec * clip.fps;
      const visible = rows.filter(r => r.frame >= f0 && r.frame <= f1);

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#0b1018';
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = '#2a3342';
      ctx.lineWidth = 1;
      ctx.fillStyle = '#9aa6b2';
      ctx.font = '12px Segoe UI, sans-serif';
      for (let yv = 0; yv <= 1.0001; yv += 0.25) {{
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
        ctx.fillText((s >= 0 ? '+' : '') + s + 's', x - 11, h - 13);
      }}

      function xFor(frame) {{ return pad.left + ((frame - f0) / (f1 - f0)) * plotW; }}
      function yFor(v) {{ return pad.top + (1 - Math.max(0, Math.min(1, v))) * plotH; }}

      if (visible.length) {{
        ctx.strokeStyle = '#63b3ff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        visible.forEach((r, i) => {{
          const x = xFor(r.frame);
          const y = yFor(r.win);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }});
        ctx.stroke();

        for (const r of visible) {{
          const x = xFor(r.frame);
          const y = yFor(r.win);
          const hasGt = r.gt > 0;
          ctx.fillStyle = hasGt ? (r.win > 0 ? '#26d07c' : '#ff5f57') : '#778399';
          ctx.beginPath();
          ctx.arc(x, y, hasGt ? 4 : 2.5, 0, Math.PI * 2);
          ctx.fill();
        }}
      }}

      const cursorX = xFor(currentFrame);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cursorX, pad.top - 4);
      ctx.lineTo(cursorX, h - pad.bottom + 4);
      ctx.stroke();

      const cur = nearestRow(rows, currentFrame);
      if (cur) {{
        frameMetric.textContent = `${{currentFrame}} / sample ${{cur.frame}}`;
        winMetric.textContent = cur.win.toFixed(3);
        statusMetric.textContent = cur.status;
        roiMetric.textContent = `${{cur.gt}} / ${{cur.positive_roi}} of ${{cur.roi}}`;
      }}
    }}

    function loadClip(name) {{
      const clip = DATA.clips[name];
      video.src = clip.video_url;
      clipMeta.textContent = `${{clip.fps.toFixed(2)}} FPS | ${{clip.frames}} frames | ${{clip.width}}x${{clip.height}} | ${{clip.rows.length}} scored samples`;
      video.load();
      drawChart();
    }}

    function jumpTo(predicate) {{
      const clip = currentClip();
      const currentFrame = frameAtTime(clip, video.currentTime || 0);
      const next = clip.rows.find(r => r.frame > currentFrame && predicate(r));
      if (next) {{
        video.currentTime = next.frame / clip.fps;
        drawChart();
      }}
    }}

    document.getElementById('jumpMiss').addEventListener('click', () => jumpTo(r => r.gt > 0 && r.win === 0));
    document.getElementById('jumpHit').addEventListener('click', () => jumpTo(r => r.gt > 0 && r.win > 0));
    clipSelect.addEventListener('change', () => loadClip(clipSelect.value));
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
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--video-root", type=Path, default=Path("Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Videos"))
    p.add_argument("--proxy-video-root", type=Path, default=None)
    p.add_argument("--proxy-ext", default="webm")
    p.add_argument("--include-clips", default="", help="Comma-separated clip ids, e.g. 1,10,Clip_015")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="NPS Frame Winrate Player")
    args = p.parse_args()

    by_video = load_rows(args.csv)
    include = parse_clip_filter(args.include_clips)
    proxy_ext = args.proxy_ext.lstrip(".")
    clips = {}
    for video, rows in by_video.items():
        if include and video not in include:
            continue
        num = int(video.split("_", 1)[1])
        video_path = args.video_root / f"Clip_{num}.mov"
        playback_path = video_path
        if args.proxy_video_root:
            proxy_path = args.proxy_video_root / f"Clip_{num}.{proxy_ext}"
            if proxy_path.exists():
                playback_path = proxy_path
        meta = video_meta(video_path)
        clips[video] = {
            **meta,
            "video_path": str(video_path.resolve()),
            "playback_path": str(playback_path.resolve()),
            "video_url": asset_url(playback_path, args.out),
            "rows": rows,
        }
    if not clips:
        raise SystemExit("no clips matched the requested filters")
    payload = {"csv": str(args.csv.resolve()), "clips": clips}
    html = make_html(json.dumps(payload, ensure_ascii=False), args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"clips={len(clips)}")


if __name__ == "__main__":
    main()
