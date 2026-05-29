from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(r"C:\Users\aaron\Desktop\URAP")
RUN_ROOT = REPO / r"URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\motion_process_gradcam"
DEFAULT_WEB_ROOT = REPO / r"artifacts\yolomg_motion_process_site"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a local website for YOLOMG motion-process videos.")
    p.add_argument("--web-root", default=str(DEFAULT_WEB_ROOT))
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--convert", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--encoder", default="h264_nvenc", choices=["h264_nvenc", "libx264"])
    return p.parse_args()


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - diagnostic path
        raise RuntimeError("imageio-ffmpeg is required. Install it in the YOLOMG venv.") from exc


def collect_items() -> list[dict[str, object]]:
    roots = [
        ("test", RUN_ROOT / "phantom02_full"),
        ("test", RUN_ROOT / "test_rest_full"),
        ("train", RUN_ROOT / "train_full"),
    ]
    items: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for split, root in roots:
        for avi in sorted(root.rglob("*_model_motion_process_gradcam.avi")):
            video = avi.parent.name
            key = (split, video)
            if key in seen:
                continue
            seen.add(key)
            manifest = avi.parent / "manifest.txt"
            frames = None
            if manifest.exists():
                for line in manifest.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("frames="):
                        try:
                            frames = int(line.split("=", 1)[1])
                        except ValueError:
                            frames = None
            poster_candidates = sorted(avi.parent.glob("*_model_motion_process.jpg"))
            items.append(
                {
                    "split": split,
                    "video": video,
                    "frames": frames,
                    "avi": str(avi),
                    "poster_src": str(poster_candidates[0]) if poster_candidates else "",
                    "bytes": avi.stat().st_size,
                }
            )
    return sorted(items, key=lambda x: (str(x["split"]), str(x["video"])))


def media_paths(web_root: Path, item: dict[str, object]) -> tuple[Path, Path]:
    split = str(item["split"])
    video = str(item["video"])
    return web_root / "media" / split / f"{video}.mp4", web_root / "posters" / split / f"{video}.jpg"


def convert_one(ffmpeg: str, web_root: Path, item: dict[str, object], encoder: str, overwrite: bool) -> dict[str, object]:
    mp4, poster = media_paths(web_root, item)
    mp4.parent.mkdir(parents=True, exist_ok=True)
    poster.parent.mkdir(parents=True, exist_ok=True)

    poster_src = str(item.get("poster_src") or "")
    if poster_src and not poster.exists():
        shutil.copy2(poster_src, poster)

    if mp4.exists() and mp4.stat().st_size > 0 and not overwrite:
        return {**item, "mp4": str(mp4), "poster": str(poster), "status": "SKIP"}

    tmp = mp4.with_suffix(".tmp.mp4")
    if tmp.exists():
        tmp.unlink()

    if encoder == "h264_nvenc":
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "24", "-b:v", "0"]
    else:
        codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "24"]

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(item["avi"]),
        *codec_args,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 and encoder == "h264_nvenc":
        fallback_cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(item["avi"]),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp),
        ]
        proc = subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ffmpeg failed for {item['avi']}")
    tmp.replace(mp4)
    return {**item, "mp4": str(mp4), "poster": str(poster), "status": "DONE"}


def relative_item(web_root: Path, item: dict[str, object]) -> dict[str, object]:
    mp4, poster = media_paths(web_root, item)
    return {
        "split": item["split"],
        "video": item["video"],
        "frames": item.get("frames"),
        "bytes": item.get("bytes"),
        "src": mp4.relative_to(web_root).as_posix(),
        "poster": poster.relative_to(web_root).as_posix() if poster.exists() else "",
        "original": str(item["avi"]),
        "ready": mp4.exists() and mp4.stat().st_size > 0,
    }


def write_site(web_root: Path, items: list[dict[str, object]]) -> None:
    web_root.mkdir(parents=True, exist_ok=True)
    data = [relative_item(web_root, item) for item in items]
    (web_root / "videos.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (web_root / "index.html").write_text(HTML, encoding="utf-8")


def main() -> None:
    args = parse_args()
    web_root = Path(args.web_root)
    items = collect_items()
    if not items:
        raise SystemExit("No model motion-process AVI videos found.")

    if args.convert:
        ffmpeg = ffmpeg_exe()
        print(f"[INFO] ffmpeg={ffmpeg}", flush=True)
        print(f"[INFO] videos={len(items)} workers={args.workers} encoder={args.encoder}", flush=True)
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(convert_one, ffmpeg, web_root, item, args.encoder, args.overwrite) for item in items]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                completed += 1
                print(f"[{result['status']}] {completed}/{len(items)} {result['split']}/{result['video']}", flush=True)

    write_site(web_root, items)
    ready = sum(1 for item in items if media_paths(web_root, item)[0].exists())
    print(f"[SITE] {web_root / 'index.html'}", flush=True)
    print(f"[READY] {ready}/{len(items)}", flush=True)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLOMG Motion Process Gallery</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #191d20;
      --panel-2: #20262a;
      --text: #edf1f4;
      --muted: #9aa6ad;
      --line: #343c42;
      --accent: #7dd3fc;
      --good: #86efac;
      --warn: #fbbf24;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      overflow: hidden;
    }
    .app {
      display: grid;
      grid-template-columns: 360px 1fr;
      height: 100vh;
      min-height: 620px;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      flex-direction: column;
      min-width: 0;
    }
    header {
      padding: 18px 18px 14px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0 0 8px;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 12px;
    }
    .stat {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
    }
    .stat strong { display: block; font-size: 16px; }
    .stat span { color: var(--muted); font-size: 12px; }
    .controls {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
    }
    input, select {
      width: 100%;
      background: #0f1214;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      outline: none;
    }
    .list {
      overflow: auto;
      padding: 10px;
      display: grid;
      gap: 8px;
      align-content: start;
    }
    .item {
      display: grid;
      grid-template-columns: 96px 1fr;
      gap: 10px;
      padding: 8px;
      background: transparent;
      border: 1px solid transparent;
      border-radius: 6px;
      color: inherit;
      text-align: left;
      cursor: pointer;
    }
    .item:hover, .item.active {
      background: var(--panel-2);
      border-color: var(--line);
    }
    .thumb {
      width: 96px;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      border-radius: 4px;
      background: #0b0d0e;
    }
    .meta { min-width: 0; }
    .name {
      font-weight: 650;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .sub {
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
    }
    main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-width: 0;
      height: 100vh;
    }
    .topbar {
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }
    .title {
      min-width: 0;
    }
    .title h2 {
      margin: 0;
      font-size: 19px;
      font-weight: 650;
    }
    .title p {
      margin: 4px 0 0;
      color: var(--muted);
    }
    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    button, a.button {
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 12px;
      text-decoration: none;
      cursor: pointer;
    }
    button:hover, a.button:hover { border-color: var(--accent); }
    .stage {
      min-height: 0;
      display: grid;
      place-items: center;
      padding: 16px 20px;
    }
    video {
      width: min(100%, 1480px);
      max-height: calc(100vh - 190px);
      background: black;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .footer {
      padding: 12px 20px;
      color: var(--muted);
      border-top: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 12px;
    }
    .ready { color: var(--good); }
    .missing { color: var(--warn); }
    @media (max-width: 900px) {
      body { overflow: auto; }
      .app { grid-template-columns: 1fr; height: auto; }
      aside, main { height: auto; }
      .list { max-height: 420px; }
      .topbar, .footer { flex-direction: column; align-items: flex-start; }
      video { max-height: 54vh; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <header>
        <h1>YOLOMG Motion Process Gallery</h1>
        <div class="sub">RGB | input motion | Grad-CAM L3 motion | Grad-CAM L5 fusion</div>
        <div class="stats">
          <div class="stat"><strong id="totalCount">0</strong><span>videos</span></div>
          <div class="stat"><strong id="trainCount">0</strong><span>train</span></div>
          <div class="stat"><strong id="testCount">0</strong><span>test</span></div>
        </div>
      </header>
      <div class="controls">
        <input id="search" placeholder="Search phantom ID">
        <select id="split">
          <option value="all">All splits</option>
          <option value="test">Test</option>
          <option value="train">Train</option>
        </select>
      </div>
      <div id="list" class="list"></div>
    </aside>
    <main>
      <div class="topbar">
        <div class="title">
          <h2 id="videoTitle">Loading...</h2>
          <p id="videoMeta"></p>
        </div>
        <div class="actions">
          <button id="prev">Previous</button>
          <button id="next">Next</button>
          <a id="openFile" class="button" href="#" target="_blank" rel="noreferrer">Open MP4</a>
        </div>
      </div>
      <div class="stage">
        <video id="player" controls playsinline preload="metadata"></video>
      </div>
      <div class="footer">
        <span id="path"></span>
        <span>Use search/filter on the left; only the selected clip is loaded.</span>
      </div>
    </main>
  </div>
  <script>
    const state = { videos: [], filtered: [], current: 0 };
    const els = {
      list: document.getElementById('list'),
      search: document.getElementById('search'),
      split: document.getElementById('split'),
      player: document.getElementById('player'),
      title: document.getElementById('videoTitle'),
      meta: document.getElementById('videoMeta'),
      path: document.getElementById('path'),
      openFile: document.getElementById('openFile'),
      prev: document.getElementById('prev'),
      next: document.getElementById('next'),
      total: document.getElementById('totalCount'),
      train: document.getElementById('trainCount'),
      test: document.getElementById('testCount')
    };
    const fmtBytes = n => {
      if (!n) return '';
      const gb = n / (1024 ** 3);
      return gb >= 1 ? `${gb.toFixed(2)} GB AVI` : `${(n / (1024 ** 2)).toFixed(1)} MB AVI`;
    };
    function applyFilters() {
      const q = els.search.value.trim().toLowerCase();
      const split = els.split.value;
      state.filtered = state.videos.filter(v => {
        const splitOk = split === 'all' || v.split === split;
        const qOk = !q || v.video.toLowerCase().includes(q);
        return splitOk && qOk;
      });
      state.current = Math.min(state.current, Math.max(0, state.filtered.length - 1));
      renderList();
      selectVideo(state.current);
    }
    function renderList() {
      els.list.innerHTML = '';
      state.filtered.forEach((v, idx) => {
        const btn = document.createElement('button');
        btn.className = `item ${idx === state.current ? 'active' : ''}`;
        btn.innerHTML = `
          <img class="thumb" src="${v.poster || ''}" alt="">
          <span class="meta">
            <span class="name">${v.video}</span>
            <span class="sub">${v.split} | ${v.frames || '?'} frames</span>
            <span class="sub ${v.ready ? 'ready' : 'missing'}">${v.ready ? 'MP4 ready' : 'MP4 missing'}</span>
          </span>`;
        btn.addEventListener('click', () => selectVideo(idx));
        els.list.appendChild(btn);
      });
    }
    function selectVideo(idx) {
      if (!state.filtered.length) {
        els.title.textContent = 'No videos';
        els.meta.textContent = '';
        els.player.removeAttribute('src');
        return;
      }
      state.current = Math.max(0, Math.min(idx, state.filtered.length - 1));
      const v = state.filtered[state.current];
      els.title.textContent = `${v.split.toUpperCase()} / ${v.video}`;
      els.meta.textContent = `${v.frames || '?'} frames | ${fmtBytes(v.bytes)} | ${state.current + 1}/${state.filtered.length}`;
      els.path.textContent = v.original;
      els.player.poster = v.poster || '';
      els.player.src = v.src;
      els.openFile.href = v.src;
      [...els.list.children].forEach((el, i) => el.classList.toggle('active', i === state.current));
      const active = els.list.children[state.current];
      if (active) active.scrollIntoView({ block: 'nearest' });
    }
    els.search.addEventListener('input', applyFilters);
    els.split.addEventListener('change', applyFilters);
    els.prev.addEventListener('click', () => selectVideo(state.current - 1));
    els.next.addEventListener('click', () => selectVideo(state.current + 1));
    window.addEventListener('keydown', (e) => {
      if (e.target.matches('input, select')) return;
      if (e.key === 'ArrowUp') selectVideo(state.current - 1);
      if (e.key === 'ArrowDown') selectVideo(state.current + 1);
    });
    fetch('videos.json')
      .then(r => r.json())
      .then(videos => {
        state.videos = videos;
        els.total.textContent = videos.length;
        els.train.textContent = videos.filter(v => v.split === 'train').length;
        els.test.textContent = videos.filter(v => v.split === 'test').length;
        state.filtered = videos;
        renderList();
        selectVideo(0);
      })
      .catch(err => {
        els.title.textContent = 'Failed to load videos.json';
        els.meta.textContent = String(err);
      });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
