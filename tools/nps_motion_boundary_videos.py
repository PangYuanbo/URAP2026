#!/usr/bin/env python3
"""Render NPS optical-flow motion-boundary demos as gallery-ready MP4 files."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_FRAMES = Path(r"D:\URAP_datasets\TransVisDrone\NPS\AllFrames")
DEFAULT_OUT = REPO / r"artifacts\nps_motion_boundary_site"
FRAME_RE = re.compile(r"^(Clip_\d+)_(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render NPS paper optical-flow motion-boundary demo videos.")
    p.add_argument("--frames-root", default=str(DEFAULT_FRAMES))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--splits", nargs="+", default=["test", "val"])
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--display-width", type=int, default=640)
    p.add_argument("--flow-width", type=int, default=640)
    p.add_argument("--threshold-percentile", type=float, default=97.0)
    p.add_argument("--threshold-floor", type=int, default=32)
    p.add_argument("--dilate", type=int, default=5)
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def dense_flow_factory():
    if hasattr(cv2, "DualTVL1OpticalFlow_create"):
        return "DualTVL1", cv2.DualTVL1OpticalFlow_create()
    optflow = getattr(cv2, "optflow", None)
    if optflow is not None and hasattr(optflow, "DualTVL1OpticalFlow_create"):
        return "DualTVL1", optflow.DualTVL1OpticalFlow_create()
    return "Farneback", None


def calc_flow(method: str, engine, prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    if method == "DualTVL1":
        return engine.calc(prev_gray, gray, None)
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def motion_boundary_from_flow(flow: np.ndarray) -> np.ndarray:
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)
    ux, uy = np.gradient(u)
    vx, vy = np.gradient(v)
    u_mag = np.sqrt(np.square(ux) + np.square(uy))
    v_mag = np.sqrt(np.square(vx) + np.square(vy))
    mb = np.maximum(u_mag, v_mag)
    return cv2.normalize(mb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def resize_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        out = frame
    else:
        new_h = int(round(h * width / w))
        out = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
    if out.shape[0] % 2:
        out = out[:-1]
    return out


def label_panel(frame: np.ndarray, title: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def ffmpeg_writer(out_path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def group_frames(frames_root: Path, splits: list[str]) -> list[dict[str, object]]:
    clips: list[dict[str, object]] = []
    for split in splits:
        split_dir = frames_root / split
        grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
        for path in split_dir.glob("*"):
            match = FRAME_RE.match(path.name)
            if match:
                grouped[match.group(1)].append((int(match.group(2)), path))
        for clip, items in sorted(grouped.items()):
            items.sort()
            clips.append({"split": split, "clip": clip, "frames": [p for _, p in items]})
    return clips


def write_progress(progress_path: Path, payload: dict[str, object]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_clip(
    split: str,
    clip: str,
    frames: list[Path],
    out_dir: Path,
    method: str,
    engine,
    args: argparse.Namespace,
    progress_path: Path,
    clip_index: int,
    total_clips: int,
) -> dict[str, object]:
    media_dir = out_dir / "media" / split
    poster_dir = out_dir / "posters" / split
    out_path = media_dir / f"{clip}_nps_motion_boundary.mp4"
    poster_path = poster_dir / f"{clip}_poster.jpg"
    rel_src = out_path.relative_to(out_dir).as_posix()
    rel_poster = poster_path.relative_to(out_dir).as_posix()
    if out_path.exists() and not args.overwrite:
        return {
            "video": clip,
            "split": split,
            "frames": len(frames),
            "src": rel_src,
            "poster": rel_poster if poster_path.exists() else "",
            "method": method,
        }

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frames[0]}")
    rgb_panel = resize_width(first, args.display_width)
    panel_h, panel_w = rgb_panel.shape[:2]
    writer = ffmpeg_writer(out_path, panel_w * 3, panel_h, args.fps)
    assert writer.stdin is not None

    prev_gray = None
    written = 0
    start = time.time()
    try:
        for idx, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frame_disp = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resize_width(frame, args.flow_width), cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                mb = np.zeros_like(gray, dtype=np.uint8)
            else:
                flow = calc_flow(method, engine, prev_gray, gray)
                mb = motion_boundary_from_flow(flow)
            prev_gray = gray

            threshold = max(args.threshold_floor, int(np.percentile(mb, args.threshold_percentile)))
            mask = (mb >= threshold).astype(np.uint8) * 255
            if args.dilate > 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate, args.dilate))
                mask = cv2.dilate(mask, kernel, iterations=1)

            mb_disp = cv2.resize(mb, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            motion_color = cv2.applyColorMap(mb_disp, cv2.COLORMAP_TURBO)
            mask_disp = cv2.resize(mask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
            cutout = np.zeros_like(frame_disp)
            cutout[mask_disp > 0] = frame_disp[mask_disp > 0]

            stamp = f"{clip} {split} frame {idx + 1}/{len(frames)}"
            rgb = label_panel(frame_disp, f"RGB | {stamp}")
            flow_panel = label_panel(motion_color, f"NPS optical-flow motion boundary | {method}")
            cut_panel = label_panel(cutout, "moving region cutout")
            panel = np.concatenate([rgb, flow_panel, cut_panel], axis=1)
            writer.stdin.write(panel.tobytes())
            written += 1

            if idx == 0:
                poster_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(poster_path), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if idx % 25 == 0 or idx == len(frames) - 1:
                write_progress(
                    progress_path,
                    {
                        "status": "running",
                        "clip": clip,
                        "split": split,
                        "clip_index": clip_index,
                        "total_clips": total_clips,
                        "frame_index": idx + 1,
                        "frames_in_clip": len(frames),
                        "written_frames": written,
                        "last_completed_unit": f"{split}/{clip}",
                        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
    finally:
        writer.stdin.close()
        code = writer.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed for {clip} with exit code {code}")

    seconds = time.time() - start
    return {
        "video": clip,
        "split": split,
        "frames": written,
        "src": rel_src,
        "poster": rel_poster,
        "method": method,
        "seconds": round(seconds, 2),
    }


def build_html(out_dir: Path) -> None:
    (out_dir / "index.html").write_text(HTML, encoding="utf-8")


def main() -> None:
    args = parse_args()
    frames_root = Path(args.frames_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"

    clips = group_frames(frames_root, args.splits)
    if args.max_clips > 0:
        clips = clips[: args.max_clips]
    if not clips:
        raise RuntimeError(f"No NPS clips found under {frames_root}")

    method, engine = dense_flow_factory()
    if method != "DualTVL1":
        print("[WARN] DualTVL1 unavailable in this OpenCV build; using Farneback dense optical flow fallback.", flush=True)

    items: list[dict[str, object]] = []
    for index, item in enumerate(clips, start=1):
        split = str(item["split"])
        clip = str(item["clip"])
        frames = list(item["frames"])
        print(f"[{index}/{len(clips)}] rendering {split}/{clip} frames={len(frames)}", flush=True)
        result = render_clip(split, clip, frames, out_dir, method, engine, args, progress_path, index, len(clips))
        items.append(result)
        (out_dir / "videos.json").write_text(json.dumps(items, indent=2), encoding="utf-8")

    build_html(out_dir)
    write_progress(
        progress_path,
        {
            "status": "complete",
            "clip_index": len(clips),
            "total_clips": len(clips),
            "last_completed_unit": f"{items[-1]['split']}/{items[-1]['video']}" if items else "",
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(f"videos={len(items)}", flush=True)
    print(f"site={out_dir / 'index.html'}", flush=True)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NPS Optical-Flow Motion Boundary Gallery</title>
  <style>
    :root { color-scheme: dark; --bg:#101214; --panel:#191d20; --panel2:#22282c; --text:#edf1f4; --muted:#9aa6ad; --line:#343c42; --accent:#27d35f; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; overflow: hidden; }
    .app { height: 100vh; display: grid; grid-template-columns: 360px 1fr; }
    aside { min-width: 0; background: var(--panel); border-right: 1px solid var(--line); display: flex; flex-direction: column; }
    header { padding: 18px; border-bottom: 1px solid var(--line); }
    h1 { margin: 0 0 8px; font-size: 18px; }
    .sub { color: var(--muted); font-size: 12px; }
    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }
    .stat { background: var(--panel2); border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
    .stat strong { display:block; font-size: 17px; }
    .stat span { color: var(--muted); font-size: 12px; }
    .controls { display: grid; gap: 10px; padding: 14px 18px; border-bottom: 1px solid var(--line); }
    input, select { width: 100%; background: #0f1214; color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 10px 11px; }
    .list { overflow: auto; padding: 10px; display: grid; gap: 8px; align-content: start; }
    .item { display: grid; grid-template-columns: 96px 1fr; gap: 10px; padding: 8px; border: 1px solid transparent; border-radius: 6px; background: transparent; color: inherit; cursor: pointer; text-align: left; }
    .item:hover, .item.active { background: var(--panel2); border-color: var(--line); }
    .thumb { width: 96px; aspect-ratio: 16/9; object-fit: cover; border-radius: 4px; background: #050606; }
    .name { font-weight: 650; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .mini { display: block; color: var(--muted); font-size: 12px; margin-top: 3px; }
    main { min-width: 0; display: grid; grid-template-rows: auto 1fr auto; }
    .topbar { padding: 16px 20px; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; gap: 16px; align-items: center; }
    .title h2 { margin: 0; font-size: 19px; }
    .title p { margin: 4px 0 0; color: var(--muted); }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; }
    button, a.button { background: var(--panel2); color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 9px 12px; text-decoration: none; cursor: pointer; }
    button:hover, a.button:hover { border-color: var(--accent); }
    .stage { min-height: 0; display: grid; place-items: center; padding: 16px 20px; }
    video, iframe { width: min(100%, 1480px); max-height: 76vh; background: black; border: 1px solid var(--line); border-radius: 6px; }
    .footer { padding: 12px 20px; color: var(--muted); border-top: 1px solid var(--line); display: flex; justify-content: space-between; gap: 12px; font-size: 12px; }
    @media (max-width: 900px) { body { overflow:auto; } .app { grid-template-columns: 1fr; height:auto; } .list { max-height: 420px; } .topbar, .footer { flex-direction: column; align-items: flex-start; } video, iframe { max-height: 56vh; } }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <header>
        <h1>NPS Optical-Flow Motion Boundary</h1>
        <div class="sub">RGB | motion boundary | moving-region cutout</div>
        <div class="stats">
          <div class="stat"><strong id="totalCount">0</strong><span>videos</span></div>
          <div class="stat"><strong id="testCount">0</strong><span>test</span></div>
          <div class="stat"><strong id="valCount">0</strong><span>val</span></div>
        </div>
      </header>
      <div class="controls">
        <input id="search" placeholder="Search Clip ID">
        <select id="split">
          <option value="all">All splits</option>
          <option value="test">Test</option>
          <option value="val">Val</option>
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
          <a id="openFile" class="button" href="#" target="_blank" rel="noreferrer">Open Video</a>
        </div>
      </div>
      <div class="stage"><video id="player" controls playsinline></video></div>
      <div class="footer">
        <span id="path"></span>
        <span>NPS formula: dense optical flow -> gradients of U/V -> motion boundary response.</span>
      </div>
    </main>
  </div>
  <script>
    const state = { videos: [], filtered: [], current: 0 };
    const els = {
      list: document.getElementById('list'), search: document.getElementById('search'), split: document.getElementById('split'),
      player: document.getElementById('player'), title: document.getElementById('videoTitle'), meta: document.getElementById('videoMeta'),
      path: document.getElementById('path'), openFile: document.getElementById('openFile'), prev: document.getElementById('prev'), next: document.getElementById('next'),
      total: document.getElementById('totalCount'), test: document.getElementById('testCount'), val: document.getElementById('valCount')
    };
    function renderList() {
      els.list.innerHTML = '';
      state.filtered.forEach((v, idx) => {
        const btn = document.createElement('button');
        btn.className = `item ${idx === state.current ? 'active' : ''}`;
        btn.innerHTML = `<img class="thumb" src="${v.poster || ''}" alt=""><span><span class="name">${v.video}</span><span class="mini">${v.split} | ${v.frames} frames</span><span class="mini">${v.method}</span></span>`;
        btn.addEventListener('click', () => selectVideo(idx));
        els.list.appendChild(btn);
      });
    }
    function applyFilters() {
      const q = els.search.value.trim().toLowerCase();
      const split = els.split.value;
      state.filtered = state.videos.filter(v => (split === 'all' || v.split === split) && (!q || v.video.toLowerCase().includes(q)));
      state.current = Math.min(state.current, Math.max(0, state.filtered.length - 1));
      renderList();
      selectVideo(state.current);
    }
    function selectVideo(idx) {
      if (!state.filtered.length) return;
      state.current = Math.max(0, Math.min(idx, state.filtered.length - 1));
      const v = state.filtered[state.current];
      els.title.textContent = `${v.split.toUpperCase()} / ${v.video}`;
      els.meta.textContent = `${v.frames} frames | ${state.current + 1}/${state.filtered.length}`;
      els.path.textContent = v.src;
      els.player.src = v.src;
      els.openFile.href = v.drive_url || v.src;
      [...els.list.children].forEach((el, i) => el.classList.toggle('active', i === state.current));
    }
    els.search.addEventListener('input', applyFilters);
    els.split.addEventListener('change', applyFilters);
    els.prev.addEventListener('click', () => selectVideo(state.current - 1));
    els.next.addEventListener('click', () => selectVideo(state.current + 1));
    fetch('videos.json').then(r => r.json()).then(videos => {
      state.videos = videos;
      state.filtered = videos;
      els.total.textContent = videos.length;
      els.test.textContent = videos.filter(v => v.split === 'test').length;
      els.val.textContent = videos.filter(v => v.split === 'val').length;
      renderList();
      selectVideo(0);
    }).catch(err => { els.title.textContent = 'Failed to load videos.json'; els.meta.textContent = String(err); });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise
