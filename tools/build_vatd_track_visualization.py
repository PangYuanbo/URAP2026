from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an interactive VATD track-selection visualization.")
    parser.add_argument("--tracklets", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--video-frame-offset", type=int, default=0)
    parser.add_argument("--rgb-panel-width", type=int, default=480)
    parser.add_argument("--rgb-panel-height", type=int, default=270)
    parser.add_argument("--display-scale", type=float, default=2.0)
    parser.add_argument("--selection-threshold", type=float, default=0.5)
    parser.add_argument("--trail-length", type=int, default=45)
    parser.add_argument("--max-active-tracks", type=int, default=30)
    return parser.parse_args()


def color_for_score(score: float | None, threshold: float) -> tuple[int, int, int]:
    if score is None:
        return (130, 130, 130)
    if score >= threshold:
        return (70, 220, 80)
    if score >= max(0.2, threshold * 0.5):
        return (0, 190, 255)
    return (145, 145, 145)


def load_tracks(path: Path, sequence: str, start_frame: int, end_frame: int) -> tuple[list[dict], dict[int, list[dict]]]:
    tracks: list[dict] = []
    rows_by_frame: dict[int, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            meta = payload.get("meta", {})
            if meta.get("seq") != sequence:
                continue
            rows = [
                row
                for row in payload.get("rows", [])
                if start_frame <= int(row["frame_id"]) <= end_frame
            ]
            if not rows:
                continue
            track = {
                "track_id": str(meta.get("track_id", "")),
                "vatd_score": None if meta.get("vatd_score") is None else float(meta["vatd_score"]),
                "label": meta.get("label"),
                "best_iou": float(meta.get("best_iou") or 0.0),
                "mean_objectness": float(meta.get("mean_objectness") or 0.0),
                "rows": rows,
            }
            tracks.append(track)
            for row in rows:
                rows_by_frame[int(row["frame_id"])].append({"track": track, "row": row})
    tracks.sort(key=lambda item: item["vatd_score"] if item["vatd_score"] is not None else -1.0, reverse=True)
    return tracks, rows_by_frame


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, image.shape[1] - width - 8))
    y = max(height + 8, min(y, image.shape[0] - baseline - 4))
    cv2.rectangle(image, (x, y - height - 7), (x + width + 7, y + baseline + 3), (12, 15, 18), -1)
    cv2.rectangle(image, (x, y - height - 7), (x + width + 7, y + baseline + 3), color, 1)
    cv2.putText(image, text, (x + 4, y - 3), font, scale, (245, 248, 250), thickness, cv2.LINE_AA)


def build_html(out_dir: Path, manifest: dict) -> None:
    manifest_json = json.dumps(manifest, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VATD Real-Test Trajectory Selection</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b0f14; --panel:#141b23; --line:#2a3745; --text:#edf3f8; --muted:#91a0ad; --green:#46dc50; --amber:#ffc000; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,"Segoe UI",sans-serif; }}
    header {{ padding:16px 20px; border-bottom:1px solid var(--line); background:#10161d; }}
    h1 {{ margin:0 0 5px; font-size:20px; }}
    header p {{ margin:0; color:var(--muted); }}
    main {{ max-width:1440px; margin:auto; padding:16px; display:grid; gap:12px; }}
    .viewer {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
    #frameImage {{ display:block; width:100%; height:auto; background:#050709; }}
    .controls {{ padding:12px; display:grid; grid-template-columns:auto auto 1fr auto auto; gap:10px; align-items:center; }}
    button,input {{ accent-color:var(--green); }}
    button {{ border:1px solid var(--line); background:#202b36; color:var(--text); border-radius:7px; padding:7px 12px; cursor:pointer; }}
    input[type=range] {{ width:100%; }}
    .cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:12px; }}
    .card b {{ display:block; font-size:21px; margin-top:3px; }}
    .legend {{ display:flex; gap:18px; flex-wrap:wrap; color:var(--muted); }}
    .dot {{ width:11px; height:11px; border-radius:50%; display:inline-block; margin-right:6px; }}
    .note {{ color:var(--muted); }}
    @media(max-width:800px) {{ .cards {{ grid-template-columns:1fr 1fr; }} .controls {{ grid-template-columns:auto auto 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>VATD on Real Test Data: Proposals → Trajectories → Drone-Like Selection</h1>
    <p>Sequence <b>{manifest['sequence']}</b>, rendered from existing VATD inference outputs. Green tracks are selected by the VATD score threshold, not manually chosen.</p>
  </header>
  <main>
    <section class="viewer">
      <img id="frameImage" alt="VATD trajectory frame" />
      <div class="controls">
        <button id="playBtn">Play</button>
        <button id="prevBtn">Previous</button>
        <input id="slider" type="range" min="0" max="{len(manifest['frames']) - 1}" value="0" />
        <button id="nextBtn">Next</button>
        <span id="frameText"></span>
      </div>
    </section>
    <section class="cards">
      <div class="card">Real Test Sequence<b>{manifest['sequence']}</b></div>
      <div class="card">Rendered Frames<b>{len(manifest['frames'])}</b></div>
      <div class="card">Candidate Tracks<b>{manifest['track_count']}</b></div>
      <div class="card">VATD Threshold<b>{manifest['selection_threshold']:.2f}</b></div>
    </section>
    <section class="card">
      <div class="legend">
        <span><i class="dot" style="background:#46dc50"></i>VATD selected: score ≥ {manifest['selection_threshold']:.2f}</span>
        <span><i class="dot" style="background:#ffc000"></i>Medium-score candidate</span>
        <span><i class="dot" style="background:#919191"></i>Low-score or unscored candidate</span>
      </div>
      <p class="note">The left panel shows real RGB frames. Polylines connect recent box centers. The zoom panel follows the highest-scoring active VATD track, and the list is ranked by VATD score. Ground truth is used only for offline evaluation and never controls the green selection.</p>
    </section>
  </main>
  <script>
    const manifest = {manifest_json};
    const image = document.getElementById('frameImage');
    const slider = document.getElementById('slider');
    const frameText = document.getElementById('frameText');
    const playBtn = document.getElementById('playBtn');
    let index = 0;
    let timer = null;
    function render() {{
      index = Math.max(0, Math.min(index, manifest.frames.length - 1));
      slider.value = index;
      const item = manifest.frames[index];
      image.src = item.image;
      frameText.textContent = `frame ${{item.frame_id}} · ${{index + 1}}/${{manifest.frames.length}}`;
    }}
    function stop() {{ if (timer) clearInterval(timer); timer = null; playBtn.textContent = 'Play'; }}
    function play() {{
      if (timer) {{ stop(); return; }}
      playBtn.textContent = 'Pause';
      timer = setInterval(() => {{ index = index + 1 >= manifest.frames.length ? 0 : index + 1; render(); }}, 80);
    }}
    playBtn.addEventListener('click', play);
    document.getElementById('prevBtn').addEventListener('click', () => {{ stop(); index -= 1; render(); }});
    document.getElementById('nextBtn').addEventListener('click', () => {{ stop(); index += 1; render(); }});
    slider.addEventListener('input', () => {{ stop(); index = Number(slider.value); render(); }});
    window.addEventListener('keydown', event => {{ if (event.key === 'ArrowLeft') {{ stop(); index -= 1; render(); }} if (event.key === 'ArrowRight') {{ stop(); index += 1; render(); }} if (event.key === ' ') {{ event.preventDefault(); play(); }} }});
    render();
  </script>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    tracks, rows_by_frame = load_tracks(args.tracklets, args.sequence, args.start_frame, args.end_frame)
    if not tracks:
        raise RuntimeError(f"No tracks found for sequence {args.sequence}")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    rgb_width = round(args.rgb_panel_width * args.display_scale)
    rgb_height = round(args.rgb_panel_height * args.display_scale)
    side_width = 410
    canvas_width = rgb_width + side_width
    canvas_height = max(rgb_height, 560)
    source_scale_x = rgb_width / 1920.0
    source_scale_y = rgb_height / 1080.0
    histories: dict[str, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=args.trail_length))
    frame_manifest: list[dict] = []

    for frame_id in range(args.start_frame, args.end_frame + 1):
        video_index = frame_id - args.video_frame_offset
        capture.set(cv2.CAP_PROP_POS_FRAMES, video_index)
        ok, source = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read video frame {video_index} for dataset frame {frame_id}")
        rgb = source[: args.rgb_panel_height, : args.rgb_panel_width]
        rgb = cv2.resize(rgb, (rgb_width, rgb_height), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((canvas_height, canvas_width, 3), (12, 16, 21), dtype=np.uint8)
        canvas[:rgb_height, :rgb_width] = rgb

        active = sorted(
            rows_by_frame.get(frame_id, []),
            key=lambda item: item["track"]["vatd_score"] if item["track"]["vatd_score"] is not None else -1.0,
            reverse=True,
        )[: args.max_active_tracks]
        for item in active:
            track = item["track"]
            row = item["row"]
            score = track["vatd_score"]
            color = color_for_score(score, args.selection_threshold)
            x1, y1, x2, y2 = row["bbox"]
            x1, x2 = int(round(x1 * source_scale_x)), int(round(x2 * source_scale_x))
            y1, y2 = int(round(y1 * source_scale_y)), int(round(y2 * source_scale_y))
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            histories[track["track_id"]].append(center)
            points = list(histories[track["track_id"]])
            if len(points) > 1:
                cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            thickness = 3 if score is not None and score >= args.selection_threshold else 1
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        cv2.line(canvas, (rgb_width, 0), (rgb_width, canvas_height), (42, 55, 69), 1)
        cv2.putText(canvas, f"{args.sequence}  frame {frame_id}", (rgb_width + 18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 245, 248), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"active candidates: {len(active)}", (rgb_width + 18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155, 170, 184), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"selected if VATD >= {args.selection_threshold:.2f}", (rgb_width + 18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 220, 80), 1, cv2.LINE_AA)

        if active:
            top = active[0]
            bbox = top["row"]["bbox"]
            x1, y1, x2, y2 = [int(round(value * factor)) for value, factor in zip(bbox, [source_scale_x, source_scale_y, source_scale_x, source_scale_y])]
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            half = 70
            crop = rgb[max(0, center_y - half) : min(rgb_height, center_y + half), max(0, center_x - half) : min(rgb_width, center_x + half)]
            if crop.size:
                zoom = cv2.resize(crop, (360, 260), interpolation=cv2.INTER_NEAREST)
                canvas[104:364, rgb_width + 24 : rgb_width + 384] = zoom
                cv2.rectangle(canvas, (rgb_width + 24, 104), (rgb_width + 384, 364), (70, 220, 80), 2)
                cv2.putText(canvas, "top VATD track zoom", (rgb_width + 34, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        list_y = 398
        cv2.putText(canvas, "Top active trajectories", (rgb_width + 18, list_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 240, 244), 2, cv2.LINE_AA)
        for rank, item in enumerate(active[:6], start=1):
            track = item["track"]
            score = track["vatd_score"]
            color = color_for_score(score, args.selection_threshold)
            score_text = "n/a" if score is None else f"{score:.3f}"
            selected = "SELECT" if score is not None and score >= args.selection_threshold else "keep watching"
            text = f"{rank}. {track['track_id']}  {score_text}  {selected}"
            cv2.putText(canvas, text, (rgb_width + 22, list_y + 28 * rank), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        output_name = f"frame_{frame_id:04d}.jpg"
        cv2.imwrite(str(frames_dir / output_name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        frame_manifest.append({"frame_id": frame_id, "image": f"frames/{output_name}"})

    capture.release()
    manifest = {
        "sequence": args.sequence,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "selection_threshold": args.selection_threshold,
        "track_count": len(tracks),
        "frames": frame_manifest,
        "tracklets": str(args.tracklets.resolve()),
        "video": str(args.video.resolve()),
        "video_frame_offset": args.video_frame_offset,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    build_html(args.out_dir, manifest)
    print(json.dumps({"out_dir": str(args.out_dir), "frames": len(frame_manifest), "tracks": len(tracks)}, indent=2))


if __name__ == "__main__":
    main()
