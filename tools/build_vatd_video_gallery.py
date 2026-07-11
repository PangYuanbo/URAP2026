from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VATD frame galleries into browser-ready H.264 video players.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument("--width", type=int, default=960)
    return parser.parse_args()


def encode_case(case_dir: Path, out_dir: Path, ffmpeg: Path, fps: float, target_width: int) -> dict:
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest.get("frames", [])
    if not frames:
        raise RuntimeError(f"No frames listed in {case_dir / 'manifest.json'}")
    first = cv2.imread(str(case_dir / frames[0]["image"]))
    if first is None:
        raise RuntimeError(f"Could not read first frame for {case_dir.name}")
    source_height, source_width = first.shape[:2]
    width = target_width
    height = round(source_height * target_width / source_width)
    if height % 2:
        height += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / "trajectory.temp.mp4"
    final_video = out_dir / "trajectory.mp4"
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create temporary video: {temporary}")
    try:
        for item in frames:
            image = cv2.imread(str(case_dir / item["image"]))
            if image is None:
                raise RuntimeError(f"Could not read {case_dir / item['image']}")
            writer.write(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    finally:
        writer.release()
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(temporary),
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            "-an",
            str(final_video),
        ],
        check=True,
    )
    temporary.unlink()
    poster_source = case_dir / frames[len(frames) // 2]["image"]
    shutil.copy2(poster_source, out_dir / "poster.jpg")
    output_manifest = {
        **manifest,
        "video": "trajectory.mp4",
        "poster": "poster.jpg",
        "playback_fps": fps,
        "duration_seconds": round(len(frames) / fps, 3),
    }
    (out_dir / "manifest.json").write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")
    (out_dir / "index.html").write_text(case_html(output_manifest), encoding="utf-8")
    return output_manifest


def case_html(manifest: dict) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VATD Real-Test Trajectory Selection — {html.escape(manifest['sequence'])}</title>
  <style>
    :root {{ color-scheme:dark; --bg:#090d12; --panel:#121922; --line:#273442; --text:#edf4f9; --muted:#91a0ae; --green:#46dc50; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,"Segoe UI",sans-serif; }}
    header,main {{ max-width:1450px; margin:auto; padding:18px 22px; }}
    header {{ display:flex; align-items:end; justify-content:space-between; gap:20px; }}
    h1 {{ margin:0; font-size:clamp(24px,4vw,44px); }}
    p {{ color:var(--muted); margin:5px 0 0; }}
    a {{ color:var(--green); font-weight:700; text-decoration:none; }}
    .player {{ overflow:hidden; border:1px solid var(--line); border-radius:14px; background:#050709; box-shadow:0 20px 60px rgba(0,0,0,.35); }}
    video {{ display:block; width:100%; height:auto; background:#050709; }}
    .stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-top:14px; }}
    .stat {{ border:1px solid var(--line); border-radius:10px; background:var(--panel); padding:13px; color:var(--muted); }}
    .stat b {{ display:block; color:var(--text); font-size:22px; margin-top:3px; }}
    .note {{ border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:13px; margin-top:12px; color:var(--muted); }}
    @media(max-width:760px) {{ header {{ align-items:start; flex-direction:column; }} .stats {{ grid-template-columns:1fr 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div><h1>{html.escape(manifest['sequence'])}</h1><p>VATD trajectory selection on real YOLOMG test data · frames {manifest['start_frame']}–{manifest['end_frame']}</p></div>
    <a href="../">← All Sequences</a>
  </header>
  <main>
    <div class="player"><video controls autoplay muted loop playsinline preload="metadata" poster="poster.jpg" src="trajectory.mp4"></video></div>
    <section class="stats">
      <div class="stat">Rendered Frames<b>{len(manifest['frames'])}</b></div>
      <div class="stat">Candidate Tracks<b>{manifest['track_count']}</b></div>
      <div class="stat">VATD Threshold<b>{manifest['selection_threshold']:.2f}</b></div>
      <div class="stat">Duration<b>{manifest['duration_seconds']:.1f}s</b></div>
    </section>
    <div class="note">Green tracks pass the VATD score threshold. Yellow tracks have intermediate scores, and gray tracks have low or unavailable VATD scores. The video is a replay of saved model inference outputs, not a manually annotated animation.</div>
  </main>
</body>
</html>
"""


def gallery_html(cases: list[dict]) -> str:
    cards = []
    for case in cases:
        directory = html.escape(case["directory"])
        cards.append(
            f"""
            <article class="case">
              <video controls muted loop playsinline preload="none" poster="{directory}/poster.jpg" src="{directory}/trajectory.mp4"></video>
              <div class="content">
                <div class="topline"><h2>{html.escape(case['sequence'])}</h2><a href="{directory}/">Open Player →</a></div>
                <p>Real YOLOMG test sequence · frames {case['start_frame']}–{case['end_frame']}</p>
                <div class="metrics"><b>{len(case['frames'])} frames</b><b>{case['track_count']} tracks</b><b>{case['duration_seconds']:.1f}s video</b></div>
              </div>
            </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VATD Real-Test Trajectory Gallery</title>
  <style>
    :root {{ color-scheme:dark; --bg:#090d12; --panel:#121922; --line:#273442; --text:#edf4f9; --muted:#91a0ae; --green:#46dc50; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at top,#15212c 0,#090d12 48%); color:var(--text); font:15px/1.5 system-ui,"Segoe UI",sans-serif; }}
    header {{ max-width:1380px; margin:auto; padding:52px 22px 28px; }}
    .eyebrow {{ color:var(--green); font-weight:750; letter-spacing:.14em; text-transform:uppercase; }}
    h1 {{ max-width:900px; margin:10px 0; font-size:clamp(34px,6vw,68px); line-height:1.02; }}
    header p {{ max-width:900px; color:var(--muted); font-size:18px; }}
    main {{ max-width:1380px; margin:auto; padding:10px 22px 60px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
    .case {{ overflow:hidden; border:1px solid var(--line); border-radius:14px; background:rgba(18,25,34,.92); box-shadow:0 12px 34px rgba(0,0,0,.22); }}
    .case video {{ width:100%; aspect-ratio:1370/560; object-fit:cover; display:block; background:#050709; }}
    .content {{ padding:17px 18px 19px; }}
    .topline {{ display:flex; align-items:center; justify-content:space-between; gap:12px; }}
    h2 {{ margin:0; font-size:25px; }}
    a {{ color:var(--green); font-weight:700; text-decoration:none; }}
    .content p {{ color:var(--muted); margin:6px 0 15px; }}
    .metrics {{ display:flex; gap:8px; flex-wrap:wrap; }}
    .metrics b {{ padding:6px 9px; border:1px solid var(--line); border-radius:999px; background:#0d131a; font-size:12px; }}
    footer {{ max-width:1380px; margin:auto; padding:0 22px 45px; color:var(--muted); }}
    @media(max-width:900px) {{ main {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header><div class="eyebrow">Video-Action Tiny Drone Detector</div><h1>VATD Real-Test Trajectory Selection</h1><p>Six real YOLOMG test sequences delivered as streaming H.264 videos. Every card is a native video player, or open a sequence for a larger view.</p></header>
  <main>{''.join(cards)}</main>
  <footer>Green selection is based on saved VATD scores. Ground truth does not control the rendered decision.</footer>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True)
    cases = []
    for manifest_path in sorted(args.source_root.glob("*/manifest.json")):
        case_dir = manifest_path.parent
        output = encode_case(case_dir, args.out_root / case_dir.name, args.ffmpeg, args.fps, args.width)
        output["directory"] = case_dir.name
        cases.append(output)
        print(f"encoded {output['sequence']}: {output['duration_seconds']:.1f}s")
    (args.out_root / "index.html").write_text(gallery_html(cases), encoding="utf-8")
    (args.out_root / "_headers").write_text(
        "/*.mp4\n  Cache-Control: public, max-age=31536000, immutable\n  Content-Type: video/mp4\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_root": str(args.out_root), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
