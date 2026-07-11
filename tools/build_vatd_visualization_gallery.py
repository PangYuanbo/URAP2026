from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an English gallery for VATD trajectory visualizations.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = []
    for manifest_path in sorted(args.root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = manifest.get("frames", [])
        if not frames:
            continue
        cases.append(
            {
                "directory": manifest_path.parent.name,
                "sequence": manifest["sequence"],
                "start": manifest["start_frame"],
                "end": manifest["end_frame"],
                "frames": len(frames),
                "tracks": manifest["track_count"],
                "threshold": manifest["selection_threshold"],
                "preview": frames[len(frames) // 2]["image"],
            }
        )
    cards = []
    for case in cases:
        directory = html.escape(case["directory"])
        cards.append(
            f"""
            <a class="case" href="{directory}/index.html">
              <img src="{directory}/{html.escape(case['preview'])}" alt="{html.escape(case['sequence'])} VATD preview" />
              <div class="content">
                <div class="topline"><h2>{html.escape(case['sequence'])}</h2><span>Open Player →</span></div>
                <p>Real YOLOMG test sequence · frames {case['start']}–{case['end']}</p>
                <div class="metrics">
                  <b>{case['frames']} frames</b><b>{case['tracks']} tracks</b><b>threshold {case['threshold']:.2f}</b>
                </div>
              </div>
            </a>
            """
        )
    page = f"""<!doctype html>
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
    .legend {{ display:flex; flex-wrap:wrap; gap:18px; color:var(--muted); margin-top:24px; }}
    .dot {{ width:11px; height:11px; border-radius:50%; display:inline-block; margin-right:7px; }}
    main {{ max-width:1380px; margin:auto; padding:10px 22px 60px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
    .case {{ display:block; overflow:hidden; border:1px solid var(--line); border-radius:14px; background:rgba(18,25,34,.92); color:var(--text); text-decoration:none; transition:.18s ease; }}
    .case:hover {{ transform:translateY(-3px); border-color:var(--green); box-shadow:0 16px 44px rgba(0,0,0,.38); }}
    .case img {{ width:100%; aspect-ratio:1370/560; object-fit:cover; display:block; background:#050709; }}
    .content {{ padding:17px 18px 19px; }}
    .topline {{ display:flex; align-items:center; justify-content:space-between; gap:12px; }}
    h2 {{ margin:0; font-size:25px; }}
    .topline span {{ color:var(--green); font-weight:700; }}
    .content p {{ color:var(--muted); margin:6px 0 15px; }}
    .metrics {{ display:flex; gap:8px; flex-wrap:wrap; }}
    .metrics b {{ padding:6px 9px; border:1px solid var(--line); border-radius:999px; background:#0d131a; font-size:12px; }}
    footer {{ max-width:1380px; margin:auto; padding:0 22px 45px; color:var(--muted); }}
    @media(max-width:900px) {{ main {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">Video-Action Tiny Drone Detector</div>
    <h1>VATD Real-Test Trajectory Selection</h1>
    <p>Six real YOLOMG test sequences rendered from existing VATD inference results. Each player shows raw candidate boxes, connected trajectory history, ranked VATD scores, and the tracks selected as drone-like.</p>
    <div class="legend">
      <span><i class="dot" style="background:#46dc50"></i>VATD-selected track</span>
      <span><i class="dot" style="background:#ffc000"></i>Medium-score candidate</span>
      <span><i class="dot" style="background:#919191"></i>Low-score candidate</span>
    </div>
  </header>
  <main>{''.join(cards)}</main>
  <footer>Selection is based on VATD scores only. Ground-truth labels were used for offline evaluation, not for rendering the green decision.</footer>
</body>
</html>
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(json.dumps({"out": str(args.out), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
