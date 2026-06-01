from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh NPS motion-boundary gallery posters from non-initial video frames.")
    p.add_argument("--site", default=r"C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_boundary_site")
    p.add_argument("--frame-index", type=int, default=45)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    site = Path(args.site)
    items = json.loads((site / "videos.json").read_text(encoding="utf-8"))
    for item in items:
        video_path = site / item["src"]
        poster_path = site / item["poster"]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"skip open_failed {video_path}")
            continue
        target = min(max(0, args.frame_index), max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"skip read_failed {video_path}")
            continue
        poster_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(poster_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print(f"poster {poster_path}")


if __name__ == "__main__":
    main()
