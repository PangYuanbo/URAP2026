import csv
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np


REPO = Path(r"C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dualdetector import Yolov5Detector, draw_predictions  # noqa: E402


VIDEO_PATH = Path(r"D:\URAP_datasets\ARD100\test_videos\phantom97.mp4")
RGB_DIR = Path(r"D:\URAP_datasets\ARD100_YOLOMG\images\test")
MASK_DIR = Path(r"D:\URAP_datasets\ARD100_YOLOMG\images2\test")
WEIGHTS = Path(r"C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\train\yolomg_ard100_e50_b4_img1280_20260221_181641\weights\best.pt")
OUT_DIR = Path(r"C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\doc\yolomg_phantom97_50s_55s")

START_SEC = 50.0
END_SEC = 55.0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_dir = OUT_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_idx = int(math.floor(START_SEC * fps)) + 1
    end_idx = int(math.floor(END_SEC * fps))

    detector = Yolov5Detector(weights=str(WEIGHTS))

    rows = []
    for idx in range(start_idx, end_idx + 1):
        stem = f"phantom97_{idx:04d}"
        rgb_path = RGB_DIR / f"{stem}.jpg"
        mask_path = MASK_DIR / f"{stem}.jpg"
        if not rgb_path.exists() or not mask_path.exists():
            rows.append({
                "frame_idx": idx,
                "time_sec": idx / fps,
                "num_det": 0,
                "top_score": "",
                "boxes": "",
                "status": "missing_rgb_or_mask",
            })
            continue

        img1 = cv2.imread(str(rgb_path))
        img2 = cv2.imread(str(mask_path))
        labels, scores, boxes = detector.run(img1, img2, conf_thres=0.001, iou_thres=0.4, classes=None)
        top_score = max(scores) if scores else 0.0

        annotated = img1.copy()
        for label, score, box in zip(labels, scores, boxes):
            annotated = draw_predictions(annotated, label, score, box)
        cv2.putText(
            annotated,
            f"{stem}  t={idx/fps:.2f}s  top={top_score:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(frames_dir / f"{stem}.jpg"), annotated)

        rows.append({
            "frame_idx": idx,
            "time_sec": f"{idx / fps:.4f}",
            "num_det": len(scores),
            "top_score": f"{top_score:.6f}",
            "boxes": "; ".join(
                [f"{label}:{score:.4f}:{list(map(int, box.tolist()))}" for label, score, box in zip(labels, scores, boxes)]
            ),
            "status": "ok",
        })

    csv_path = OUT_DIR / "scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "time_sec", "num_det", "top_score", "boxes", "status"])
        writer.writeheader()
        writer.writerows(rows)

    valid_scores = [float(r["top_score"]) for r in rows if r["status"] == "ok" and r["top_score"] != ""]
    nonzero_scores = [s for s in valid_scores if s > 0]

    summary = [
        f"video={VIDEO_PATH}",
        f"weights={WEIGHTS}",
        f"fps={fps}",
        f"total_frames={total_frames}",
        f"segment_start_sec={START_SEC}",
        f"segment_end_sec={END_SEC}",
        f"frame_start={start_idx}",
        f"frame_end={end_idx}",
        f"num_segment_frames={len(rows)}",
        f"num_valid_frames={len(valid_scores)}",
        f"num_frames_with_detection={len(nonzero_scores)}",
        f"max_top_score={(max(valid_scores) if valid_scores else 0):.6f}",
        f"mean_top_score_all={(sum(valid_scores) / len(valid_scores) if valid_scores else 0):.6f}",
        f"mean_top_score_nonzero={(sum(nonzero_scores) / len(nonzero_scores) if nonzero_scores else 0):.6f}",
        f"csv={csv_path}",
        f"frames_dir={frames_dir}",
    ]
    (OUT_DIR / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))


if __name__ == "__main__":
    main()
