from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("labels", 0)) > 0:
            rows.append(row)
    return rows


def read_frame(source: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read {source} frame {frame_idx}")
    return frame


def draw_box(img: np.ndarray, box: list[int], color: tuple[int, int, int], label: str, scale: float = 1.0) -> None:
    x1, y1, x2, y2 = [int(round(v * scale)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def fit_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = height / float(h)
    return cv2.resize(img, (int(round(w * scale)), height), interpolation=cv2.INTER_AREA)


def make_panel(row: dict, out_path: Path) -> None:
    frame = read_frame(Path(row["source"]), int(row["frame"]))
    crop = cv2.imread(row["image"], cv2.IMREAD_COLOR)
    if crop is None:
        raise RuntimeError(f"Failed to read crop {row['image']}")

    full = frame.copy()
    for gt in row.get("gt_boxes_xyxy", []):
        draw_box(full, gt, (0, 255, 255), "GT")
    draw_box(full, row["candidate_xyxy"], (0, 220, 0), "motion candidate")
    draw_box(full, row["crop_xyxy"], (255, 255, 0), "ESOD crop")
    full_small = cv2.resize(full, (960, 540), interpolation=cv2.INTER_AREA)

    crop_panel = cv2.resize(crop, (540, 540), interpolation=cv2.INTER_CUBIC)
    label_path = Path(row["label"])
    label_text = label_path.read_text(encoding="utf-8", errors="ignore").strip().replace("\n", " | ")
    if label_text:
        cv2.putText(crop_panel, "enlarged ROI + YOLO label", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(crop_panel, label_text[:70], (18, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    header = np.full((64, 1500, 3), 28, dtype=np.uint8)
    title = f"{row['dataset'].upper()} {row['video']} frame {row['frame']} | motion score {float(row['motion_score']):.1f}"
    cv2.putText(header, title, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    body = np.concatenate([full_small, crop_panel], axis=1)
    panel = np.concatenate([header, body], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 94])


def make_contact_sheet(paths: list[Path], out_path: Path, thumb_w: int = 750) -> None:
    thumbs = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_w / float(w)
        thumbs.append(cv2.resize(img, (thumb_w, int(round(h * scale))), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    rows = []
    for i in range(0, len(thumbs), 2):
        row_imgs = thumbs[i : i + 2]
        if len(row_imgs) == 1:
            pad = np.full_like(row_imgs[0], 28)
            row_imgs.append(pad)
        rows.append(np.concatenate(row_imgs, axis=1))
    sheet = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("artifacts/motion_esod_rois_smoke3"))
    p.add_argument("--out", type=Path, default=Path("artifacts/motion_esod_rois_demos"))
    p.add_argument("--per-dataset", type=int, default=4)
    args = p.parse_args()

    panel_paths: list[Path] = []
    for dataset in ["nps", "ard100"]:
        manifests = sorted(args.root.glob(f"{dataset}_*_manifest.jsonl"))
        rows: list[dict] = []
        for manifest in manifests:
            rows.extend(read_manifest(manifest))
        for i, row in enumerate(rows[: args.per_dataset]):
            out_path = args.out / dataset / f"{dataset}_demo_{i + 1:02d}.jpg"
            make_panel(row, out_path)
            panel_paths.append(out_path)

    make_contact_sheet(panel_paths, args.out / "motion_esod_demo_contact_sheet.jpg")
    print(f"wrote {len(panel_paths)} panels to {args.out}")
    print(f"contact_sheet={args.out / 'motion_esod_demo_contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
