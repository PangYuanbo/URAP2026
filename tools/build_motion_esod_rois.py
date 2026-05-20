from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def parse_name_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def load_nps_annotations(path: Path) -> dict[int, list[Box]]:
    anns: dict[int, list[Box]] = {}
    if not path.exists():
        return anns
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        items = [x.strip() for x in raw.split(",") if x.strip()]
        if len(items) < 2:
            continue
        frame_id = int(items[0])
        n = int(items[1])
        vals = [int(float(x)) for x in items[2:]]
        boxes: list[Box] = []
        for i in range(min(n, len(vals) // 4)):
            x1, y1, x2, y2 = vals[i * 4 : i * 4 + 4]
            if x2 > x1 and y2 > y1:
                boxes.append(Box(x1, y1, x2, y2))
        anns[frame_id] = boxes
    return anns


def load_ard_xml(path: Path) -> list[Box]:
    if not path.exists():
        return []
    root = ET.parse(path).getroot()
    out: list[Box] = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        if b is None:
            continue
        x1 = int(float(b.findtext("xmin", "0")))
        y1 = int(float(b.findtext("ymin", "0")))
        x2 = int(float(b.findtext("xmax", "0")))
        y2 = int(float(b.findtext("ymax", "0")))
        if x2 > x1 and y2 > y1:
            out.append(Box(x1, y1, x2, y2))
    return out


def iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def nms_boxes(scored: list[tuple[Box, float]], thr: float) -> list[tuple[Box, float]]:
    kept: list[tuple[Box, float]] = []
    for box, score in sorted(scored, key=lambda x: x[1], reverse=True):
        if all(iou(box, k[0]) < thr for k in kept):
            kept.append((box, score))
    return kept


def motion_boundary(prev_gray: np.ndarray, cur_gray: np.ndarray, downscale: int) -> np.ndarray:
    h, w = cur_gray.shape[:2]
    ds = max(1, int(downscale))
    if ds > 1:
        size = (max(1, w // ds), max(1, h // ds))
        prev_small = cv2.resize(prev_gray, size, interpolation=cv2.INTER_AREA)
        cur_small = cv2.resize(cur_gray, size, interpolation=cv2.INTER_AREA)
    else:
        prev_small = prev_gray
        cur_small = cur_gray

    if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
        flow = cv2.optflow.DualTVL1OpticalFlow_create().calc(prev_small, cur_small, None)
    elif hasattr(cv2, "DualTVL1OpticalFlow_create"):
        flow = cv2.DualTVL1OpticalFlow_create().calc(prev_small, cur_small, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(prev_small, cur_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    fx = flow[..., 0].astype(np.float32)
    fy = flow[..., 1].astype(np.float32)
    ux, uy = np.gradient(fx)
    vx, vy = np.gradient(fy)
    mb = np.maximum(np.hypot(ux, uy), np.hypot(vx, vy))
    mb = cv2.normalize(mb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if ds > 1:
        mb = cv2.resize(mb, (w, h), interpolation=cv2.INTER_LINEAR)
    return mb


def candidate_boxes(
    mb: np.ndarray,
    *,
    percentile: float,
    min_value: int,
    dilate: int,
    ignore_border: int,
    min_area: int,
    max_area_frac: float,
    max_boxes: int,
) -> list[tuple[Box, float]]:
    thresh = max(float(min_value), float(np.percentile(mb, percentile)))
    mask = (mb >= thresh).astype(np.uint8) * 255
    b = max(0, int(ignore_border))
    if b:
        mask[:b, :] = 0
        mask[-b:, :] = 0
        mask[:, :b] = 0
        mask[:, -b:] = 0
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.dilate(mask, k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mb.shape[:2]
    max_area = int(h * w * max_area_frac)
    scored: list[tuple[Box, float]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        box = Box(x, y, x + bw, y + bh)
        if box.area < min_area or box.area > max_area:
            continue
        score = float(mb[y : y + bh, x : x + bw].mean())
        scored.append((box, score))
    return nms_boxes(scored, 0.25)[:max_boxes]


def esod_square_crop(box: Box, w: int, h: int, context: float, min_crop: int) -> Box:
    cx, cy = box.center
    side = max(box.x2 - box.x1, box.y2 - box.y1)
    side = max(int(math.ceil(side * context)), int(min_crop))
    side = min(side, max(w, h))
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x1 = max(0, min(max(0, w - side), x1))
    y1 = max(0, min(max(0, h - side), y1))
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    return Box(x1, y1, x2, y2)


def labels_for_crop(gt_boxes: list[Box], crop: Box, out_size: int, min_iou: float) -> list[str]:
    labels: list[str] = []
    cw = crop.x2 - crop.x1
    ch = crop.y2 - crop.y1
    if cw <= 0 or ch <= 0:
        return labels
    for gt in gt_boxes:
        cx, cy = gt.center
        inside = crop.x1 <= cx <= crop.x2 and crop.y1 <= cy <= crop.y2
        if not inside and iou(gt, crop) < min_iou:
            continue
        ix1 = max(gt.x1, crop.x1)
        iy1 = max(gt.y1, crop.y1)
        ix2 = min(gt.x2, crop.x2)
        iy2 = min(gt.y2, crop.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        x1 = (ix1 - crop.x1) / cw
        y1 = (iy1 - crop.y1) / ch
        x2 = (ix2 - crop.x1) / cw
        y2 = (iy2 - crop.y1) / ch
        xc = (x1 + x2) * 0.5
        yc = (y1 + y2) * 0.5
        bw = x2 - x1
        bh = y2 - y1
        if bw * out_size >= 1 and bh * out_size >= 1:
            labels.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return labels


def draw_overlay(frame: np.ndarray, candidates: list[tuple[Box, float]], gt_boxes: list[Box]) -> np.ndarray:
    out = frame.copy()
    for b in gt_boxes:
        cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 255), 2)
    for b, score in candidates:
        cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)
        cv2.putText(out, f"{score:.1f}", (b.x1, max(0, b.y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return out


def process_video(video_path: Path, video_name: str, gt_by_frame: dict[int, list[Box]], args: argparse.Namespace) -> dict[str, int]:
    image_dir = args.out / "images" / args.dataset / video_name
    label_dir = args.out / "labels" / args.dataset / video_name
    overlay_dir = args.out / "overlays" / args.dataset / video_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out / f"{args.dataset}_{video_name}_manifest.jsonl"
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    prev_gray: np.ndarray | None = None
    frame_idx = 0
    saved = 0
    candidates_total = 0
    hit_gt = 0
    gt_frames_labeled: set[int] = set()

    with manifest_path.open("w", encoding="utf-8") as mf:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if args.max_frames and frame_idx >= args.max_frames:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gt = gt_by_frame.get(frame_idx, [])
            if prev_gray is not None and frame_idx % args.stride == 0:
                mb = motion_boundary(prev_gray, gray, args.downscale)
                cands = candidate_boxes(
                    mb,
                    percentile=args.percentile,
                    min_value=args.min_motion,
                    dilate=args.dilate,
                    ignore_border=args.ignore_border,
                    min_area=args.min_candidate_area,
                    max_area_frac=args.max_candidate_area_frac,
                    max_boxes=args.max_boxes,
                )
                candidates_total += len(cands)
                if gt and any(iou(c[0], g) > 0 or (c[0].x1 <= g.center[0] <= c[0].x2 and c[0].y1 <= g.center[1] <= c[0].y2) for c in cands for g in gt):
                    hit_gt += 1
                if args.save_overlays and (args.overlay_limit <= 0 or frame_idx < args.overlay_limit):
                    cv2.imwrite(str(overlay_dir / f"{video_name}_{frame_idx:06d}.jpg"), draw_overlay(frame, cands, gt))
                h, w = frame.shape[:2]
                for roi_idx, (cand, score) in enumerate(cands):
                    crop_box = esod_square_crop(cand, w, h, args.context, args.min_crop)
                    crop = frame[crop_box.y1 : crop_box.y2, crop_box.x1 : crop_box.x2]
                    if crop.size == 0:
                        continue
                    crop_up = cv2.resize(crop, (args.out_size, args.out_size), interpolation=cv2.INTER_CUBIC)
                    stem = f"{video_name}_{frame_idx:06d}_roi{roi_idx:02d}"
                    img_path = image_dir / f"{stem}.jpg"
                    lab_path = label_dir / f"{stem}.txt"
                    cv2.imwrite(str(img_path), crop_up, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    labels = labels_for_crop(gt, crop_box, args.out_size, args.gt_iou)
                    if labels:
                        gt_frames_labeled.add(frame_idx)
                    lab_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
                    mf.write(json.dumps({
                        "dataset": args.dataset,
                        "video": video_name,
                        "frame": frame_idx,
                        "source": str(video_path),
                        "image": str(img_path),
                        "label": str(lab_path),
                        "candidate_xyxy": [cand.x1, cand.y1, cand.x2, cand.y2],
                        "crop_xyxy": [crop_box.x1, crop_box.y1, crop_box.x2, crop_box.y2],
                        "motion_score": score,
                        "gt_boxes_xyxy": [[g.x1, g.y1, g.x2, g.y2] for g in gt],
                        "labels": len(labels),
                    }, ensure_ascii=True) + "\n")
                    saved += 1
            prev_gray = gray
            frame_idx += 1

    cap.release()
    return {
        "frames_seen": frame_idx,
        "frames_total": total_frames,
        "candidates": candidates_total,
        "patches_saved": saved,
        "candidate_gt_frames_hit": hit_gt,
        "crop_gt_frames_labeled": len(gt_frames_labeled),
    }


def nps_jobs(args: argparse.Namespace) -> list[tuple[Path, str, dict[int, list[Box]]]]:
    jobs = []
    for vid in parse_int_list(args.videos):
        video = args.nps_root / "Videos" / f"Clip_{vid}.mov"
        ann = args.nps_annotations / f"Clip_{vid:03d}.txt"
        jobs.append((video, f"Clip_{vid:03d}", load_nps_annotations(ann)))
    return jobs


def ard_jobs(args: argparse.Namespace) -> list[tuple[Path, str, dict[int, list[Box]]]]:
    jobs = []
    for name in parse_name_list(args.videos):
        video = args.ard_root / f"{args.ard_split}_videos" / f"{name}.mp4"
        ann_dir = args.ard_yolomg_root / "annotations" / name
        gt_by_frame: dict[int, list[Box]] = {}
        for xml_path in ann_dir.glob(f"{name}_*.xml"):
            try:
                frame_id_1 = int(xml_path.stem.rsplit("_", 1)[1])
            except Exception:
                continue
            gt_by_frame[frame_id_1 - 1] = load_ard_xml(xml_path)
        jobs.append((video, name, gt_by_frame))
    return jobs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["nps", "ard100"], required=True)
    p.add_argument("--videos", required=True, help="NPS: '1,2,10-12'; ARD100: 'phantom09,phantom10'")
    p.add_argument("--out", type=Path, default=Path("artifacts/motion_esod_rois"))
    p.add_argument("--nps-root", type=Path, default=Path("Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data"))
    p.add_argument("--nps-annotations", type=Path, default=Path("datasets/Drone-Detection/annotations/NPS-Drones-Dataset"))
    p.add_argument("--ard-root", type=Path, default=Path("D:/URAP_datasets/ARD100"))
    p.add_argument("--ard-yolomg-root", type=Path, default=Path("D:/URAP_datasets/ARD100_YOLOMG"))
    p.add_argument("--ard-split", choices=["train", "test"], default="train")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--downscale", type=int, default=4)
    p.add_argument("--percentile", type=float, default=99.6)
    p.add_argument("--min-motion", type=int, default=25)
    p.add_argument("--dilate", type=int, default=17)
    p.add_argument("--ignore-border", type=int, default=48)
    p.add_argument("--min-candidate-area", type=int, default=16)
    p.add_argument("--max-candidate-area-frac", type=float, default=0.08)
    p.add_argument("--max-boxes", type=int, default=4)
    p.add_argument("--context", type=float, default=8.0)
    p.add_argument("--min-crop", type=int, default=256)
    p.add_argument("--out-size", type=int, default=640)
    p.add_argument("--gt-iou", type=float, default=0.02)
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--overlay-limit", type=int, default=200)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = nps_jobs(args) if args.dataset == "nps" else ard_jobs(args)
    summary = {"dataset": args.dataset, "jobs": []}
    for video_path, video_name, gt in jobs:
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        stats = process_video(video_path, video_name, gt, args)
        summary["jobs"].append({"video": video_name, "path": str(video_path), **stats})
        print(f"{video_name}: {stats}")

    all_images = sorted((args.out / "images" / args.dataset).glob("**/*.jpg"))
    positive_images = []
    for img in all_images:
        lab = Path(str(img).replace(f"{args.out}\\images", f"{args.out}\\labels").replace(f"{args.out}/images", f"{args.out}/labels")).with_suffix(".txt")
        if lab.exists() and lab.stat().st_size > 0:
            positive_images.append(img)
    all_list = args.out / f"{args.dataset}_all.txt"
    pos_list = args.out / f"{args.dataset}_positive.txt"
    all_list.write_text("\n".join(str(p) for p in all_images) + ("\n" if all_images else ""), encoding="utf-8")
    pos_list.write_text("\n".join(str(p) for p in positive_images) + ("\n" if positive_images else ""), encoding="utf-8")
    (args.out / f"{args.dataset}_esod.yaml").write_text(
        "train: " + str(all_list) + "\n"
        "val: " + str(pos_list if positive_images else all_list) + "\n"
        "nc: 1\n"
        "names: ['UAV']\n",
        encoding="utf-8",
    )
    summary["images"] = {
        "all": len(all_images),
        "positive": len(positive_images),
        "all_list": str(all_list),
        "positive_list": str(pos_list),
        "yaml": str(args.out / f"{args.dataset}_esod.yaml"),
    }
    (args.out / f"{args.dataset}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
