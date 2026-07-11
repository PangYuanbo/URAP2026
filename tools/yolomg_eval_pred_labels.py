from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "URAP-UAV-to-UAV-Detection-and-Tracking" / "papers" / "YOLOMG"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.metrics import ap_per_class  # type: ignore


VIDEO_STEM_RE = re.compile(r"^(?P<video>.+)_(?P<frame>\d+)$")


def remap_stale_dataset_path(path: Path) -> Path:
    if path.exists():
        return path
    text = str(path)
    for old, new in (("D:\\URAP_datasets\\", "U:\\URAP_datasets\\"), ("D:/URAP_datasets/", "U:/URAP_datasets/")):
        if text.startswith(old):
            candidate = Path(new + text[len(old) :])
            if candidate.exists():
                return candidate
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved YOLO prediction labels with YOLOMG-style AP50/F1 metrics.")
    parser.add_argument("--images-list", type=Path, required=True)
    parser.add_argument("--pred-label-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    return parser.parse_args()


def parse_frame_identity(image_path: Path) -> tuple[str, int]:
    match = VIDEO_STEM_RE.match(image_path.stem)
    if not match:
        raise ValueError(f"Unable to parse video/frame from image stem: {image_path.stem}")
    return match.group("video"), int(match.group("frame"))


def yolo_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() in {"images", "images2"}:
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def read_image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[0] <= 0 or img.shape[1] <= 0:
        raise ValueError(f"could not read image: {path}")
    return int(img.shape[1]), int(img.shape[0])


def xywhn_to_xyxy(cx: float, cy: float, bw: float, bh: float, width: int, height: int) -> list[float]:
    x = cx * width
    y = cy * height
    w = bw * width
    h = bh * height
    return [max(0.0, x - w / 2.0), max(0.0, y - h / 2.0), min(float(width), x + w / 2.0), min(float(height), y + h / 2.0)]


def load_gt(image_path: Path, width: int, height: int) -> np.ndarray:
    path = yolo_label_path(image_path)
    rows: list[list[float]] = []
    if not path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = float(parts[0])
        box = xywhn_to_xyxy(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), width, height)
        rows.append([cls, *box])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def load_pred(label_dir: Path, image_path: Path, width: int, height: int, conf_thres: float) -> np.ndarray:
    path = label_dir / f"{image_path.stem}.txt"
    rows: list[list[float]] = []
    if not path.exists():
        return np.zeros((0, 6), dtype=np.float32)
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        conf = float(parts[5]) if len(parts) >= 6 else 1.0
        if conf < conf_thres:
            continue
        cls = float(parts[0])
        box = xywhn_to_xyxy(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), width, height)
        rows.append([*box, conf, cls])
    rows.sort(key=lambda row: row[4], reverse=True)
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 6), dtype=np.float32)


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def process_batch_numpy(pred: np.ndarray, labels: np.ndarray, match_iou: float) -> np.ndarray:
    correct = np.zeros((pred.shape[0], 1), dtype=bool)
    if pred.shape[0] == 0 or labels.shape[0] == 0:
        return correct
    iou = box_iou_matrix(labels[:, 1:5], pred[:, :4])
    class_match = labels[:, 0:1] == pred[:, 5][None, :]
    candidates = np.argwhere((iou >= match_iou) & class_match)
    if candidates.size == 0:
        return correct
    matches = [(int(label_i), int(pred_i), float(iou[label_i, pred_i])) for label_i, pred_i in candidates]
    matches.sort(key=lambda item: item[2], reverse=True)
    used_labels: set[int] = set()
    used_preds: set[int] = set()
    for label_i, pred_i, _ in matches:
        if label_i in used_labels or pred_i in used_preds:
            continue
        used_labels.add(label_i)
        used_preds.add(pred_i)
        correct[pred_i, 0] = True
    return correct


def metric_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_ap50(correct: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray) -> float:
    if target_cls.size == 0 or conf.size == 0:
        return 0.0
    _, _, _, _, _, ap, _ = ap_per_class(correct, conf, pred_cls, target_cls, plot=False, names={0: "target"})
    return float(ap[:, 0].mean()) if ap.size else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if (args.image_width is None) != (args.image_height is None):
        raise ValueError("--image-width and --image-height must be provided together")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_frame_dir = args.out_dir / "per_frame"
    per_frame_dir.mkdir(parents=True, exist_ok=True)
    images = [remap_stale_dataset_path(Path(line.strip())) for line in args.images_list.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

    per_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image_index, image_path in enumerate(images, start=1):
        video, frame_index = parse_frame_identity(image_path)
        if args.image_width is not None and args.image_height is not None:
            width, height = int(args.image_width), int(args.image_height)
        else:
            width, height = read_image_size(image_path)
        labels = load_gt(image_path, width, height)
        pred = load_pred(args.pred_label_dir, image_path, width, height, args.conf_thres)
        correct = process_batch_numpy(pred, labels, args.match_iou)
        tp = int(correct[:, 0].sum()) if pred.shape[0] else 0
        fp = int(pred.shape[0] - tp)
        fn = int(labels.shape[0] - tp)
        metrics = metric_from_counts(tp, fp, fn)
        conf = pred[:, 4].astype(np.float32) if pred.shape[0] else np.zeros((0,), dtype=np.float32)
        matched_confidence = float(conf[correct[:, 0]].max()) if correct.size and correct[:, 0].any() else 0.0
        per_video[video].append(
            {
                "video": video,
                "frame_index": frame_index,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "gt_count": int(labels.shape[0]),
                "pred_count": int(pred.shape[0]),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "matched_confidence": matched_confidence,
                "confidence_mean": float(conf.mean()) if conf.size else 0.0,
                "image_path": str(image_path),
                "_correct": correct,
                "_conf": conf,
                "_pred_cls": pred[:, 5].astype(np.float32) if pred.shape[0] else np.zeros((0,), dtype=np.float32),
                "_target_cls": labels[:, 0].astype(np.float32) if labels.shape[0] else np.zeros((0,), dtype=np.float32),
            }
        )
        if image_index == 1 or image_index % 5000 == 0 or image_index == len(images):
            print(
                json.dumps(
                    {
                        "kind": "yolomg_eval_pred_labels_progress",
                        "images_done": image_index,
                        "images_total": len(images),
                        "videos_seen": len(per_video),
                    }
                ),
                flush=True,
            )

    video_summaries: list[dict[str, Any]] = []
    for video, rows in sorted(per_video.items()):
        rows = sorted(rows, key=lambda row: int(row["frame_index"]))
        public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
        write_csv(
            per_frame_dir / f"{video}_per_frame.csv",
            public_rows,
            ["video", "frame_index", "tp", "fp", "fn", "gt_count", "pred_count", "precision", "recall", "f1", "matched_confidence", "confidence_mean", "image_path"],
        )
        tp = int(sum(row["tp"] for row in rows))
        fp = int(sum(row["fp"] for row in rows))
        fn = int(sum(row["fn"] for row in rows))
        metrics = metric_from_counts(tp, fp, fn)
        ap50 = compute_ap50(
            np.concatenate([row["_correct"] for row in rows], axis=0) if rows else np.zeros((0, 1), dtype=bool),
            np.concatenate([row["_conf"] for row in rows], axis=0) if rows else np.zeros((0,), dtype=np.float32),
            np.concatenate([row["_pred_cls"] for row in rows], axis=0) if rows else np.zeros((0,), dtype=np.float32),
            np.concatenate([row["_target_cls"] for row in rows], axis=0) if rows else np.zeros((0,), dtype=np.float32),
        )
        video_summaries.append({"video": video, "frames": len(rows), "tp": tp, "fp": fp, "fn": fn, "ap50": ap50, **metrics})

    total_frames = sum(int(row["frames"]) for row in video_summaries)
    weighted = {
        "clips": len(video_summaries),
        "frames": total_frames,
        "weighted_ap50": sum(float(row["ap50"]) * int(row["frames"]) for row in video_summaries) / max(total_frames, 1),
        "weighted_f1": sum(float(row["f1"]) * int(row["frames"]) for row in video_summaries) / max(total_frames, 1),
        "weighted_recall": sum(float(row["recall"]) * int(row["frames"]) for row in video_summaries) / max(total_frames, 1),
        "weighted_precision": sum(float(row["precision"]) * int(row["frames"]) for row in video_summaries) / max(total_frames, 1),
        "tp": int(sum(row["tp"] for row in video_summaries)),
        "fp": int(sum(row["fp"] for row in video_summaries)),
        "fn": int(sum(row["fn"] for row in video_summaries)),
    }
    weighted.update(metric_from_counts(weighted["tp"], weighted["fp"], weighted["fn"]))

    write_csv(args.out_dir / "yolomg_pred_label_summary.csv", video_summaries, ["video", "frames", "tp", "fp", "fn", "ap50", "precision", "recall", "f1"])
    manifest = {
        "images_list": str(args.images_list),
        "pred_label_dir": str(args.pred_label_dir),
        "conf_thres": args.conf_thres,
        "match_iou": args.match_iou,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "summary": weighted,
        "per_video_csv": str(args.out_dir / "yolomg_pred_label_summary.csv"),
        "per_frame_dir": str(per_frame_dir),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
