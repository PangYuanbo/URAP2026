from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_RE = re.compile(r"^Clip_(?P<video>\d+)_(?P<frame>\d+)$")


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class Counters:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tp_small: int = 0
    fn_small: int = 0
    tp_medium: int = 0
    fn_medium: int = 0
    tp_large: int = 0
    fn_large: int = 0

    def as_dict(self, frames: int) -> dict[str, float]:
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        return {
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(self.fn),
            "precision": float(precision),
            "recall": float(recall),
            "recall_small": float(self.tp_small / max(1, self.tp_small + self.fn_small)),
            "recall_medium": float(self.tp_medium / max(1, self.tp_medium + self.fn_medium)),
            "recall_large": float(self.tp_large / max(1, self.tp_large + self.fn_large)),
            "fp_per_frame": float(self.fp / max(1, frames)),
        }


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    return inter / max(1e-9, a.area() + b.area() - inter)


def parse_li_tetc_line(line: str) -> tuple[int, list[Box]] | None:
    line = line.strip()
    if not line.startswith("time_layer:"):
        return None
    parts = line.split()
    try:
        frame_id = int(parts[1])
    except (IndexError, ValueError):
        return None
    if "detections:" not in parts:
        return frame_id, []
    tail = line.split("detections:", 1)[1].strip()
    boxes: list[Box] = []
    for chunk in tail.split("),"):
        chunk = chunk.strip().lstrip(",").strip().lstrip("(").rstrip(")").strip()
        if not chunk:
            continue
        vals = [v.strip() for v in chunk.split(",")]
        if len(vals) < 4:
            continue
        try:
            y1, x1, y2, x2 = (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
        except ValueError:
            continue
        if x2 > x1 and y2 > y1:
            boxes.append(Box(x1=x1, y1=y1, x2=x2, y2=y2))
    return frame_id, boxes


def load_gt(repo_root: Path, videos: list[int]) -> dict[tuple[int, int], list[Box]]:
    out: dict[tuple[int, int], list[Box]] = {}
    for video_id in videos:
        path = repo_root / "Data" / "Annotation_update_180925" / f"Video_{video_id}_gt.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parsed = parse_li_tetc_line(line)
            if parsed is None:
                continue
            frame_id, boxes = parsed
            out[(video_id, frame_id)] = boxes
    return out


def load_predictions(path: Path, videos: set[int]) -> dict[tuple[int, int], list[tuple[float, Box]]]:
    with path.open("rb") as f:
        rows = pickle.load(f)
    if not isinstance(rows, list):
        raise TypeError(f"{path}: expected list, got {type(rows)}")
    out: dict[tuple[int, int], list[tuple[float, Box]]] = {}
    skipped = 0
    for row in rows:
        if not isinstance(row, dict):
            skipped += 1
            continue
        match = IMAGE_RE.match(str(row.get("image_id") or ""))
        if match is None:
            skipped += 1
            continue
        video_id = int(match.group("video"))
        if video_id not in videos:
            continue
        frame_id = int(match.group("frame"))
        try:
            x, y, w, h = [float(v) for v in row["bbox"]]
            score = float(row.get("score", 0.0))
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if w <= 0.0 or h <= 0.0:
            skipped += 1
            continue
        out.setdefault((video_id, frame_id), []).append((score, Box(x1=x, y1=y, x2=x + w, y2=y + h)))
    return out


def size_bin(box: Box) -> str:
    area = box.area()
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def add_fn(counter: Counters, box: Box) -> None:
    counter.fn += 1
    bucket = size_bin(box)
    if bucket == "small":
        counter.fn_small += 1
    elif bucket == "medium":
        counter.fn_medium += 1
    else:
        counter.fn_large += 1


def add_tp(counter: Counters, box: Box) -> None:
    counter.tp += 1
    bucket = size_bin(box)
    if bucket == "small":
        counter.tp_small += 1
    elif bucket == "medium":
        counter.tp_medium += 1
    else:
        counter.tp_large += 1


def evaluate(
    gt: dict[tuple[int, int], list[Box]],
    preds: dict[tuple[int, int], list[tuple[float, Box]]],
    scores: list[float],
    iou_thr: float,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    frames = len(gt)
    for score_thr in scores:
        c = Counters()
        for key, gt_boxes in gt.items():
            frame_preds = [(s, b) for s, b in preds.get(key, []) if s >= score_thr]
            frame_preds.sort(key=lambda item: item[0], reverse=True)
            matched = [False] * len(gt_boxes)
            for _score, pred_box in frame_preds:
                best_iou = -1.0
                best_idx = -1
                for idx, gt_box in enumerate(gt_boxes):
                    if matched[idx]:
                        continue
                    value = iou(pred_box, gt_box)
                    if value > best_iou:
                        best_iou = value
                        best_idx = idx
                if best_iou >= iou_thr and best_idx >= 0:
                    matched[best_idx] = True
                    add_tp(c, gt_boxes[best_idx])
                else:
                    c.fp += 1
            for idx, gt_box in enumerate(gt_boxes):
                if not matched[idx]:
                    add_fn(c, gt_box)
        results[f"{score_thr:g}"] = c.as_dict(frames=frames)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate TransVisDrone COCO-style prediction PKL on Li-TETC NPS time_layer GT.")
    parser.add_argument("--repo-root", type=Path, default=Path("Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking"))
    parser.add_argument("--pred-pkl", type=Path, required=True)
    parser.add_argument("--videos", type=int, nargs="*", default=list(range(41, 51)))
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--scores", type=float, nargs="*", default=[0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--match-pt-pipeline-sampling", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--empty-stride", type=int, default=10)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    pred_path = args.pred_pkl.resolve()
    gt = load_gt(repo_root, args.videos)
    if args.match_pt_pipeline_sampling:
        frame_stride = max(1, int(args.frame_stride))
        empty_stride = max(1, int(args.empty_stride))
        gt = {
            key: boxes
            for key, boxes in gt.items()
            if ((key[1] - 1) % frame_stride == 0) and (boxes or (key[1] % empty_stride == 0))
        }
    preds = load_predictions(pred_path, set(args.videos))
    summary: dict[str, Any] = {
        "repo_root": str(repo_root),
        "pred_pkl": str(pred_path),
        "videos": args.videos,
        "iou": float(args.iou),
        "frames": len(gt),
        "gt_boxes": int(sum(len(v) for v in gt.values())),
        "prediction_frames": len(preds),
        "prediction_boxes": int(sum(len(v) for v in preds.values())),
        "scores": evaluate(gt, preds, args.scores, float(args.iou)),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
