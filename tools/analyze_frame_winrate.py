from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_frame_key_from_name(path: str) -> tuple[str, int] | None:
    name = Path(path).stem
    m = re.match(r"(Clip_\d{3})_(\d{6})_roi\d+", name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def load_predictions(pred_labels: Path, manifest_rows: list[dict[str, Any]]) -> dict[str, list[list[float]]]:
    by_image: dict[str, list[list[float]]] = {}
    row_by_stem = {Path(r["image"]).stem: r for r in manifest_rows}
    for txt in pred_labels.rglob("*.txt"):
        stem = txt.stem
        row = row_by_stem.get(stem)
        if row is None:
            continue
        crop = row["crop_xyxy"]
        cw = max(1.0, float(crop[2] - crop[0]))
        ch = max(1.0, float(crop[3] - crop[1]))
        boxes: list[list[float]] = []
        for raw in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = raw.split()
            if len(parts) < 5:
                continue
            vals = [float(x) for x in parts]
            _, xc, yc, bw, bh = vals[:5]
            conf = vals[5] if len(vals) >= 6 else 1.0
            x1 = crop[0] + (xc - bw / 2.0) * cw
            y1 = crop[1] + (yc - bh / 2.0) * ch
            x2 = crop[0] + (xc + bw / 2.0) * cw
            y2 = crop[1] + (yc + bh / 2.0) * ch
            boxes.append([x1, y1, x2, y2, conf])
        by_image[stem] = boxes
    return by_image


def analyze(manifest_paths: list[Path], pred_labels: Path | None, iou_thr: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in manifest_paths:
        for line in manifest.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    preds = load_predictions(pred_labels, rows) if pred_labels else {}
    frames: dict[tuple[str, int], dict[str, Any]] = defaultdict(lambda: {
        "dataset": "",
        "video": "",
        "frame": 0,
        "gt_boxes": [],
        "roi_count": 0,
        "positive_roi_count": 0,
        "max_motion_score": 0.0,
        "pred_count": 0,
        "max_pred_conf": 0.0,
        "best_iou": 0.0,
    })

    for row in rows:
        key = (row["video"], int(row["frame"]))
        f = frames[key]
        f["dataset"] = row["dataset"]
        f["video"] = row["video"]
        f["frame"] = int(row["frame"])
        f["roi_count"] += 1
        f["positive_roi_count"] += int(row.get("labels", 0) > 0)
        f["max_motion_score"] = max(float(f["max_motion_score"]), float(row.get("motion_score", 0.0)))
        if not f["gt_boxes"]:
            f["gt_boxes"] = row.get("gt_boxes_xyxy", [])

        stem = Path(row["image"]).stem
        for pred in preds.get(stem, []):
            f["pred_count"] += 1
            f["max_pred_conf"] = max(float(f["max_pred_conf"]), float(pred[4]))
            for gt in f["gt_boxes"]:
                f["best_iou"] = max(float(f["best_iou"]), iou(pred[:4], [float(x) for x in gt]))

    out: list[dict[str, Any]] = []
    for f in sorted(frames.values(), key=lambda x: (x["video"], int(x["frame"]))):
        gt_count = len(f["gt_boxes"])
        proposal_hit = int(gt_count > 0 and f["positive_roi_count"] > 0)
        pred_hit = int(gt_count > 0 and f["best_iou"] >= iou_thr)
        if preds:
            win_score = float(f["max_pred_conf"]) if pred_hit else 0.0
            status = "hit" if pred_hit else ("miss" if gt_count else "background")
        else:
            win_score = min(1.0, f["positive_roi_count"] / max(1, gt_count)) if gt_count else 0.0
            status = "proposal_hit" if proposal_hit else ("proposal_miss" if gt_count else "background")
        out.append({
            "dataset": f["dataset"],
            "video": f["video"],
            "frame": f["frame"],
            "gt_count": gt_count,
            "roi_count": f["roi_count"],
            "positive_roi_count": f["positive_roi_count"],
            "proposal_hit": proposal_hit,
            "pred_count": f["pred_count"],
            "max_pred_conf": round(float(f["max_pred_conf"]), 6),
            "best_iou": round(float(f["best_iou"]), 6),
            "pred_hit": pred_hit,
            "win_score": round(float(win_score), 6),
            "status": status,
            "max_motion_score": round(float(f["max_motion_score"]), 6),
        })
    return out


def write_outputs(rows: list[dict[str, Any]], out_csv: Path, out_json: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "video", "frame", "gt_count", "roi_count", "positive_roi_count",
        "proposal_hit", "pred_count", "max_pred_conf", "best_iou", "pred_hit",
        "win_score", "status", "max_motion_score",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    gt_frames = [r for r in rows if r["gt_count"] > 0]
    pred_frames = [r for r in rows if r["pred_count"] > 0]
    by_video = {}
    for video in sorted({r["video"] for r in rows}):
        vr = [r for r in rows if r["video"] == video]
        vgt = [r for r in vr if r["gt_count"] > 0]
        by_video[video] = {
            "frames": len(vr),
            "gt_frames": len(vgt),
            "proposal_hits": sum(int(r["proposal_hit"]) for r in vgt),
            "proposal_frame_recall": (sum(int(r["proposal_hit"]) for r in vgt) / len(vgt)) if vgt else 0.0,
            "prediction_hits": sum(int(r["pred_hit"]) for r in vgt),
            "prediction_frame_recall": (sum(int(r["pred_hit"]) for r in vgt) / len(vgt)) if vgt else 0.0,
            "mean_win_score_on_gt_frames": (sum(float(r["win_score"]) for r in vgt) / len(vgt)) if vgt else 0.0,
        }

    summary = {
        "frames": len(rows),
        "gt_frames": len(gt_frames),
        "proposal_hits": sum(int(r["proposal_hit"]) for r in gt_frames),
        "proposal_frame_recall": (sum(int(r["proposal_hit"]) for r in gt_frames) / len(gt_frames)) if gt_frames else 0.0,
        "prediction_frames": len(pred_frames),
        "prediction_hits": sum(int(r["pred_hit"]) for r in gt_frames),
        "prediction_frame_recall": (sum(int(r["pred_hit"]) for r in gt_frames) / len(gt_frames)) if gt_frames else 0.0,
        "mean_win_score_on_gt_frames": (sum(float(r["win_score"]) for r in gt_frames) / len(gt_frames)) if gt_frames else 0.0,
        "csv": str(out_csv),
        "by_video": by_video,
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, action="append", required=True, help="One or more *_manifest.jsonl files.")
    p.add_argument("--pred-labels", type=Path, default=None, help="Optional YOLO/ESOD prediction label txt directory.")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()
    rows = analyze(args.manifest, args.pred_labels, args.iou)
    write_outputs(rows, args.out_csv, args.out_json)
    print(f"frames={len(rows)}")
    print(f"csv={args.out_csv}")
    print(f"summary={args.out_json}")


if __name__ == "__main__":
    main()
