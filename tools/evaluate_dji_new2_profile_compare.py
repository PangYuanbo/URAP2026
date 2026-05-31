from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_gt(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("class") != "drone":
                continue
            frame_id = int(float(row["frame_id"]))
            bbox = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
            rows.append(
                {
                    "frame_id": frame_id,
                    "bbox": bbox,
                    "tag": row.get("tag", ""),
                    "frame_path": row.get("frame_path", ""),
                    "source_video": row.get("source_video", ""),
                }
            )
    return rows


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / max(area, 1e-6))


def _center_distance(a: list[float], b: list[float]) -> float:
    return float(math.hypot((a[0] + a[2] - b[0] - b[2]) / 2.0, (a[1] + a[3] - b[1] - b[3]) / 2.0))


def _best_for_gt(gt: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[float, float, dict[str, Any] | None]:
    frame_rows = [r for r in rows if int(r.get("frame_id", -1)) == gt["frame_id"]]
    if not frame_rows:
        return 0.0, float("inf"), None
    best_iou_row = max(frame_rows, key=lambda r: _iou(r["bbox"], gt["bbox"]))
    best_iou = _iou(best_iou_row["bbox"], gt["bbox"])
    best_center_row = min(frame_rows, key=lambda r: _center_distance(r["bbox"], gt["bbox"]))
    best_center = _center_distance(best_center_row["bbox"], gt["bbox"])
    return best_iou, best_center, best_iou_row


def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _run_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _run_frame_count(run_dir: Path, rows: list[dict[str, Any]], fallback: int, meta: dict[str, Any] | None = None) -> int:
    meta = meta if meta is not None else _run_meta(run_dir)
    for key in ("evaluated_frames", "max_frames"):
        value = meta.get(key)
        if value is not None and int(value) > 0:
            return int(value)
    row_frames = len({int(r.get("frame_id", -1)) for r in rows if int(r.get("frame_id", -1)) >= 0})
    return row_frames or fallback


def evaluate(root: Path, sequences: list[str], profiles: list[str], iou_threshold: float, center_threshold: float) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, float]] = {}
    for profile in profiles:
        aggregate[profile] = {
            "gt": 0,
            "candidate_iou_hits": 0,
            "candidate_center_hits": 0,
            "final_iou_hits": 0,
            "final_center_hits": 0,
            "raw_drone": 0,
            "final_drone": 0,
            "frames": 0,
            "transformer_rejected": 0,
        }
        for sequence in sequences:
            run_dir = root / profile / sequence
            gt_path = run_dir / "frame_annotations.csv"
            if not gt_path.exists():
                raise FileNotFoundError(f"Missing GT file copied into run dir: {gt_path}")
            gt_rows = _load_gt(gt_path)
            raw_rows = _load_jsonl(run_dir / "predictions_raw.jsonl")
            final_rows = _load_jsonl(run_dir / "predictions.jsonl")
            meta = _run_meta(run_dir)
            frames = _run_frame_count(run_dir, final_rows, len(gt_rows), meta)
            if meta.get("stream") and meta.get("evaluated_frames"):
                frame_start = int(meta.get("frame_start", 0))
                frame_stride = int(meta.get("frame_stride", 1))
                sampled_frames = {frame_start + idx * frame_stride for idx in range(frames)}
                gt_rows = [gt for gt in gt_rows if int(gt["frame_id"]) in sampled_frames]
            raw_drone = [r for r in raw_rows if r.get("predicted_class") == "drone"]
            final_drone = [r for r in final_rows if r.get("predicted_class") == "drone"]
            candidate_iou_hits = 0
            candidate_center_hits = 0
            final_iou_hits = 0
            final_center_hits = 0
            for gt in gt_rows:
                cand_iou, cand_center, cand_best = _best_for_gt(gt, raw_rows)
                final_iou, final_center, final_best = _best_for_gt(gt, final_drone)
                cand_iou_hit = cand_iou >= iou_threshold
                cand_center_hit = cand_center <= center_threshold
                final_iou_hit = final_iou >= iou_threshold
                final_center_hit = final_center <= center_threshold
                candidate_iou_hits += int(cand_iou_hit)
                candidate_center_hits += int(cand_center_hit)
                final_iou_hits += int(final_iou_hit)
                final_center_hits += int(final_center_hit)
                frame_rows.append(
                    {
                        "profile": profile,
                        "sequence": sequence,
                        "frame_id": gt["frame_id"],
                        "tag": gt["tag"],
                        "candidate_best_iou": cand_iou,
                        "candidate_best_center": cand_center,
                        "candidate_hit_iou": cand_iou_hit,
                        "candidate_hit_center": cand_center_hit,
                        "candidate_best_source": cand_best.get("source") if cand_best else "",
                        "candidate_best_predicted_class": cand_best.get("predicted_class") if cand_best else "",
                        "candidate_best_score": cand_best.get("final_drone_score") if cand_best else "",
                        "final_best_iou": final_iou,
                        "final_best_center": final_center,
                        "final_hit_iou": final_iou_hit,
                        "final_hit_center": final_center_hit,
                        "final_best_source": final_best.get("source") if final_best else "",
                        "final_best_score": final_best.get("final_drone_score") if final_best else "",
                        "final_best_cause": final_best.get("diagnostic_cause") if final_best else "",
                    }
                )
            transformer_rejected = sum(1 for r in final_rows if "transformer_rejected" in str(r.get("diagnostic_cause")))
            row = {
                "profile": profile,
                "sequence": sequence,
                "gt": len(gt_rows),
                "frames": frames,
                "candidate_recall_iou": candidate_iou_hits / max(1, len(gt_rows)),
                "candidate_recall_center": candidate_center_hits / max(1, len(gt_rows)),
                "final_recall_iou": final_iou_hits / max(1, len(gt_rows)),
                "final_recall_center": final_center_hits / max(1, len(gt_rows)),
                "raw_drone": len(raw_drone),
                "final_drone": len(final_drone),
                "final_drone_per_frame": len(final_drone) / max(1, frames),
                "approx_final_precision_iou": final_iou_hits / max(1, len(final_drone)),
                "approx_final_fp_iou": max(0, len(final_drone) - final_iou_hits),
                "transformer_rejected": transformer_rejected,
                "final_source_counts": json.dumps(_counter(final_drone, "source"), ensure_ascii=False),
                "final_cause_counts": json.dumps(_counter(final_rows, "diagnostic_cause"), ensure_ascii=False),
            }
            summary_rows.append(row)
            agg = aggregate[profile]
            agg["gt"] += len(gt_rows)
            agg["candidate_iou_hits"] += candidate_iou_hits
            agg["candidate_center_hits"] += candidate_center_hits
            agg["final_iou_hits"] += final_iou_hits
            agg["final_center_hits"] += final_center_hits
            agg["raw_drone"] += len(raw_drone)
            agg["final_drone"] += len(final_drone)
            agg["frames"] += frames
            agg["transformer_rejected"] += transformer_rejected
    aggregate_rows = []
    for profile, agg in aggregate.items():
        aggregate_rows.append(
            {
                "profile": profile,
                "sequence": "ALL",
                "gt": int(agg["gt"]),
                "frames": int(agg["frames"]),
                "candidate_recall_iou": agg["candidate_iou_hits"] / max(1.0, agg["gt"]),
                "candidate_recall_center": agg["candidate_center_hits"] / max(1.0, agg["gt"]),
                "final_recall_iou": agg["final_iou_hits"] / max(1.0, agg["gt"]),
                "final_recall_center": agg["final_center_hits"] / max(1.0, agg["gt"]),
                "raw_drone": int(agg["raw_drone"]),
                "final_drone": int(agg["final_drone"]),
                "final_drone_per_frame": agg["final_drone"] / max(1.0, agg["frames"]),
                "approx_final_precision_iou": agg["final_iou_hits"] / max(1.0, agg["final_drone"]),
                "approx_final_fp_iou": max(0, int(agg["final_drone"] - agg["final_iou_hits"])),
                "transformer_rejected": int(agg["transformer_rejected"]),
            }
        )
    return {"summary_rows": summary_rows, "aggregate_rows": aggregate_rows, "frame_rows": frame_rows}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--center-threshold", type=float, default=16.0)
    args = parser.parse_args()
    root = Path(args.root)
    result = evaluate(root, args.sequences, args.profiles, args.iou_threshold, args.center_threshold)
    _write_csv(root / "summary.csv", result["aggregate_rows"] + result["summary_rows"])
    _write_csv(root / "frame_timeline.csv", result["frame_rows"])
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate_rows"], indent=2))


if __name__ == "__main__":
    main()
