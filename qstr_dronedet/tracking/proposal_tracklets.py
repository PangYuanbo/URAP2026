from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qstr_dronedet.candidates.merge import bbox_iou, center_distance
from qstr_dronedet.tracking.tracklet_classifier import TRACKLET_FEATURES, _features, _load_gt_csv, _prob


@dataclass
class ProposalTrackletDatasetResult:
    csv_path: Path
    json_path: Path
    summary: dict[str, Any]


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in row.get("bbox", [0.0, 0.0, 0.0, 0.0]))


def _box_side(row: dict[str, Any]) -> float:
    x1, y1, x2, y2 = _bbox(row)
    return max(0.0, max(x2 - x1, y2 - y1))


def _row_score(row: dict[str, Any]) -> float:
    return max(float(row.get("objectness", 0.0)), float(row.get("final_drone_score", 0.0)))


def _source_has_detector(row: dict[str, Any]) -> bool:
    source = str(row.get("source", ""))
    return any(token in source for token in ("yolo", "fallback", "motion", "seed", "oracle"))


def _artifact_score(row: dict[str, Any]) -> float:
    return max(
        _prob(row, "crop_probs", "alignment_artifact"),
        _prob(row, "feature_probs", "alignment_artifact"),
        _prob(row, "temporal_probs", "alignment_artifact"),
        _prob(row, "final_probs", "alignment_artifact"),
    )


def _load_run_diagnostics(
    run_roots: list[str | Path],
    profile: str,
    diagnostics_name: str,
    max_frames: int | None,
) -> dict[str, list[dict[str, Any]]]:
    by_seq: dict[str, list[dict[str, Any]]] = {}
    for run_root in run_roots:
        profile_root = Path(run_root) / profile
        for seq_dir in sorted(p for p in profile_root.glob("*") if p.is_dir()):
            diag_path = seq_dir / diagnostics_name
            if not diag_path.exists() and diagnostics_name == "diagnostics_raw.jsonl":
                diag_path = seq_dir / "diagnostics.jsonl"
            if not diag_path.exists():
                continue
            seq = seq_dir.name
            for row in _load_jsonl(diag_path):
                frame_id = int(row.get("frame_id", -1))
                if max_frames is not None and frame_id >= max_frames:
                    continue
                if "bbox" not in row:
                    continue
                item = dict(row)
                item["seq"] = seq
                by_seq.setdefault(seq, []).append(item)
    return by_seq


def _relink_sequence_rows(
    rows: list[dict[str, Any]],
    max_gap: int,
    base_radius: float,
    radius_per_side: float,
    min_iou: float,
    min_score: float,
    detector_only: bool,
) -> list[list[dict[str, Any]]]:
    rows = [r for r in rows if _row_score(r) >= min_score and (not detector_only or _source_has_detector(r))]
    rows = sorted(rows, key=lambda r: (int(r.get("frame_id", 0)), -_row_score(r)))
    active: list[dict[str, Any]] = []
    tracklets: list[list[dict[str, Any]]] = []

    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_frame.setdefault(int(row.get("frame_id", 0)), []).append(row)

    next_id = 1
    for frame_id in sorted(by_frame):
        frame_rows = by_frame[frame_id]
        used_track_indices: set[int] = set()
        for row in frame_rows:
            best_idx = None
            best_score = -1e9
            side = _box_side(row)
            for idx, tr in enumerate(active):
                if idx in used_track_indices:
                    continue
                gap = frame_id - int(tr["last_frame"])
                if gap <= 0 or gap > max_gap:
                    continue
                last_box = tr["last_bbox"]
                dist = center_distance(last_box, _bbox(row))
                radius = base_radius + radius_per_side * max(side, max(last_box[2] - last_box[0], last_box[3] - last_box[1])) + 4.0 * max(0, gap - 1)
                ov = bbox_iou(last_box, _bbox(row))
                if dist > radius and ov < min_iou:
                    continue
                score = 2.0 * ov - dist / max(radius, 1e-6) + 0.1 * _row_score(row)
                if score > best_score:
                    best_idx = idx
                    best_score = score
            if best_idx is None:
                item = dict(row)
                item["proposal_track_id"] = f"proposal_{next_id}"
                item["track_id"] = item["proposal_track_id"]
                next_id += 1
                tracklets.append([item])
                active.append({"tracklet": tracklets[-1], "last_bbox": _bbox(item), "last_frame": frame_id})
            else:
                tr = active[best_idx]
                item = dict(row)
                item["proposal_track_id"] = tr["tracklet"][0]["proposal_track_id"]
                item["track_id"] = item["proposal_track_id"]
                tr["tracklet"].append(item)
                tr["last_bbox"] = _bbox(item)
                tr["last_frame"] = frame_id
                used_track_indices.add(best_idx)
        active = [tr for tr in active if frame_id - int(tr["last_frame"]) <= max_gap]
    return tracklets


def _label_tracklet(
    seq: str,
    rows: list[dict[str, Any]],
    gt_by_key: dict[tuple[str, int], list[tuple[float, float, float, float]]],
    iou_threshold: float,
    center_threshold: float,
) -> tuple[int, float, int]:
    best_iou = 0.0
    matched_frames = 0
    for row in rows:
        frame_id = int(row.get("frame_id", -1))
        box = _bbox(row)
        matched = False
        for gt in gt_by_key.get((seq, frame_id), []):
            ov = bbox_iou(box, gt)
            best_iou = max(best_iou, ov)
            if ov >= iou_threshold or center_distance(box, gt) <= center_threshold:
                matched = True
        matched_frames += int(matched)
    return int(matched_frames > 0), best_iou, matched_frames


def _bucket(label: int, rows: list[dict[str, Any]], hard_tiny_side: float, hard_low_score: float) -> str:
    scores = [_row_score(r) for r in rows]
    sides = [_box_side(r) for r in rows]
    artifact = max([_artifact_score(r) for r in rows] or [0.0])
    low_alignment = np.mean([float(r.get("alignment_quality", 1.0)) < 0.3 for r in rows]) if rows else 0.0
    source = "+".join(str(r.get("source", "")) for r in rows)
    if label:
        if (np.mean(sides) if sides else 0.0) <= hard_tiny_side or max(scores or [0.0]) <= hard_low_score:
            return "hard_tiny_positive"
        return "positive"
    if artifact >= 0.35 or ("motion" in source and low_alignment >= 0.5):
        return "motion_alignment_artifact"
    if max(scores or [0.0]) >= hard_low_score or "fallback" in source:
        return "high_score_detector_fp"
    return "easy_background"


def build_proposal_tracklet_dataset(
    run_roots: list[str | Path],
    gt_csv: str | Path,
    out: str | Path,
    profile: str = "hard_recovery",
    diagnostics_name: str = "diagnostics_raw.jsonl",
    max_frames: int | None = None,
    max_gap: int = 3,
    base_radius: float = 18.0,
    radius_per_side: float = 0.75,
    min_iou: float = 0.05,
    min_score: float = 0.0,
    detector_only: bool = False,
    min_tracklet_rows: int = 1,
    iou_threshold: float = 0.3,
    center_threshold: float = 24.0,
    hard_tiny_side: float = 24.0,
    hard_low_score: float = 0.25,
) -> ProposalTrackletDatasetResult:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_by_key = _load_gt_csv(gt_csv, max_frames=max_frames)
    rows_by_seq = _load_run_diagnostics(run_roots, profile, diagnostics_name, max_frames)
    csv_path = out_dir / "proposal_tracklets.csv"
    json_path = out_dir / "proposal_tracklets.jsonl"
    fields = ["seq", "track_id", "label", "bucket", "best_iou", "matched_frames", "num_rows_raw"] + TRACKLET_FEATURES
    counts: dict[str, int] = {}
    positives = 0
    total = 0

    with csv_path.open("w", encoding="utf-8", newline="") as f_csv, json_path.open("w", encoding="utf-8") as f_json:
        writer = csv.DictWriter(f_csv, fieldnames=fields)
        writer.writeheader()
        for seq, rows in sorted(rows_by_seq.items()):
            for tracklet in _relink_sequence_rows(rows, max_gap, base_radius, radius_per_side, min_iou, min_score, detector_only):
                if len(tracklet) < min_tracklet_rows:
                    continue
                tracklet = sorted(tracklet, key=lambda r: int(r.get("frame_id", 0)))
                label, best_iou, matched_frames = _label_tracklet(seq, tracklet, gt_by_key, iou_threshold, center_threshold)
                bucket = _bucket(label, tracklet, hard_tiny_side, hard_low_score)
                feats = _features(tracklet)
                track_id = str(tracklet[0].get("proposal_track_id", tracklet[0].get("track_id", "")))
                meta = {
                    "seq": seq,
                    "track_id": track_id,
                    "label": label,
                    "bucket": bucket,
                    "best_iou": best_iou,
                    "matched_frames": matched_frames,
                    "num_rows_raw": len(tracklet),
                    **feats,
                }
                writer.writerow(meta)
                f_json.write(json.dumps({"meta": meta, "rows": tracklet}, ensure_ascii=False) + "\n")
                counts[bucket] = counts.get(bucket, 0) + 1
                positives += int(label)
                total += 1

    summary = {
        "run_roots": [str(p) for p in run_roots],
        "gt_csv": str(gt_csv),
        "profile": profile,
        "diagnostics_name": diagnostics_name,
        "num_sequences": len(rows_by_seq),
        "num_tracklets": total,
        "positives": positives,
        "negatives": total - positives,
        "bucket_counts": counts,
        "params": {
            "max_frames": max_frames,
            "max_gap": max_gap,
            "base_radius": base_radius,
            "radius_per_side": radius_per_side,
            "min_iou": min_iou,
            "min_score": min_score,
            "detector_only": detector_only,
            "min_tracklet_rows": min_tracklet_rows,
            "iou_threshold": iou_threshold,
            "center_threshold": center_threshold,
            "hard_tiny_side": hard_tiny_side,
            "hard_low_score": hard_low_score,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ProposalTrackletDatasetResult(csv_path=csv_path, json_path=json_path, summary=summary)
