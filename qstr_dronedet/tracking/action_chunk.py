from __future__ import annotations

import json
from dataclasses import dataclass
from math import exp, isfinite, log
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EPS = 1e-6
LOG_SCALE_LIMIT = 6.0


@dataclass(frozen=True)
class ActionChunkSample:
    seq: str
    track_id: str
    anchor_frame: int
    past_boxes: np.ndarray
    past_scores: np.ndarray
    past_visible: np.ndarray
    future_actions: np.ndarray
    future_boxes: np.ndarray


@dataclass(frozen=True)
class ActionChunkDatasetResult:
    jsonl_path: Path
    summary: dict[str, Any]


def _row_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    value = row.get("bbox") or row.get("bbox_xyxy")
    if value is None:
        raise KeyError("row must contain bbox or bbox_xyxy")
    x1, y1, x2, y2 = value
    return float(x1), float(y1), float(x2), float(y2)


def _row_score(row: dict[str, Any]) -> float:
    return float(max(row.get("objectness", 0.0), row.get("final_drone_score", 0.0), row.get("score", 0.0)))


def _row_image_size(row: dict[str, Any]) -> tuple[int, int] | None:
    width = row.get("image_width")
    height = row.get("image_height")
    if width is None or height is None:
        size = row.get("image_size")
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            width, height = size[0], size[1]
    if width is None or height is None:
        return None
    width_i = int(float(width))
    height_i = int(float(height))
    if width_i <= 0 or height_i <= 0:
        return None
    return width_i, height_i


def xyxy_to_cxcywh(box: Iterable[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(EPS, x2 - x1)
    h = max(EPS, y2 - y1)
    return x1 + w / 2.0, y1 + h / 2.0, w, h


def cxcywh_to_xyxy(box: Iterable[float]) -> tuple[float, float, float, float]:
    cx, cy, w, h = [float(v) for v in box]
    w = max(EPS, w)
    h = max(EPS, h)
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def normalize_xyxy(box: Iterable[float], image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = image_size
    x1, y1, x2, y2 = [float(v) for v in box]
    return x1 / max(1, width), y1 / max(1, height), x2 / max(1, width), y2 / max(1, height)


def denormalize_xyxy(box: Iterable[float], image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = image_size
    x1, y1, x2, y2 = [float(v) for v in box]
    return x1 * width, y1 * height, x2 * width, y2 * height


def box_action(prev_box_xyxy: Iterable[float], next_box_xyxy: Iterable[float]) -> tuple[float, float, float, float]:
    prev = xyxy_to_cxcywh(prev_box_xyxy)
    nxt = xyxy_to_cxcywh(next_box_xyxy)
    return (
        nxt[0] - prev[0],
        nxt[1] - prev[1],
        log(max(EPS, nxt[2]) / max(EPS, prev[2])),
        log(max(EPS, nxt[3]) / max(EPS, prev[3])),
    )


def apply_box_action(prev_box_xyxy: Iterable[float], action: Iterable[float]) -> tuple[float, float, float, float]:
    cx, cy, w, h = xyxy_to_cxcywh(prev_box_xyxy)
    dx, dy, dlogw, dlogh = [float(v) for v in action]
    dx = dx if isfinite(dx) else 0.0
    dy = dy if isfinite(dy) else 0.0
    dlogw = min(LOG_SCALE_LIMIT, max(-LOG_SCALE_LIMIT, dlogw if isfinite(dlogw) else 0.0))
    dlogh = min(LOG_SCALE_LIMIT, max(-LOG_SCALE_LIMIT, dlogh if isfinite(dlogh) else 0.0))
    return cxcywh_to_xyxy((cx + dx, cy + dy, w * exp(dlogw), h * exp(dlogh)))


def actions_from_boxes(boxes_xyxy: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape [T, 4]")
    if len(boxes) < 2:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray([box_action(boxes[i], boxes[i + 1]) for i in range(len(boxes) - 1)], dtype=np.float32)


def reconstruct_boxes(start_box_xyxy: Iterable[float], actions: np.ndarray) -> np.ndarray:
    current = tuple(float(v) for v in start_box_xyxy)
    out: list[tuple[float, float, float, float]] = []
    for action in np.asarray(actions, dtype=np.float32):
        current = apply_box_action(current, action)
        out.append(current)
    return np.asarray(out, dtype=np.float32)


def action_reconstruction_error(start_box_xyxy: Iterable[float], actions: np.ndarray, target_boxes_xyxy: np.ndarray) -> float:
    pred = reconstruct_boxes(start_box_xyxy, actions)
    target = np.asarray(target_boxes_xyxy, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: predicted {pred.shape}, target {target.shape}")
    if pred.size == 0:
        return 0.0
    pred_centers = np.column_stack(((pred[:, 0] + pred[:, 2]) / 2.0, (pred[:, 1] + pred[:, 3]) / 2.0))
    target_centers = np.column_stack(((target[:, 0] + target[:, 2]) / 2.0, (target[:, 1] + target[:, 3]) / 2.0))
    return float(np.mean(np.linalg.norm(pred_centers - target_centers, axis=1)))


def build_action_chunk_samples_from_rows(
    rows: list[dict[str, Any]],
    past_len: int = 8,
    future_len: int = 8,
    image_size: tuple[int, int] | None = None,
    normalize_by_row_image_size: bool = False,
    seq: str = "",
    track_id: str = "",
) -> list[ActionChunkSample]:
    if past_len <= 0 or future_len <= 0:
        raise ValueError("past_len and future_len must be positive")
    ordered = sorted(rows, key=lambda r: int(r.get("frame_id", 0)))
    if len(ordered) < past_len + future_len:
        return []
    boxes = []
    scores = []
    visible = []
    frame_ids = []
    for row in ordered:
        box = _row_box(row)
        row_image_size = _row_image_size(row) if normalize_by_row_image_size else None
        if row_image_size is not None:
            box = normalize_xyxy(box, row_image_size)
        elif image_size is not None:
            box = normalize_xyxy(box, image_size)
        boxes.append(box)
        scores.append(_row_score(row))
        visible.append(float(row.get("visible", True)))
        frame_ids.append(int(row.get("frame_id", 0)))

    boxes_arr = np.asarray(boxes, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    visible_arr = np.asarray(visible, dtype=np.float32)
    samples: list[ActionChunkSample] = []
    total = past_len + future_len
    for start in range(0, len(ordered) - total + 1):
        end_past = start + past_len
        end_future = end_past + future_len
        future_boxes = boxes_arr[end_past:end_future]
        chunk_boxes = boxes_arr[end_past - 1 : end_future]
        samples.append(
            ActionChunkSample(
                seq=seq or str(ordered[start].get("seq", "")),
                track_id=track_id or str(ordered[start].get("track_id", "")),
                anchor_frame=frame_ids[end_past - 1],
                past_boxes=boxes_arr[start:end_past].copy(),
                past_scores=scores_arr[start:end_past].copy(),
                past_visible=visible_arr[start:end_past].copy(),
                future_actions=actions_from_boxes(chunk_boxes),
                future_boxes=future_boxes.copy(),
            )
        )
    return samples


def gaussian_prior_heatmap(
    boxes_xyxy: np.ndarray,
    image_size: tuple[int, int],
    sigma_scale: float = 1.5,
    min_sigma: float = 2.0,
) -> np.ndarray:
    width, height = image_size
    heatmap = np.zeros((height, width), dtype=np.float32)
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    yy, xx = np.mgrid[0:height, 0:width]
    for box in np.asarray(boxes_xyxy, dtype=np.float32):
        cx, cy, bw, bh = xyxy_to_cxcywh(box)
        sigma = max(float(min_sigma), float(sigma_scale) * max(bw, bh, 1.0))
        heatmap = np.maximum(heatmap, np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)))
    peak = float(heatmap.max())
    if peak > 0:
        heatmap /= peak
    return heatmap


def _load_tracklet_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _sample_json(sample: ActionChunkSample, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "seq": sample.seq,
        "track_id": sample.track_id,
        "anchor_frame": sample.anchor_frame,
        "label": int(float(meta.get("label", 0))),
        "bucket": str(meta.get("bucket", "")),
        "dataset_source": str(meta.get("dataset_source", "")),
        "past_boxes": sample.past_boxes.tolist(),
        "past_scores": sample.past_scores.tolist(),
        "past_visible": sample.past_visible.tolist(),
        "future_actions": sample.future_actions.tolist(),
        "future_boxes": sample.future_boxes.tolist(),
    }


def export_action_chunk_dataset_from_tracklets(
    tracklet_jsonl: str | Path,
    out: str | Path,
    past_len: int = 8,
    future_len: int = 8,
    image_size: tuple[int, int] | None = None,
    normalize_by_row_image_size: bool = False,
    positives_only: bool = False,
    min_tracklet_rows: int = 0,
) -> ActionChunkDatasetResult:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = _load_tracklet_jsonl(tracklet_jsonl)
    total_tracklets = 0
    used_tracklets = 0
    total_samples = 0
    positive_samples = 0
    negative_samples = 0
    bucket_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            total_tracklets += 1
            meta = dict(item.get("meta") or {})
            rows = list(item.get("rows") or [])
            if min_tracklet_rows > 0 and len(rows) < min_tracklet_rows:
                continue
            label = int(float(meta.get("label", 0)))
            if positives_only and label <= 0:
                continue
            seq = str(meta.get("seq", ""))
            track_id = str(meta.get("track_id", ""))
            samples = build_action_chunk_samples_from_rows(
                rows,
                past_len=past_len,
                future_len=future_len,
                image_size=image_size,
                normalize_by_row_image_size=normalize_by_row_image_size,
                seq=seq,
                track_id=track_id,
            )
            if not samples:
                continue
            used_tracklets += 1
            bucket = str(meta.get("bucket", ""))
            source = str(meta.get("dataset_source", ""))
            for sample in samples:
                f.write(json.dumps(_sample_json(sample, meta), ensure_ascii=False) + "\n")
                total_samples += 1
                positive_samples += int(label > 0)
                negative_samples += int(label <= 0)
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
                source_counts[source] = source_counts.get(source, 0) + 1

    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "jsonl": str(out_path),
        "past_len": past_len,
        "future_len": future_len,
        "image_size": list(image_size) if image_size is not None else None,
        "normalize_by_row_image_size": normalize_by_row_image_size,
        "positives_only": positives_only,
        "min_tracklet_rows": min_tracklet_rows,
        "total_tracklets": total_tracklets,
        "used_tracklets": used_tracklets,
        "samples": total_samples,
        "positive_samples": positive_samples,
        "negative_samples": negative_samples,
        "bucket_counts": bucket_counts,
        "dataset_source_counts": source_counts,
    }
    return ActionChunkDatasetResult(jsonl_path=out_path, summary=summary)


def merge_action_chunk_datasets(
    inputs: list[str | Path],
    out: str | Path,
    source_names: list[str] | None = None,
    manifest_out: str | Path | None = None,
) -> ActionChunkDatasetResult:
    if not inputs:
        raise ValueError("inputs must contain at least one action-chunk JSONL")
    if source_names is not None and len(source_names) != len(inputs):
        raise ValueError("source_names must have the same length as inputs")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_out) if manifest_out is not None else out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    positive_samples = 0
    negative_samples = 0
    bucket_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    dataset_summaries: list[dict[str, Any]] = []

    with out_path.open("w", encoding="utf-8") as f:
        for index, input_path in enumerate(inputs):
            source_name = source_names[index] if source_names is not None else None
            per_dataset_samples = 0
            per_dataset_positive = 0
            per_dataset_negative = 0
            per_dataset_buckets: dict[str, int] = {}
            per_dataset_sources: dict[str, int] = {}
            for sample in _iter_jsonl(input_path):
                if source_name:
                    sample["dataset_source"] = source_name
                source = str(sample.get("dataset_source", source_name or Path(input_path).stem))
                sample["dataset_source"] = source
                bucket = str(sample.get("bucket", ""))
                label = int(float(sample.get("label", 0)))

                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total_samples += 1
                per_dataset_samples += 1
                positive_samples += int(label > 0)
                negative_samples += int(label <= 0)
                per_dataset_positive += int(label > 0)
                per_dataset_negative += int(label <= 0)
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
                source_counts[source] = source_counts.get(source, 0) + 1
                per_dataset_buckets[bucket] = per_dataset_buckets.get(bucket, 0) + 1
                per_dataset_sources[source] = per_dataset_sources.get(source, 0) + 1

            dataset_summaries.append(
                {
                    "input": str(input_path),
                    "source_name": source_name,
                    "samples": per_dataset_samples,
                    "positive_samples": per_dataset_positive,
                    "negative_samples": per_dataset_negative,
                    "bucket_counts": per_dataset_buckets,
                    "dataset_source_counts": per_dataset_sources,
                }
            )

    summary = {
        "inputs": [str(path) for path in inputs],
        "jsonl": str(out_path),
        "manifest": str(manifest_path),
        "samples": total_samples,
        "positive_samples": positive_samples,
        "negative_samples": negative_samples,
        "bucket_counts": bucket_counts,
        "dataset_source_counts": source_counts,
        "datasets": dataset_summaries,
        "schema": {
            "seq": "video/flight sequence id",
            "track_id": "candidate tube id",
            "anchor_frame": "last observed frame before future action chunk",
            "label": "1 for UAV-like positive, 0 for negative/background",
            "bucket": "sampling bucket such as hard_tiny_positive or hard_negative",
            "dataset_source": "canonical dataset/source name for balanced multi-dataset training",
            "past_boxes": "[past_len, 4] xyxy boxes, normalized when image_size was used during export",
            "past_scores": "[past_len] detector/proposal confidence",
            "past_visible": "[past_len] visibility mask",
            "future_actions": "[future_len, 4] dx, dy, dlogw, dlogh action chunk",
            "future_boxes": "[future_len, 4] target boxes at the action horizon",
        },
    }
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionChunkDatasetResult(jsonl_path=out_path, summary=summary)


def split_action_chunk_dataset(
    jsonl_path: str | Path,
    out_dir: str | Path,
    calib_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 59,
    group_field: str = "seq",
    source_field: str = "dataset_source",
) -> ActionChunkDatasetResult:
    if calib_fraction < 0.0 or test_fraction < 0.0 or calib_fraction + test_fraction >= 1.0:
        raise ValueError("calib_fraction and test_fraction must be nonnegative and sum to less than 1")
    rows = list(_iter_jsonl(jsonl_path))
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    split_paths = {
        "train": out_root / "train_action_chunks.jsonl",
        "calib": out_root / "calib_action_chunks.jsonl",
        "test": out_root / "test_action_chunks.jsonl",
    }
    for path in split_paths.values():
        path.write_text("", encoding="utf-8")

    by_source_group: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        source = str(row.get(source_field, "") or "unknown")
        group = str(row.get(group_field, "") or row.get("seq", "") or row.get("track_id", "") or "ungrouped")
        by_source_group.setdefault(source, {}).setdefault(group, []).append(row)

    rng = np.random.default_rng(seed)
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "calib": [], "test": []}
    assignments: dict[str, dict[str, str]] = {}
    source_summaries: dict[str, Any] = {}
    for source, groups in sorted(by_source_group.items()):
        group_names = sorted(groups)
        order = rng.permutation(len(group_names)).tolist() if group_names else []
        ordered_groups = [group_names[i] for i in order]
        n = len(ordered_groups)
        test_count = int(round(n * test_fraction))
        calib_count = int(round(n * calib_fraction))
        if n > 1 and test_fraction > 0.0 and test_count == 0:
            test_count = 1
        if n - test_count > 1 and calib_fraction > 0.0 and calib_count == 0:
            calib_count = 1
        if test_count + calib_count >= n and n > 0:
            overflow = test_count + calib_count - (n - 1)
            calib_count = max(0, calib_count - overflow)
            if test_count + calib_count >= n:
                test_count = max(0, n - 1 - calib_count)

        source_assignments: dict[str, str] = {}
        for pos, group in enumerate(ordered_groups):
            if pos < test_count:
                split = "test"
            elif pos < test_count + calib_count:
                split = "calib"
            else:
                split = "train"
            source_assignments[group] = split
            for row in groups[group]:
                row_out = dict(row)
                row_out["split"] = split
                split_rows[split].append(row_out)
        assignments[source] = source_assignments
        source_summaries[source] = {
            "groups": n,
            "samples": sum(len(items) for items in groups.values()),
            "train_groups": sum(1 for value in source_assignments.values() if value == "train"),
            "calib_groups": sum(1 for value in source_assignments.values() if value == "calib"),
            "test_groups": sum(1 for value in source_assignments.values() if value == "test"),
            "train_samples": sum(len(groups[group]) for group, value in source_assignments.items() if value == "train"),
            "calib_samples": sum(len(groups[group]) for group, value in source_assignments.items() if value == "calib"),
            "test_samples": sum(len(groups[group]) for group, value in source_assignments.items() if value == "test"),
        }

    for split, path in split_paths.items():
        with path.open("w", encoding="utf-8") as f:
            for row in split_rows[split]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "jsonl": str(jsonl_path),
        "out_dir": str(out_root),
        "seed": seed,
        "group_field": group_field,
        "source_field": source_field,
        "calib_fraction": calib_fraction,
        "test_fraction": test_fraction,
        "total_samples": len(rows),
        "total_groups": sum(len(groups) for groups in by_source_group.values()),
        "train_jsonl": str(split_paths["train"]),
        "calib_jsonl": str(split_paths["calib"]),
        "test_jsonl": str(split_paths["test"]),
        "train_samples": len(split_rows["train"]),
        "calib_samples": len(split_rows["calib"]),
        "test_samples": len(split_rows["test"]),
        "sources": source_summaries,
        "assignments": assignments,
        "leakage_guard": f"splits are assigned by {source_field}/{group_field} groups, not individual samples",
    }
    manifest_path = out_root / "action_chunk_split_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionChunkDatasetResult(jsonl_path=split_paths["train"], summary=summary)


def export_action_prior_heatmaps_from_sample_scores(
    sample_scores_jsonl: str | Path,
    out_dir: str | Path,
    image_size: tuple[int, int],
    sigma_scale: float = 1.5,
    min_sigma: float = 2.0,
    box_field: str = "learned_boxes",
    split_horizon: bool = False,
) -> ActionChunkDatasetResult:
    out_root = Path(out_dir)
    heatmap_dir = out_root / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "action_prior_heatmaps.jsonl"
    summary_path = out_root / "action_prior_heatmaps_summary.json"

    total = 0
    exported = 0
    skipped = 0
    max_values = []
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in _iter_jsonl(sample_scores_jsonl):
            total += 1
            boxes = np.asarray(row.get(box_field) or [], dtype=np.float32)
            if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) == 0:
                skipped += 1
                continue
            heatmap = gaussian_prior_heatmap(boxes, image_size=image_size, sigma_scale=sigma_scale, min_sigma=min_sigma)
            seq = str(row.get("seq", ""))
            track_id = str(row.get("track_id", ""))
            anchor_frame = int(row.get("anchor_frame", 0))
            safe_seq = seq.replace("/", "_").replace("\\", "_") or "seq"
            safe_track = track_id.replace("/", "_").replace("\\", "_") or "track"
            if split_horizon:
                for horizon_index, box in enumerate(boxes):
                    target_frame_id = anchor_frame + horizon_index + 1
                    heatmap = gaussian_prior_heatmap(
                        np.asarray([box], dtype=np.float32),
                        image_size=image_size,
                        sigma_scale=sigma_scale,
                        min_sigma=min_sigma,
                    )
                    heatmap_path = heatmap_dir / f"{safe_seq}_{safe_track}_{anchor_frame:06d}_h{horizon_index + 1:02d}.npy"
                    np.save(heatmap_path, heatmap)
                    max_values.append(float(heatmap.max()))
                    f.write(
                        json.dumps(
                            {
                                "seq": seq,
                                "track_id": track_id,
                                "anchor_frame": anchor_frame,
                                "target_frame_id": target_frame_id,
                                "horizon_index": horizon_index,
                                "label": int(float(row.get("label", 0))),
                                "heatmap": str(heatmap_path),
                                "image_size": list(image_size),
                                "box_field": box_field,
                                "boxes": [box.tolist()],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    exported += 1
            else:
                heatmap = gaussian_prior_heatmap(boxes, image_size=image_size, sigma_scale=sigma_scale, min_sigma=min_sigma)
                heatmap_path = heatmap_dir / f"{safe_seq}_{safe_track}_{anchor_frame:06d}.npy"
                np.save(heatmap_path, heatmap)
                max_values.append(float(heatmap.max()))
                f.write(
                    json.dumps(
                        {
                            "seq": seq,
                            "track_id": track_id,
                            "anchor_frame": anchor_frame,
                            "target_frame_id": None,
                            "horizon_index": None,
                            "label": int(float(row.get("label", 0))),
                            "heatmap": str(heatmap_path),
                            "image_size": list(image_size),
                            "box_field": box_field,
                            "boxes": boxes.tolist(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                exported += 1

    summary = {
        "sample_scores_jsonl": str(sample_scores_jsonl),
        "out_dir": str(out_root),
        "manifest": str(manifest_path),
        "heatmap_dir": str(heatmap_dir),
        "image_size": list(image_size),
        "sigma_scale": sigma_scale,
        "min_sigma": min_sigma,
        "box_field": box_field,
        "split_horizon": split_horizon,
        "total_rows": total,
        "exported": exported,
        "skipped": skipped,
        "mean_peak": float(np.mean(max_values)) if max_values else 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionChunkDatasetResult(jsonl_path=manifest_path, summary=summary)


def build_frame_prior_index_from_heatmaps(
    prior_manifest_jsonl: str | Path,
    out_dir: str | Path,
    merge_mode: str = "max",
) -> ActionChunkDatasetResult:
    if merge_mode not in {"max", "mean"}:
        raise ValueError("merge_mode must be 'max' or 'mean'")
    rows = list(_iter_jsonl(prior_manifest_jsonl))
    out_root = Path(out_dir)
    frame_dir = out_root / "frame_priors"
    frame_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_root / "frame_prior_index.jsonl"
    summary_path = out_root / "frame_prior_index_summary.json"

    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    skipped = 0
    for row in rows:
        target = row.get("target_frame_id")
        if target is None:
            skipped += 1
            continue
        key = (str(row.get("seq", "")), int(target))
        groups.setdefault(key, []).append(row)

    exported = 0
    with index_path.open("w", encoding="utf-8") as f:
        for (seq, frame_id), items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
            heatmaps = [np.load(str(item["heatmap"])).astype(np.float32) for item in items]
            stack = np.stack(heatmaps, axis=0)
            if merge_mode == "max":
                merged = np.max(stack, axis=0)
            else:
                merged = np.mean(stack, axis=0)
            peak = float(merged.max())
            if peak > 0:
                merged = merged / peak
            safe_seq = seq.replace("/", "_").replace("\\", "_") or "seq"
            prior_path = frame_dir / f"{safe_seq}_{frame_id:06d}.npy"
            np.save(prior_path, merged)
            f.write(
                json.dumps(
                    {
                        "seq": seq,
                        "frame_id": frame_id,
                        "prior": str(prior_path),
                        "merge_mode": merge_mode,
                        "num_tracklet_priors": len(items),
                        "source_heatmaps": [str(item["heatmap"]) for item in items],
                        "track_ids": [str(item.get("track_id", "")) for item in items],
                        "image_size": items[0].get("image_size", []),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            exported += 1

    summary = {
        "prior_manifest_jsonl": str(prior_manifest_jsonl),
        "out_dir": str(out_root),
        "index": str(index_path),
        "frame_prior_dir": str(frame_dir),
        "merge_mode": merge_mode,
        "input_rows": len(rows),
        "skipped_without_target_frame": skipped,
        "frames": exported,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionChunkDatasetResult(jsonl_path=index_path, summary=summary)


def _frame_prior_fields(prior: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_frame_prior": str(prior.get("prior", "")),
        "action_frame_prior_merge_mode": str(prior.get("merge_mode", "")),
        "action_frame_prior_num_tracklet_priors": int(prior.get("num_tracklet_priors", 0)),
        "action_frame_prior_track_ids": list(prior.get("track_ids") or []),
        "action_frame_prior_image_size": list(prior.get("image_size") or []),
    }


def _resolve_prior_path(path_value: str, index_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return index_dir / path


def _prior_bbox_stats(prior: dict[str, Any], row: dict[str, Any], index_dir: Path) -> dict[str, float]:
    path_value = str(prior.get("prior", ""))
    if not path_value:
        return {}
    prior_path = _resolve_prior_path(path_value, index_dir)
    if not prior_path.exists():
        return {}
    try:
        heatmap = np.load(prior_path).astype(np.float32)
        x1, y1, x2, y2 = _row_box(row)
    except (OSError, ValueError, KeyError):
        return {}
    if heatmap.ndim != 2 or heatmap.size == 0:
        return {}
    height, width = heatmap.shape
    ix1 = max(0, min(width - 1, int(np.floor(min(x1, x2)))))
    iy1 = max(0, min(height - 1, int(np.floor(min(y1, y2)))))
    ix2 = max(ix1 + 1, min(width, int(np.ceil(max(x1, x2)))))
    iy2 = max(iy1 + 1, min(height, int(np.ceil(max(y1, y2)))))
    patch = heatmap[iy1:iy2, ix1:ix2]
    cx = max(0, min(width - 1, int(round((x1 + x2) * 0.5))))
    cy = max(0, min(height - 1, int(round((y1 + y2) * 0.5))))
    if patch.size == 0:
        return {}
    patch_max = float(np.max(patch))
    patch_mean = float(np.mean(patch))
    center = float(heatmap[cy, cx])
    return {
        "action_frame_prior_score": max(patch_max, center),
        "action_frame_prior_bbox_max": patch_max,
        "action_frame_prior_bbox_mean": patch_mean,
        "action_frame_prior_center": center,
    }


def attach_frame_priors_to_tracklets(
    tracklet_jsonl: str | Path,
    frame_prior_index_jsonl: str | Path,
    out: str | Path,
) -> ActionChunkDatasetResult:
    """Attach frame-level action prior heatmap paths to nested tracklet or flat proposal JSONL rows."""
    index_dir = Path(frame_prior_index_jsonl).parent
    priors: dict[tuple[str, int], dict[str, Any]] = {}
    for prior in _iter_jsonl(frame_prior_index_jsonl):
        priors[(str(prior.get("seq", "")), int(prior.get("frame_id", 0)))] = prior

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_items = 0
    total_rows = 0
    attached_rows = 0
    missing_rows = 0
    attached_items = 0
    flat_rows = 0
    nested_tracklets = 0
    scored_rows = 0

    with out_path.open("w", encoding="utf-8") as f:
        for item in _iter_jsonl(tracklet_jsonl):
            total_items += 1
            item_attached = 0
            if isinstance(item.get("rows"), list):
                nested_tracklets += 1
                meta = dict(item.get("meta") or {})
                seq = str(meta.get("seq", item.get("seq", "")))
                rows = []
                for row in item.get("rows") or []:
                    row_out = dict(row)
                    row_seq = str(row_out.get("seq", seq))
                    frame_id = int(row_out.get("frame_id", -1))
                    prior = priors.get((row_seq, frame_id))
                    total_rows += 1
                    if prior is None:
                        missing_rows += 1
                    else:
                        row_out.update(_frame_prior_fields(prior))
                        stats = _prior_bbox_stats(prior, row_out, index_dir)
                        row_out.update(stats)
                        scored_rows += int(bool(stats))
                        attached_rows += 1
                        item_attached += 1
                    rows.append(row_out)
                item = dict(item)
                item["meta"] = meta
                item["rows"] = rows
                meta["action_frame_prior_rows"] = item_attached
                meta["action_frame_prior_missing_rows"] = len(rows) - item_attached
            else:
                flat_rows += 1
                row_out = dict(item)
                seq = str(row_out.get("seq", ""))
                frame_id = int(row_out.get("frame_id", -1))
                prior = priors.get((seq, frame_id))
                total_rows += 1
                if prior is None:
                    missing_rows += 1
                else:
                    row_out.update(_frame_prior_fields(prior))
                    stats = _prior_bbox_stats(prior, row_out, index_dir)
                    row_out.update(stats)
                    scored_rows += int(bool(stats))
                    attached_rows += 1
                    item_attached += 1
                item = row_out
            if item_attached:
                attached_items += 1
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "frame_prior_index_jsonl": str(frame_prior_index_jsonl),
        "jsonl": str(out_path),
        "frame_priors": len(priors),
        "total_items": total_items,
        "nested_tracklets": nested_tracklets,
        "flat_rows": flat_rows,
        "total_rows": total_rows,
        "attached_items": attached_items,
        "attached_rows": attached_rows,
        "scored_rows": scored_rows,
        "missing_rows": missing_rows,
        "attach_rate": attached_rows / total_rows if total_rows else 0.0,
        "score_rate": scored_rows / attached_rows if attached_rows else 0.0,
    }
    return ActionChunkDatasetResult(jsonl_path=out_path, summary=summary)
