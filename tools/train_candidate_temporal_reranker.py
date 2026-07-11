from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FEATURE_NAMES = [
    "score",
    "raw_objectness",
    "detector_raw_objectness",
    "motion_memory_score",
    "ncc_score",
    "memory_quality",
    "rank_norm",
    "rank_inv",
    "num_merged_norm",
    "has_detector_member",
    "has_yolo_source",
    "has_ncc_source",
    "is_mixed_source",
    "is_support_only",
    "cx_norm",
    "cy_norm",
    "w_norm",
    "h_norm",
    "area_norm",
    "aspect_ratio_log",
    "detector_center_delta_norm",
    "detector_size_delta_norm",
    "temporal_minus_raw",
    "motion_minus_raw",
]


@dataclass
class CandidateSample:
    frame_key: str
    candidate: dict[str, Any]
    features: list[float]
    label: float | None
    best_iou: float | None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _source_parts(source: str) -> set[str]:
    return {part for part in str(source).split("+") if part}


def _has_detector_source(source: str) -> bool:
    parts = _source_parts(source)
    return any(part in {"yolo", "yolo_tile", "yolov5_dual", "zoom_redetect", "crop_yolo"} or part.startswith("yolo") for part in parts)


def _has_ncc_source(source: str) -> bool:
    return "gray_ncc" in _source_parts(source)


def _extra(candidate: dict[str, Any]) -> dict[str, Any]:
    extra = candidate.get("extra")
    return extra if isinstance(extra, dict) else {}


def _value(candidate: dict[str, Any], key: str, default: float = 0.0) -> float:
    if key in candidate and candidate.get(key) is not None:
        return _safe_float(candidate.get(key), default)
    return _safe_float(_extra(candidate).get(key), default)


def _bbox(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        _safe_float(candidate.get("x1")),
        _safe_float(candidate.get("y1")),
        _safe_float(candidate.get("x2")),
        _safe_float(candidate.get("y2")),
    )


def _detector_bbox(candidate: dict[str, Any]) -> tuple[float, float, float, float] | None:
    top_level = [candidate.get("detector_x1"), candidate.get("detector_y1"), candidate.get("detector_x2"), candidate.get("detector_y2")]
    if all(value is not None for value in top_level):
        return tuple(_safe_float(value) for value in top_level)  # type: ignore[return-value]
    nested = _extra(candidate).get("detector_bbox_xyxy")
    if isinstance(nested, (list, tuple)) and len(nested) == 4:
        return tuple(_safe_float(value) for value in nested)  # type: ignore[return-value]
    return None


def _xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = min(max(0.0, float(x1)), float(width))
    x2 = min(max(0.0, float(x2)), float(width))
    y1 = min(max(0.0, float(y1)), float(height))
    y2 = min(max(0.0, float(y2)), float(height))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return (x1 + bw / 2.0) / width, (y1 + bh / 2.0) / height, bw / width, bh / height


def _iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(1e-6, area_a + area_b - inter))


def label_path_from_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part.lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    text = str(image_path)
    for old, new in (("\\images\\", "\\labels\\"), ("/images/", "/labels/")):
        if old in text:
            return Path(text.replace(old, new)).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_gt_boxes(image_path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    label_path = label_path_from_image(image_path)
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = [_safe_float(value) for value in parts[1:5]]
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def candidate_features(candidate: dict[str, Any], width: int, height: int, max_rank: int) -> list[float]:
    score = _value(candidate, "score")
    raw = _value(candidate, "raw_objectness", score)
    detector_raw = _value(candidate, "detector_raw_objectness", raw)
    motion = _value(candidate, "motion_memory_score")
    ncc = _value(candidate, "ncc_score")
    quality = _value(candidate, "memory_quality")
    rank = int(_safe_float(candidate.get("rank"), 0.0))
    source = str(candidate.get("source") or "")
    has_yolo = 1.0 if _has_detector_source(source) or bool(candidate.get("has_detector_member") or _extra(candidate).get("has_detector_member")) else 0.0
    has_ncc = 1.0 if _has_ncc_source(source) else 0.0
    x1, y1, x2, y2 = _bbox(candidate)
    cx, cy, bw, bh = _xyxy_to_yolo(x1, y1, x2, y2, width, height)
    area = bw * bh
    aspect = math.log(max(1e-6, bw) / max(1e-6, bh))
    det_bbox = _detector_bbox(candidate)
    if det_bbox is None:
        det_center_delta = 0.0
        det_size_delta = 0.0
    else:
        dcx, dcy, dbw, dbh = _xyxy_to_yolo(*det_bbox, width, height)
        det_center_delta = math.hypot(cx - dcx, cy - dcy)
        det_size_delta = abs(bw - dbw) + abs(bh - dbh)
    num_merged = _value(candidate, "num_merged", 1.0)
    return [
        score,
        raw,
        detector_raw,
        motion,
        ncc,
        quality,
        rank / max(1, max_rank),
        1.0 / (1.0 + rank),
        min(1.0, num_merged / 8.0),
        has_yolo,
        1.0 if _has_detector_source(source) else 0.0,
        has_ncc,
        1.0 if "+" in source else 0.0,
        1.0 if has_ncc and not has_yolo else 0.0,
        cx,
        cy,
        bw,
        bh,
        area,
        aspect,
        det_center_delta,
        det_size_delta,
        score - raw,
        motion - raw,
    ]


def load_candidate_samples(
    candidate_jsonl: Path,
    *,
    label: bool,
    match_iou: float,
    progress_stage: str | None = None,
    progress_every: int = 10000,
) -> list[CandidateSample]:
    samples: list[CandidateSample] = []
    gt_cache: dict[tuple[str, int, int], list[tuple[float, float, float, float]]] = {}
    frames_seen = 0
    with candidate_jsonl.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            frame = json.loads(line)
            frames_seen += 1
            image_text = frame.get("image_path")
            if image_text is None:
                continue
            image_path = Path(str(image_text))
            width = int(frame.get("width") or 1920)
            height = int(frame.get("height") or 1080)
            candidates = list(frame.get("candidates") or [])
            max_rank = max(1, len(candidates) - 1)
            gt_boxes: list[tuple[float, float, float, float]] = []
            if label:
                key = (str(image_path), width, height)
                if key not in gt_cache:
                    gt_cache[key] = read_gt_boxes(image_path, width, height)
                gt_boxes = gt_cache[key]
            for candidate in candidates:
                candidate = dict(candidate)
                candidate["width"] = width
                candidate["height"] = height
                bbox = _bbox(candidate)
                best_iou = max((_iou(bbox, gt) for gt in gt_boxes), default=0.0) if label else None
                y = 1.0 if best_iou is not None and best_iou >= match_iou else 0.0 if label else None
                samples.append(
                    CandidateSample(
                        frame_key=image_path.stem,
                        candidate=candidate,
                        features=candidate_features(candidate, width, height, max_rank),
                        label=y,
                        best_iou=best_iou,
                    )
                )
            if progress_stage and frames_seen % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "kind": "candidate_reranker_progress",
                            "stage": progress_stage,
                            "frames_read": frames_seen,
                            "candidates": len(samples),
                        }
                    ),
                    flush=True,
                )
    return samples


def fit_logistic(x: np.ndarray, y: np.ndarray, *, epochs: int, batch_size: int, lr: float, seed: int) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = np.float32(0.0)
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = np.float32(0.0)
    v_b = np.float32(0.0)
    pos = float(y.sum())
    neg = float(len(y) - pos)
    pos_weight = neg / max(pos, 1.0)
    history: list[dict[str, float]] = []
    step = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y))
        losses = []
        for start in range(0, len(order), batch_size):
            step += 1
            idx = order[start : start + batch_size]
            bx = x[idx]
            by = y[idx]
            logits = bx @ weights + bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
            sample_weight = np.where(by > 0.5, pos_weight, 1.0).astype(np.float32)
            grad = (probs - by) * sample_weight / max(1, len(by))
            grad_w = bx.T @ grad
            grad_b = np.float32(grad.sum())
            m_w = 0.9 * m_w + 0.1 * grad_w
            v_w = 0.999 * v_w + 0.001 * (grad_w * grad_w)
            m_b = np.float32(0.9 * m_b + 0.1 * grad_b)
            v_b = np.float32(0.999 * v_b + 0.001 * (grad_b * grad_b))
            lr_t = lr * math.sqrt(1.0 - 0.999**step) / (1.0 - 0.9**step)
            weights -= lr_t * m_w / (np.sqrt(v_w) + 1e-8)
            bias -= np.float32(lr_t * m_b / (math.sqrt(float(v_b)) + 1e-8))
            loss = -(sample_weight * (by * np.log(probs + 1e-7) + (1.0 - by) * np.log(1.0 - probs + 1e-7))).mean()
            losses.append(float(loss))
        epoch_loss = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": float(epoch), "loss": epoch_loss})
        print(json.dumps({"kind": "candidate_reranker_progress", "stage": "train_epoch", "epoch": epoch, "epochs": epochs, "loss": epoch_loss}), flush=True)
    return weights, float(bias), history


def predict_scores(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    logits = x @ weights + np.float32(bias)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def write_top1_labels(samples: list[CandidateSample], scores: np.ndarray, out_label_dir: Path, class_id: int) -> dict[str, Any]:
    out_label_dir.mkdir(parents=True, exist_ok=True)
    best_by_frame: dict[str, tuple[CandidateSample, float]] = {}
    for sample, score in zip(samples, scores):
        current = best_by_frame.get(sample.frame_key)
        if current is None or float(score) > current[1]:
            best_by_frame[sample.frame_key] = (sample, float(score))
    for path in out_label_dir.glob("*.txt"):
        path.unlink()
    source_counts: dict[str, int] = {}
    for frame_key, (sample, score) in best_by_frame.items():
        candidate = sample.candidate
        width = int(candidate.get("width") or 1920)
        height = int(candidate.get("height") or 1080)
        cx, cy, bw, bh = _xyxy_to_yolo(*_bbox(candidate), width, height)
        line = ""
        if bw > 0.0 and bh > 0.0:
            line = f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f} {score:.8f}\n"
        (out_label_dir / f"{frame_key}.txt").write_text(line, encoding="utf-8")
        source = str(candidate.get("source") or "")
        source_counts[source] = source_counts.get(source, 0) + 1
    return {"frames_written": len(best_by_frame), "selected_sources": source_counts}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a SAMURAI-feature candidate reranker and emit top-1 YOLO labels.")
    parser.add_argument("--train-candidate-jsonl", type=Path, required=True)
    parser.add_argument("--test-candidate-jsonl", type=Path, required=True)
    parser.add_argument("--out-label-dir", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    print(json.dumps({"kind": "candidate_reranker_progress", "stage": "load_train_start", "path": str(args.train_candidate_jsonl)}), flush=True)
    train_samples = load_candidate_samples(args.train_candidate_jsonl, label=True, match_iou=args.match_iou, progress_stage="load_train")
    print(json.dumps({"kind": "candidate_reranker_progress", "stage": "load_train_done", "candidates": len(train_samples)}), flush=True)
    print(json.dumps({"kind": "candidate_reranker_progress", "stage": "load_test_start", "path": str(args.test_candidate_jsonl)}), flush=True)
    test_samples = load_candidate_samples(args.test_candidate_jsonl, label=False, match_iou=args.match_iou, progress_stage="load_test")
    print(json.dumps({"kind": "candidate_reranker_progress", "stage": "load_test_done", "candidates": len(test_samples)}), flush=True)
    x_train = np.asarray([sample.features for sample in train_samples], dtype=np.float32)
    y_train = np.asarray([float(sample.label or 0.0) for sample in train_samples], dtype=np.float32)
    x_test = np.asarray([sample.features for sample in test_samples], dtype=np.float32)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    x_train_norm = (x_train - mean) / std
    x_test_norm = (x_test - mean) / std
    weights, bias, history = fit_logistic(x_train_norm, y_train, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    test_scores = predict_scores(x_test_norm, weights, bias)
    label_summary = write_top1_labels(test_samples, test_scores, args.out_label_dir, args.class_id)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_model, feature_names=np.asarray(FEATURE_NAMES), mean=mean, std=std, weights=weights, bias=np.asarray([bias], dtype=np.float32))
    summary = {
        "train_candidate_jsonl": str(args.train_candidate_jsonl),
        "test_candidate_jsonl": str(args.test_candidate_jsonl),
        "out_label_dir": str(args.out_label_dir),
        "out_model": str(args.out_model),
        "feature_names": FEATURE_NAMES,
        "train_candidates": int(len(train_samples)),
        "train_positive": int(y_train.sum()),
        "train_negative": int(len(y_train) - y_train.sum()),
        "test_candidates": int(len(test_samples)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "history": history,
        "label_summary": label_summary,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
