from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from qstr_dronedet.features.roi import crop_with_context, extract_temporal_tube
from qstr_dronedet.fusion.rule_fusion import fuse_rule_based
from qstr_dronedet.recognition.crop_recognizer import CropRecognizer
from qstr_dronedet.recognition.feature_recognizer import FeatureRecognitionModel
from qstr_dronedet.recognition.temporal_recognizer import TemporalRecognizer
from qstr_dronedet.types import CLASSES, normalize_probs


def _load_torch_model(model: torch.nn.Module, weights: str | Path | None) -> torch.nn.Module | None:
    if weights is None:
        return None
    ckpt = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _crop_probs(model: torch.nn.Module | None, crop_bgr: np.ndarray) -> dict[str, float]:
    if model is None:
        return normalize_probs({"unknown": 1.0})
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        probs = model(x).softmax(1)[0].cpu().numpy().tolist()
    return normalize_probs({cls: float(probs[i]) for i, cls in enumerate(CLASSES)})


def _feature_probs(model: torch.nn.Module | None, frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> dict[str, float]:
    if model is None:
        return normalize_probs({"unknown": 1.0})
    img = cv2.resize(frame_bgr, (640, 640))
    sx = 640.0 / frame_bgr.shape[1]
    sy = 640.0 / frame_bgr.shape[0]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    box = torch.tensor([[bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]], dtype=torch.float32)
    with torch.no_grad():
        probs = model(x, box).softmax(1)[0].cpu().numpy().tolist()
    return normalize_probs({cls: float(probs[i]) for i, cls in enumerate(CLASSES)})


def _temporal_probs(model: torch.nn.Module | None, tube: np.ndarray) -> dict[str, float]:
    if model is None:
        return normalize_probs({"unknown": 1.0})
    x = torch.from_numpy(tube).unsqueeze(0).float()
    with torch.no_grad():
        probs = model(x).softmax(1)[0].cpu().numpy().tolist()
    return normalize_probs({cls: float(probs[i]) for i, cls in enumerate(CLASSES)})


def _read_frames(video: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _random_negative_box(rng: np.random.Generator, width: int, height: int, pos: list[float]) -> list[float]:
    bw, bh = max(2.0, pos[2] - pos[0]), max(2.0, pos[3] - pos[1])
    for _ in range(100):
        cx = float(rng.integers(int(bw), max(int(bw) + 1, width - int(bw))))
        cy = float(rng.integers(int(bh), max(int(bh) + 1, height - int(bh))))
        box = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
        if not (pos[0] <= cx <= pos[2] and pos[1] <= cy <= pos[3]):
            return box
    return [0.0, 0.0, bw, bh]


def _argmax(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)


def _metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    conf = Counter((r["gt_class"], r[key]) for r in rows)
    correct = sum(1 for r in rows if r["gt_class"] == r[key])
    total = len(rows)
    false_drone = sum(1 for r in rows if r["gt_class"] != "drone" and r[key] == "drone")
    return {
        "accuracy": correct / total if total else 0.0,
        "false_drone_rate": false_drone / max(1, sum(1 for r in rows if r["gt_class"] != "drone")),
        "confusion": {f"{a}->{b}": c for (a, b), c in sorted(conf.items())},
    }


def _diagnostic_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    artifact_rows = [r for r in rows if r["gt_class"] == "alignment_artifact"]
    non_artifact_rows = [r for r in rows if r["gt_class"] != "alignment_artifact"]
    artifact_hits = sum(1 for r in artifact_rows if r.get("diagnostic_cause") == "alignment_artifact")
    false_artifact = sum(1 for r in non_artifact_rows if r.get("diagnostic_cause") == "alignment_artifact")
    return {
        "artifact_diagnostic_recall": artifact_hits / max(1, len(artifact_rows)),
        "false_artifact_diagnostic_rate": false_artifact / max(1, len(non_artifact_rows)),
        "diagnostic_cause_counts": dict(Counter(str(r.get("diagnostic_cause")) for r in rows)),
    }


def run_stage_b_oracle_benchmark(
    metadata_paths: list[str | Path],
    out_dir: str | Path,
    crop_weights: str | Path | None = None,
    feature_weights: str | Path | None = None,
    temporal_weights: str | Path | None = None,
    hard_negative_manifests: list[str | Path] | None = None,
    frame_stride: int = 5,
    negative_per_positive: int = 1,
    t: int = 5,
    seed: int = 71,
    max_samples: int | None = None,
    max_hard_negatives: int | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_model = _load_torch_model(CropRecognizer(), crop_weights)
    feature_model = _load_torch_model(FeatureRecognitionModel(), feature_weights)
    temporal_model = _load_torch_model(TemporalRecognizer(), temporal_weights)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    frames_cache: dict[str, list[np.ndarray]] = {}

    def evaluate_one(video: Path, frames: list[np.ndarray], frame_id: int, gt_class: str, bbox: tuple[float, float, float, float], metadata: str = "") -> None:
        frame = frames[frame_id]
        crop = crop_with_context(frame, bbox)
        tube = extract_temporal_tube(frames[: frame_id + 1], bbox, T=t)
        crop_p = _crop_probs(crop_model, crop)
        feat_p = _feature_probs(feature_model, frame, bbox)
        temp_p = _temporal_probs(temporal_model, tube)
        rec = fuse_rule_based(
            objectness=0.9,
            crop_probs=crop_p,
            feature_probs=feat_p,
            temporal_probs=temp_p,
            motion_score=0.0,
            alignment_quality=0.75,
            track_score=0.8 if gt_class == "drone" else 0.2,
            mode="static_or_hovering",
        )
        rows.append(
            {
                "metadata": metadata,
                "video": str(video),
                "frame_id": frame_id,
                "bbox_xyxy": list(bbox),
                "gt_class": gt_class,
                "crop_pred": _argmax(crop_p),
                "feature_pred": _argmax(feat_p),
                "temporal_pred": _argmax(temp_p),
                "fusion_pred": rec.predicted_class,
                "diagnostic_cause": rec.diagnostic_cause,
                "crop_probs": crop_p,
                "feature_probs": feat_p,
                "temporal_probs": temp_p,
                "final_probs": rec.final_probs,
                "disagreement": rec.disagreement,
                "final_drone_score": rec.final_drone_score,
            }
        )

    for meta_like in metadata_paths:
        meta_path = Path(meta_like)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        video = Path(meta["output"])
        if not video.exists():
            video = meta_path.parent / video.name
        frames = _read_frames(video)
        if not frames:
            continue
        frames_cache[str(video)] = frames
        boxes = {int(b["frame_id"]): [float(v) for v in b["bbox_xyxy"]] for b in meta.get("boxes", [])}
        h, w = frames[0].shape[:2]
        for frame_id in sorted(boxes):
            if frame_id < t - 1 or frame_id % max(1, frame_stride) != 0:
                continue
            sample_boxes: list[tuple[str, list[float]]] = [("drone", boxes[frame_id])]
            for _ in range(max(0, negative_per_positive)):
                sample_boxes.append(("background", _random_negative_box(rng, w, h, boxes[frame_id])))
            for gt_class, box_list in sample_boxes:
                bbox = tuple(float(v) for v in box_list)
                evaluate_one(video, frames, frame_id, gt_class, bbox, metadata=str(meta_path))
                if max_samples is not None and len(rows) >= max_samples:
                    break
            if max_samples is not None and len(rows) >= max_samples:
                break
        if max_samples is not None and len(rows) >= max_samples:
            break

    hard_count = 0
    for manifest_like in hard_negative_manifests or []:
        manifest_path = Path(manifest_like)
        hard_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for hard in hard_rows:
            if max_hard_negatives is not None and hard_count >= max_hard_negatives:
                break
            video = Path(hard["source_video"])
            frame_id = int(hard["frame_id"])
            if frame_id < t - 1:
                continue
            key = str(video)
            if key not in frames_cache:
                frames_cache[key] = _read_frames(video)
            frames = frames_cache[key]
            if frame_id >= len(frames):
                continue
            gt_class = str(hard.get("class", "background"))
            if gt_class not in CLASSES:
                gt_class = "background"
            bbox = tuple(float(v) for v in hard["bbox_xyxy"])
            evaluate_one(video, frames, frame_id, gt_class, bbox, metadata=str(manifest_path))
            hard_count += 1
            if max_samples is not None and len(rows) >= max_samples:
                break
        if max_hard_negatives is not None and hard_count >= max_hard_negatives:
            break
        if max_samples is not None and len(rows) >= max_samples:
            break

    (out_dir / "stage_b_oracle_predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    summary = {
        "num_samples": len(rows),
        "class_counts": dict(Counter(r["gt_class"] for r in rows)),
        "crop": _metrics(rows, "crop_pred"),
        "feature": _metrics(rows, "feature_pred"),
        "temporal": _metrics(rows, "temporal_pred"),
        "fusion": _metrics(rows, "fusion_pred"),
        "diagnostics": _diagnostic_metrics(rows),
        "mean_disagreement": float(np.mean([r["disagreement"] for r in rows])) if rows else 0.0,
        "mean_final_drone_score": float(np.mean([r["final_drone_score"] for r in rows])) if rows else 0.0,
    }
    (out_dir / "stage_b_oracle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
