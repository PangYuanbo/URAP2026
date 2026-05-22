from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qstr_dronedet.candidates import candidates_from_motion, candidates_from_yolo, candidates_from_yolo_tiled, merge_candidates
from qstr_dronedet.candidates.merge import bbox_iou, center_distance
from qstr_dronedet.candidates.yolo_p2_train import (
    build_class_agnostic_yolo_dataset,
    build_tiled_class_agnostic_yolo_dataset,
    train_yolo_p2,
    write_yolov8_p2_model_yaml,
)
from qstr_dronedet.data_augment import (
    build_static_hover_crop_dataset,
    build_static_hover_temporal_dataset,
    build_detector_proposal_stage_b_dataset,
    export_hard_negative_feature_csv,
    export_static_hover_feature_csv,
    export_static_hover_frames_csv,
    make_speed_augmented_frame_csv,
    make_speed_augmented_video_dir,
    make_speed_augmented_videos,
    make_moving_target_sample,
    make_static_hover_sample,
    mine_hard_negative_temporal_dataset,
    mine_motion_hard_negative_crops,
    split_feature_csv,
    split_folder_dataset,
)
from qstr_dronedet.evaluation.metrics import evaluate_predictions
from qstr_dronedet.evaluation.fusion_calibration import calibrate_fusion_from_diagnostics
from qstr_dronedet.evaluation.frame_failure_analysis import analyze_frame_failures
from qstr_dronedet.evaluation.stage_b_benchmark import run_stage_b_oracle_benchmark
from qstr_dronedet.evaluation.tracker_benchmark import run_tracker_oracle_benchmark
from qstr_dronedet.evaluation.tracklet_filter_sweep import run_tracklet_filter_sweep
from qstr_dronedet.evaluation.tracklet_model_selection import run_tracklet_model_selection
from qstr_dronedet.evaluation.tracklet_sequence_model_selection import run_tracklet_sequence_model_selection
from qstr_dronedet.features.roi import crop_with_context, extract_temporal_tube
from qstr_dronedet.fusion.modes import determine_mode
from qstr_dronedet.fusion.rule_fusion import fuse_rule_based
from qstr_dronedet.motion.difference import compute_multik_motion, motion_score_in_bbox
from qstr_dronedet.recognition.train import train_crop_recognizer, train_feature_recognizer, train_temporal_recognizer
from qstr_dronedet.real_data import (
    build_real_stage_b_datasets,
    build_real_detector_proposal_stage_b_dataset,
    build_real_yolo_candidate_dataset,
    ensure_real_data_layout,
    export_anti_uav300_subset_from_zip,
    extract_real_annotated_frames,
)
from qstr_dronedet.tracking.kalman import ConstantVelocityTracker
from qstr_dronedet.tracking.proposal_tracklets import build_proposal_tracklet_dataset
from qstr_dronedet.tracking.tracklet_classifier import (
    apply_tracklet_filter_to_infer_outputs,
    build_tracklet_dataset,
    evaluate_tracklet_classifier,
    train_tracklet_classifier,
)
from qstr_dronedet.types import CLASSES, AlignmentResult, DetectionCandidate, RecognitionResult, normalize_probs
from qstr_dronedet.visualization.draw import draw_overlay, make_side_by_side


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, AlignmentResult):
        d = asdict(obj)
        d["transform"] = _jsonable(obj.transform)
        return d
    if isinstance(obj, DetectionCandidate):
        return asdict(obj)
    if isinstance(obj, RecognitionResult):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if k != "map"}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def _open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return cap


def _limit_candidates(candidates: list[DetectionCandidate], limit: int | None) -> list[DetectionCandidate]:
    if limit is None or limit <= 0 or len(candidates) <= limit:
        return list(candidates)
    return sorted(candidates, key=lambda c: c.objectness, reverse=True)[:limit]


def _filter_candidates_by_box_size(
    candidates: list[DetectionCandidate],
    max_box_side: float | None = None,
    min_box_side: float | None = None,
) -> list[DetectionCandidate]:
    max_side_enabled = max_box_side is not None and max_box_side > 0
    min_side_enabled = min_box_side is not None and min_box_side > 0
    if not max_side_enabled and not min_side_enabled:
        return list(candidates)
    kept: list[DetectionCandidate] = []
    for cand in candidates:
        x1, y1, x2, y2 = cand.bbox_xyxy
        side = max(float(x2 - x1), float(y2 - y1))
        if max_side_enabled and side > float(max_box_side):
            continue
        if min_side_enabled and side < float(min_box_side):
            continue
        kept.append(cand)
    return kept


def _candidate_source_priority(candidate: DetectionCandidate) -> tuple[int, float]:
    source = candidate.source
    if "seed" in source:
        return (0, -candidate.objectness)
    if "track" in source:
        return (1, -candidate.objectness)
    if "yolo" in source:
        return (2, -candidate.objectness)
    if "motion" in source:
        return (3, -candidate.objectness)
    return (4, -candidate.objectness)


def _limit_merged_candidates(candidates: list[DetectionCandidate], limit: int | None) -> list[DetectionCandidate]:
    if limit is None or limit <= 0 or len(candidates) <= limit:
        return list(candidates)
    return sorted(candidates, key=_candidate_source_priority)[:limit]


def _run_yolo_candidates(
    frame: np.ndarray,
    weights: str | None,
    conf: float,
    tile_size: int,
    tile_stride: int,
    device: str | None,
    max_det: int,
    source_suffix: str | None = None,
) -> list[DetectionCandidate]:
    if tile_size and tile_size > 0:
        cands = candidates_from_yolo_tiled(
            frame,
            weights,
            tile_size=tile_size,
            stride=tile_stride,
            conf=conf,
            device=device,
            max_det=max_det,
        )
    else:
        cands = candidates_from_yolo(frame, weights, conf=conf)
    if source_suffix:
        for cand in cands:
            cand.extra["base_source"] = cand.source
            cand.source = f"{cand.source}_{source_suffix}"
    return cands


def _should_run_fallback_yolo(
    primary_candidates: list[DetectionCandidate],
    min_primary_candidates: int,
    trigger_objectness: float,
) -> bool:
    if len(primary_candidates) < min_primary_candidates:
        return True
    best = max((c.objectness for c in primary_candidates), default=0.0)
    return best < trigger_objectness


def _should_run_fallback_after_recognition(
    recognitions: list[RecognitionResult],
    trigger_final_score: float,
    primary_candidates: list[DetectionCandidate] | None = None,
    max_primary_objectness: float = 1.0,
) -> bool:
    if trigger_final_score <= 0:
        return False
    if primary_candidates and max_primary_objectness < 1.0:
        best_primary_objectness = max((c.objectness for c in primary_candidates), default=0.0)
        if best_primary_objectness > max_primary_objectness:
            return False
    best = max((r.final_drone_score for r in recognitions), default=0.0)
    return best < trigger_final_score


def _safe_default_probs(candidate: DetectionCandidate, tube: np.ndarray | None = None) -> dict[str, float]:
    probs = {c: 0.02 for c in CLASSES}
    probs["unknown"] = 0.55
    probs["background"] = 0.18
    if candidate.objectness > 0.45:
        probs["drone"] += 0.08
    if candidate.track_score > 0.55:
        probs["drone"] += 0.08
        probs["unknown"] -= 0.05
    if candidate.motion_score > 0.12 and candidate.alignment_quality > 0.3:
        probs["drone"] += 0.06
    if candidate.alignment_quality < 0.25 and candidate.motion_score > 0.15:
        probs["alignment_artifact"] += 0.18
    if candidate.mode == "static_or_hovering" and candidate.track_score > 0.6:
        probs["drone"] += 0.10
        probs["unknown"] -= 0.08
    return normalize_probs(probs)


def _load_crop_model(weights: str | None):
    if not weights:
        return None
    try:
        import torch

        from qstr_dronedet.recognition.crop_recognizer import CropRecognizer

        ckpt = torch.load(weights, map_location="cpu")
        model = CropRecognizer()
        model.load_state_dict(ckpt["state_dict"])
        model.qstr_target_mode = ckpt.get("target_mode", "multiclass")
        model.eval()
        return model
    except Exception as exc:
        print(f"Warning: could not load crop recognizer weights {weights}: {exc}")
        return None


def _load_feature_model(weights: str | None):
    if not weights:
        return None
    try:
        import torch

        from qstr_dronedet.recognition.feature_recognizer import FeatureRecognitionModel

        ckpt = torch.load(weights, map_location="cpu")
        model = FeatureRecognitionModel()
        model.load_state_dict(ckpt["state_dict"])
        model.qstr_target_mode = ckpt.get("target_mode", "multiclass")
        model.eval()
        return model
    except Exception as exc:
        print(f"Warning: could not load feature recognizer weights {weights}: {exc}")
        return None


def _load_temporal_model(weights: str | None):
    if not weights:
        return None
    try:
        import torch

        from qstr_dronedet.recognition.temporal_recognizer import TemporalRecognizer

        ckpt = torch.load(weights, map_location="cpu")
        model = TemporalRecognizer()
        model.load_state_dict(ckpt["state_dict"])
        model.qstr_target_mode = ckpt.get("target_mode", "multiclass")
        model.eval()
        return model
    except Exception as exc:
        print(f"Warning: could not load temporal recognizer weights {weights}: {exc}")
        return None


def _predict_crop_probs(model, crop_bgr: np.ndarray) -> dict[str, float]:
    try:
        import torch

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            if getattr(model, "qstr_target_mode", "multiclass") == "drone_binary":
                p = logits[:, [CLASSES.index("drone"), CLASSES.index("background")]].softmax(1)[0].cpu().numpy().tolist()
                return normalize_probs({"drone": float(p[0]), "background": float(p[1])})
            probs = logits.softmax(1)[0].cpu().numpy().tolist()
        return normalize_probs({cls: float(probs[i]) for i, cls in enumerate(CLASSES)})
    except Exception as exc:
        print(f"Warning: crop recognizer inference failed: {exc}")
        return normalize_probs({"unknown": 1.0})


def _predict_feature_probs(model, frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> dict[str, float]:
    try:
        import torch

        img = cv2.resize(frame_bgr, (640, 640))
        sx = 640.0 / frame_bgr.shape[1]
        sy = 640.0 / frame_bgr.shape[0]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        box = torch.tensor([[bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]], dtype=torch.float32)
        with torch.no_grad():
            logits = model(x, box)
            if getattr(model, "qstr_target_mode", "multiclass") == "drone_binary":
                p = logits[:, [CLASSES.index("drone"), CLASSES.index("background")]].softmax(1)[0].cpu().numpy().tolist()
                return normalize_probs({"drone": float(p[0]), "background": float(p[1])})
            probs = logits.softmax(1)[0].cpu().numpy().tolist()
        return normalize_probs({cls: float(probs[i]) for i, cls in enumerate(CLASSES)})
    except Exception as exc:
        print(f"Warning: feature recognizer inference failed: {exc}")
        return normalize_probs({"unknown": 1.0})


def _predict_temporal_probs(model, tube: np.ndarray) -> dict[str, float]:
    try:
        import torch

        x = torch.from_numpy(tube).unsqueeze(0).float()
        with torch.no_grad():
            logits = model(x)
            if getattr(model, "qstr_target_mode", "multiclass") == "drone_binary":
                p = logits[:, [CLASSES.index("drone"), CLASSES.index("background")]].softmax(1)[0].cpu().numpy().tolist()
                return normalize_probs({"drone": float(p[0]), "background": float(p[1])})
            probs = logits.softmax(1)[0].cpu().numpy().tolist()
        return normalize_probs({cls: float(probs[i]) for i, cls in enumerate(CLASSES)})
    except Exception as exc:
        print(f"Warning: temporal recognizer inference failed: {exc}")
        return normalize_probs({"unknown": 1.0})


def _read_video_frames(path: str) -> tuple[list[np.ndarray], float]:
    cap = _open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, float(fps)


def cmd_motion_debug(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, _ = _read_video_frames(args.video)
    diag_path = out / "diagnostics.jsonl"
    with diag_path.open("w", encoding="utf-8") as log:
        for i in range(len(frames)):
            if i == 0:
                continue
            motion = compute_multik_motion(frames[: i + 1], i, tuple(args.k_values))
            motion_map = motion["motion_map"]
            cv2.imwrite(str(out / f"motion_{i:06d}.png"), motion_map)
            cands = candidates_from_motion(motion_map)
            for c in cands:
                c.alignment_quality = motion["best_quality"]
            overlay = draw_overlay(frames[i], cands)
            cv2.imwrite(str(out / f"overlay_{i:06d}.jpg"), make_side_by_side(frames[i], motion_map, overlay))
            row = {
                "frame_id": i,
                "best_k": motion["best_k"],
                "best_quality": motion["best_quality"],
                "per_k": {k: _jsonable(v) for k, v in motion["per_k"].items()},
            }
            log.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_infer(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vis_dir = out / "frames"
    vis_dir.mkdir(exist_ok=True)
    frames, fps = _read_video_frames(args.video)
    fusion_weights = _load_fusion_calibration(args.fusion_calibration)
    crop_model = _load_crop_model(args.crop_weights)
    feature_model = _load_feature_model(args.feature_weights)
    temporal_model = _load_temporal_model(args.temporal_weights)
    tracker = ConstantVelocityTracker()
    pred_f = (out / "predictions.jsonl").open("w", encoding="utf-8")
    diag_f = (out / "diagnostics.jsonl").open("w", encoding="utf-8")
    writer = None
    def prepare_candidates(raw_candidates: list[DetectionCandidate]) -> list[DetectionCandidate]:
        for cand in raw_candidates:
            cand.motion_score = max(cand.motion_score, motion_score_in_bbox(motion_map, cand.bbox_xyxy))
            cand.alignment_quality = float(motion["best_quality"])
        return _limit_merged_candidates(merge_candidates(raw_candidates), args.max_candidates_per_frame)

    def recognize_candidates(candidates: list[DetectionCandidate]) -> list[RecognitionResult]:
        recognitions: list[RecognitionResult] = []
        for cand in candidates:
            track_speed = tracker.compute_track_speed()
            track_conf = max(cand.track_score, tracker.track_confidence())
            blur_score = 0.0
            if motion["best_k"] is not None:
                aln = motion["per_k"][motion["best_k"]]["alignment"]
                blur_score = getattr(aln, "blur_score", 0.0) if aln is not None else 0.0
            cand.mode = determine_mode(cand.motion_score, cand.alignment_quality, track_speed, blur_score, track_conf)
            crop = crop_with_context(frame, cand.bbox_xyxy, scale=args.recognition_crop_scale)
            tube = extract_temporal_tube(frames[: i + 1], cand.bbox_xyxy, scale=args.recognition_tube_scale)
            crop_probs = _predict_crop_probs(crop_model, crop) if crop_model is not None else _safe_default_probs(cand, tube)
            feature_probs = _predict_feature_probs(feature_model, frame, cand.bbox_xyxy) if feature_model is not None else _safe_default_probs(cand, tube)
            temporal_probs = _predict_temporal_probs(temporal_model, tube) if temporal_model is not None else _safe_default_probs(cand, tube)
            if cand.mode == "static_or_hovering":
                temporal_probs["drone"] += 0.08
                temporal_probs = normalize_probs(temporal_probs)
            recognitions.append(
                fuse_rule_based(
                    cand.objectness,
                    crop_probs,
                    feature_probs,
                    temporal_probs,
                    cand.motion_score,
                    cand.alignment_quality,
                    cand.track_score,
                    cand.mode,
                candidate_source=cand.source,
                fusion_weights=fusion_weights,
                fallback_gate=not args.disable_fallback_gate,
                fallback_min_branch_drone=args.fallback_gate_min_branch_drone,
                fallback_min_crop_temporal_mean=args.fallback_gate_min_crop_temporal_mean,
                fallback_max_negative_evidence=args.fallback_gate_max_negative_evidence,
                verified_objectness=not args.disable_verified_objectness,
                verified_objectness_mode=args.verified_objectness_mode,
                verified_min_branch_drone=args.verified_min_branch_drone,
                verified_min_crop_temporal_mean=args.verified_min_crop_temporal_mean,
                verified_max_negative_evidence=args.verified_max_negative_evidence,
                verified_objectness_floor=args.verified_objectness_floor,
                hard_tiny_recovery=args.enable_hard_tiny_recovery,
                hard_tiny_min_crop_drone=args.hard_tiny_min_crop_drone,
                hard_tiny_min_temporal_drone=args.hard_tiny_min_temporal_drone,
                hard_tiny_min_temporal_crop_delta=args.hard_tiny_min_temporal_crop_delta,
                hard_tiny_max_bg_minus_drone=args.hard_tiny_max_bg_minus_drone,
                hard_tiny_min_support=args.hard_tiny_min_support,
                hard_tiny_score_floor=args.hard_tiny_score_floor,
                hard_tiny_allow_tracker_only=args.hard_tiny_allow_tracker_only,
                hard_tiny_require_validated_track=not args.hard_tiny_disable_track_validation,
                hard_tiny_max_track_frames_since_detector=args.hard_tiny_max_track_frames_since_detector,
                hard_tiny_min_track_detector_updates=args.hard_tiny_min_track_detector_updates,
                hard_tiny_max_track_drift=args.hard_tiny_max_track_drift,
                hard_tiny_min_track_history=args.hard_tiny_min_track_history,
                candidate_extra=cand.extra,
            )
        )
        return recognitions

    for i, frame in enumerate(frames):
        if args.max_frames is not None and i >= args.max_frames:
            break
        if i == 0:
            motion = {"motion_map": np.zeros(frame.shape[:2], np.uint8), "best_quality": 0.0, "best_k": None, "per_k": {}}
        else:
            motion = compute_multik_motion(frames[: i + 1], i, tuple(args.k_values))
        motion_map = motion["motion_map"]
        motion_cands = [] if args.disable_motion_candidates else candidates_from_motion(motion_map, min_area=args.min_area, max_area=args.max_area)
        yolo_cands = _run_yolo_candidates(
            frame,
            args.yolo_weights,
            args.yolo_conf,
            args.yolo_tile_size,
            args.yolo_tile_stride,
            args.yolo_device,
            args.yolo_max_det,
        )
        yolo_cands = _limit_candidates(yolo_cands, args.max_yolo_candidates_per_frame)
        fallback_ran = False
        if args.fallback_yolo_weights and _should_run_fallback_yolo(
            yolo_cands,
            args.fallback_min_primary_candidates,
            args.fallback_trigger_objectness,
        ):
            fallback_cands = _run_yolo_candidates(
                frame,
                args.fallback_yolo_weights,
                args.fallback_yolo_conf,
                args.fallback_yolo_tile_size or args.yolo_tile_size,
                args.fallback_yolo_tile_stride or args.yolo_tile_stride,
                args.yolo_device,
                args.yolo_max_det,
                source_suffix="fallback",
            )
            fallback_cands = _filter_candidates_by_box_size(
                fallback_cands,
                max_box_side=args.fallback_max_box_side,
                min_box_side=args.fallback_min_box_side,
            )
            fallback_cands = _limit_candidates(fallback_cands, args.max_fallback_yolo_candidates_per_frame)
            for cand in fallback_cands:
                cand.extra["fallback_reason"] = {
                    "primary_count": len(yolo_cands),
                    "primary_best_objectness": max((c.objectness for c in yolo_cands), default=0.0),
                    "min_primary_candidates": args.fallback_min_primary_candidates,
                    "trigger_objectness": args.fallback_trigger_objectness,
                }
            yolo_cands = yolo_cands + fallback_cands
            fallback_ran = True
        track_cands = tracker.get_track_candidates()
        seed_cands: list[DetectionCandidate] = []
        if args.seed_box is not None:
            seed_cands.append(
                DetectionCandidate(
                    bbox_xyxy=tuple(float(v) for v in args.seed_box),
                    objectness=float(args.seed_objectness),
                    source="seed",
                    track_score=float(args.seed_track_score),
                    extra={"oracle_seed": True},
                )
            )
        raw_candidates = motion_cands + yolo_cands + track_cands + seed_cands
        candidates = prepare_candidates(raw_candidates)
        recognitions = recognize_candidates(candidates)
        if args.fallback_yolo_weights and not fallback_ran and _should_run_fallback_after_recognition(
            recognitions,
            args.fallback_trigger_final_score,
            primary_candidates=yolo_cands,
            max_primary_objectness=args.fallback_post_trigger_max_primary_objectness,
        ):
            fallback_cands = _run_yolo_candidates(
                frame,
                args.fallback_yolo_weights,
                args.fallback_yolo_conf,
                args.fallback_yolo_tile_size or args.yolo_tile_size,
                args.fallback_yolo_tile_stride or args.yolo_tile_stride,
                args.yolo_device,
                args.yolo_max_det,
                source_suffix="fallback",
            )
            fallback_cands = _filter_candidates_by_box_size(
                fallback_cands,
                max_box_side=args.fallback_max_box_side,
                min_box_side=args.fallback_min_box_side,
            )
            fallback_cands = _limit_candidates(fallback_cands, args.max_fallback_yolo_candidates_per_frame)
            for cand in fallback_cands:
                cand.extra["fallback_reason"] = {
                    "primary_best_final_drone_score": max((r.final_drone_score for r in recognitions), default=0.0),
                    "trigger_final_score": args.fallback_trigger_final_score,
                    "primary_best_objectness": max((c.objectness for c in yolo_cands), default=0.0),
                    "post_trigger_max_primary_objectness": args.fallback_post_trigger_max_primary_objectness,
                }
            raw_candidates = raw_candidates + fallback_cands
            candidates = prepare_candidates(raw_candidates)
            recognitions = recognize_candidates(candidates)
            fallback_ran = True
        external_raw_candidates = [
            c for c in raw_candidates
            if c.source != "tracker" and not (c.source == "motion" and c.objectness < 0.15)
        ]
        merged_track_updates = [
            c for c in candidates
            if "tracker" in c.source and c.source != "tracker" and any(s in c.source for s in ("yolo", "fallback", "motion", "seed"))
        ]
        tracker_update_candidates = external_raw_candidates + merged_track_updates
        tracker.update(tracker_update_candidates, alignment_quality=float(motion["best_quality"]))
        for cand, rec in zip(candidates, recognitions):
            if cand.extra.get("track_id") is not None:
                tracker.update_evidence(cand.extra.get("track_id"), rec.crop_probs, rec.temporal_probs, rec.final_probs)
        for cand, rec in zip(candidates, recognitions):
            row = {
                "frame_id": i,
                "bbox": list(cand.bbox_xyxy),
                "objectness": cand.objectness,
                "source": cand.source,
                "motion_score": cand.motion_score,
                "alignment_quality": cand.alignment_quality,
                "track_score": cand.track_score,
                "track_id": cand.extra.get("track_id"),
                "track_age": cand.extra.get("track_age"),
                "track_history_len": cand.extra.get("track_history_len"),
                "track_detector_updates": cand.extra.get("track_detector_updates"),
                "track_last_detector_source": cand.extra.get("track_last_detector_source"),
                "track_frames_since_detector_update": cand.extra.get("track_frames_since_detector_update"),
                "track_drift": cand.extra.get("track_drift"),
                "track_validated": cand.extra.get("track_validated"),
                "track_evidence_len": cand.extra.get("track_evidence_len"),
                "track_crop_drone_mean": cand.extra.get("track_crop_drone_mean"),
                "track_temporal_drone_mean": cand.extra.get("track_temporal_drone_mean"),
                "track_background_mean": cand.extra.get("track_background_mean"),
                "track_temporal_gain_rate": cand.extra.get("track_temporal_gain_rate"),
                "track_recognition_confirmed": cand.extra.get("track_recognition_confirmed"),
                "mode": cand.mode,
                "final_drone_score": rec.final_drone_score,
                "predicted_class": rec.predicted_class,
                "diagnostic_cause": rec.diagnostic_cause,
                "final_probs": rec.final_probs,
                "fallback_yolo_ran": fallback_ran,
            }
            pred_f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
            diag = {**row, "crop_probs": rec.crop_probs, "feature_probs": rec.feature_probs, "temporal_probs": rec.temporal_probs, "disagreement": rec.disagreement, "error_type": rec.error_type}
            diag_f.write(json.dumps(_jsonable(diag), ensure_ascii=False) + "\n")
        overlay = draw_overlay(frame, candidates, recognitions)
        side = make_side_by_side(frame, motion_map, overlay)
        cv2.imwrite(str(vis_dir / f"frame_{i:06d}.jpg"), side)
        if args.save_video:
            if writer is None:
                h, w = side.shape[:2]
                writer = cv2.VideoWriter(str(out / "visualization.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(side)
    if writer is not None:
        writer.release()
    pred_f.close()
    diag_f.close()
    if args.tracklet_classifier_weights:
        summary = apply_tracklet_filter_to_infer_outputs(
            out / "predictions.jsonl",
            out / "diagnostics.jsonl",
        args.tracklet_classifier_weights,
        threshold=args.tracklet_classifier_threshold,
        untracked_policy=args.tracklet_filter_untracked,
        promote_positive_tracklets=not args.disable_tracklet_promotion,
        promotion_score_floor=args.tracklet_promotion_score_floor,
        promotion_min_branch_drone=args.tracklet_promotion_min_branch_drone,
        promotion_max_background=args.tracklet_promotion_max_background,
        selective_promotion=args.tracklet_selective_promotion,
        selective_min_temporal_crop_delta=args.tracklet_selective_min_temporal_crop_delta,
        selective_min_temporal_background_margin=args.tracklet_selective_min_temporal_background_margin,
        selective_max_tracklet_background=args.tracklet_selective_max_tracklet_background,
        selective_max_tracklet_objectness=args.tracklet_selective_max_tracklet_objectness,
        selective_min_tracklet_rows=args.tracklet_selective_min_tracklet_rows,
        selective_min_temporal_gain_rate=args.tracklet_selective_min_temporal_gain_rate,
        selective_min_weak_detector_temporal_signal=args.tracklet_selective_min_weak_detector_temporal_signal,
        selective_require_recovery_source=not args.tracklet_selective_allow_non_recovery_source,
        selective_max_promoted_tracklets_per_sequence=args.tracklet_selective_max_promoted_tracklets_per_sequence,
    )
        print(json.dumps(summary, indent=2))


def _load_fusion_calibration(path: str | None) -> dict[str, float] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "weights" in data:
        weights = data["weights"]
    else:
        best = data.get("best") or []
        if not best:
            raise ValueError(f"No fusion weights found in calibration file: {path}")
        weights = best[0]["weights"]
    return {str(k): float(v) for k, v in weights.items()}


def _iter_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cmd_build_crop_dataset(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _iter_csv_rows(Path(args.annotations))
    for row in rows:
        cls = row.get("class", "unknown")
        dst_dir = out / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        frame_path = Path(row["frame_path"])
        if not frame_path.is_absolute() and args.frames:
            frame_path = Path(args.frames) / frame_path
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        bbox = tuple(float(row[k]) for k in ["x1", "y1", "x2", "y2"])
        crop = crop_with_context(frame, bbox, scale=args.scale, out_size=args.size)
        cv2.imwrite(str(dst_dir / f"{frame_path.stem}_{len(list(dst_dir.glob('*.jpg'))):06d}.jpg"), crop)


def cmd_train_recognizer(args: argparse.Namespace) -> None:
    if args.type == "crop":
        train_crop_recognizer(args.data, args.out, args.epochs, args.balance, args.target_mode)
    elif args.type == "temporal":
        train_temporal_recognizer(args.data, args.out, args.epochs, args.balance, args.target_mode)
    elif args.type == "feature":
        train_feature_recognizer(args.data, args.out, args.epochs, args.target_mode)
    else:
        raise ValueError(f"Unsupported recognizer type: {args.type}")


def cmd_build_yolo_candidate_dataset(args: argparse.Namespace) -> None:
    data_yaml = build_class_agnostic_yolo_dataset(args.annotations, args.out, args.images_root, args.val_fraction, args.seed, args.min_box_px)
    print(f"Wrote class-agnostic YOLO dataset: {data_yaml}")


def cmd_build_tiled_yolo_candidate_dataset(args: argparse.Namespace) -> None:
    data_yaml = build_tiled_class_agnostic_yolo_dataset(
        args.annotations,
        args.out,
        images_root=args.images_root,
        tile_size=args.tile_size,
        positives_per_box=args.positives_per_box,
        negatives_per_image=args.negatives_per_image,
        val_fraction=args.val_fraction,
        seed=args.seed,
        min_box_px=args.min_box_px,
        negative_pad_px=args.negative_pad_px,
        photometric_augmentations=args.photometric_augmentations,
        low_contrast_injections=args.low_contrast_injections,
        positive_repeat_patterns=tuple(args.positive_repeat_pattern or ()),
        positive_repeat_factor=args.positive_repeat_factor,
    )
    print(f"Wrote tiled class-agnostic YOLO dataset: {data_yaml}")


def cmd_train_yolo_p2(args: argparse.Namespace) -> None:
    model_yaml = args.model_yaml
    if args.write_model_yaml:
        model_yaml = str(write_yolov8_p2_model_yaml(args.write_model_yaml))
        print(f"Wrote YOLO-P2 model yaml: {model_yaml}")
        if args.no_train:
            return
    run_dir = train_yolo_p2(args.data, args.out, model_yaml, args.pretrained, args.epochs, args.imgsz, args.batch, args.device)
    print(f"YOLO-P2 candidate training run: {run_dir}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    metrics = evaluate_predictions(args.pred, args.gt, args.out)
    print(json.dumps(metrics, indent=2))


def cmd_build_tracklet_dataset(args: argparse.Namespace) -> None:
    result = build_tracklet_dataset(
        args.diagnostics,
        args.gt_csv,
        args.out,
        max_frames=args.max_frames,
        iou_threshold=args.iou_threshold,
        center_threshold=args.center_threshold,
    )
    print(json.dumps({"csv": str(result.csv_path), "jsonl": str(result.json_path), **result.summary}, indent=2))


def cmd_build_proposal_tracklet_dataset(args: argparse.Namespace) -> None:
    result = build_proposal_tracklet_dataset(
        args.run_roots,
        args.gt_csv,
        args.out,
        profile=args.profile,
        diagnostics_name=args.diagnostics_name,
        max_frames=args.max_frames,
        max_gap=args.max_gap,
        base_radius=args.base_radius,
        radius_per_side=args.radius_per_side,
        min_iou=args.min_iou,
        min_score=args.min_score,
        detector_only=args.detector_only,
        min_tracklet_rows=args.min_tracklet_rows,
        iou_threshold=args.iou_threshold,
        center_threshold=args.center_threshold,
        hard_tiny_side=args.hard_tiny_side,
        hard_low_score=args.hard_low_score,
    )
    print(json.dumps({"csv": str(result.csv_path), "jsonl": str(result.json_path), **result.summary}, indent=2))


def cmd_train_tracklet_classifier(args: argparse.Namespace) -> None:
    out = train_tracklet_classifier(
        args.csv,
        args.out,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        hard_tiny_positive_augments=args.hard_tiny_positive_augments,
    )
    print(f"Wrote tracklet classifier: {out}")


def cmd_eval_tracklet_classifier(args: argparse.Namespace) -> None:
    metrics = evaluate_tracklet_classifier(args.csv, args.weights, args.out, threshold=args.threshold)
    print(json.dumps(metrics, indent=2))


def cmd_stage_b_oracle_benchmark(args: argparse.Namespace) -> None:
    metadata = _expand_paths(args.metadata)
    if not metadata:
        raise FileNotFoundError(f"No metadata files matched: {args.metadata}")
    summary = run_stage_b_oracle_benchmark(
        metadata,
        args.out,
        crop_weights=args.crop_weights,
        feature_weights=args.feature_weights,
        temporal_weights=args.temporal_weights,
        hard_negative_manifests=args.hard_negative_manifest,
        frame_stride=args.frame_stride,
        negative_per_positive=args.negative_per_positive,
        t=args.t,
        seed=args.seed,
        max_samples=args.max_samples,
        max_hard_negatives=args.max_hard_negatives,
    )
    print(json.dumps(summary, indent=2))


def cmd_tracker_oracle_benchmark(args: argparse.Namespace) -> None:
    metadata = _expand_paths(args.metadata)
    if not metadata:
        raise FileNotFoundError(f"No metadata files matched: {args.metadata}")
    summary = run_tracker_oracle_benchmark(
        metadata,
        args.out,
        detection_stride=args.detection_stride,
        max_frames=args.max_frames,
        match_iou=args.match_iou,
        match_center_px=args.match_center_px,
        tracker_r0=args.tracker_r0,
        tracker_alpha=args.tracker_alpha,
        tracker_beta=args.tracker_beta,
        tracker_reacquire=args.tracker_reacquire,
    )
    print(json.dumps(summary, indent=2))


def cmd_mode_sweep(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    counts: dict[str, int] = {}
    with Path(args.diagnostics).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            best_k = row.get("best_k")
            alignment = None
            if best_k is not None:
                alignment = row.get("per_k", {}).get(str(best_k), {}).get("alignment")
            q = float(row.get("best_quality", 0.0))
            residual = float(alignment.get("photometric_residual", 0.0)) if alignment else 0.0
            blur = float(alignment.get("blur_score", 0.0)) if alignment else 0.0
            inlier = float(alignment.get("inlier_ratio", 0.0)) if alignment else 0.0
            # Motion-debug does not log candidate motion scores, so use residual as a proxy for frame motion here.
            motion_proxy = min(1.0, residual / max(args.high_residual_threshold, 1e-6))
            mode = determine_mode(
                motion_score=motion_proxy,
                alignment_quality=q,
                track_speed=0.0,
                blur_score=blur,
                track_confidence=0.0,
                static_motion_threshold=args.static_motion_threshold,
            )
            if q < args.weak_alignment_q_threshold and mode == "normal":
                mode = "fast_target"
            counts[mode] = counts.get(mode, 0) + 1
            rows_out.append(
                {
                    "frame_id": row.get("frame_id"),
                    "mode": mode,
                    "alignment_quality": q,
                    "motion_proxy": motion_proxy,
                    "photometric_residual": residual,
                    "inlier_ratio": inlier,
                    "blur_score": blur,
                    "best_k": best_k,
                }
            )
    (out / "mode_labels.jsonl").write_text("\n".join(json.dumps(r) for r in rows_out) + "\n", encoding="utf-8")
    summary = {"counts": counts, "num_frames": len(rows_out)}
    (out / "mode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cmd_augment_speed(args: argparse.Namespace) -> None:
    if args.input_dir:
        outputs = make_speed_augmented_video_dir(
            args.input_dir,
            args.out,
            args.strides,
            args.pattern,
            args.max_videos,
            args.keep_fps,
            args.motion_blur,
            args.blur_direction,
        )
        for p in outputs:
            print(f"Wrote {p}")
    if args.video:
        outputs = make_speed_augmented_videos(args.video, args.out, args.strides, args.keep_fps, args.motion_blur, args.blur_direction)
        for p in outputs:
            print(f"Wrote {p}")
    if args.annotations:
        out_csv = Path(args.out) / "speed_augmented_annotations.csv"
        make_speed_augmented_frame_csv(args.annotations, out_csv, args.strides, args.frame_index_column)
        print(f"Wrote {out_csv}")
    if not args.input_dir and not args.video and not args.annotations:
        raise ValueError("Provide --input-dir, --video, and/or --annotations")


def cmd_make_static_hover(args: argparse.Namespace) -> None:
    out = make_static_hover_sample(
        args.video,
        args.out_video,
        args.x,
        args.y,
        args.radius,
        tuple(args.color),
        args.max_frames,
        args.freeze_background,
        args.jitter_px,
        args.seed,
    )
    print(f"Wrote {out}")


def cmd_make_moving_target(args: argparse.Namespace) -> None:
    out = make_moving_target_sample(
        args.video,
        args.out_video,
        args.start_x,
        args.start_y,
        args.vx,
        args.vy,
        args.radius,
        tuple(args.color),
        args.max_frames,
        args.freeze_background,
    )
    print(f"Wrote {out}")


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        paths.extend(matches)
    return paths


def cmd_build_static_hover_crops(args: argparse.Namespace) -> None:
    metadata = _expand_paths(args.metadata)
    if not metadata:
        raise FileNotFoundError(f"No metadata files matched: {args.metadata}")
    manifest = build_static_hover_crop_dataset(
        metadata,
        args.out,
        class_name=args.class_name,
        negative_class=args.negative_class,
        negative_per_positive=args.negative_per_positive,
        crop_scale=args.scale,
        crop_size=args.size,
        seed=args.seed,
    )
    print(f"Wrote crop dataset manifest: {manifest}")


def cmd_mine_hard_negative_crops(args: argparse.Namespace) -> None:
    videos = _expand_paths(args.video)
    excludes = _expand_paths(args.exclude_metadata) if args.exclude_metadata else []
    if not videos:
        raise FileNotFoundError(f"No videos matched: {args.video}")
    manifest = mine_motion_hard_negative_crops(
        videos,
        args.out,
        exclude_metadata_paths=excludes,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        max_crops_per_video=args.max_crops_per_video,
        min_area=args.min_area,
        max_area=args.max_area,
        min_motion_score=args.min_motion_score,
        artifact_q_threshold=args.artifact_q_threshold,
        crop_scale=args.scale,
        crop_size=args.size,
        exclude_padding_px=args.exclude_padding_px,
    )
    print(f"Wrote hard-negative crop manifest: {manifest}")


def cmd_build_static_hover_yolo_dataset(args: argparse.Namespace) -> None:
    metadata = _expand_paths(args.metadata)
    if not metadata:
        raise FileNotFoundError(f"No metadata files matched: {args.metadata}")
    export_dir = Path(args.out) / "source"
    csv_path = export_static_hover_frames_csv(metadata, export_dir, args.frame_stride, args.max_frames_per_video)
    if args.tiled:
        data_yaml = build_tiled_class_agnostic_yolo_dataset(
            csv_path,
            Path(args.out) / "yolo_tiled",
            images_root=None,
            tile_size=args.tile_size,
            positives_per_box=args.positives_per_box,
            negatives_per_image=args.negatives_per_image,
            val_fraction=args.val_fraction,
            seed=args.seed,
            min_box_px=args.min_box_px,
            negative_pad_px=args.negative_pad_px,
            photometric_augmentations=args.photometric_augmentations,
            low_contrast_injections=args.low_contrast_injections,
            positive_repeat_patterns=tuple(args.positive_repeat_pattern or ()),
            positive_repeat_factor=args.positive_repeat_factor,
        )
    else:
        data_yaml = build_class_agnostic_yolo_dataset(csv_path, Path(args.out) / "yolo", images_root=None, val_fraction=args.val_fraction, seed=args.seed, min_box_px=args.min_box_px)
    print(f"Wrote synthetic frame CSV: {csv_path}")
    print(f"Wrote YOLO data.yaml: {data_yaml}")


def cmd_build_static_hover_temporal(args: argparse.Namespace) -> None:
    metadata = _expand_paths(args.metadata)
    if not metadata:
        raise FileNotFoundError(f"No metadata files matched: {args.metadata}")
    manifest = build_static_hover_temporal_dataset(
        metadata,
        args.out,
        t=args.t,
        frame_stride=args.frame_stride,
        negative_per_positive=args.negative_per_positive,
        crop_scale=args.scale,
        crop_size=args.size,
        seed=args.seed,
    )
    print(f"Wrote temporal dataset manifest: {manifest}")


def cmd_build_detector_proposal_stage_b(args: argparse.Namespace) -> None:
    metadata = _expand_paths(args.metadata)
    if not metadata:
        raise FileNotFoundError(f"No metadata files matched: {args.metadata}")
    summary = build_detector_proposal_stage_b_dataset(
        metadata,
        args.out,
        args.yolo_weights,
        yolo_conf=args.yolo_conf,
        tile_size=args.yolo_tile_size,
        tile_stride=args.yolo_tile_stride,
        frame_stride=args.frame_stride,
        max_frames_per_video=args.max_frames_per_video,
        max_proposals_per_frame=args.max_proposals_per_frame,
        max_negatives_per_frame=args.max_negatives_per_frame,
        match_iou=args.match_iou,
        match_center_px=args.match_center_px,
        crop_scale=args.scale,
        crop_size=args.crop_size,
        tube_t=args.t,
        tube_size=args.tube_size,
    )
    print(json.dumps(summary, indent=2))


def cmd_build_static_hover_feature_dataset(args: argparse.Namespace) -> None:
    metadata = _expand_paths(args.metadata)
    if not metadata:
        raise FileNotFoundError(f"No metadata files matched: {args.metadata}")
    csv_path = export_static_hover_feature_csv(metadata, args.out, args.frame_stride, args.negative_per_positive, args.seed)
    print(f"Wrote feature ROI CSV: {csv_path}")


def cmd_mine_hard_negative_temporal(args: argparse.Namespace) -> None:
    manifest = mine_hard_negative_temporal_dataset(
        args.manifest,
        args.out,
        t=args.t,
        crop_scale=args.scale,
        crop_size=args.size,
        max_samples_per_class=args.max_samples_per_class,
    )
    print(f"Wrote hard-negative temporal manifest: {manifest}")


def cmd_split_folder_dataset(args: argparse.Namespace) -> None:
    manifest = split_folder_dataset(args.input, args.out, args.val_fraction, args.seed)
    print(f"Wrote split manifest: {manifest}")


def cmd_split_feature_csv(args: argparse.Namespace) -> None:
    train_csv, val_csv = split_feature_csv(args.csv, args.out, args.val_fraction, args.seed)
    print(f"Wrote train CSV: {train_csv}")
    print(f"Wrote val CSV: {val_csv}")


def cmd_build_hard_negative_feature_dataset(args: argparse.Namespace) -> None:
    manifests = _expand_paths(args.manifest)
    if not manifests:
        raise FileNotFoundError(f"No manifests matched: {args.manifest}")
    csv_path = export_hard_negative_feature_csv(manifests, args.out, args.max_samples_per_class)
    print(f"Wrote hard-negative feature CSV: {csv_path}")


def _metadata_video_and_boxes(metadata_path: Path) -> tuple[Path, dict[int, tuple[float, float, float, float]]]:
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    video = Path(meta["output"])
    if not video.exists():
        video = metadata_path.parent / video.name
    boxes = {
        int(row["frame_id"]): tuple(float(v) for v in row["bbox_xyxy"])
        for row in meta.get("boxes", [])
    }
    return video, boxes


def cmd_stage_a_yolo_recall(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    total = 0
    hit_iou = 0
    hit_center = 0
    for metadata in _expand_paths(args.metadata):
        video, boxes = _metadata_video_and_boxes(metadata)
        frames, _ = _read_video_frames(str(video))
        for frame_id, frame in enumerate(frames):
            if args.max_frames is not None and frame_id >= args.max_frames:
                break
            if frame_id % args.frame_stride != 0:
                continue
            gt = boxes.get(frame_id)
            if gt is None:
                continue
            total += 1
            cands = candidates_from_yolo_tiled(
                frame,
                args.yolo_weights,
                tile_size=args.yolo_tile_size,
                stride=args.yolo_tile_stride,
                conf=args.yolo_conf,
                device=args.device,
                max_det=args.max_det,
            )
            best_iou = max((bbox_iou(c.bbox_xyxy, gt) for c in cands), default=0.0)
            best_center = min((center_distance(c.bbox_xyxy, gt) for c in cands), default=float("inf"))
            ok_iou = best_iou >= args.match_iou
            ok_center = best_center <= args.match_center_px
            hit_iou += int(ok_iou)
            hit_center += int(ok_center)
            rows_out.append(
                {
                    "metadata": str(metadata),
                    "video": str(video),
                    "frame_id": frame_id,
                    "gt_bbox": list(gt),
                    "num_candidates": len(cands),
                    "best_iou": best_iou,
                    "best_center_distance": best_center,
                    "hit_iou": ok_iou,
                    "hit_center": ok_center,
                    "top_candidates": [
                        {"bbox": list(c.bbox_xyxy), "objectness": c.objectness, "source": c.source}
                        for c in sorted(cands, key=lambda x: x.objectness, reverse=True)[: args.keep_top]
                    ],
                }
            )
    summary = {
        "total_gt_frames": total,
        "recall_iou": hit_iou / max(1, total),
        "recall_center": hit_center / max(1, total),
        "match_iou": args.match_iou,
        "match_center_px": args.match_center_px,
        "yolo_weights": args.yolo_weights,
        "tile_size": args.yolo_tile_size,
        "tile_stride": args.yolo_tile_stride,
        "frame_stride": args.frame_stride,
    }
    with (out / "stage_a_yolo_recall.jsonl").open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cmd_init_real_data_layout(args: argparse.Namespace) -> None:
    dirs = ensure_real_data_layout(args.root)
    print("Created/verified real data layout:")
    for d in dirs:
        print(d)


def cmd_extract_real_annotated_frames(args: argparse.Namespace) -> None:
    csv_path = extract_real_annotated_frames(
        args.annotations,
        args.frames_dir,
        args.out_csv,
        video_root=args.video_root,
        strict_labels=not args.allow_unknown_labels,
    )
    print(f"Wrote frame annotations: {csv_path}")


def cmd_prepare_real_yolo_dataset(args: argparse.Namespace) -> None:
    result = build_real_yolo_candidate_dataset(
        args.annotations,
        args.out,
        video_root=args.video_root,
        tiled=not args.full_frame,
        tile_size=args.tile_size,
        positives_per_box=args.positives_per_box,
        negatives_per_image=args.negatives_per_image,
        val_fraction=args.val_fraction,
        seed=args.seed,
        min_box_px=args.min_box_px,
        negative_pad_px=args.negative_pad_px,
        strict_labels=not args.allow_unknown_labels,
    )
    print(f"Wrote frame annotations: {result.frame_annotations_csv}")
    print(f"Wrote frames: {result.frames_dir}")
    print(f"Wrote YOLO data yaml: {result.data_yaml}")
    print(f"Wrote summary: {result.summary_json}")


def cmd_export_anti_uav300_subset(args: argparse.Namespace) -> None:
    result = export_anti_uav300_subset_from_zip(
        args.zip,
        args.out,
        split=args.split,
        modality=args.modality,
        max_sequences=args.max_sequences,
        start_index=args.start_index,
        frame_stride=args.frame_stride,
        max_frames_per_sequence=args.max_frames_per_sequence,
    )
    print(f"Wrote annotations: {result.annotations_csv}")
    print(f"Wrote manifest: {result.manifest_csv}")
    print(f"Extracted videos under: {result.extracted_root}")
    print(f"Wrote summary: {result.summary_json}")


def cmd_build_real_stage_b_dataset(args: argparse.Namespace) -> None:
    result = build_real_stage_b_datasets(
        args.annotations,
        args.out,
        negative_per_positive=args.negative_per_positive,
        crop_scale=args.crop_scale,
        crop_size=args.crop_size,
        tube_t=args.tube_t,
        tube_size=args.tube_size,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    print(f"Wrote crop dataset: {result.crop_root}")
    print(f"Wrote temporal dataset: {result.temporal_root}")
    print(f"Wrote manifest: {result.manifest_jsonl}")
    print(f"Wrote summary: {result.summary_json}")


def cmd_build_real_detector_proposal_stage_b(args: argparse.Namespace) -> None:
    result = build_real_detector_proposal_stage_b_dataset(
        args.annotations,
        args.out,
        args.yolo_weights,
        yolo_conf=args.yolo_conf,
        fallback_yolo_weights=args.fallback_yolo_weights,
        fallback_yolo_conf=args.fallback_yolo_conf,
        tile_size=args.yolo_tile_size,
        tile_stride=args.yolo_tile_stride,
        max_samples=args.max_samples,
        max_proposals_per_frame=args.max_proposals_per_frame,
        max_fallback_proposals_per_frame=args.max_fallback_proposals_per_frame,
        max_negatives_per_frame=args.max_negatives_per_frame,
        match_iou=args.match_iou,
        match_center_px=args.match_center_px,
        artifact_score_threshold=args.artifact_score_threshold,
        high_score_fp_threshold=args.high_score_fp_threshold,
        non_drone_label_mode=args.non_drone_label_mode,
        hard_positive_max_size_px=args.hard_positive_max_size_px,
        hard_positive_max_score=args.hard_positive_max_score,
        hard_positive_repeat=args.hard_positive_repeat,
        crop_scale=args.crop_scale,
        crop_size=args.crop_size,
        tube_t=args.tube_t,
        tube_size=args.tube_size,
        device=args.device,
    )
    print(f"Wrote crop dataset: {result.crop_root}")
    print(f"Wrote temporal dataset: {result.temporal_root}")
    print(f"Wrote feature CSV: {result.feature_csv}")
    print(f"Wrote manifest: {result.manifest_jsonl}")
    print(f"Wrote summary: {result.summary_json}")


def cmd_calibrate_fusion(args: argparse.Namespace) -> None:
    summary = calibrate_fusion_from_diagnostics(
        args.diagnostics,
        args.gt,
        args.out,
        video_hint=args.video,
        match_iou=args.match_iou,
        match_center_px=args.match_center_px,
        threshold=args.threshold,
        frame_tolerance=args.frame_tolerance,
    )
    print(json.dumps(summary, indent=2))


def cmd_analyze_frame_failures(args: argparse.Namespace) -> None:
    summary = analyze_frame_failures(
        args.run_root,
        args.gt,
        args.out,
        profile=args.profile,
        raw_prediction_name=args.raw_prediction_name,
        filtered_prediction_name=args.filtered_prediction_name,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2))


def cmd_sweep_tracklet_filter(args: argparse.Namespace) -> None:
    summary = run_tracklet_filter_sweep(
        args.run_roots,
        args.gt,
        args.weights,
        args.out,
        profile=args.profile,
        classifier_thresholds=args.classifier_thresholds,
        promotion_score_floors=args.promotion_score_floors,
        promotion_max_backgrounds=args.promotion_max_backgrounds,
        promotion_min_branch_drone=args.promotion_min_branch_drone,
        selective_promotion=args.selective_promotion,
        selective_min_temporal_crop_deltas=args.selective_min_temporal_crop_deltas,
        selective_min_temporal_background_margins=args.selective_min_temporal_background_margins,
        selective_max_promoted_tracklets_per_sequence_values=args.selective_max_promoted_tracklets_per_sequence_values,
        selective_max_tracklet_background=args.selective_max_tracklet_background,
        selective_max_tracklet_objectness=args.selective_max_tracklet_objectness,
        selective_min_tracklet_rows=args.selective_min_tracklet_rows,
        selective_min_temporal_gain_rate=args.selective_min_temporal_gain_rate,
        selective_min_weak_detector_temporal_signal=args.selective_min_weak_detector_temporal_signal,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2))


def cmd_select_tracklet_model(args: argparse.Namespace) -> None:
    summary = run_tracklet_model_selection(
        args.tracklets,
        args.run_roots,
        args.gt,
        args.out,
        profile=args.profile,
        calib_seqs=args.calib_seqs,
        calib_seq_patterns=args.calib_seq_patterns,
        calib_fraction=args.calib_fraction,
        epochs_values=args.epochs_values,
        hidden_values=args.hidden_values,
        lr_values=args.lr_values,
        hard_tiny_positive_augments_values=args.hard_tiny_positive_augments_values,
        hard_negative_augments_values=args.hard_negative_augments_values,
        classifier_thresholds=args.classifier_thresholds,
        promotion_enabled_values=[bool(v) for v in args.promotion_enabled_values] if args.promotion_enabled_values is not None else None,
        promotion_score_floors=args.promotion_score_floors,
        promotion_max_backgrounds=args.promotion_max_backgrounds,
        promotion_min_branch_drone=args.promotion_min_branch_drone,
        selective_promotion=args.selective_promotion,
        selective_max_promoted_tracklets_per_sequence_values=args.selective_max_promoted_tracklets_per_sequence_values,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_frames=args.max_frames,
        max_recall_drop=args.max_recall_drop,
    )
    print(json.dumps(summary, indent=2))


def cmd_select_tracklet_sequence_model(args: argparse.Namespace) -> None:
    summary = run_tracklet_sequence_model_selection(
        args.tracklets_jsonl,
        args.run_roots,
        args.gt,
        args.out,
        profile=args.profile,
        calib_seqs=args.calib_seqs,
        calib_seq_patterns=args.calib_seq_patterns,
        calib_fraction=args.calib_fraction,
        epochs_values=args.epochs_values,
        hidden_values=args.hidden_values,
        max_len_values=args.max_len_values,
        hard_negative_augments_values=args.hard_negative_augments_values,
        classifier_thresholds=args.classifier_thresholds,
        promotion_enabled_values=[bool(v) for v in args.promotion_enabled_values] if args.promotion_enabled_values is not None else None,
        promotion_score_floors=args.promotion_score_floors,
        promotion_max_backgrounds=args.promotion_max_backgrounds,
        selective_promotion=args.selective_promotion,
        selective_max_promoted_tracklets_per_sequence_values=args.selective_max_promoted_tracklets_per_sequence_values,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_frames=args.max_frames,
        max_recall_drop=args.max_recall_drop,
    )
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qstr-dronedet", description="QSTR-DroneDet runnable research MVP")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("motion-debug")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 4])
    p.set_defaults(func=cmd_motion_debug)

    p = sub.add_parser("infer")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--yolo-weights", default=None)
    p.add_argument("--yolo-conf", type=float, default=0.05)
    p.add_argument("--yolo-tile-size", type=int, default=0, help="Run YOLO over tiled crops; 0 disables tiled inference")
    p.add_argument("--yolo-tile-stride", type=int, default=192)
    p.add_argument("--yolo-device", default=None, help="Optional YOLO device, e.g. 0 or cpu")
    p.add_argument("--yolo-max-det", type=int, default=300, help="Per YOLO predict max_det before QSTR proposal budgeting")
    p.add_argument("--max-yolo-candidates-per-frame", type=int, default=None, help="Keep only top-N primary YOLO candidates before merge")
    p.add_argument("--max-candidates-per-frame", type=int, default=None, help="Keep only top-N merged candidates per frame after source-priority sorting")
    p.add_argument("--fallback-yolo-weights", default=None, help="Optional fallback YOLO weights used only when primary detector has weak/no candidates")
    p.add_argument("--fallback-yolo-conf", type=float, default=0.15)
    p.add_argument("--fallback-yolo-tile-size", type=int, default=0, help="Fallback tiled inference size; 0 reuses --yolo-tile-size")
    p.add_argument("--fallback-yolo-tile-stride", type=int, default=0, help="Fallback tiled stride; 0 reuses --yolo-tile-stride")
    p.add_argument("--fallback-min-primary-candidates", type=int, default=1, help="Run fallback if primary YOLO returns fewer candidates than this")
    p.add_argument("--fallback-trigger-objectness", type=float, default=0.20, help="Run fallback if the best primary YOLO objectness is below this")
    p.add_argument("--fallback-trigger-final-score", type=float, default=0.0, help="Run fallback after primary recognition if best final drone score is below this; 0 disables post-fusion fallback")
    p.add_argument("--fallback-post-trigger-max-primary-objectness", type=float, default=1.0, help="For post-fusion fallback, require primary best objectness at or below this value; 1.0 disables this guard")
    p.add_argument("--max-fallback-yolo-candidates-per-frame", type=int, default=10, help="Keep only top-N fallback YOLO candidates before merge")
    p.add_argument("--fallback-max-box-side", type=float, default=0.0, help="Drop fallback proposals whose max box side exceeds this many pixels; 0 disables")
    p.add_argument("--fallback-min-box-side", type=float, default=0.0, help="Drop fallback proposals whose max box side is below this many pixels; 0 disables")
    p.add_argument("--disable-fallback-gate", action="store_true", help="Do not suppress fallback-source detections with weak crop/temporal or strong background/artifact evidence")
    p.add_argument("--fallback-gate-min-branch-drone", type=float, default=0.45, help="Fallback acceptance requires crop or temporal drone probability at least this high")
    p.add_argument("--fallback-gate-min-crop-temporal-mean", type=float, default=0.35, help="Fallback acceptance requires mean crop/temporal drone probability at least this high")
    p.add_argument("--fallback-gate-max-negative-evidence", type=float, default=0.55, help="Reject fallback if background/artifact evidence exceeds this")
    p.add_argument("--disable-verified-objectness", action="store_true", help="Do not raise effective objectness for tracker/fallback candidates verified by Stage B")
    p.add_argument("--verified-objectness-mode", choices=["always", "hard_recovery"], default="hard_recovery", help="hard_recovery boosts only fallback or low-objectness tracker recovery candidates; always preserves the older tracker/fallback behavior")
    p.add_argument("--verified-min-branch-drone", type=float, default=0.45)
    p.add_argument("--verified-min-crop-temporal-mean", type=float, default=0.48)
    p.add_argument("--verified-max-negative-evidence", type=float, default=0.62)
    p.add_argument("--verified-objectness-floor", type=float, default=0.55)
    p.add_argument("--enable-hard-tiny-recovery", action="store_true", help="Allow a narrow fallback/tracker recovery rule when crop+temporal support a tiny drone but fusion still predicts background")
    p.add_argument("--hard-tiny-min-crop-drone", type=float, default=0.40)
    p.add_argument("--hard-tiny-min-temporal-drone", type=float, default=0.55)
    p.add_argument("--hard-tiny-min-temporal-crop-delta", type=float, default=0.0, help="Require temporal drone probability to exceed crop drone probability by this margin")
    p.add_argument("--hard-tiny-max-bg-minus-drone", type=float, default=0.08)
    p.add_argument("--hard-tiny-min-support", type=float, default=0.15, help="Minimum tracker support unless the candidate source is fallback")
    p.add_argument("--hard-tiny-score-floor", type=float, default=0.22, help="Effective objectness floor for hard-tiny recovered candidates")
    p.add_argument("--hard-tiny-allow-tracker-only", action="store_true", help="Allow hard-tiny recovery for tracker-only candidates; off by default because stale tracks can create many false positives")
    p.add_argument("--hard-tiny-disable-track-validation", action="store_true", help="Disable track metadata validation for tracker-only hard-tiny recovery")
    p.add_argument("--hard-tiny-max-track-frames-since-detector", type=int, default=3)
    p.add_argument("--hard-tiny-min-track-detector-updates", type=int, default=1)
    p.add_argument("--hard-tiny-max-track-drift", type=float, default=48.0)
    p.add_argument("--hard-tiny-min-track-history", type=int, default=2)
    p.add_argument("--crop-weights", default=None, help="Optional CropRecognizer .pt weights")
    p.add_argument("--feature-weights", default=None, help="Optional FeatureRecognitionModel .pt weights")
    p.add_argument("--temporal-weights", default=None, help="Optional TemporalRecognizer .pt weights")
    p.add_argument("--recognition-crop-scale", type=float, default=4.0, help="Context scale for inference crop recognizer crops")
    p.add_argument("--recognition-tube-scale", type=float, default=4.0, help="Context scale for inference temporal tube crops")
    p.add_argument("--fusion-calibration", default=None, help="Optional fusion_calibration_summary.json or {'weights': ...} JSON")
    p.add_argument("--tracklet-classifier-weights", default=None, help="Optional TrackletMLP .pt checkpoint used as a post-infer filter")
    p.add_argument("--tracklet-classifier-threshold", type=float, default=0.5)
    p.add_argument("--tracklet-filter-untracked", choices=["keep", "suppress"], default="keep", help="How to handle drone predictions without a track_id when tracklet filtering is enabled")
    p.add_argument("--disable-tracklet-promotion", action="store_true", help="Only reject non-drone tracklets; do not promote positive tracklets to drone detections")
    p.add_argument("--tracklet-promotion-score-floor", type=float, default=0.22)
    p.add_argument("--tracklet-promotion-min-branch-drone", type=float, default=0.40)
    p.add_argument("--tracklet-promotion-max-background", type=float, default=0.68)
    p.add_argument("--tracklet-selective-promotion", action="store_true", help="Require tracklet-level recovery evidence and a per-sequence promotion budget before promoting background rows")
    p.add_argument("--tracklet-selective-min-temporal-crop-delta", type=float, default=0.05)
    p.add_argument("--tracklet-selective-min-temporal-background-margin", type=float, default=-0.05)
    p.add_argument("--tracklet-selective-max-tracklet-background", type=float, default=0.60)
    p.add_argument("--tracklet-selective-max-tracklet-objectness", type=float, default=0.50)
    p.add_argument("--tracklet-selective-min-tracklet-rows", type=int, default=2)
    p.add_argument("--tracklet-selective-min-temporal-gain-rate", type=float, default=0.40)
    p.add_argument("--tracklet-selective-min-weak-detector-temporal-signal", type=float, default=0.05)
    p.add_argument("--tracklet-selective-allow-non-recovery-source", action="store_true")
    p.add_argument("--tracklet-selective-max-promoted-tracklets-per-sequence", type=int, default=2, help="0 disables the per-sequence promotion budget")
    p.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--min-area", type=int, default=3)
    p.add_argument("--max-area", type=int, default=5000)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--disable-motion-candidates", action="store_true", help="Skip motion candidate generation; useful for oracle/seed recognition experiments")
    p.add_argument("--seed-box", nargs=4, type=float, default=None, metavar=("X1", "Y1", "X2", "Y2"), help="Experimental/oracle candidate box injected every frame")
    p.add_argument("--seed-objectness", type=float, default=0.75)
    p.add_argument("--seed-track-score", type=float, default=0.8)
    p.add_argument("--save-video", action="store_true")
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("build-crop-dataset")
    p.add_argument("--frames", default=None)
    p.add_argument("--annotations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=["csv", "coco"], default="csv")
    p.add_argument("--scale", type=float, default=4.0)
    p.add_argument("--size", type=int, default=128)
    p.set_defaults(func=cmd_build_crop_dataset)

    p = sub.add_parser("train-recognizer")
    p.add_argument("--type", choices=["crop", "temporal", "feature"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--balance", choices=["sampler", "class_weight", "none"], default="sampler")
    p.add_argument("--target-mode", choices=["multiclass", "drone_binary"], default="multiclass", help="Use drone_binary to train drone vs non-drone while keeping the 8-class output head")
    p.set_defaults(func=cmd_train_recognizer)

    p = sub.add_parser("build-yolo-candidate-dataset")
    p.add_argument("--annotations", required=True, help="CSV with frame_path,x1,y1,x2,y2,class; class is ignored for candidate training")
    p.add_argument("--images-root", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-box-px", type=float, default=1.0, help="Expand tiny labels to at least this many pixels for candidate training")
    p.set_defaults(func=cmd_build_yolo_candidate_dataset)

    p = sub.add_parser("build-tiled-yolo-candidate-dataset")
    p.add_argument("--annotations", required=True, help="CSV with frame_path,x1,y1,x2,y2,class")
    p.add_argument("--images-root", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--positives-per-box", type=int, default=2)
    p.add_argument("--negatives-per-image", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--min-box-px", type=float, default=8.0, help="Expand tiny labels inside each tile")
    p.add_argument("--negative-pad-px", type=float, default=32.0, help="Reject negative tiles this close to a known box")
    p.add_argument("--photometric-augmentations", type=int, default=0, help="Extra brightness/contrast variants per positive tile")
    p.add_argument("--low-contrast-injections", type=int, default=0, help="Extra positive variants with a locally redrawn low-contrast tiny target")
    p.add_argument("--positive-repeat-pattern", action="append", default=[], help="Repeat positives when this substring appears in the source frame path; can be passed multiple times")
    p.add_argument("--positive-repeat-factor", type=int, default=1, help="Positive oversampling factor for matching repeat patterns")
    p.set_defaults(func=cmd_build_tiled_yolo_candidate_dataset)

    p = sub.add_parser("train-yolo-p2")
    p.add_argument("--data", required=True, help="YOLO data.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--model-yaml", default=None)
    p.add_argument("--write-model-yaml", default=None, help="Write default YOLO-P2 yaml here before training")
    p.add_argument("--no-train", action="store_true", help="Only write model yaml")
    p.add_argument("--pretrained", default=None, help="Optional local pretrained YOLO weights")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_train_yolo_p2)

    p = sub.add_parser("evaluate")
    p.add_argument("--pred", required=True)
    p.add_argument("--gt", default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("build-tracklet-dataset")
    p.add_argument("--diagnostics", nargs="+", required=True, help="One or more diagnostics.jsonl files from infer")
    p.add_argument("--gt-csv", required=True, help="Unified CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--out", required=True)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--center-threshold", type=float, default=24.0)
    p.set_defaults(func=cmd_build_tracklet_dataset)

    p = sub.add_parser("build-proposal-tracklet-dataset")
    p.add_argument("--run-roots", nargs="+", required=True, help="Profile benchmark roots containing <profile>/<seq>/diagnostics*.jsonl")
    p.add_argument("--gt-csv", required=True, help="Unified CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--out", required=True)
    p.add_argument("--profile", default="hard_recovery")
    p.add_argument("--diagnostics-name", default="diagnostics_raw.jsonl")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-gap", type=int, default=3)
    p.add_argument("--base-radius", type=float, default=18.0)
    p.add_argument("--radius-per-side", type=float, default=0.75)
    p.add_argument("--min-iou", type=float, default=0.05)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--detector-only", action="store_true")
    p.add_argument("--min-tracklet-rows", type=int, default=1)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--center-threshold", type=float, default=24.0)
    p.add_argument("--hard-tiny-side", type=float, default=24.0)
    p.add_argument("--hard-low-score", type=float, default=0.25)
    p.set_defaults(func=cmd_build_proposal_tracklet_dataset)

    p = sub.add_parser("train-tracklet-classifier")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--hard-tiny-positive-augments", type=int, default=0, help="Synthetic hard-tiny positive variants per positive tracklet")
    p.set_defaults(func=cmd_train_tracklet_classifier)

    p = sub.add_parser("eval-tracklet-classifier")
    p.add_argument("--csv", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.set_defaults(func=cmd_eval_tracklet_classifier)

    p = sub.add_parser("stage-b-oracle-benchmark")
    p.add_argument("--metadata", nargs="+", required=True, help="Static-hover JSON metadata files or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--crop-weights", default=None)
    p.add_argument("--feature-weights", default=None)
    p.add_argument("--temporal-weights", default=None)
    p.add_argument("--hard-negative-manifest", nargs="*", default=None, help="Optional hard_negative_manifest.jsonl files to include as oracle negatives")
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--negative-per-positive", type=int, default=1)
    p.add_argument("--t", type=int, default=5)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-hard-negatives", type=int, default=None)
    p.set_defaults(func=cmd_stage_b_oracle_benchmark)

    p = sub.add_parser("tracker-oracle-benchmark")
    p.add_argument("--metadata", nargs="+", required=True, help="JSON metadata files or glob patterns with oracle boxes")
    p.add_argument("--out", required=True)
    p.add_argument("--detection-stride", type=int, default=5, help="Only feed oracle detections every N frames")
    p.add_argument("--max-frames", type=int, default=80)
    p.add_argument("--match-iou", type=float, default=0.1)
    p.add_argument("--match-center-px", type=float, default=16.0)
    p.add_argument("--tracker-r0", type=float, default=24.0)
    p.add_argument("--tracker-alpha", type=float, default=1.5)
    p.add_argument("--tracker-beta", type=float, default=20.0)
    p.add_argument("--tracker-reacquire", type=float, default=18.0)
    p.set_defaults(func=cmd_tracker_oracle_benchmark)

    p = sub.add_parser("mode-sweep")
    p.add_argument("--diagnostics", required=True, help="motion-debug diagnostics.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--weak-alignment-q-threshold", type=float, default=0.66)
    p.add_argument("--high-residual-threshold", type=float, default=0.0072)
    p.add_argument("--static-motion-threshold", type=float, default=0.08)
    p.set_defaults(func=cmd_mode_sweep)

    p = sub.add_parser("augment-speed")
    p.add_argument("--input-dir", default=None, help="Directory of videos to batch augment")
    p.add_argument("--pattern", default="*.mp4", help="Glob pattern for --input-dir")
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--video", default=None, help="Input video to temporally subsample")
    p.add_argument("--annotations", default=None, help="Optional annotation CSV to subsample by frame_index")
    p.add_argument("--out", required=True)
    p.add_argument("--strides", nargs="+", type=int, default=[2, 4], help="Temporal stride values; speed factor when --keep-fps is set")
    p.add_argument("--keep-fps", action="store_true", default=True, help="Keep original FPS so apparent motion is faster")
    p.add_argument("--motion-blur", type=int, default=0, help="Optional odd/even kernel size; 0 disables blur")
    p.add_argument("--blur-direction", choices=["horizontal", "vertical", "diagonal"], default="horizontal")
    p.add_argument("--frame-index-column", default="frame_index")
    p.set_defaults(func=cmd_augment_speed)

    p = sub.add_parser("make-static-hover")
    p.add_argument("--video", required=True)
    p.add_argument("--out-video", required=True)
    p.add_argument("--x", type=int, default=None)
    p.add_argument("--y", type=int, default=None)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--color", nargs=3, type=int, default=[15, 15, 15], help="BGR dot color")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--freeze-background", action="store_true", help="Repeat the first frame before drawing the fixed target")
    p.add_argument("--jitter-px", type=int, default=0, help="Per-frame integer jitter range around the target center")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_make_static_hover)

    p = sub.add_parser("make-moving-target")
    p.add_argument("--video", required=True)
    p.add_argument("--out-video", required=True)
    p.add_argument("--start-x", type=int, default=None)
    p.add_argument("--start-y", type=int, default=None)
    p.add_argument("--vx", type=float, default=16.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--color", nargs=3, type=int, default=[0, 0, 0], help="BGR dot color")
    p.add_argument("--max-frames", type=int, default=80)
    p.add_argument("--freeze-background", action="store_true", default=True)
    p.set_defaults(func=cmd_make_moving_target)

    p = sub.add_parser("build-static-hover-crops")
    p.add_argument("--metadata", nargs="+", required=True, help="Static-hover JSON metadata files or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--class-name", default="drone")
    p.add_argument("--negative-class", default="background")
    p.add_argument("--negative-per-positive", type=int, default=3)
    p.add_argument("--scale", type=float, default=8.0)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--seed", type=int, default=17)
    p.set_defaults(func=cmd_build_static_hover_crops)

    p = sub.add_parser("mine-hard-negative-crops")
    p.add_argument("--video", nargs="+", required=True, help="Videos or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--exclude-metadata", nargs="*", default=None, help="Static-hover JSON metadata boxes to avoid")
    p.add_argument("--max-frames", type=int, default=120)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-crops-per-video", type=int, default=300)
    p.add_argument("--min-area", type=int, default=3)
    p.add_argument("--max-area", type=int, default=5000)
    p.add_argument("--min-motion-score", type=float, default=0.08)
    p.add_argument("--artifact-q-threshold", type=float, default=0.66)
    p.add_argument("--scale", type=float, default=8.0)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--exclude-padding-px", type=float, default=24.0)
    p.set_defaults(func=cmd_mine_hard_negative_crops)

    p = sub.add_parser("build-static-hover-yolo-dataset")
    p.add_argument("--metadata", nargs="+", required=True, help="Static-hover JSON metadata files or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--max-frames-per-video", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-box-px", type=float, default=1.0, help="Expand tiny labels to at least this many pixels for candidate training")
    p.add_argument("--tiled", action="store_true", help="Build tiled/high-res YOLO dataset instead of full-frame dataset")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--positives-per-box", type=int, default=2)
    p.add_argument("--negatives-per-image", type=int, default=2)
    p.add_argument("--negative-pad-px", type=float, default=32.0)
    p.add_argument("--photometric-augmentations", type=int, default=0, help="Extra brightness/contrast variants per positive tile")
    p.add_argument("--low-contrast-injections", type=int, default=0, help="Extra positive variants with a locally redrawn low-contrast tiny target")
    p.add_argument("--positive-repeat-pattern", action="append", default=[], help="Repeat positives when this substring appears in the source frame path; can be passed multiple times")
    p.add_argument("--positive-repeat-factor", type=int, default=1, help="Positive oversampling factor for matching repeat patterns")
    p.set_defaults(func=cmd_build_static_hover_yolo_dataset)

    p = sub.add_parser("build-detector-proposal-stage-b")
    p.add_argument("--metadata", nargs="+", required=True, help="Synthetic metadata JSON files or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--yolo-conf", type=float, default=0.05)
    p.add_argument("--yolo-tile-size", type=int, default=256)
    p.add_argument("--yolo-tile-stride", type=int, default=128)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--max-frames-per-video", type=int, default=80)
    p.add_argument("--max-proposals-per-frame", type=int, default=8)
    p.add_argument("--max-negatives-per-frame", type=int, default=4)
    p.add_argument("--match-iou", type=float, default=0.1)
    p.add_argument("--match-center-px", type=float, default=24.0)
    p.add_argument("--scale", type=float, default=4.0)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--t", type=int, default=5)
    p.add_argument("--tube-size", type=int, default=96)
    p.set_defaults(func=cmd_build_detector_proposal_stage_b)

    p = sub.add_parser("build-static-hover-temporal")
    p.add_argument("--metadata", nargs="+", required=True, help="Static-hover JSON metadata files or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--t", type=int, default=5)
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--negative-per-positive", type=int, default=1)
    p.add_argument("--scale", type=float, default=8.0)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--seed", type=int, default=29)
    p.set_defaults(func=cmd_build_static_hover_temporal)

    p = sub.add_parser("build-static-hover-feature-dataset")
    p.add_argument("--metadata", nargs="+", required=True, help="Static-hover JSON metadata files or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--frame-stride", type=int, default=4)
    p.add_argument("--negative-per-positive", type=int, default=1)
    p.add_argument("--seed", type=int, default=37)
    p.set_defaults(func=cmd_build_static_hover_feature_dataset)

    p = sub.add_parser("mine-hard-negative-temporal")
    p.add_argument("--manifest", required=True, help="hard_negative_manifest.jsonl from mine-hard-negative-crops")
    p.add_argument("--out", required=True)
    p.add_argument("--t", type=int, default=5)
    p.add_argument("--scale", type=float, default=8.0)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--max-samples-per-class", type=int, default=80)
    p.set_defaults(func=cmd_mine_hard_negative_temporal)

    p = sub.add_parser("split-folder-dataset")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=53)
    p.set_defaults(func=cmd_split_folder_dataset)

    p = sub.add_parser("split-feature-csv")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=59)
    p.set_defaults(func=cmd_split_feature_csv)

    p = sub.add_parser("build-hard-negative-feature-dataset")
    p.add_argument("--manifest", nargs="+", required=True, help="Hard-negative manifest files or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--max-samples-per-class", type=int, default=120)
    p.set_defaults(func=cmd_build_hard_negative_feature_dataset)

    p = sub.add_parser("stage-a-yolo-recall")
    p.add_argument("--metadata", nargs="+", required=True, help="Synthetic metadata JSON files with output video and boxes")
    p.add_argument("--out", required=True)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--yolo-conf", type=float, default=0.05)
    p.add_argument("--yolo-tile-size", type=int, default=256)
    p.add_argument("--yolo-tile-stride", type=int, default=128)
    p.add_argument("--device", default="0")
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--match-iou", type=float, default=0.1)
    p.add_argument("--match-center-px", type=float, default=24.0)
    p.add_argument("--keep-top", type=int, default=5)
    p.set_defaults(func=cmd_stage_a_yolo_recall)

    p = sub.add_parser("init-real-data-layout")
    p.add_argument("--root", default="data/real")
    p.set_defaults(func=cmd_init_real_data_layout)

    p = sub.add_parser("extract-real-annotated-frames")
    p.add_argument("--annotations", required=True, help="CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--video-root", default=None)
    p.add_argument("--allow-unknown-labels", action="store_true")
    p.set_defaults(func=cmd_extract_real_annotated_frames)

    p = sub.add_parser("prepare-real-yolo-dataset")
    p.add_argument("--annotations", required=True, help="CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--out", required=True)
    p.add_argument("--video-root", default=None)
    p.add_argument("--full-frame", action="store_true", help="Build full-frame YOLO dataset instead of tiled dataset")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--positives-per-box", type=int, default=2)
    p.add_argument("--negatives-per-image", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--min-box-px", type=float, default=8.0)
    p.add_argument("--negative-pad-px", type=float, default=32.0)
    p.add_argument("--allow-unknown-labels", action="store_true")
    p.set_defaults(func=cmd_prepare_real_yolo_dataset)

    p = sub.add_parser("export-anti-uav300-subset")
    p.add_argument("--zip", required=True, help="Path to Anti-UAV300.zip")
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--modality", choices=["visible", "infrared"], default="visible")
    p.add_argument("--max-sequences", type=int, default=5)
    p.add_argument("--start-index", type=int, default=0, help="Skip this many sorted sequences before exporting")
    p.add_argument("--frame-stride", type=int, default=10)
    p.add_argument("--max-frames-per-sequence", type=int, default=80)
    p.set_defaults(func=cmd_export_anti_uav300_subset)

    p = sub.add_parser("build-real-stage-b-dataset")
    p.add_argument("--annotations", required=True, help="CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--out", required=True)
    p.add_argument("--negative-per-positive", type=int, default=1)
    p.add_argument("--crop-scale", type=float, default=4.0)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--tube-t", type=int, default=5)
    p.add_argument("--tube-size", type=int, default=96)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--max-samples", type=int, default=None)
    p.set_defaults(func=cmd_build_real_stage_b_dataset)

    p = sub.add_parser("build-real-detector-proposal-stage-b")
    p.add_argument("--annotations", required=True, help="CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--out", required=True)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--yolo-conf", type=float, default=0.05)
    p.add_argument("--fallback-yolo-weights", default=None)
    p.add_argument("--fallback-yolo-conf", type=float, default=0.15)
    p.add_argument("--yolo-tile-size", type=int, default=256)
    p.add_argument("--yolo-tile-stride", type=int, default=128)
    p.add_argument("--device", default="0")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-proposals-per-frame", type=int, default=8)
    p.add_argument("--max-fallback-proposals-per-frame", type=int, default=4)
    p.add_argument("--max-negatives-per-frame", type=int, default=4)
    p.add_argument("--match-iou", type=float, default=0.1)
    p.add_argument("--match-center-px", type=float, default=24.0)
    p.add_argument("--artifact-score-threshold", type=float, default=0.25, help="Nonmatched fallback proposals above this score become alignment_artifact samples")
    p.add_argument("--high-score-fp-threshold", type=float, default=0.5, help="Nonmatched detector proposals above this score become high_score_detector_fp diagnostic samples")
    p.add_argument("--non-drone-label-mode", choices=["multiclass_artifact", "binary_buckets"], default="multiclass_artifact", help="binary_buckets trains all non-drone proposals as background while preserving diagnostic_bucket metadata")
    p.add_argument("--hard-positive-max-size-px", type=float, default=24.0, help="Matched positives with proposal max side at or below this are hard_tiny_positive candidates")
    p.add_argument("--hard-positive-max-score", type=float, default=0.5, help="Matched tiny positives at or below this score are repeated as hard positives")
    p.add_argument("--hard-positive-repeat", type=int, default=1, help="Repeat each hard tiny positive this many times in Stage B datasets")
    p.add_argument("--crop-scale", type=float, default=2.0)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--tube-t", type=int, default=5)
    p.add_argument("--tube-size", type=int, default=96)
    p.set_defaults(func=cmd_build_real_detector_proposal_stage_b)

    p = sub.add_parser("calibrate-fusion")
    p.add_argument("--diagnostics", required=True)
    p.add_argument("--gt", required=True, help="QSTR real CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--out", required=True)
    p.add_argument("--video", default=None, help="Optional video path when diagnostics come from one video")
    p.add_argument("--match-iou", type=float, default=0.1)
    p.add_argument("--match-center-px", type=float, default=24.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--frame-tolerance", type=int, default=0, help="Allow nearest GT annotation within this many frames")
    p.set_defaults(func=cmd_calibrate_fusion)

    p = sub.add_parser("analyze-frame-failures")
    p.add_argument("--run-root", required=True, help="Profile benchmark output root containing hard_recovery/<seq>/predictions*.jsonl")
    p.add_argument("--gt", required=True, help="QSTR real CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--out", required=True)
    p.add_argument("--profile", default="hard_recovery")
    p.add_argument("--raw-prediction-name", default="predictions_raw.jsonl")
    p.add_argument("--filtered-prediction-name", default="predictions.jsonl")
    p.add_argument("--score-threshold", type=float, default=0.2)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=cmd_analyze_frame_failures)

    p = sub.add_parser("sweep-tracklet-filter")
    p.add_argument("--run-roots", nargs="+", required=True, help="Train/adapt profile output roots; do not pass frozen test roots for calibration")
    p.add_argument("--gt", required=True, help="QSTR real CSV covering the run roots")
    p.add_argument("--weights", required=True, help="Tracklet classifier checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--profile", default="hard_recovery")
    p.add_argument("--classifier-thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.85, 0.95])
    p.add_argument("--promotion-score-floors", nargs="+", type=float, default=[0.20, 0.22, 0.30])
    p.add_argument("--promotion-max-backgrounds", nargs="+", type=float, default=[0.55, 0.60, 0.68])
    p.add_argument("--promotion-min-branch-drone", type=float, default=0.40)
    p.add_argument("--selective-promotion", action="store_true")
    p.add_argument("--selective-min-temporal-crop-deltas", nargs="+", type=float, default=[0.05])
    p.add_argument("--selective-min-temporal-background-margins", nargs="+", type=float, default=[-0.05])
    p.add_argument("--selective-max-promoted-tracklets-per-sequence-values", nargs="+", type=int, default=[2])
    p.add_argument("--selective-max-tracklet-background", type=float, default=0.60)
    p.add_argument("--selective-max-tracklet-objectness", type=float, default=0.50)
    p.add_argument("--selective-min-tracklet-rows", type=int, default=2)
    p.add_argument("--selective-min-temporal-gain-rate", type=float, default=0.40)
    p.add_argument("--selective-min-weak-detector-temporal-signal", type=float, default=0.05)
    p.add_argument("--score-threshold", type=float, default=0.2)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=cmd_sweep_tracklet_filter)

    p = sub.add_parser("select-tracklet-model")
    p.add_argument("--tracklets", required=True, help="Tracklet CSV built from train/adapt diagnostics")
    p.add_argument("--run-roots", nargs="+", required=True, help="Train/adapt profile output roots used for downstream calibration")
    p.add_argument("--gt", required=True, help="QSTR real CSV covering the run roots")
    p.add_argument("--out", required=True)
    p.add_argument("--profile", default="hard_recovery")
    p.add_argument("--calib-seqs", nargs="+", default=None, help="Exact sequence names reserved for calibration")
    p.add_argument("--calib-seq-patterns", nargs="+", default=None, help="fnmatch or substring patterns reserved for calibration")
    p.add_argument("--calib-fraction", type=float, default=0.4, help="Fallback deterministic sequence fraction when no calibration names/patterns are given")
    p.add_argument("--epochs-values", nargs="+", type=int, default=[30, 60])
    p.add_argument("--hidden-values", nargs="+", type=int, default=[32])
    p.add_argument("--lr-values", nargs="+", type=float, default=[1e-3])
    p.add_argument("--hard-tiny-positive-augments-values", nargs="+", type=int, default=[0, 2])
    p.add_argument("--hard-negative-augments-values", nargs="+", type=int, default=[0, 2])
    p.add_argument("--classifier-thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.85])
    p.add_argument("--promotion-enabled-values", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    p.add_argument("--promotion-score-floors", nargs="+", type=float, default=[0.22, 0.30])
    p.add_argument("--promotion-max-backgrounds", nargs="+", type=float, default=[0.55, 0.60])
    p.add_argument("--promotion-min-branch-drone", type=float, default=0.40)
    p.add_argument("--selective-promotion", action="store_true", help="Apply selective promotion budget while evaluating promoted candidates")
    p.add_argument("--selective-max-promoted-tracklets-per-sequence-values", nargs="+", type=int, default=[1, 2])
    p.add_argument("--score-threshold", type=float, default=0.2)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-recall-drop", type=float, default=0.02, help="Selection prefers configs within this recall drop from raw before minimizing FP")
    p.set_defaults(func=cmd_select_tracklet_model)

    p = sub.add_parser("select-tracklet-sequence-model")
    p.add_argument("--tracklets-jsonl", required=True, help="Tracklet JSONL built from train/adapt diagnostics")
    p.add_argument("--run-roots", nargs="+", required=True, help="Train/adapt profile output roots used for downstream calibration")
    p.add_argument("--gt", required=True, help="QSTR real CSV covering the run roots")
    p.add_argument("--out", required=True)
    p.add_argument("--profile", default="hard_recovery")
    p.add_argument("--calib-seqs", nargs="+", default=None)
    p.add_argument("--calib-seq-patterns", nargs="+", default=None)
    p.add_argument("--calib-fraction", type=float, default=0.4)
    p.add_argument("--epochs-values", nargs="+", type=int, default=[30])
    p.add_argument("--hidden-values", nargs="+", type=int, default=[32])
    p.add_argument("--max-len-values", nargs="+", type=int, default=[12, 24])
    p.add_argument("--hard-negative-augments-values", nargs="+", type=int, default=[0, 2])
    p.add_argument("--classifier-thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.85])
    p.add_argument("--promotion-enabled-values", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    p.add_argument("--promotion-score-floors", nargs="+", type=float, default=[0.22, 0.30])
    p.add_argument("--promotion-max-backgrounds", nargs="+", type=float, default=[0.55, 0.60])
    p.add_argument("--selective-promotion", action="store_true")
    p.add_argument("--selective-max-promoted-tracklets-per-sequence-values", nargs="+", type=int, default=[1, 2])
    p.add_argument("--score-threshold", type=float, default=0.2)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-recall-drop", type=float, default=0.02)
    p.set_defaults(func=cmd_select_tracklet_sequence_model)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
