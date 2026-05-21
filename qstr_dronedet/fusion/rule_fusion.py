from __future__ import annotations

from qstr_dronedet.types import CLASSES, RecognitionResult, normalize_probs


def _safe(probs: dict[str, float] | None) -> dict[str, float]:
    if probs is None:
        probs = {"unknown": 1.0}
    return normalize_probs(probs)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {
        "crop": max(0.0, float(weights.get("crop", 0.0))),
        "feature": max(0.0, float(weights.get("feature", 0.0))),
        "temporal": max(0.0, float(weights.get("temporal", 0.0))),
        "motion_prior": max(0.0, float(weights.get("motion_prior", weights.get("motion", 0.0)))),
        "tracker": max(0.0, float(weights.get("tracker", 0.0))),
    }
    total = sum(clean.values())
    if total <= 1e-12:
        return {"crop": 0.30, "feature": 0.25, "temporal": 0.25, "motion_prior": 0.15, "tracker": 0.05}
    return {k: v / total for k, v in clean.items()}


def _weights(mode: str, alignment_quality: float, track_score: float) -> dict[str, float]:
    if mode == "static_or_hovering":
        return {"crop": 0.35, "feature": 0.25, "temporal": 0.30, "motion_prior": 0.02, "tracker": 0.08}
    if mode == "bad_alignment_fast_egomotion":
        return {"crop": 0.40, "feature": 0.20, "temporal": 0.35, "motion_prior": 0.0, "tracker": 0.05}
    if mode == "fast_target":
        return {"crop": 0.20, "feature": 0.15, "temporal": 0.40, "motion_prior": 0.10, "tracker": 0.15}
    if mode == "uncertain":
        return {"crop": 0.25, "feature": 0.20, "temporal": 0.25, "motion_prior": 0.05, "tracker": 0.05}
    motion_w = 0.15 * max(0.0, min(1.0, alignment_quality))
    return {"crop": 0.30, "feature": 0.25, "temporal": 0.25, "motion_prior": motion_w, "tracker": 0.05 + 0.05 * track_score}


def fuse_rule_based(
    objectness: float,
    crop_probs: dict[str, float] | None,
    feature_probs: dict[str, float] | None,
    temporal_probs: dict[str, float] | None,
    motion_score: float,
    alignment_quality: float,
    track_score: float,
    mode: str,
    candidate_source: str = "",
    fusion_weights: dict[str, float] | None = None,
    fallback_gate: bool = True,
    fallback_min_branch_drone: float = 0.45,
    fallback_min_crop_temporal_mean: float = 0.35,
    fallback_max_negative_evidence: float = 0.55,
    verified_objectness: bool = True,
    verified_objectness_mode: str = "hard_recovery",
    verified_min_branch_drone: float = 0.45,
    verified_min_crop_temporal_mean: float = 0.48,
    verified_max_negative_evidence: float = 0.62,
    verified_objectness_floor: float = 0.55,
    hard_tiny_recovery: bool = False,
    hard_tiny_min_crop_drone: float = 0.40,
    hard_tiny_min_temporal_drone: float = 0.55,
    hard_tiny_min_temporal_crop_delta: float = 0.0,
    hard_tiny_max_bg_minus_drone: float = 0.08,
    hard_tiny_min_support: float = 0.15,
    hard_tiny_score_floor: float = 0.22,
    hard_tiny_allow_tracker_only: bool = False,
    hard_tiny_require_validated_track: bool = True,
    hard_tiny_max_track_frames_since_detector: int = 3,
    hard_tiny_min_track_detector_updates: int = 1,
    hard_tiny_max_track_drift: float = 48.0,
    hard_tiny_min_track_history: int = 2,
    candidate_extra: dict | None = None,
) -> RecognitionResult:
    if verified_objectness_mode not in {"always", "hard_recovery"}:
        raise ValueError("verified_objectness_mode must be 'always' or 'hard_recovery'")
    crop = _safe(crop_probs)
    feat = _safe(feature_probs)
    temp = _safe(temporal_probs)
    disagreement = abs(crop.get("drone", 0.0) - feat.get("drone", 0.0))
    w = _normalize_weights(fusion_weights) if fusion_weights is not None else _weights(mode, alignment_quality, track_score)
    fused = {c: 0.0 for c in CLASSES}
    for c in CLASSES:
        fused[c] += w["crop"] * crop[c] + w["feature"] * feat[c] + w["temporal"] * temp[c]
    fused["drone"] += w["motion_prior"] * max(0.0, min(1.0, motion_score))
    fused["drone"] += w["tracker"] * max(0.0, min(1.0, track_score))
    if objectness < 0.15:
        fused["background"] += 0.5
    sources = {s for s in candidate_source.split("+") if s}
    motion_only = sources == {"motion"}
    isolated_motion_artifact = (
        motion_only
        and track_score < 0.2
        and motion_score > 0.4
        and disagreement > 0.45
        and feat["background"] > 0.40
        and mode not in {"fast_target", "static_or_hovering"}
    )
    if isolated_motion_artifact:
        fused["background"] += 0.45
        fused["alignment_artifact"] += 0.20
    final = normalize_probs(fused)
    fallback_source = any("fallback" in s for s in sources)
    tracker_source = any("tracker" in s for s in sources)
    crop_temporal_mean = 0.5 * (crop["drone"] + temp["drone"])
    strongest_crop_temporal = max(crop["drone"], temp["drone"])
    negative_evidence = max(
        crop["background"],
        crop["alignment_artifact"],
        feat["background"],
        feat["alignment_artifact"],
        temp["background"],
        temp["alignment_artifact"],
        final["background"],
        final["alignment_artifact"],
    )
    fallback_rejected = False
    if fallback_gate and fallback_source:
        fallback_rejected = (
            strongest_crop_temporal < fallback_min_branch_drone
            or crop_temporal_mean < fallback_min_crop_temporal_mean
            or negative_evidence > fallback_max_negative_evidence
        )
        if fallback_rejected:
            final["drone"] *= 0.20
            final["background"] += 0.45
            final = normalize_probs(final)
    predicted = max(final, key=final.get)
    if final["unknown"] > 0.45:
        predicted = "unknown"
    diagnostic_cause = None
    artifact_votes = sum(1 for p in (crop, feat, temp) if p["alignment_artifact"] > 0.35)
    artifact_signal = max(crop["alignment_artifact"], feat["alignment_artifact"], temp["alignment_artifact"], final["alignment_artifact"])
    if fallback_rejected:
        diagnostic_cause = "fallback_rejected"
        predicted = "background"
    elif isolated_motion_artifact:
        diagnostic_cause = "isolated_motion_artifact"
        if final["drone"] < 0.55:
            predicted = "background"
    elif artifact_votes >= 2 or final["alignment_artifact"] > 0.45 or (alignment_quality < 0.5 and motion_score > 0.18):
        diagnostic_cause = "alignment_artifact"
        if predicted == "alignment_artifact" and final["drone"] < 0.35:
            predicted = "background"
    error_type = None
    if objectness < 0.2:
        error_type = "candidate/localization_failure"
    elif disagreement > 0.45:
        error_type = "feature_crop_disagreement"
    elif alignment_quality < 0.3 and motion_score > 0.2:
        error_type = "motion_alignment_failure"
    elif diagnostic_cause in {"alignment_artifact", "isolated_motion_artifact", "fallback_rejected"} and predicted != "drone":
        error_type = diagnostic_cause
    effective_objectness = float(objectness)
    verified_source_allowed = tracker_source or fallback_source
    if verified_objectness_mode == "hard_recovery":
        verified_source_allowed = fallback_source or (tracker_source and objectness < 0.2)
    verified_by_stage_b = (
        verified_objectness
        and verified_source_allowed
        and not fallback_rejected
        and strongest_crop_temporal >= verified_min_branch_drone
        and crop_temporal_mean >= verified_min_crop_temporal_mean
        and negative_evidence <= verified_max_negative_evidence
    )
    if verified_by_stage_b:
        effective_objectness = max(effective_objectness, float(verified_objectness_floor))
        if error_type == "candidate/localization_failure":
            error_type = None
    extra = candidate_extra or {}
    track_validated = bool(extra.get("track_validated", False))
    if tracker_source and not track_validated:
        try:
            frames_since_detector = int(extra.get("track_frames_since_detector_update", 999))
            detector_updates = int(extra.get("track_detector_updates", 0))
            drift = float(extra.get("track_drift", float("inf")))
            history_len = int(extra.get("track_history_len", 0))
            track_validated = (
                frames_since_detector <= hard_tiny_max_track_frames_since_detector
                and detector_updates >= hard_tiny_min_track_detector_updates
                and drift <= hard_tiny_max_track_drift
                and history_len >= hard_tiny_min_track_history
            )
        except (TypeError, ValueError):
            track_validated = False
    hard_tiny_supported_source = fallback_source or (hard_tiny_allow_tracker_only and tracker_source)
    crop_temporal_drone_mean = 0.5 * (crop["drone"] + temp["drone"])
    crop_temporal_background_mean = 0.5 * (crop["background"] + temp["background"])
    crop_temporal_artifact_mean = 0.5 * (crop["alignment_artifact"] + temp["alignment_artifact"])
    hard_tiny_stage_b_support = (
        crop["drone"] >= hard_tiny_min_crop_drone
        and temp["drone"] >= hard_tiny_min_temporal_drone
        and temp["drone"] - crop["drone"] >= hard_tiny_min_temporal_crop_delta
        and crop_temporal_background_mean - crop_temporal_drone_mean <= hard_tiny_max_bg_minus_drone
        and crop_temporal_artifact_mean <= max(0.35, crop_temporal_drone_mean + hard_tiny_max_bg_minus_drone)
        and feat["background"] <= 0.90
    )
    hard_tiny_metadata_support = fallback_source or (
        tracker_source
        and track_score >= hard_tiny_min_support
        and (track_validated or not hard_tiny_require_validated_track)
    )
    hard_tiny_recovered = (
        hard_tiny_recovery
        and not fallback_rejected
        and hard_tiny_supported_source
        and hard_tiny_stage_b_support
        and hard_tiny_metadata_support
        and final["unknown"] <= 0.45
    )
    if hard_tiny_recovered and predicted == "background":
        predicted = "drone"
        diagnostic_cause = "hard_tiny_recovery"
        error_type = None
        effective_objectness = max(effective_objectness, float(hard_tiny_score_floor))
    elif hard_tiny_recovered and predicted == "drone" and diagnostic_cause is None:
        diagnostic_cause = "hard_tiny_recovery"
    return RecognitionResult(crop, feat, temp, final, disagreement, predicted, float(effective_objectness * final["drone"]), error_type, diagnostic_cause)
