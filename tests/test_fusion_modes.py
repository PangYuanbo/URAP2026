from qstr_dronedet.fusion.modes import determine_mode
from qstr_dronedet.fusion.rule_fusion import fuse_rule_based


def test_static_mode_preserves_drone_with_strong_crop_temporal():
    mode = determine_mode(0.01, 0.8, 0.0, 100.0, 0.8)
    assert mode == "static_or_hovering"
    crop = {"drone": 0.8, "unknown": 0.1}
    feat = {"drone": 0.6, "unknown": 0.2}
    temp = {"drone": 0.9, "unknown": 0.05}
    rec = fuse_rule_based(0.8, crop, feat, temp, 0.01, 0.8, 0.8, mode)
    assert rec.predicted_class == "drone"


def test_bad_alignment_downweights_motion_artifact():
    mode = determine_mode(0.4, 0.1, 0.0, 5.0, 0.1)
    assert mode == "bad_alignment_fast_egomotion"
    rec = fuse_rule_based(0.7, {"unknown": 0.7}, {"unknown": 0.7}, {"unknown": 0.7}, 0.9, 0.1, 0.0, mode)
    assert rec.predicted_class == "unknown"


def test_unknown_not_forced_to_drone():
    rec = fuse_rule_based(0.9, {"unknown": 0.9}, {"unknown": 0.9}, {"unknown": 0.9}, 1.0, 1.0, 1.0, "normal")
    assert rec.predicted_class == "unknown"


def test_artifact_becomes_diagnostic_cause_not_identity():
    probs = {"alignment_artifact": 0.8, "background": 0.15, "drone": 0.05}
    rec = fuse_rule_based(0.9, probs, probs, probs, 0.2, 0.5, 0.0, "normal")
    assert rec.predicted_class == "background"
    assert rec.diagnostic_cause == "alignment_artifact"
    assert rec.error_type == "alignment_artifact"


def test_isolated_motion_artifact_is_not_promoted_to_drone():
    crop = {"drone": 0.75, "background": 0.10, "alignment_artifact": 0.14}
    feat = {"drone": 0.09, "background": 0.52, "alignment_artifact": 0.07, "unknown": 0.08}
    temp = {"drone": 0.61, "background": 0.15, "alignment_artifact": 0.24}
    rec = fuse_rule_based(
        1.0,
        crop,
        feat,
        temp,
        motion_score=1.0,
        alignment_quality=0.75,
        track_score=0.0,
        mode="normal",
        candidate_source="motion",
    )
    assert rec.predicted_class == "background"
    assert rec.diagnostic_cause == "isolated_motion_artifact"
    assert rec.final_drone_score < 0.55


def test_supported_motion_candidate_can_still_be_drone():
    crop = {"drone": 0.75, "background": 0.10, "alignment_artifact": 0.14}
    feat = {"drone": 0.09, "background": 0.52, "alignment_artifact": 0.07, "unknown": 0.08}
    temp = {"drone": 0.61, "background": 0.15, "alignment_artifact": 0.24}
    rec = fuse_rule_based(
        1.0,
        crop,
        feat,
        temp,
        motion_score=1.0,
        alignment_quality=0.75,
        track_score=0.7,
        mode="normal",
        candidate_source="motion+tracker",
    )
    assert rec.predicted_class == "drone"
    assert rec.diagnostic_cause != "isolated_motion_artifact"


def test_calibrated_weights_can_disable_failed_feature_branch():
    crop = {"drone": 0.8, "background": 0.1, "unknown": 0.1}
    feat = {"drone": 0.01, "background": 0.98, "unknown": 0.01}
    temp = {"drone": 0.7, "background": 0.2, "unknown": 0.1}
    default = fuse_rule_based(0.9, crop, feat, temp, 0.0, 0.8, 0.0, "normal")
    calibrated = fuse_rule_based(
        0.9,
        crop,
        feat,
        temp,
        0.0,
        0.8,
        0.0,
        "normal",
        fusion_weights={"crop": 0.7, "feature": 0.0, "temporal": 0.3, "tracker": 0.0, "motion": 0.0},
    )
    assert calibrated.final_probs["drone"] > default.final_probs["drone"]
    assert calibrated.predicted_class == "drone"


def test_fallback_source_requires_crop_temporal_support():
    crop = {"drone": 0.20, "background": 0.65, "alignment_artifact": 0.05, "unknown": 0.10}
    feat = {"drone": 0.10, "background": 0.75, "alignment_artifact": 0.05, "unknown": 0.10}
    temp = {"drone": 0.30, "background": 0.45, "alignment_artifact": 0.05, "unknown": 0.20}
    rec = fuse_rule_based(
        0.9,
        crop,
        feat,
        temp,
        0.0,
        0.8,
        0.0,
        "normal",
        candidate_source="yolo_tile_fallback",
    )
    assert rec.predicted_class == "background"
    assert rec.diagnostic_cause == "fallback_rejected"
    assert rec.final_drone_score < 0.15


def test_fallback_source_passes_with_strong_crop_temporal_support():
    crop = {"drone": 0.72, "background": 0.12, "alignment_artifact": 0.04, "unknown": 0.12}
    feat = {"drone": 0.15, "background": 0.35, "alignment_artifact": 0.05, "unknown": 0.45}
    temp = {"drone": 0.68, "background": 0.12, "alignment_artifact": 0.04, "unknown": 0.16}
    rec = fuse_rule_based(
        0.8,
        crop,
        feat,
        temp,
        0.0,
        0.8,
        0.0,
        "normal",
        candidate_source="yolo_tile_fallback",
    )
    assert rec.predicted_class == "drone"
    assert rec.diagnostic_cause != "fallback_rejected"


def test_verified_tracker_candidate_gets_effective_objectness_floor():
    crop = {"drone": 0.56, "background": 0.44}
    feat = {"unknown": 1.0}
    temp = {"drone": 0.62, "background": 0.38}
    rec = fuse_rule_based(
        0.12,
        crop,
        feat,
        temp,
        motion_score=0.0,
        alignment_quality=0.8,
        track_score=0.5,
        mode="normal",
        candidate_source="tracker",
        verified_objectness_floor=0.55,
    )
    unverified = fuse_rule_based(
        0.12,
        crop,
        feat,
        temp,
        motion_score=0.0,
        alignment_quality=0.8,
        track_score=0.5,
        mode="normal",
        candidate_source="tracker",
        verified_objectness=False,
    )
    assert rec.final_drone_score > unverified.final_drone_score * 3.0


def test_verified_objectness_does_not_apply_to_plain_yolo_candidate():
    crop = {"drone": 0.56, "background": 0.44}
    feat = {"unknown": 1.0}
    temp = {"drone": 0.62, "background": 0.38}
    rec = fuse_rule_based(
        0.12,
        crop,
        feat,
        temp,
        motion_score=0.0,
        alignment_quality=0.8,
        track_score=0.0,
        mode="normal",
        candidate_source="yolo_tile",
        verified_objectness_floor=0.55,
    )
    assert rec.final_drone_score < 0.1


def test_hard_recovery_verified_objectness_skips_high_objectness_tracker():
    crop = {"drone": 0.58, "background": 0.42}
    feat = {"unknown": 1.0}
    temp = {"drone": 0.64, "background": 0.36}
    rec = fuse_rule_based(
        0.65,
        crop,
        feat,
        temp,
        motion_score=0.0,
        alignment_quality=0.8,
        track_score=0.6,
        mode="normal",
        candidate_source="tracker",
        verified_objectness_mode="hard_recovery",
        verified_objectness_floor=0.95,
    )
    always = fuse_rule_based(
        0.65,
        crop,
        feat,
        temp,
        motion_score=0.0,
        alignment_quality=0.8,
        track_score=0.6,
        mode="normal",
        candidate_source="tracker",
        verified_objectness_mode="always",
        verified_objectness_floor=0.95,
    )
    assert always.final_drone_score > rec.final_drone_score


def test_hard_recovery_verified_objectness_allows_fallback():
    crop = {"drone": 0.58, "background": 0.42}
    feat = {"unknown": 1.0}
    temp = {"drone": 0.64, "background": 0.36}
    rec = fuse_rule_based(
        0.22,
        crop,
        feat,
        temp,
        motion_score=0.0,
        alignment_quality=0.8,
        track_score=0.0,
        mode="normal",
        candidate_source="yolo_tile_fallback",
        verified_objectness_mode="hard_recovery",
        verified_objectness_floor=0.55,
        fallback_max_negative_evidence=0.8,
    )
    unverified = fuse_rule_based(
        0.22,
        crop,
        feat,
        temp,
        motion_score=0.0,
        alignment_quality=0.8,
        track_score=0.0,
        mode="normal",
        candidate_source="yolo_tile_fallback",
        verified_objectness=False,
        fallback_max_negative_evidence=0.8,
    )
    assert rec.final_drone_score > unverified.final_drone_score
