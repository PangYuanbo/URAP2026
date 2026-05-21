from __future__ import annotations


def determine_mode(
    motion_score: float,
    alignment_quality: float,
    track_speed: float,
    blur_score: float,
    track_confidence: float,
    static_motion_threshold: float = 0.08,
    fast_speed_threshold: float = 12.0,
    blur_threshold: float = 20.0,
) -> str:
    if motion_score < static_motion_threshold and track_confidence > 0.6:
        return "static_or_hovering"
    if alignment_quality < 0.3 and blur_score < blur_threshold:
        return "bad_alignment_fast_egomotion"
    if track_speed > fast_speed_threshold:
        return "fast_target"
    if alignment_quality < 0.66 and motion_score >= static_motion_threshold:
        return "fast_target"
    if alignment_quality <= 0.05 and track_confidence <= 0.2:
        return "uncertain"
    return "normal"
