from qstr_dronedet.tracking.kalman import ConstantVelocityTracker
from qstr_dronedet.types import DetectionCandidate


def test_tracker_candidates_include_validation_metadata():
    tracker = ConstantVelocityTracker()
    det = DetectionCandidate((10, 10, 20, 20), 0.8, "yolo_tile")
    tracker.update([det], alignment_quality=1.0)
    cands = tracker.get_track_candidates()

    assert len(cands) == 1
    extra = cands[0].extra
    assert extra["track_id"] == 1
    assert extra["track_detector_updates"] == 1
    assert extra["track_last_detector_source"] == "yolo_tile"
    assert extra["track_frames_since_detector_update"] == 0
    assert extra["track_history_len"] >= 1
    assert extra["track_drift"] == 0.0


def test_tracker_does_not_validate_stale_prediction_without_detector_update():
    tracker = ConstantVelocityTracker()
    det = DetectionCandidate((10, 10, 20, 20), 0.8, "yolo_tile")
    tracker.update([det], alignment_quality=1.0)
    for _ in range(4):
        tracker.update([], alignment_quality=1.0)

    cands = tracker.get_track_candidates()
    assert cands
    extra = cands[0].extra
    assert extra["track_frames_since_detector_update"] >= 4
    assert extra["track_validated"] is False


def test_pure_tracker_candidate_does_not_refresh_detector_age():
    tracker = ConstantVelocityTracker()
    tracker.update([DetectionCandidate((10, 10, 20, 20), 0.8, "yolo_tile")], alignment_quality=1.0)
    track_candidate = tracker.get_track_candidates()[0]
    tracker.update([track_candidate], alignment_quality=1.0)
    refreshed = tracker.get_track_candidates()[0]

    assert refreshed.extra["track_detector_updates"] == 1
    assert refreshed.extra["track_frames_since_detector_update"] == 1


def test_fallback_candidate_can_reacquire_with_wider_radius():
    tracker = ConstantVelocityTracker(r0=8.0, fallback_bonus=50.0, tiny_bonus=0.0)
    tracker.update([DetectionCandidate((10, 10, 20, 20), 0.8, "yolo_tile")], alignment_quality=1.0)
    for _ in range(2):
        tracker.update([], alignment_quality=1.0)

    # Center is 35 px away from the original center: outside r0 but inside fallback-boosted radius.
    tracker.update([DetectionCandidate((45, 10, 55, 20), 0.2, "yolo_tile_fallback")], alignment_quality=1.0)
    cands = tracker.get_track_candidates()

    assert len(cands) == 1
    assert cands[0].extra["track_id"] == 1
    assert cands[0].extra["track_detector_updates"] == 2
    assert "fallback" in cands[0].extra["track_last_detector_source"]


def test_low_score_fallback_can_spawn_but_low_motion_cannot():
    tracker = ConstantVelocityTracker()
    tracker.update(
        [
            DetectionCandidate((10, 10, 20, 20), 0.09, "yolo_tile_fallback"),
            DetectionCandidate((100, 100, 110, 110), 0.09, "motion"),
        ],
        alignment_quality=1.0,
    )
    cands = tracker.get_track_candidates()

    assert len(cands) == 1
    assert "fallback" in cands[0].extra["track_last_detector_source"]
