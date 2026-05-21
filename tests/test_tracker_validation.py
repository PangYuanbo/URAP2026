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
