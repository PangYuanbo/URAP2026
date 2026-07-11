import numpy as np

from qstr_dronedet.pipelines.temporal_recovery import (
    MotionMemoryTrack,
    TemporalRecoveryConfig,
    apply_final_output_score,
    hard_reset_bbox_correction,
    ncc_proposal_from_memory,
    run_temporal_recovery_frames,
    score_candidates_with_motion_memory,
    select_final_candidate,
    should_emit_detection,
    should_write_motion_memory,
    zoom_in_redetect,
)
from qstr_dronedet.types import DetectionCandidate


def _cand(bbox, score=0.2, source="yolo_tile"):
    return DetectionCandidate(tuple(float(v) for v in bbox), float(score), source)


def test_motion_memory_promotes_low_score_candidate_near_prediction():
    memory = MotionMemoryTrack((10, 10, 18, 18), velocity_xy=(4, 0), score=0.7)
    near = _cand((14, 10, 22, 18), 0.08)
    far = _cand((90, 90, 98, 98), 0.32)

    scored = score_candidates_with_motion_memory([far, near], memory, (120, 120, 3), TemporalRecoveryConfig(max_center_distance=32))

    assert scored[0].bbox_xyxy == near.bbox_xyxy
    assert scored[0].extra["motion_memory_score"] > scored[1].extra["motion_memory_score"]
    assert scored[0].extra["raw_objectness"] == near.objectness


def test_final_selection_defaults_to_temporal_order():
    temporal_top = DetectionCandidate(
        bbox_xyxy=(10.0, 10.0, 18.0, 18.0),
        objectness=0.9,
        source="gray_ncc+yolov5_dual",
        extra={"raw_objectness": 0.2},
    )
    raw_top = DetectionCandidate(
        bbox_xyxy=(30.0, 30.0, 38.0, 38.0),
        objectness=0.4,
        source="yolov5_dual",
        extra={"raw_objectness": 0.8},
    )

    selected = select_final_candidate([temporal_top, raw_top], TemporalRecoveryConfig(final_selection_score="temporal"))

    assert selected is temporal_top


def test_final_selection_can_use_raw_detector_score():
    temporal_top = DetectionCandidate(
        bbox_xyxy=(10.0, 10.0, 18.0, 18.0),
        objectness=0.9,
        source="gray_ncc+yolov5_dual",
        extra={"raw_objectness": 0.2},
    )
    raw_top = DetectionCandidate(
        bbox_xyxy=(30.0, 30.0, 38.0, 38.0),
        objectness=0.4,
        source="yolov5_dual",
        extra={"raw_objectness": 0.8},
    )

    selected = select_final_candidate([temporal_top, raw_top], TemporalRecoveryConfig(final_selection_score="raw"))

    assert selected is raw_top


def test_final_output_score_can_use_raw_objectness():
    candidate = DetectionCandidate(
        bbox_xyxy=(10.0, 10.0, 18.0, 18.0),
        objectness=0.4,
        source="yolov5_dual",
        extra={"raw_objectness": 0.8},
    )

    scored = apply_final_output_score(candidate, TemporalRecoveryConfig(final_output_score="raw"))

    assert scored.objectness == 0.8
    assert scored.extra["final_output_score"] == 0.8


def test_raw_output_uses_preserved_detector_evidence_from_merged_candidate():
    candidate = DetectionCandidate(
        bbox_xyxy=(10.0, 10.0, 18.0, 18.0),
        objectness=0.9,
        source="gray_ncc+yolov5_dual",
        extra={
            "raw_objectness": 0.34,
            "detector_raw_objectness": 0.82,
            "detector_bbox_xyxy": [30.0, 30.0, 38.0, 38.0],
            "detector_source": "yolov5_dual",
        },
    )

    scored = apply_final_output_score(candidate, TemporalRecoveryConfig(final_output_score="raw"))

    assert scored.objectness == 0.82
    assert scored.bbox_xyxy == (30.0, 30.0, 38.0, 38.0)
    assert scored.source == "yolov5_dual"
    assert scored.extra["final_output_bbox_source"] == "yolov5_dual"


def test_run_temporal_recovery_can_decouple_output_from_memory_update():
    frames = [np.zeros((120, 120, 3), dtype=np.uint8) for _ in range(2)]

    def detector(frame):
        idx = detector.calls
        detector.calls += 1
        if idx == 0:
            return [_cand((10, 10, 18, 18), 0.6, source="yolov5_dual")]
        return [
            _cand((70, 70, 78, 78), 0.6, source="yolov5_dual"),
            _cand((10, 10, 18, 18), 0.05, source="yolov5_dual"),
        ]

    detector.calls = 0
    rows = run_temporal_recovery_frames(
        frames,
        detector,
        None,
        TemporalRecoveryConfig(
            final_selection_score="raw",
            final_output_score="raw",
            memory_update_selection="temporal",
            apply_output_gate=False,
            max_center_distance=32,
            ncc_min_score=1.1,
            hard_reset_min_score=1.0,
            memory_detector_min=0.0,
        ),
    )

    assert rows[1].selected is not None
    assert rows[1].selected.bbox_xyxy == (70.0, 70.0, 78.0, 78.0)
    assert rows[1].selected.objectness == 0.6
    assert np.allclose(rows[1].memory_bbox, (10.0, 10.0, 18.0, 18.0))
    assert rows[1].diagnostics["memory_write"] is True


def test_dji_tiny_profile_triggers_zoom_redetect_and_remaps_crop_bbox():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    predicted = (40, 40, 48, 48)

    def crop_detector(crop):
        assert crop.shape[0] >= 64
        return [_cand((28, 28, 36, 36), 0.44, source="crop_yolo")]

    found = zoom_in_redetect(frame, predicted, crop_detector, TemporalRecoveryConfig(profile="dji-tiny", zoom_crop_scale=8.0))

    assert len(found) == 1
    assert found[0].source == "zoom_redetect"
    assert found[0].bbox_xyxy == (40.0, 40.0, 48.0, 48.0)
    assert found[0].extra["crop_origin_xy"] == [12, 12]


def test_ncc_support_proposes_bbox_when_detector_misses():
    prev = np.zeros((80, 80, 3), dtype=np.uint8)
    curr = np.zeros((80, 80, 3), dtype=np.uint8)
    patch = np.zeros((10, 10, 3), dtype=np.uint8)
    patch[2:8, 2:8] = 180
    patch[4:6, 4:6] = 255
    prev[20:30, 20:30] = patch
    curr[23:33, 25:35] = patch
    memory = MotionMemoryTrack((20, 20, 30, 30), velocity_xy=(5, 3), score=0.6)

    prop = ncc_proposal_from_memory(prev, curr, memory, TemporalRecoveryConfig(ncc_min_score=0.5))

    assert prop is not None
    assert prop.source == "gray_ncc"
    assert prop.bbox_xyxy == (25.0, 23.0, 35.0, 33.0)
    assert prop.extra["ncc_score"] >= 0.5


def test_support_only_ncc_needs_enough_clean_detector_history_to_emit():
    memory = MotionMemoryTrack((20, 20, 30, 30), velocity_xy=(5, 3), score=0.6, detector_updates=1)
    candidate = DetectionCandidate(
        bbox_xyxy=(25.0, 23.0, 35.0, 33.0),
        objectness=0.34,
        source="gray_ncc",
        motion_score=1.0,
        track_score=1.0,
        extra={"raw_objectness": 0.34, "motion_memory_score": 1.0, "ncc_score": 1.0},
    )

    emit, quality, reason = should_emit_detection(candidate, memory, TemporalRecoveryConfig())

    assert emit is False
    assert quality > 0.7
    assert reason == "support_only_insufficient_detector_history"


def test_support_only_ncc_can_emit_after_clean_detector_history_and_quality_gate():
    memory = MotionMemoryTrack((20, 20, 30, 30), velocity_xy=(5, 3), score=0.6, detector_updates=2)
    candidate = DetectionCandidate(
        bbox_xyxy=(25.0, 23.0, 35.0, 33.0),
        objectness=0.34,
        source="gray_ncc",
        motion_score=1.0,
        track_score=1.0,
        extra={"raw_objectness": 0.34, "motion_memory_score": 1.0, "ncc_score": 1.0},
    )

    emit, quality, reason = should_emit_detection(candidate, memory, TemporalRecoveryConfig(support_only_output_min_quality=0.70))

    assert emit is True
    assert quality > 0.7
    assert reason == "support_only_quality_pass"


def test_run_temporal_recovery_suppresses_early_support_only_ncc_output():
    prev = np.zeros((80, 80, 3), dtype=np.uint8)
    curr = np.zeros((80, 80, 3), dtype=np.uint8)
    patch = np.zeros((10, 10, 3), dtype=np.uint8)
    patch[2:8, 2:8] = 180
    patch[4:6, 4:6] = 255
    prev[20:30, 20:30] = patch
    curr[23:33, 25:35] = patch

    def detector(frame):
        idx = detector.calls
        detector.calls += 1
        if idx == 0:
            return [_cand((20, 20, 30, 30), 0.5)]
        return []

    detector.calls = 0

    rows = run_temporal_recovery_frames([prev, curr], detector, None, TemporalRecoveryConfig(ncc_min_score=0.5))

    assert rows[0].selected is not None
    assert rows[1].selected is None
    assert rows[1].diagnostics["support_candidates"] == 1
    assert rows[1].diagnostics["emit_reason"] == "support_only_insufficient_detector_history"


def test_hard_reset_keeps_strong_detector_far_from_stale_memory():
    memory = MotionMemoryTrack((10, 10, 18, 18), velocity_xy=(0, 0), score=0.2, misses=3)
    selected = _cand((130, 130, 146, 146), 0.8, source="yolov5_dual")

    corrected = hard_reset_bbox_correction(selected, memory, (180, 180, 3), TemporalRecoveryConfig(hard_reset_min_distance=32))

    assert corrected is not None
    assert corrected.bbox_xyxy == selected.bbox_xyxy
    assert corrected.extra["hard_reset"] is True
    assert corrected.extra["hard_reset_reason"] == "strong_detector_far_from_stale_memory"


def test_hard_reset_uses_raw_detector_score_before_temporal_suppression():
    memory = MotionMemoryTrack((10, 10, 18, 18), velocity_xy=(0, 0), score=0.2, misses=3)
    selected = DetectionCandidate(
        bbox_xyxy=(130.0, 130.0, 146.0, 146.0),
        objectness=0.46,
        source="yolov5_dual",
        extra={"raw_objectness": 0.8, "motion_memory_score": 0.0},
    )

    corrected = hard_reset_bbox_correction(selected, memory, (180, 180, 3), TemporalRecoveryConfig(hard_reset_min_distance=32, hard_reset_min_score=0.55))

    assert corrected is not None
    assert corrected.extra["hard_reset"] is True


def test_motion_aware_memory_gate_rejects_low_detector_quality():
    candidate = DetectionCandidate(
        bbox_xyxy=(14.0, 10.0, 22.0, 18.0),
        objectness=0.44,
        source="yolo_tile",
        motion_score=1.0,
        extra={"raw_objectness": 0.03, "motion_memory_score": 1.0},
    )

    write_memory, quality, reason = should_write_motion_memory(candidate, TemporalRecoveryConfig(memory_detector_min=0.05))

    assert write_memory is False
    assert quality > 0.0
    assert reason == "detector_score_below_memory_gate"


def test_run_temporal_recovery_updates_state_without_polluting_memory_bank():
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(2)]

    def detector(frame):
        idx = detector.calls
        detector.calls += 1
        if idx == 0:
            return [_cand((10, 10, 18, 18), 0.5)]
        return [_cand((14, 10, 22, 18), 0.03)]

    detector.calls = 0

    rows = run_temporal_recovery_frames(
        frames,
        detector,
        None,
        TemporalRecoveryConfig(max_center_distance=32, ncc_min_score=1.1, memory_detector_min=0.05),
    )

    assert rows[0].diagnostics["memory_bank_size"] == 1
    assert rows[1].selected is not None
    assert rows[1].selected.extra["memory_write"] is False
    assert rows[1].diagnostics["memory_bank_size"] == 1
    assert rows[1].memory_bbox == (14.0, 10.0, 22.0, 18.0)


def test_run_temporal_recovery_frames_uses_zoom_after_miss_for_tiny_profile():
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]

    def detector(frame):
        idx = detector.calls
        detector.calls += 1
        if idx == 0:
            return [_cand((40, 40, 48, 48), 0.5)]
        return []

    detector.calls = 0

    def crop_detector(crop):
        return [_cand((28, 28, 36, 36), 0.35, source="crop_yolo")]

    rows = run_temporal_recovery_frames(
        frames,
        detector,
        crop_detector,
        TemporalRecoveryConfig(profile="dji-tiny", zoom_crop_scale=8.0, zoom_trigger_misses=0, ncc_min_score=1.1),
    )

    assert rows[0].selected is not None
    assert rows[1].selected is not None
    assert rows[1].selected.source == "zoom_redetect"
    assert rows[1].selected.bbox_xyxy == (40.0, 40.0, 48.0, 48.0)
