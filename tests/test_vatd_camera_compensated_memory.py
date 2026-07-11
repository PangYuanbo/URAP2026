from __future__ import annotations

import cv2
import numpy as np

from qstr_dronedet.camera_motion import estimate_background_homography, transform_bbox_xyxy
from qstr_dronedet.pipelines.temporal_recovery import (
    MotionMemoryTrack,
    TemporalRecoveryConfig,
    score_candidates_with_motion_memory,
)
from qstr_dronedet.types import DetectionCandidate


def _candidate(bbox, score):
    return DetectionCandidate(tuple(float(value) for value in bbox), float(score), "yolo_tile")


def test_numpy_homography_estimator_recovers_camera_translation() -> None:
    rng = np.random.default_rng(17)
    previous = rng.integers(0, 256, size=(320, 480, 3), dtype=np.uint8)
    previous = cv2.GaussianBlur(previous, (5, 5), 0)
    expected = np.asarray([[1.0, 0.0, 11.0], [0.0, 1.0, -7.0], [0.0, 0.0, 1.0]])
    current = cv2.warpPerspective(previous, expected, (480, 320))

    estimate = estimate_background_homography(previous, current, max_size=480)

    assert estimate.valid
    bbox = (100.0, 90.0, 120.0, 110.0)
    actual = transform_bbox_xyxy(bbox, estimate.matrix)
    assert np.allclose(actual, (111.0, 83.0, 131.0, 103.0), atol=1.5)


def test_camera_compensated_prediction_does_not_treat_camera_shift_as_object_velocity() -> None:
    camera = np.asarray([[1.0, 0.0, 12.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]])
    memory = MotionMemoryTrack((100.0, 80.0, 120.0, 100.0), velocity_xy=(0.0, 0.0), score=0.7)

    predicted = memory.predict((300, 400, 3), camera)

    assert np.allclose(predicted, (112.0, 75.0, 132.0, 95.0))


def test_memory_update_learns_only_residual_target_motion() -> None:
    camera = np.asarray([[1.0, 0.0, 10.0], [0.0, 1.0, -4.0], [0.0, 0.0, 1.0]])
    memory = MotionMemoryTrack((100.0, 80.0, 120.0, 100.0), velocity_xy=(0.0, 0.0), score=0.7)
    candidate = _candidate((113.0, 78.0, 133.0, 98.0), 0.6)

    memory.update(candidate, (300, 400, 3), camera_previous_to_current=camera, residual_velocity_momentum=0.0)

    assert np.allclose(memory.velocity_xy, (3.0, 2.0))
    next_prediction = memory.predict((300, 400, 3), camera)
    assert np.allclose(next_prediction, (126.0, 76.0, 146.0, 96.0))


def test_samurai_motion_iou_prefers_camera_compensated_candidate() -> None:
    camera = np.asarray([[1.0, 0.0, 15.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    memory = MotionMemoryTrack((20.0, 20.0, 30.0, 30.0), velocity_xy=(0.0, 0.0), score=0.7)
    correct = _candidate((35.0, 20.0, 45.0, 30.0), 0.08)
    distractor = _candidate((20.0, 20.0, 30.0, 30.0), 0.30)

    scored = score_candidates_with_motion_memory(
        [distractor, correct],
        memory,
        (120, 120, 3),
        TemporalRecoveryConfig(max_center_distance=32, samurai_motion_iou_weight=0.5),
        camera,
    )

    assert scored[0].bbox_xyxy == correct.bbox_xyxy
    assert scored[0].extra["samurai_motion_iou"] == 1.0
    assert scored[1].extra["samurai_motion_iou"] == 0.0
