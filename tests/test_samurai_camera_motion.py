from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SAM2_ROOT = ROOT / "third_party" / "samurai" / "sam2"
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

from sam2.utils.camera_motion import (  # noqa: E402
    estimate_background_homography,
    transform_xyxy,
)


def _normalized_tensor(image: np.ndarray) -> torch.Tensor:
    rgb = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    mean = rgb.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = rgb.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    return (rgb - mean) / std


def test_homography_estimator_recovers_camera_translation() -> None:
    rng = np.random.default_rng(7)
    previous = rng.integers(0, 256, size=(384, 512, 3), dtype=np.uint8)
    previous = cv2.GaussianBlur(previous, (5, 5), 0)
    expected = np.asarray(
        [[1.0, 0.002, 9.0], [-0.001, 1.0, -6.0], [2e-6, -1e-6, 1.0]],
        dtype=np.float64,
    )
    current = cv2.warpPerspective(previous, expected, (512, 384))
    estimate = estimate_background_homography(
        _normalized_tensor(previous),
        _normalized_tensor(current),
        max_size=512,
        min_tracks=24,
    )
    assert estimate.valid
    assert estimate.inliers >= 24
    assert estimate.inlier_ratio >= 0.45

    points = np.asarray([[[50.0, 70.0], [250.0, 180.0], [460.0, 320.0]]], dtype=np.float64)
    expected_points = cv2.perspectiveTransform(points, expected)
    actual_points = cv2.perspectiveTransform(points, estimate.matrix)
    assert np.max(np.linalg.norm(expected_points - actual_points, axis=2)) < 1.5


def test_stabilized_prediction_adds_camera_motion_once() -> None:
    camera_previous_to_current = np.asarray(
        [[1.0, 0.0, 12.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    previous_to_reference = np.eye(3, dtype=np.float64)
    current_to_reference = previous_to_reference @ np.linalg.inv(camera_previous_to_current)

    previous_image_bbox = [100.0, 80.0, 120.0, 100.0]
    residual_velocity = np.asarray([3.0, 2.0])
    stable_bbox = transform_xyxy(previous_image_bbox, previous_to_reference)
    stable_prediction = [
        stable_bbox[0] + residual_velocity[0],
        stable_bbox[1] + residual_velocity[1],
        stable_bbox[2] + residual_velocity[0],
        stable_bbox[3] + residual_velocity[1],
    ]
    image_prediction = transform_xyxy(stable_prediction, np.linalg.inv(current_to_reference))

    assert np.allclose(image_prediction, [115.0, 77.0, 135.0, 97.0], atol=1e-6)


def test_degenerate_frames_fall_back_to_identity() -> None:
    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    estimate = estimate_background_homography(
        _normalized_tensor(blank), _normalized_tensor(blank), min_tracks=24
    )
    assert not estimate.valid
    assert np.allclose(estimate.matrix, np.eye(3))
