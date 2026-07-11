from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HomographyEstimate:
    matrix: np.ndarray
    valid: bool
    tracked_points: int
    inliers: int
    inlier_ratio: float
    median_reprojection_error: float


def _gray(image: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, float(max_size) / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, scale


def _grid_points(width: int, height: int, columns: int, rows: int) -> np.ndarray:
    xs = np.linspace(width * 0.05, width * 0.95, columns, dtype=np.float32)
    ys = np.linspace(height * 0.05, height * 0.95, rows, dtype=np.float32)
    return np.asarray([(x, y) for y in ys for x in xs], dtype=np.float32).reshape(-1, 1, 2)


def estimate_background_homography(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    *,
    max_size: int = 512,
    grid_columns: int = 24,
    grid_rows: int = 18,
    min_tracks: int = 24,
    min_inlier_ratio: float = 0.45,
    ransac_threshold: float = 2.5,
    max_forward_backward_error: float = 1.5,
) -> HomographyEstimate:
    identity = np.eye(3, dtype=np.float64)
    previous_gray, scale = _gray(previous_image, max_size)
    current_gray, current_scale = _gray(current_image, max_size)
    if previous_gray.shape != current_gray.shape or abs(scale - current_scale) > 1e-6:
        return HomographyEstimate(identity, False, 0, 0, 0.0, float("inf"))

    points = _grid_points(previous_gray.shape[1], previous_gray.shape[0], grid_columns, grid_rows)
    lk_args = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None, **lk_args)
    if forward is None or forward_status is None:
        return HomographyEstimate(identity, False, 0, 0, 0.0, float("inf"))
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray, forward, None, **lk_args)
    if backward is None or backward_status is None:
        return HomographyEstimate(identity, False, 0, 0, 0.0, float("inf"))

    valid = (forward_status.ravel() == 1) & (backward_status.ravel() == 1)
    forward_backward_error = np.linalg.norm(backward.reshape(-1, 2) - points.reshape(-1, 2), axis=1)
    valid &= np.isfinite(forward_backward_error) & (forward_backward_error <= max_forward_backward_error)
    source = points.reshape(-1, 2)[valid]
    destination = forward.reshape(-1, 2)[valid]
    tracked_points = len(source)
    if tracked_points < min_tracks:
        return HomographyEstimate(identity, False, tracked_points, 0, 0.0, float("inf"))

    homography_small, inlier_mask = cv2.findHomography(source, destination, cv2.RANSAC, ransac_threshold)
    if homography_small is None or inlier_mask is None or not np.isfinite(homography_small).all():
        return HomographyEstimate(identity, False, tracked_points, 0, 0.0, float("inf"))
    inlier_mask = inlier_mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / tracked_points
    if inliers < min_tracks or inlier_ratio < min_inlier_ratio:
        return HomographyEstimate(identity, False, tracked_points, inliers, inlier_ratio, float("inf"))

    projected = cv2.perspectiveTransform(source[inlier_mask, None, :], homography_small)[:, 0, :]
    median_error = float(np.median(np.linalg.norm(projected - destination[inlier_mask], axis=1)))
    determinant = float(np.linalg.det(homography_small[:2, :2]))
    perspective = float(np.linalg.norm(homography_small[2, :2]))
    if not (
        np.isfinite(median_error)
        and median_error <= ransac_threshold
        and 0.25 <= abs(determinant) <= 4.0
        and perspective <= 0.02
        and abs(homography_small[2, 2]) > 1e-8
    ):
        return HomographyEstimate(identity, False, tracked_points, inliers, inlier_ratio, median_error)

    homography_small /= homography_small[2, 2]
    scale_matrix = np.diag((scale, scale, 1.0))
    homography = np.linalg.inv(scale_matrix) @ homography_small @ scale_matrix
    homography /= homography[2, 2]
    return HomographyEstimate(homography.astype(np.float64), True, tracked_points, inliers, inlier_ratio, median_error)


def transform_bbox_xyxy(bbox: tuple[float, float, float, float], homography: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    corners = np.asarray([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float64)
    transformed = cv2.perspectiveTransform(corners, homography)[0]
    if not np.isfinite(transformed).all():
        return x1, y1, x2, y2
    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)
    return float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1])
