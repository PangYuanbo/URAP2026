from __future__ import annotations

import cv2
import numpy as np

from qstr_dronedet.types import AlignmentResult


def _project_points(points: np.ndarray, transform: np.ndarray, transform_type: str) -> np.ndarray:
    pts = points.reshape(-1, 2).astype(np.float32)
    if transform_type in {"translation", "affine"}:
        homog = np.c_[pts, np.ones(len(pts), dtype=np.float32)]
        return (homog @ transform.T).astype(np.float32)
    if transform_type == "homography":
        out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), transform)
        return out.reshape(-1, 2).astype(np.float32)
    raise ValueError(f"Unsupported transform type: {transform_type}")


def _corner_deformation_ok(transform: np.ndarray, transform_type: str, width: int, height: int) -> bool:
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    try:
        warped = _project_points(corners, transform, transform_type)
    except Exception:
        return False
    if not np.isfinite(warped).all():
        return False
    diag = float(np.hypot(width, height))
    if np.max(np.linalg.norm(warped - corners, axis=1)) > 0.75 * diag:
        return False
    area = cv2.contourArea(warped.reshape(-1, 1, 2))
    return 0.25 * width * height <= area <= 4.0 * width * height


def _warp_gray(prev_gray: np.ndarray, transform: np.ndarray, transform_type: str, out_size: tuple[int, int]) -> np.ndarray:
    if transform_type in {"translation", "affine"}:
        return cv2.warpAffine(prev_gray, transform, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.warpPerspective(prev_gray, transform, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def compute_alignment_quality(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    transform: np.ndarray | None,
    transform_type: str,
    inlier_mask: np.ndarray | None,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> AlignmentResult:
    h, w = curr_gray.shape[:2]
    blur_score = float(cv2.Laplacian(curr_gray, cv2.CV_64F).var())
    debug: dict[str, float | bool] = {}
    if transform is None or src_pts is None or dst_pts is None or len(src_pts) < 3:
        return AlignmentResult(None, transform_type, 0.0, 0.0, 0, 1e6, 1.0, blur_score, {"reason": "missing_transform_or_points"})
    if not _corner_deformation_ok(transform, transform_type, w, h):
        return AlignmentResult(transform, transform_type, 0.0, 0.0, 0, 1e6, 1.0, blur_score, {"reason": "corner_deformation"})

    src = src_pts.reshape(-1, 2)
    dst = dst_pts.reshape(-1, 2)
    mask = np.ones(len(src), dtype=bool) if inlier_mask is None else inlier_mask.reshape(-1).astype(bool)
    num_inliers = int(mask.sum())
    inlier_ratio = float(num_inliers / max(1, len(src)))
    if num_inliers < 3:
        return AlignmentResult(transform, transform_type, 0.0, inlier_ratio, num_inliers, 1e6, 1.0, blur_score, {"reason": "too_few_inliers"})

    projected = _project_points(src[mask], transform, transform_type)
    errors = np.linalg.norm(projected - dst[mask], axis=1)
    median_err = float(np.median(errors)) if len(errors) else 1e6
    aligned = _warp_gray(prev_gray, transform, transform_type, (w, h))
    residual = float(np.mean(cv2.absdiff(aligned, curr_gray)) / 255.0)

    inlier_score = min(1.0, inlier_ratio / 0.65)
    err_score = float(np.exp(-median_err / 3.0))
    photo_score = float(np.exp(-residual / 0.12))
    blur_penalty = 0.75 if blur_score < 12.0 else 1.0
    quality = float(np.clip((0.45 * inlier_score + 0.35 * err_score + 0.20 * photo_score) * blur_penalty, 0.0, 1.0))
    debug.update({"inlier_score": inlier_score, "err_score": err_score, "photo_score": photo_score})
    return AlignmentResult(transform, transform_type, quality, inlier_ratio, num_inliers, median_err, residual, blur_score, debug)

