from __future__ import annotations

import cv2
import numpy as np

from qstr_dronedet.motion.quality import compute_alignment_quality
from qstr_dronedet.types import AlignmentResult


def preprocess_gray(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr.copy()
    return cv2.GaussianBlur(gray, (5, 5), 0)


def make_grid_points(width: int, height: int, step: int = 32, margin: int = 16) -> np.ndarray:
    xs = np.arange(margin, max(margin + 1, width - margin), step, dtype=np.float32)
    ys = np.arange(margin, max(margin + 1, height - margin), step, dtype=np.float32)
    pts = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
    return pts.reshape(-1, 1, 2)


def track_points_lk(prev_gray: np.ndarray, curr_gray: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points is None or len(points) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), np.empty((0,), np.float32)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points, None, winSize=(21, 21), maxLevel=3)
    back, st_back, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, nxt, None, winSize=(21, 21), maxLevel=3)
    src = points.reshape(-1, 2)
    dst = nxt.reshape(-1, 2) if nxt is not None else np.empty_like(src)
    bck = back.reshape(-1, 2) if back is not None else np.empty_like(src)
    fb_error = np.linalg.norm(src - bck, axis=1)
    h, w = curr_gray.shape[:2]
    valid = (
        (st.reshape(-1) == 1)
        & (st_back.reshape(-1) == 1)
        & (fb_error < 2.5)
        & (dst[:, 0] >= 0)
        & (dst[:, 0] < w)
        & (dst[:, 1] >= 0)
        & (dst[:, 1] < h)
    )
    return src[valid].astype(np.float32), dst[valid].astype(np.float32), fb_error[valid].astype(np.float32)


def estimate_translation(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    if len(src_pts) < 1:
        return None, np.zeros((0,), dtype=np.uint8)
    flow = dst_pts.reshape(-1, 2) - src_pts.reshape(-1, 2)
    med = np.median(flow, axis=0)
    residual = np.linalg.norm(flow - med, axis=1)
    mad = np.median(np.abs(residual - np.median(residual))) + 1e-6
    mask = residual <= max(2.5, 3.5 * mad)
    mat = np.array([[1.0, 0.0, med[0]], [0.0, 1.0, med[1]]], dtype=np.float32)
    return mat, mask.astype(np.uint8)


def estimate_affine(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    if len(src_pts) < 3:
        return None, np.zeros((len(src_pts),), dtype=np.uint8)
    mat, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return (mat.astype(np.float32) if mat is not None else None), (mask.reshape(-1).astype(np.uint8) if mask is not None else np.zeros((len(src_pts),), dtype=np.uint8))


def estimate_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    if len(src_pts) < 4:
        return None, np.zeros((len(src_pts),), dtype=np.uint8)
    mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    return (mat.astype(np.float32) if mat is not None else None), (mask.reshape(-1).astype(np.uint8) if mask is not None else np.zeros((len(src_pts),), dtype=np.uint8))


def estimate_best_alignment(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    grid_step: int = 32,
    models: tuple[str, ...] = ("translation", "affine", "homography"),
) -> AlignmentResult:
    prev_gray = preprocess_gray(prev_frame)
    curr_gray = preprocess_gray(curr_frame)
    h, w = curr_gray.shape[:2]
    pts = make_grid_points(w, h, step=grid_step)
    src, dst, fb = track_points_lk(prev_gray, curr_gray, pts)
    if len(src) < 3:
        return AlignmentResult(None, "none", 0.0, 0.0, 0, 1e6, 1.0, float(cv2.Laplacian(curr_gray, cv2.CV_64F).var()), {"reason": "too_few_tracks", "tracked": len(src)})
    results: list[AlignmentResult] = []
    for model in models:
        if model == "translation":
            transform, mask = estimate_translation(src, dst)
        elif model == "affine":
            transform, mask = estimate_affine(src, dst)
        elif model == "homography":
            transform, mask = estimate_homography(src, dst)
        else:
            continue
        res = compute_alignment_quality(src, dst, transform, model, mask, prev_gray, curr_gray)
        res.debug["tracked"] = int(len(src))
        res.debug["fb_error_median"] = float(np.median(fb)) if len(fb) else 0.0
        results.append(res)
    return max(results, key=lambda r: r.quality) if results else AlignmentResult(None, "none", 0.0, 0.0, 0, 1e6, 1.0, 0.0, {})


def warp_frame(frame_gray: np.ndarray, alignment_result: AlignmentResult, out_size: tuple[int, int]) -> np.ndarray:
    if alignment_result.transform is None:
        return np.zeros((out_size[1], out_size[0]), dtype=frame_gray.dtype)
    if alignment_result.transform_type in {"translation", "affine"}:
        return cv2.warpAffine(frame_gray, alignment_result.transform, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if alignment_result.transform_type == "homography":
        return cv2.warpPerspective(frame_gray, alignment_result.transform, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.zeros((out_size[1], out_size[0]), dtype=frame_gray.dtype)

