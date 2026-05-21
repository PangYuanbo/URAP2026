import cv2
import numpy as np

from qstr_dronedet.motion.alignment import estimate_best_alignment


def _grid_frame(shift=(0, 0), blur=False):
    img = np.zeros((160, 220, 3), np.uint8)
    for y in range(20, 145, 30):
        for x in range(20, 205, 30):
            cv2.circle(img, (x + shift[0], y + shift[1]), 3, (255, 255, 255), -1)
    if blur:
        img = cv2.GaussianBlur(img, (17, 17), 0)
    return img


def test_reliable_shift_alignment_quality():
    prev = _grid_frame()
    curr = _grid_frame((4, 3))
    res = estimate_best_alignment(prev, curr, grid_step=20)
    assert res.quality > 0.4
    assert res.num_inliers > 5


def test_textureless_alignment_low_quality():
    prev = np.zeros((120, 160, 3), np.uint8)
    curr = np.zeros((120, 160, 3), np.uint8)
    res = estimate_best_alignment(prev, curr)
    assert res.quality <= 0.3

