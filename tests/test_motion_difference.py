import cv2
import numpy as np

from qstr_dronedet.motion.alignment import estimate_best_alignment
from qstr_dronedet.motion.difference import compute_motion_map, motion_score_in_bbox


def test_moving_tiny_dot_has_motion_response():
    prev = np.zeros((96, 128, 3), np.uint8)
    curr = np.zeros_like(prev)
    for y in range(12, 90, 18):
        for x in range(12, 120, 18):
            cv2.circle(prev, (x, y), 1, (80, 80, 80), -1)
            cv2.circle(curr, (x, y), 1, (80, 80, 80), -1)
    cv2.circle(prev, (40, 50), 2, (255, 255, 255), -1)
    cv2.circle(curr, (48, 50), 2, (255, 255, 255), -1)
    align = estimate_best_alignment(prev, curr, grid_step=16, models=("translation",))
    motion = compute_motion_map(prev, curr, align, clean=False)
    assert motion_score_in_bbox(motion, (44, 46, 53, 55)) > 0.1
