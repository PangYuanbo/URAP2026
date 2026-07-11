from __future__ import annotations

import numpy as np

from qstr_dronedet.tracking.action_bank import ActionBankConfig, action_token
from tools.score_tracklets_samurai_cmc import attach_causal_camera_motion


class TranslationCache:
    def sequence_size(self, seq: str, frame_id: int):
        return 100, 100

    def between(self, seq: str, source_frame: int, target_frame: int):
        gap = target_frame - source_frame
        matrix = np.asarray([[1.0, 0.0, 2.0 * gap], [0.0, 1.0, -1.0 * gap], [0.0, 0.0, 1.0]])
        return matrix, 1.0


def test_causal_camera_motion_is_attached_per_transition() -> None:
    rows = [
        {"frame_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "image_width": 100, "image_height": 100, "fps": 10.0},
        {"frame_id": 3, "bbox": [14.0, 8.0, 24.0, 18.0], "image_width": 100, "image_height": 100, "fps": 10.0},
    ]
    attach_causal_camera_motion(rows, "sequence", TranslationCache())
    assert rows[0]["camera_dx"] == 0.0
    assert np.isclose(rows[1]["camera_dx"], 4.0)
    assert np.isclose(rows[1]["camera_dy"], -2.0)
    assert rows[1]["camera_motion_gap_frames"] == 2
    assert rows[1]["camera_motion_validity"] == 1.0
    token = action_token(rows[0], rows[1], ActionBankConfig(fps_fallback=10.0))
    assert np.isclose(token.values[1], 0.0)
    assert np.isclose(token.values[2], 0.0)

