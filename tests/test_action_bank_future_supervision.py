from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.train_action_bank_all_candidate_listwise import FEATURE_NAMES, TOKEN_FEATURE_NAMES, dataset_arrays, load_auxiliary


class FutureSupervisionIsolationTests(unittest.TestCase):
    def test_future_consistency_is_target_only(self) -> None:
        predictions = {
            "Clip_1_00001": {
                "detections": [{"bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.8, "category_id": 0}],
                "labels": [{"bbox": [0.0, 0.0, 10.0, 10.0], "category_id": 0}],
            }
        }
        record = {
            "meta": {"seq": "Clip_1", "image_id": "Clip_1_00001"},
            "rows": [{
                "seq": "Clip_1",
                "frame_id": 1,
                "prediction_index": 0,
                "online_action_bank_score": 0.9,
                "online_action_bank_future_consistency": 0.75,
                "image_width": 100,
                "image_height": 100,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            auxiliary, sizes = load_auxiliary(path)
            features, current_iou, future, groups, locations = dataset_arrays(predictions, auxiliary, sizes, {}, True)
        future_index = FEATURE_NAMES.index("online_action_bank_future_consistency")
        self.assertEqual(float(features[0, future_index]), 0.0)
        self.assertAlmostEqual(float(future[0]), 0.75)
        self.assertAlmostEqual(float(current_iou[0]), 1.0)
        self.assertEqual(groups, [(0, 1)])
        self.assertEqual(locations, [("Clip_1_00001", 0)])


    def test_action_tokens_are_loaded_and_padded(self) -> None:
        predictions = {"Clip_1_00001": {"detections": [{"bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.8}], "labels": []}}
        record = {"meta": {"seq": "Clip_1", "image_id": "Clip_1_00001"}, "rows": [{
            "seq": "Clip_1", "frame_id": 1, "prediction_index": 0,
            "online_action_bank_short_tokens": [1.0, 0.8],
            "online_action_bank_long_tokens": [1.0, 0.6],
            "image_width": 100, "image_height": 100,
        }]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            auxiliary, sizes = load_auxiliary(path)
            features, *_ = dataset_arrays(predictions, auxiliary, sizes, {}, False)
        self.assertEqual(features.shape[1], len(FEATURE_NAMES))
        token_start = FEATURE_NAMES.index(TOKEN_FEATURE_NAMES[0])
        self.assertTrue(np.allclose(features[0, token_start:token_start + 4], [1.0, 0.8, 0.0, 0.0]))

if __name__ == "__main__":
    unittest.main()
