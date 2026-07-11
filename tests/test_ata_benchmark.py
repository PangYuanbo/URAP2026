from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qstr_dronedet.ata_benchmark import evaluate_sequence, read_boxes, read_prediction_boxes


class AtaBenchmarkTest(unittest.TestCase):
    def test_read_boxes_accepts_comma_and_space_delimiters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "boxes.txt"
            path.write_text("1,2,3,4\n5 6 7 8\n", encoding="ascii")
            self.assertEqual([(1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)], read_boxes(path))

    def test_perfect_predictions_score_one(self) -> None:
        boxes = [(10.0, 20.0, 8.0, 6.0), (12.0, 21.0, 8.0, 6.0)]
        metrics = evaluate_sequence("perfect", boxes, boxes)
        self.assertAlmostEqual(1.0, metrics.auc)
        self.assertAlmostEqual(1.0, metrics.op50)
        self.assertAlmostEqual(1.0, metrics.precision_20)
        self.assertAlmostEqual(1.0, metrics.normalized_precision_auc)
        self.assertAlmostEqual(1.0, metrics.mean_iou)

    def test_read_samurai_prediction_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sequence.csv"
            path.write_text(
                "sequence,frame_index,pred_x,pred_y,pred_w,pred_h\n"
                "uav-m23,0,1,2,3,4\n",
                encoding="ascii",
            )
            self.assertEqual([(1.0, 2.0, 3.0, 4.0)], read_prediction_boxes(path))

    def test_large_offset_fails_localization_metrics(self) -> None:
        targets = [(10.0, 20.0, 8.0, 6.0)]
        predictions = [(100.0, 100.0, 8.0, 6.0)]
        metrics = evaluate_sequence("offset", predictions, targets)
        self.assertEqual(0.0, metrics.op50)
        self.assertEqual(0.0, metrics.precision_20)
        self.assertEqual(0.0, metrics.normalized_precision_auc)
        self.assertEqual(0.0, metrics.mean_iou)


if __name__ == "__main__":
    unittest.main()
