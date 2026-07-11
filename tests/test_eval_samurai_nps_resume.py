import csv
import math
import tempfile
import unittest
from pathlib import Path

from tools.eval_samurai_nps import read_prediction_csv, summarize_sequence


class SamuraiResumeTest(unittest.TestCase):
    def test_infinite_center_error_survives_csv_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prediction.csv"
            fields = [
                "sequence", "frame_index", "pred_x", "pred_y", "pred_w", "pred_h",
                "gt_x", "gt_y", "gt_w", "gt_h", "visible", "iou", "center_error",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    dict(sequence="seq", frame_index=0, pred_x=0, pred_y=0, pred_w=0, pred_h=0,
                         gt_x=10, gt_y=10, gt_w=5, gt_h=5, visible=1, iou=0, center_error="inf")
                )
            rows = read_prediction_csv(path)
            self.assertTrue(math.isinf(rows[0]["center_error"]))
            summary = summarize_sequence("seq", rows)
            self.assertEqual(summary["precision_20"], 0.0)


if __name__ == "__main__":
    unittest.main()
