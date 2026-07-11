import unittest

import numpy as np

from qstr_dronedet.samurai_bbox_readout import decode_delta, encode_delta, tracking_metrics


class SamuraiBBoxReadoutTest(unittest.TestCase):
    def test_box_delta_round_trip(self):
        previous = np.asarray([[10, 20, 8, 6], [50, 60, 12, 9]], dtype=np.float32)
        target = np.asarray([[13, 18, 10, 5], [48, 63, 9, 12]], dtype=np.float32)
        image_wh = np.asarray([[100, 100], [100, 100]], dtype=np.float32)
        reconstructed = decode_delta(previous, encode_delta(previous, target), image_wh)
        np.testing.assert_allclose(reconstructed, target, atol=1e-5)

    def test_perfect_tracking_metrics(self):
        boxes = np.asarray([[10, 20, 8, 6], [11, 21, 8, 6]], dtype=np.float32)
        metrics = tracking_metrics(boxes, boxes)
        self.assertEqual(metrics.mean_iou, 1.0)
        self.assertEqual(metrics.success_auc, 1.0)
        self.assertEqual(metrics.success_50, 1.0)
        self.assertEqual(metrics.precision_20, 1.0)


if __name__ == "__main__":
    unittest.main()
