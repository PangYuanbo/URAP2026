import unittest
import numpy as np
from tools.train_action_chunk_causal_full import AUX_D, causal_arrays

class Aux:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)
    def get_many(self, seq, frame_id, count):
        return np.repeat(self.value[None, :], count, axis=0)

class ActionChunkCausalBoundaryTests(unittest.TestCase):
    def test_backward_bank_is_supervision_not_input(self):
        predictions = {'Clip_1_7': {'detections': [{'score': .8, 'bbox': [0, 0, 10, 10]}], 'labels': [{'bbox': [0, 0, 10, 10]}]}}
        forward = Aux(np.arange(AUX_D, dtype=np.float32) / AUX_D)
        backward_low = Aux(np.zeros(AUX_D, dtype=np.float32))
        backward_high = Aux(np.ones(AUX_D, dtype=np.float32))
        x_low, _, weight_low, _, _, _ = causal_arrays(predictions, forward, backward_low, True)
        x_high, _, weight_high, _, _, _ = causal_arrays(predictions, forward, backward_high, True)
        self.assertEqual(x_low.shape, (1, 4 + AUX_D))
        np.testing.assert_array_equal(x_low, x_high)
        self.assertLess(float(weight_low[0]), float(weight_high[0]))

if __name__ == '__main__':
    unittest.main()
