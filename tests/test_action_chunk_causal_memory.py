import unittest
import numpy as np
from tools.train_action_chunk_causal_memory import future_strength,hard_rows


class CausalMemoryTest(unittest.TestCase):
    def test_future_strength_is_supervision_scalar(self):
        values=np.zeros((2,50),np.float32)
        values[0,0]=1.0
        result=future_strength(values)
        self.assertEqual(result.shape,(2,))
        self.assertAlmostEqual(float(result[0]),.3,places=6)

    def test_empty_frame_keeps_highest_negatives(self):
        features=np.asarray([[.1],[.9],[.4]],np.float32)
        quality=np.zeros(3,np.float32)
        keep=hard_rows(features,quality,[(0,3)])
        self.assertEqual(set(keep.tolist()),{0,1,2})


if __name__=='__main__':
    unittest.main()
