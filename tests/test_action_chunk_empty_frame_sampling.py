import unittest
import numpy as np
from tools.train_action_chunk_neighbor_full import hard_rows as classifier_rows
from tools.train_action_chunk_neighbor_regression import hard_rows as regression_rows

class EmptyFrameSamplingTests(unittest.TestCase):
    def test_empty_target_frame_contributes_hard_negatives(self):
        x=np.zeros((10,8),np.float32);x[:,0]=np.linspace(.1,1.,10);y=np.zeros(10,np.float32);groups=[(0,10)]
        for function in (classifier_rows,regression_rows):
            keep=function(x,y,groups)
            self.assertEqual(len(keep),6)
            self.assertIn(9,keep)

if __name__=='__main__':unittest.main()
