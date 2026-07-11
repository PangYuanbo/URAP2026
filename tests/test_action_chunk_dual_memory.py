import unittest
import numpy as np
from tools.train_action_chunk_dual_memory import hard_rows

class DualMemorySamplingTest(unittest.TestCase):
 def test_hard_rows_keep_positive_and_competing_negative(self):
  x=np.zeros((3,4),np.float32);x[:,0]=[.9,.8,.1];y=np.asarray([1.,0.,0.],np.float32);keep=hard_rows(x,y,[(0,3)])
  self.assertIn(0,keep)
  self.assertIn(1,keep)

if __name__=='__main__':unittest.main()
