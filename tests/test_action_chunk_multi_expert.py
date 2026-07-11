import unittest
import numpy as np
from tools.train_action_chunk_multi_expert import multi_target_rows

class MultiExpertSamplingTest(unittest.TestCase):
 def test_only_multi_target_frames_are_selected(self):
  x=np.zeros((7,3),np.float32);x[:,0]=[.9,.8,.7,.9,.8,.7,.6];y=np.asarray([1,0,0,1,1,0,0],np.float32);keep,weights,groups=multi_target_rows(x,y,[(0,3),(3,7)])
  self.assertEqual(groups,1)
  self.assertTrue((keep>=3).all())
  self.assertEqual(int((y[keep]>=.5).sum()),2)
  self.assertTrue((weights[y[keep]>=.5]>1).all())

if __name__=='__main__':unittest.main()
