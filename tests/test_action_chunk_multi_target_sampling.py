import unittest
import numpy as np
from tools.train_action_chunk_multi_target import selected_rows

class MultiTargetSamplingTests(unittest.TestCase):
    def test_empty_frame_cap_and_multi_positive_weight(self):
        x=np.zeros((12,8),np.float32);x[:,0]=np.linspace(.1,1.,12);y=np.zeros(12,np.float32);y[6:9]=1.;keep,weights=selected_rows(x,y,[(0,6),(6,12)])
        self.assertEqual(sum(index<6 for index in keep),2)
        positive_weights=weights[np.isin(keep,[6,7,8])]
        self.assertTrue(np.all(positive_weights>1.))

if __name__=='__main__':unittest.main()
