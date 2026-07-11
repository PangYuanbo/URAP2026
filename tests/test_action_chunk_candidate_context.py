import unittest
import numpy as np
from tools.action_chunk_candidate_context import candidate_context_features

class CandidateContextTest(unittest.TestCase):
 def test_separate_targets_survive_nms(self):
  detections=[
   {'score':.9,'bbox':[0,0,20,20]},
   {'score':.8,'bbox':[1,1,21,21]},
   {'score':.7,'bbox':[100,100,120,120]},
  ]
  features=candidate_context_features(detections)
  self.assertEqual(features.shape,(3,23))
  self.assertEqual(features[0,17],1.)
  self.assertEqual(features[1,17],0.)
  self.assertEqual(features[2,17],1.)
  self.assertEqual(features[2,16],1.)
  self.assertTrue(np.isfinite(features).all())

 def test_empty(self):
  self.assertEqual(candidate_context_features([]).shape,(0,23))

if __name__=='__main__':unittest.main()
