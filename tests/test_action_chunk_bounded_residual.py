import math
import unittest

from tools.sweep_action_chunk_bounded_residual import bounded_aux


class BoundedResidualTest(unittest.TestCase):
    def test_never_exceeds_logit_cap(self):
        base=0.8
        corrected=bounded_aux(base,0.001,'symmetric',0.25,1.0)
        delta=abs(math.log(corrected/(1-corrected))-math.log(base/(1-base)))
        self.assertLessEqual(delta,0.250001)

    def test_boost_only_never_suppresses_base(self):
        self.assertGreaterEqual(bounded_aux(0.8,0.01,'boost-only',1.0,1.0),0.8)
        self.assertGreater(bounded_aux(0.2,0.9,'boost-only',1.0,1.0),0.2)


if __name__=='__main__':
    unittest.main()
