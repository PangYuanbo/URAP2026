import unittest
from tools.sweep_action_chunk_temporal_multiplicity import temporal_gate_map

class TemporalMultiplicityGateTest(unittest.TestCase):
 def test_gate_uses_past_and_current_only(self):
  data={f'Clip_1_{i:05d}':{'detections':([{'score':.9,'bbox':[0,0,10,10]},{'score':.8,'bbox':[100,100,110,110]}] if i in (2,3) else [{'score':.9,'bbox':[0,0,10,10]}])} for i in range(1,5)}
  gates=temporal_gate_map(data,.4,1.,.5,{'Clip_1':2})
  self.assertFalse(gates['Clip_1_00001'])
  self.assertTrue(gates['Clip_1_00002'])
  self.assertTrue(gates['Clip_1_00003'])
  self.assertTrue(gates['Clip_1_00004'])

if __name__=='__main__':unittest.main()
