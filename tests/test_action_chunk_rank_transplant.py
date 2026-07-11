import unittest
from tools.sweep_action_chunk_rank_transplant import transplant_frame


class RankTransplantTest(unittest.TestCase):
    def test_preserves_frame_score_multiset(self):
        rows=[{'score':.9},{'score':.8},{'score':.2}]
        scores={('Clip_1',1,0):.1,('Clip_1',1,1):.9,('Clip_1',1,2):.2}
        output,changed=transplant_frame(rows,'Clip_1',1,scores,1.0,1.0)
        self.assertEqual(sorted(row['score'] for row in output),[.2,.8,.9])
        self.assertGreater(changed,0)
        self.assertEqual(output[1]['score'],.9)

    def test_band_leaves_low_scores_untouched(self):
        rows=[{'score':.9},{'score':.85},{'score':.2}]
        scores={('Clip_1',1,0):.1,('Clip_1',1,1):.9,('Clip_1',1,2):1.0}
        output,_=transplant_frame(rows,'Clip_1',1,scores,1.0,.1)
        self.assertEqual(output[2]['score'],.2)


if __name__=='__main__':
    unittest.main()
