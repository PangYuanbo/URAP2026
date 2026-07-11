import unittest

import numpy as np

from tools.train_action_bank_all_candidate_listwise import greedy_match_qualities


class CandidateMatchingTest(unittest.TestCase):
    def test_duplicate_candidates_only_reward_best_match(self) -> None:
        candidates = [[0, 0, 10, 10], [1, 1, 9, 9]]
        gt = np.asarray([[0, 0, 10, 10]], dtype=np.float32)

        qualities = greedy_match_qualities(candidates, gt)

        self.assertEqual(int((qualities >= 0.5).sum()), 1)
        self.assertAlmostEqual(float(qualities[0]), 1.0)
        self.assertEqual(float(qualities[1]), 0.0)

    def test_multiple_gt_receive_distinct_candidates(self) -> None:
        candidates = [[0, 0, 10, 10], [20, 20, 30, 30], [1, 1, 9, 9]]
        gt = np.asarray([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)

        qualities = greedy_match_qualities(candidates, gt)

        self.assertEqual(int((qualities >= 0.5).sum()), 2)
        self.assertAlmostEqual(float(qualities[0]), 1.0)
        self.assertAlmostEqual(float(qualities[1]), 1.0)
        self.assertEqual(float(qualities[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
