import unittest

import numpy as np

from qstr_dronedet.tracking.online_action_bank import OnlineActionTrack, OnlineCandidateScore
from tools.score_predictionsgt_action_chunk_bank import choose_updates


class OnlineActionBankAssignmentTest(unittest.TestCase):
    def test_tracks_cannot_share_candidate(self) -> None:
        tracks = [
            OnlineActionTrack(1, 0.0, (0.0, 0.0, 10.0, 10.0), 0.9),
            OnlineActionTrack(1, 0.0, (20.0, 0.0, 30.0, 10.0), 0.9),
        ]
        detections = [
            {"bbox": [1.0, 0.0, 11.0, 10.0], "score": 0.95},
            {"bbox": [21.0, 0.0, 31.0, 10.0], "score": 0.85},
        ]
        strong = OnlineCandidateScore(0.95, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9)
        medium = OnlineCandidateScore(0.80, 0.7, 0.7, 0.7, 0.7, 0.8, 0.1, 0.7, 0.8)
        weak = OnlineCandidateScore(0.10, 0.0, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8)
        candidate_scores = [[strong, medium], [weak, strong]]
        transforms = [np.eye(3), np.eye(3)]

        updates = choose_updates(tracks, detections, candidate_scores, transforms, 2, 0.1, 2.5, 0.08)

        self.assertEqual(len(updates), 2)
        self.assertEqual(len({tuple(track.bbox) for track in updates}), 2)

        updates_with_ids, assigned_tracks, assigned_candidates = choose_updates(
            tracks, detections, candidate_scores, transforms, 2, 0.1, 2.5, 0.08, return_assignments=True
        )
        self.assertEqual(len(updates_with_ids), 2)
        self.assertEqual(assigned_tracks, {0, 1})
        self.assertEqual(assigned_candidates, {0, 1})


if __name__ == "__main__":
    unittest.main()
