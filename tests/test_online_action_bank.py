from __future__ import annotations

import unittest

import numpy as np

from qstr_dronedet.tracking.online_action_bank import OnlineActionTrack


class OnlineActionBankTests(unittest.TestCase):
    def test_prefers_motion_consistent_candidate(self) -> None:
        identity = np.eye(3, dtype=np.float64)
        track = OnlineActionTrack(frame_id=0, timestamp=0.0, bbox=(0.0, 0.0, 10.0, 10.0), quality=0.9)
        track.update(1, 0.1, (2.0, 0.0, 12.0, 10.0), 0.9, 0.9, identity)
        consistent = track.score_candidate((4.0, 0.0, 14.0, 10.0), 0.2, identity, 1.0)
        jump = track.score_candidate((50.0, 30.0, 60.0, 40.0), 0.2, identity, 1.0)
        self.assertGreater(consistent.score, jump.score)
        self.assertGreater(consistent.predicted_iou, 0.9)
        self.assertGreater(consistent.acceleration_similarity, jump.acceleration_similarity)

    def test_uses_elapsed_time(self) -> None:
        identity = np.eye(3, dtype=np.float64)
        track = OnlineActionTrack(frame_id=0, timestamp=0.0, bbox=(0.0, 0.0, 10.0, 10.0), quality=1.0)
        track.update(1, 0.5, (5.0, 0.0, 15.0, 10.0), 1.0, 1.0, identity)
        prediction = track.predict(1.5, identity, short_seconds=1.0, long_seconds=3.0)
        self.assertTrue(np.allclose(prediction, (15.0, 0.0, 25.0, 10.0), atol=1e-5))

    def test_acceleration_changes_prediction_without_unbounded_jump(self) -> None:
        identity = np.eye(3, dtype=np.float64)
        track = OnlineActionTrack(frame_id=0, timestamp=0.0, bbox=(0.0, 0.0, 10.0, 10.0), quality=1.0)
        track.update(1, 0.5, (2.0, 0.0, 12.0, 10.0), 1.0, 1.0, identity)
        track.update(2, 1.0, (6.0, 0.0, 16.0, 10.0), 1.0, 1.0, identity)
        prediction = track.predict(2.0, identity, short_seconds=1.0, long_seconds=3.0)
        self.assertGreater(prediction[0], 14.0)
        self.assertLess(prediction[0], 36.1)
        self.assertTrue(track.accelerations_x)


    def test_time_binned_tokens_are_causal_and_candidate_specific(self) -> None:
        identity = np.eye(3, dtype=np.float64)
        track = OnlineActionTrack(frame_id=0, timestamp=0.0, bbox=(0.0, 0.0, 10.0, 10.0), quality=1.0)
        track.update(1, 0.25, (2.5, 0.0, 12.5, 10.0), 1.0, 1.0, identity)
        track.update(2, 0.50, (5.0, 0.0, 15.0, 10.0), 1.0, 1.0, identity)
        consistent = track.candidate_action_tokens((7.5, 0.0, 17.5, 10.0), 0.75, identity, 1.0, 8)
        jump = track.candidate_action_tokens((40.0, 20.0, 50.0, 30.0), 0.75, identity, 1.0, 8)
        self.assertEqual(len(consistent), 16)
        self.assertTrue(any(consistent[index] == 1.0 for index in range(0, len(consistent), 2)))
        self.assertGreater(sum(consistent[1::2]), sum(jump[1::2]))

    def test_motion_tokens_expose_real_time_candidate_dynamics(self) -> None:
        identity = np.eye(3, dtype=np.float64)
        track = OnlineActionTrack(frame_id=0, timestamp=0.0, bbox=(0.0, 0.0, 10.0, 10.0), quality=1.0)
        track.update(1, 0.25, (2.5, 0.0, 12.5, 10.0), 1.0, 1.0, identity)
        track.update(2, 0.50, (5.0, 0.0, 15.0, 10.0), 1.0, 1.0, identity)
        consistent = track.candidate_motion_tokens((7.5, 0.0, 17.5, 10.0), 0.75, identity, 1.0, 8, 0.9)
        jump = track.candidate_motion_tokens((40.0, 20.0, 50.0, 30.0), 0.75, identity, 1.0, 8, 0.9)
        self.assertEqual(len(consistent), 8 * 12)
        self.assertTrue(any(consistent[index] == 1.0 for index in range(0, len(consistent), 12)))
        self.assertGreater(sum(consistent[11::12]), sum(jump[11::12]))
        self.assertGreater(sum(consistent[9::12]), sum(jump[9::12]))

    def test_equivalent_motion_at_25_and_75_fps(self) -> None:
        identity = np.eye(3, dtype=np.float64)

        def build(fps: int) -> OnlineActionTrack:
            track = OnlineActionTrack(frame_id=0, timestamp=0.0, bbox=(0.0, 0.0, 10.0, 10.0), quality=1.0)
            for frame_id in range(1, fps + 1):
                timestamp = frame_id / fps
                offset = 10.0 * timestamp
                track.update(frame_id, timestamp, (offset, 0.0, offset + 10.0, 10.0), 1.0, 1.0, identity)
            return track

        track_25 = build(25)
        track_75 = build(75)
        prediction_25 = track_25.predict(1.2, identity, short_seconds=1.0, long_seconds=3.0)
        prediction_75 = track_75.predict(1.2, identity, short_seconds=1.0, long_seconds=3.0)
        self.assertTrue(np.allclose(prediction_25, prediction_75, atol=1e-3))
        self.assertTrue(np.allclose(prediction_25, (12.0, 0.0, 22.0, 10.0), atol=.15))


if __name__ == "__main__":
    unittest.main()
