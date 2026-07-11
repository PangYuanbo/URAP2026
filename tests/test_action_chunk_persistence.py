import unittest

from qstr_dronedet.tracking.action_chunk_bank import ActionChunkTrack
from tools.score_predictionsgt_action_chunk_bank import prune_tracks


class ActionChunkPersistenceTest(unittest.TestCase):
    def test_recent_track_beats_stale_equal_quality_track(self):
        stale = ActionChunkTrack(1, 0.0, (0.0, 0.0, 10.0, 10.0), 0.9)
        recent = ActionChunkTrack(20, 2.0, (20.0, 0.0, 30.0, 10.0), 0.9)
        kept = prune_tracks([stale, recent], 1, timestamp=2.5, long_seconds=3.0)
        self.assertEqual(kept[0].frame_id, 20)

    def test_boxes_from_different_frames_are_not_directly_nms_compared(self):
        first = ActionChunkTrack(1, 0.0, (0.0, 0.0, 10.0, 10.0), 0.9)
        second = ActionChunkTrack(2, 0.1, (0.0, 0.0, 10.0, 10.0), 0.8)
        kept = prune_tracks([first, second], 2, timestamp=0.1, long_seconds=3.0)
        self.assertEqual(len(kept), 2)

    def test_projected_boxes_deduplicate_cross_frame_hypotheses(self):
        first = ActionChunkTrack(1, 0.0, (0.0, 0.0, 10.0, 10.0), 0.9)
        second = ActionChunkTrack(2, 0.1, (20.0, 0.0, 30.0, 10.0), 0.8)
        projected = [(5.0, 0.0, 15.0, 10.0), (5.1, 0.0, 15.1, 10.0)]
        kept = prune_tracks([first, second], 2, timestamp=0.1, long_seconds=3.0, projected_boxes=projected)
        self.assertEqual(len(kept), 1)


if __name__ == '__main__':
    unittest.main()
