import numpy as np

from qstr_dronedet.tracking.action_chunk_bank import ActionChunkTrack


def test_clone_and_update_preserve_track_identity_and_birth_time():
    track = ActionChunkTrack(
        1,
        0.04,
        (0.0, 0.0, 10.0, 10.0),
        0.9,
        track_id=17,
        born_timestamp=0.04,
    )
    clone = track.clone()
    clone.update(
        2,
        0.08,
        (1.0, 0.0, 11.0, 10.0),
        0.8,
        0.9,
        np.eye(3),
    )
    assert clone.track_id == 17
    assert clone.born_timestamp == 0.04
    assert clone.observations == 2
    assert track.observations == 1


def test_birth_time_defaults_to_first_timestamp():
    track = ActionChunkTrack(3, 1.25, (0.0, 0.0, 4.0, 4.0), 0.5)
    assert track.born_timestamp == 1.25
