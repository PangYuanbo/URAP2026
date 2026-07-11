from pathlib import Path

from qstr_dronedet.action_chunk_camera_motion import action_chunk_frame_path


def test_action_chunk_frame_path_supports_jpg(tmp_path: Path) -> None:
    frame = tmp_path / "Clip_102_00001.jpg"
    frame.write_bytes(b"test")
    assert action_chunk_frame_path(tmp_path, "Clip_102", 1) == frame


def test_action_chunk_frame_path_prefers_png(tmp_path: Path) -> None:
    png = tmp_path / "Clip_1_00001.png"
    jpg = tmp_path / "Clip_1_00001.jpg"
    png.write_bytes(b"png")
    jpg.write_bytes(b"jpg")
    assert action_chunk_frame_path(tmp_path, "Clip_1", 1) == png
