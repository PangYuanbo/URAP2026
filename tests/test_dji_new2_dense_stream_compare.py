from tools.run_dji_new2_dense_stream_compare import _safe_name


def test_safe_name_uses_parent_for_antiuav_visible_videos():
    a = _safe_name(r"D:\datasets\Anti-UAV300\train\visible\seq_a\visible.mp4")
    b = _safe_name(r"D:\datasets\Anti-UAV300\train\visible\seq_b\visible.mp4")

    assert a == "seq_a_visible"
    assert b == "seq_b_visible"
    assert a != b


def test_safe_name_keeps_dji_stem_behavior():
    name = _safe_name(r"D:\datasets\my_video\today_raw\dji_fly_20260527_121806_13_1779921757607_hdrvideo.MP4")

    assert name == "121806_13_1779921757607"
