from argparse import Namespace

from tools.audit_dji_scene_recovery_data import audit


def test_dji_scene_recovery_data_audit_roles(tmp_path):
    ann = tmp_path / "annotations"
    ann.mkdir()
    (ann / "boxes_8col.csv").write_text(
        "\n".join(
            [
                "video_path,frame_id,x1,y1,x2,y2,class,tag",
                r"D:\datasets\my_video\today_raw\dji_fly_20260527_121932_14_clip.MP4,1,1,1,2,2,drone,train",
                r"D:\datasets\my_video\today_raw\dji_fly_20260527_121932_14_clip.MP4,2,1,1,2,2,drone,train",
                r"D:\datasets\my_video\today_raw\dji_fly_20260527_121806_13_clip.MP4,1,1,1,2,2,drone,heldout",
                r"D:\datasets\my_video\dji_fly_20260522_clip.MP4,1,1,1,2,2,drone,calibration",
                r"D:\datasets\my_video\today_raw\dji_fly_20260531_new_clip.MP4,1,1,1,2,2,drone,new",
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "reports"
    args = Namespace(
        annotations_dir=str(ann),
        glob="*8col.csv",
        out=str(out),
        train_patterns=["121932", "122540"],
        calibration_patterns=["20260522"],
        heldout_patterns=["121806"],
        min_train_frames=3,
        min_calibration_frames=2,
    )

    result = audit(args)

    assert result["role_counts"]["train"] == 1
    assert result["role_counts"]["heldout"] == 1
    assert result["role_counts"]["calibration"] == 1
    assert result["role_counts"]["candidate_new_train_or_calibration"] == 1
    assert result["unique_annotated_frames_by_role"]["train"] == 2
    assert (out / "dji_scene_recovery_data_audit.json").exists()
