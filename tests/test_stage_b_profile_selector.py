from argparse import Namespace

from tools.select_stage_b_profile_outputs import select_rows


def _row(source="yolo_tile", predicted_class="background", score=0.1, drone=0.2, background=0.7, bbox=None, mode="normal"):
    return {
        "frame_id": 1,
        "bbox": bbox or [10.0, 10.0, 50.0, 50.0],
        "source": source,
        "predicted_class": predicted_class,
        "final_drone_score": score,
        "final_probs": {"drone": drone, "background": background},
        "mode": mode,
        "diagnostic_cause": None,
    }


def _args():
    return Namespace(
        hard_tiny_max_side=32.0,
        recall_min_score=0.18,
        recall_min_prob=0.55,
        recall_max_background=0.60,
        scene_recovery_allow_untracked=False,
        scene_min_track_history_len=2,
        scene_min_track_detector_updates=2,
        scene_max_frames_since_detector_update=1,
        scene_min_track_score=0.10,
        scene_min_track_evidence_len=0,
    )


def test_stage_b_selector_keeps_strict_for_plain_yolo_fp():
    recall = _row(predicted_class="drone", score=0.7, drone=0.8, background=0.1)
    strict = _row(predicted_class="background", score=0.1, drone=0.1, background=0.8)

    selected, counts = select_rows([recall], [strict], _args())

    assert selected[0]["predicted_class"] == "background"
    assert selected[0]["stage_b_profile_selected"] == "strict_fp_control"
    assert counts["strict_default"] == 1


def test_stage_b_selector_recovers_tracker_supported_drone():
    recall = _row(source="tracker+yolo_tile", predicted_class="drone", score=0.4, drone=0.7, background=0.2)
    strict = _row(source="tracker+yolo_tile", predicted_class="background", score=0.1, drone=0.2, background=0.7)

    selected, counts = select_rows([recall], [strict], _args())

    assert selected[0]["predicted_class"] == "drone"
    assert selected[0]["stage_b_profile_selected"] == "recall_oriented"
    assert selected[0]["stage_b_profile_selection_reason"] == "recall_track_supported_recovery"
    assert counts["recall_track_supported_recovery"] == 1


def test_stage_b_selector_keeps_untracked_scene_hard_tiny_strict_by_default():
    recall = _row(
        predicted_class="drone",
        score=0.22,
        drone=0.65,
        background=0.3,
        bbox=[10.0, 10.0, 30.0, 25.0],
        mode="bad_alignment_fast_egomotion",
    )
    strict = _row(
        predicted_class="background",
        score=0.05,
        drone=0.1,
        background=0.9,
        bbox=[10.0, 10.0, 30.0, 25.0],
        mode="bad_alignment_fast_egomotion",
    )

    selected, counts = select_rows([recall], [strict], _args())

    assert selected[0]["predicted_class"] == "background"
    assert selected[0]["stage_b_profile_selection_reason"] == "strict_default"
    assert counts["recall_scene_hard_tiny_recovery"] == 0


def test_stage_b_selector_recovers_persistent_scene_hard_tiny_drone():
    recall = _row(
        predicted_class="drone",
        score=0.22,
        drone=0.65,
        background=0.3,
        bbox=[10.0, 10.0, 30.0, 25.0],
        mode="bad_alignment_fast_egomotion",
    )
    recall.update(
        {
            "track_history_len": 4,
            "track_detector_updates": 3,
            "track_frames_since_detector_update": 0,
            "track_score": 0.35,
        }
    )
    strict = _row(
        predicted_class="background",
        score=0.05,
        drone=0.1,
        background=0.9,
        bbox=[10.0, 10.0, 30.0, 25.0],
        mode="bad_alignment_fast_egomotion",
    )

    selected, counts = select_rows([recall], [strict], _args())

    assert selected[0]["predicted_class"] == "drone"
    assert selected[0]["stage_b_profile_selection_reason"] == "recall_scene_hard_tiny_recovery"
    assert counts["recall_scene_hard_tiny_recovery"] == 1


def test_stage_b_selector_uses_scene_tracklet_gate_for_scene_recovery():
    recall = _row(
        predicted_class="drone",
        score=0.22,
        drone=0.65,
        background=0.3,
        bbox=[10.0, 10.0, 30.0, 25.0],
        mode="bad_alignment_fast_egomotion",
    )
    strict = _row(
        predicted_class="background",
        score=0.05,
        drone=0.1,
        background=0.9,
        bbox=[10.0, 10.0, 30.0, 25.0],
        mode="bad_alignment_fast_egomotion",
    )
    args = _args()
    args.scene_tracklet_gate_required = True
    key = (1, (10.0, 10.0, 30.0, 25.0))

    selected, counts = select_rows([recall], [strict], args, scene_gate_scores={key: {"scene_tracklet_gate_pass": True, "scene_tracklet_gate_prob": 0.9}})

    assert selected[0]["predicted_class"] == "drone"
    assert selected[0]["stage_b_profile_selection_reason"] == "recall_scene_hard_tiny_recovery"
    assert selected[0]["scene_tracklet_gate_prob"] == 0.9
    assert counts["recall_scene_hard_tiny_recovery"] == 1
