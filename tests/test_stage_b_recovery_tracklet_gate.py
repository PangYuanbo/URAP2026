from argparse import Namespace

from tools.train_stage_b_recovery_tracklet_gate import _sample_row


def _args(sample_mode="gt_suppressed_candidate"):
    return Namespace(
        sample_mode=sample_mode,
        hard_tiny_max_side=48.0,
        recall_min_score=0.18,
        iou_threshold=0.3,
        center_threshold=16.0,
    )


def test_gt_suppressed_candidate_keeps_gt_hit_even_when_recall_background():
    row = {
        "frame_id": 10,
        "bbox": [100.0, 100.0, 108.0, 108.0],
        "predicted_class": "background",
        "final_drone_score": 0.02,
    }
    strict = {"frame_id": 10, "bbox": [100.0, 100.0, 108.0, 108.0], "predicted_class": "background"}
    gt_by_frame = {10: [{"bbox": [100.0, 100.0, 108.0, 108.0]}]}

    assert _sample_row(row, strict, gt_by_frame, _args(), _args())


def test_gt_suppressed_candidate_keeps_unmatched_high_score_drone_as_negative():
    row = {
        "frame_id": 10,
        "bbox": [100.0, 100.0, 108.0, 108.0],
        "predicted_class": "drone",
        "final_drone_score": 0.3,
    }
    strict = {"frame_id": 10, "bbox": [100.0, 100.0, 108.0, 108.0], "predicted_class": "background"}

    assert _sample_row(row, strict, {}, _args(), _args())


def test_gt_suppressed_candidate_drops_candidate_already_kept_by_strict():
    row = {
        "frame_id": 10,
        "bbox": [100.0, 100.0, 108.0, 108.0],
        "predicted_class": "background",
        "final_drone_score": 0.02,
    }
    strict = {"frame_id": 10, "bbox": [100.0, 100.0, 108.0, 108.0], "predicted_class": "drone"}
    gt_by_frame = {10: [{"bbox": [100.0, 100.0, 108.0, 108.0]}]}

    assert not _sample_row(row, strict, gt_by_frame, _args(), _args())
