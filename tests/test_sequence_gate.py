from qstr_dronedet.tracking.sequence_gate import SequenceGateConfig, sequence_consistency_gate_rows, sequence_tracklet_features


def _row(frame_id, bbox, crop, temporal, background, score=0.3, predicted="drone"):
    return {
        "seq": "seq001",
        "frame_id": frame_id,
        "bbox": bbox,
        "objectness": 0.8,
        "source": "yolo_tile",
        "predicted_class": predicted,
        "final_drone_score": score,
        "crop_probs": {"drone": crop, "background": background},
        "temporal_probs": {"drone": temporal, "background": background},
        "feature_probs": {"drone": 0.1, "background": background},
        "final_probs": {"drone": max(crop, temporal), "background": background},
    }


def test_sequence_gate_rejects_single_frame_temporal_fp():
    row = _row(10, [100, 100, 110, 110], crop=0.05, temporal=0.8, background=0.75)

    filtered_pred, _, summary = sequence_consistency_gate_rows([row], [row], SequenceGateConfig())

    assert summary["raw_drone_predictions"] == 1
    assert summary["filtered_drone_predictions"] == 0
    assert filtered_pred[0]["predicted_class"] == "background"
    assert "sequence_gate_rejected" in filtered_pred[0]["diagnostic_cause"]


def test_sequence_gate_keeps_stable_two_frame_tracklet():
    rows = [
        _row(10, [100, 100, 110, 110], crop=0.55, temporal=0.65, background=0.25),
        _row(11, [102, 101, 112, 111], crop=0.58, temporal=0.68, background=0.24),
    ]

    filtered_pred, _, summary = sequence_consistency_gate_rows(rows, rows, SequenceGateConfig())

    assert summary["raw_drone_predictions"] == 2
    assert summary["filtered_drone_predictions"] == 2
    assert all(r["predicted_class"] == "drone" for r in filtered_pred)
    assert all(r["sequence_gate_confirmed"] for r in filtered_pred)


def test_sequence_tracklet_features_include_persistence_and_contradiction_terms():
    rows = [
        _row(10, [100, 100, 110, 110], crop=0.52, temporal=0.75, background=0.70, score=0.22),
        _row(11, [101, 100, 111, 110], crop=0.50, temporal=0.77, background=0.72, score=0.24),
        _row(12, [102, 101, 112, 111], crop=0.51, temporal=0.76, background=0.73, score=0.23),
    ]

    features = sequence_tracklet_features(rows, SequenceGateConfig(candidate_min_score=0.0))

    assert features["detector_support_count"] == 3
    assert features["longest_detector_streak"] == 3
    assert features["detector_persistence"] == 1.0
    assert features["longest_objectness_streak"] == 3
    assert features["high_background_rate"] == 1.0
    assert features["detector_high_background_drone_rate"] == 1.0
    assert features["longest_detector_high_background_streak"] == 3
    assert features["detector_high_background_persistence"] == 1.0
    assert features["longest_detector_high_background_drone_streak"] == 3
    assert features["detector_high_background_drone_persistence"] == 1.0
    assert abs(features["mean_detector_objectness"] - 0.8) < 1e-6
    assert features["mean_center_step_per_side"] < 0.2
