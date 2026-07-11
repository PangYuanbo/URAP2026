from qstr_dronedet.candidates.merge import bbox_iou, merge_candidates, nms_candidates
from qstr_dronedet.cli import (
    _filter_candidates_by_box_size,
    _limit_merged_candidates,
    _should_run_fallback_after_recognition,
    _should_run_fallback_yolo,
)
from qstr_dronedet.types import DetectionCandidate, RecognitionResult


def test_iou_and_nms():
    a = DetectionCandidate((0, 0, 10, 10), 0.9, "a")
    b = DetectionCandidate((1, 1, 11, 11), 0.5, "b")
    c = DetectionCandidate((50, 50, 60, 60), 0.8, "c")
    assert bbox_iou(a.bbox_xyxy, b.bbox_xyxy) > 0.5
    assert len(nms_candidates([a, b, c], 0.5)) == 2


def test_merge_keeps_sources():
    a = DetectionCandidate((0, 0, 10, 10), 0.9, "motion")
    b = DetectionCandidate((2, 2, 12, 12), 0.7, "tracker")
    merged = merge_candidates([a, b])
    assert len(merged) == 1
    assert "motion" in merged[0].source and "tracker" in merged[0].source


def test_merge_preserves_best_detector_evidence_when_support_scores_higher():
    support = DetectionCandidate((0, 0, 10, 10), 0.9, "gray_ncc", extra={"raw_objectness": 0.34})
    detector = DetectionCandidate((1, 1, 11, 11), 0.4, "yolov5_dual", extra={"raw_objectness": 0.82})

    merged = merge_candidates([support, detector])

    assert len(merged) == 1
    assert merged[0].extra["has_detector_member"] is True
    assert merged[0].extra["detector_raw_objectness"] == 0.82
    assert merged[0].extra["detector_bbox_xyxy"] == [1.0, 1.0, 11.0, 11.0]
    assert merged[0].extra["detector_source"] == "yolov5_dual"


def test_fallback_trigger_and_source_priority_budget():
    weak_primary = [DetectionCandidate((0, 0, 4, 4), 0.12, "yolo_tile")]
    strong_primary = [DetectionCandidate((0, 0, 4, 4), 0.35, "yolo_tile")]
    assert _should_run_fallback_yolo([], min_primary_candidates=1, trigger_objectness=0.2)
    assert _should_run_fallback_yolo(weak_primary, min_primary_candidates=1, trigger_objectness=0.2)
    assert not _should_run_fallback_yolo(strong_primary, min_primary_candidates=1, trigger_objectness=0.2)

    cands = [
        DetectionCandidate((0, 0, 4, 4), 0.99, "motion"),
        DetectionCandidate((10, 10, 14, 14), 0.20, "track"),
        DetectionCandidate((20, 20, 24, 24), 0.30, "yolo_tile_fallback"),
    ]
    kept = _limit_merged_candidates(cands, 2)
    assert [c.source for c in kept] == ["track", "yolo_tile_fallback"]


def test_post_fusion_fallback_trigger():
    low = RecognitionResult({}, {}, {}, {"drone": 0.1}, 0.0, "unknown", 0.12, None)
    high = RecognitionResult({}, {}, {}, {"drone": 0.9}, 0.0, "drone", 0.55, None)
    assert not _should_run_fallback_after_recognition([low], trigger_final_score=0.0)
    assert _should_run_fallback_after_recognition([low], trigger_final_score=0.2)
    assert not _should_run_fallback_after_recognition([high], trigger_final_score=0.2)


def test_post_fusion_fallback_can_require_weak_primary_objectness():
    low = RecognitionResult({}, {}, {}, {"drone": 0.1}, 0.0, "unknown", 0.12, None)
    strong_primary = [DetectionCandidate((0, 0, 10, 10), 0.8, "yolo_tile")]
    weak_primary = [DetectionCandidate((0, 0, 10, 10), 0.2, "yolo_tile")]
    assert not _should_run_fallback_after_recognition(
        [low],
        trigger_final_score=0.5,
        primary_candidates=strong_primary,
        max_primary_objectness=0.35,
    )
    assert _should_run_fallback_after_recognition(
        [low],
        trigger_final_score=0.5,
        primary_candidates=weak_primary,
        max_primary_objectness=0.35,
    )


def test_fallback_box_size_filter_keeps_tiny_proposals():
    small = DetectionCandidate((10, 10, 50, 45), 0.6, "yolo_tile_fallback")
    large = DetectionCandidate((10, 10, 220, 170), 0.8, "yolo_tile_fallback")
    kept = _filter_candidates_by_box_size([small, large], max_box_side=128)
    assert kept == [small]
