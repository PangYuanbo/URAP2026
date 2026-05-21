from __future__ import annotations

from qstr_dronedet.candidates.merge import bbox_iou
from qstr_dronedet.types import DetectionCandidate, RecognitionResult


def classify_error(gt_objects: list[dict], candidates: list[DetectionCandidate], recognition_results: list[RecognitionResult]) -> str:
    if not gt_objects:
        if any(r.predicted_class == "drone" for r in recognition_results):
            return "false_positive_drone"
        return "correct"
    for gt in gt_objects:
        gt_box = tuple(gt["bbox_xyxy"])
        overlaps = [i for i, c in enumerate(candidates) if bbox_iou(c.bbox_xyxy, gt_box) >= 0.3]
        if not overlaps:
            return "no_candidate_for_gt"
        best = overlaps[0]
        rec = recognition_results[best] if best < len(recognition_results) else None
        gt_cls = gt.get("class", "drone")
        if rec is None:
            return "candidate_exists_but_wrong_class"
        if rec.final_probs.get("unknown", 0.0) > 0.45:
            return "high_unknown"
        if rec.crop_probs.get(gt_cls, 0.0) > 0.5 and rec.feature_probs.get(gt_cls, 0.0) < 0.25:
            return "crop_correct_feature_wrong"
        if rec.feature_probs.get(gt_cls, 0.0) > 0.5 and rec.crop_probs.get(gt_cls, 0.0) < 0.25:
            return "feature_correct_crop_wrong"
        if rec.temporal_probs.get(gt_cls, 0.0) > 0.5 and rec.crop_probs.get(gt_cls, 0.0) < 0.25:
            return "temporal_correct_single_frame_wrong"
        if rec.predicted_class != gt_cls:
            return "candidate_exists_but_wrong_class"
    return "correct"

