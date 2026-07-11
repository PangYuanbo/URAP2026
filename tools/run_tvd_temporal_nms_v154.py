from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\aaron\Desktop\URAP')
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.run_tvd_frame_budget_v146 import VAL, TEST, load, budget
from tools.run_tvd_oof_stack_v130 import metrics

RUN = ROOT / 'artifacts' / 'detached_tvd_temporal_nms_v154'
OUT = Path(r'D:\URAP_vatd_rank_results\tvd_temporal_nms_v154')
TVD = Path(r'D:\urap_modal_stage\TransVisDrone')
VATD = 0.93844


def report(stage: str, done: int, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {'stage': stage, 'done': done, 'total': 3, 'updated': datetime.now().astimezone().isoformat(), **extra}
    (RUN / 'progress.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload), flush=True)


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area + areas - intersection, 1e-9)


def boxes_for(data, locations) -> np.ndarray:
    boxes = []
    for _sequence, _frame_id, index, image_id, _raw in locations:
        row = data[image_id]['detections'][index]
        boxes.append([float(value) for value in row['bbox']])
    return np.asarray(boxes, dtype=np.float64)


def suppress(score: np.ndarray, locations, boxes: np.ndarray, threshold: float, factor: float, minimum_score: float):
    output = score.copy()
    frames = {}
    for index, location in enumerate(locations):
        frames.setdefault(location[3], []).append(index)
    changed = 0
    for indices in frames.values():
        indices = np.asarray(indices, dtype=np.int64)
        order = indices[np.argsort(score[indices])[::-1]]
        kept = []
        for index in order:
            if score[index] < minimum_score:
                continue
            if kept and np.max(iou(boxes[index], boxes[np.asarray(kept)])) >= threshold:
                output[index] *= factor
                changed += 1
            else:
                kept.append(int(index))
    return output, changed


def base(split: str, source: Path):
    data = load_predictionsgt(source)
    correct, pred_cls, target_cls, locations, labels, memory = load(split, source)
    selected = json.loads(Path(r'D:\URAP_vatd_rank_results\tvd_frame_budget_v146\official_summary.json').read_text(encoding='utf-8'))['validation_selection']
    score, _ = budget(memory, locations, int(selected['top_k']), float(selected['suppression_factor']), float(selected['score_gate']))
    return correct, pred_cls, target_cls, locations, labels, score, boxes_for(data, locations)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report('select_validation', 0)
    correct, pred_cls, target_cls, locations, labels, score, boxes = base('val', VAL)
    rows = []
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for factor in (0.0, 0.01, 0.05, 0.1, 0.3, 0.6):
            for minimum_score in (0.0, 0.01, 0.05, 0.1):
                candidate, changed = suppress(score, locations, boxes, threshold, factor, minimum_score)
                rows.append({'iou_threshold': threshold, 'suppression_factor': factor, 'minimum_score': minimum_score, 'changed_rows': changed, **metrics(correct, candidate, pred_cls, target_cls, TVD)})
    best = max(rows, key=lambda row: float(row['map50']))
    (OUT / 'val_sweep.json').write_text(json.dumps({'best': best, 'top': sorted(rows, key=lambda row: -float(row['map50']))[:50], 'labels': labels}, indent=2), encoding='utf-8')
    report('fixed_test', 2, validation_selection=best)
    qcorrect, qpred_cls, qtarget_cls, qlocations, qlabels, qscore, qboxes = base('test', TEST)
    qcandidate, qchanged = suppress(qscore, qlocations, qboxes, float(best['iou_threshold']), float(best['suppression_factor']), float(best['minimum_score']))
    test = {**metrics(qcorrect, qcandidate, qpred_cls, qtarget_cls, TVD), 'labels': qlabels, 'detections': len(qlocations), 'changed_rows': qchanged}
    gain = 100.0 * (float(test['map50']) - VATD)
    summary = {'protocol': 'validation-selected spatial duplicate suppression after V146 temporal Action Memory; fixed test', 'validation_selection': best, 'test_fixed': test, 'vatd_map50': VATD, 'gain_over_vatd_points': gain, 'target_3_to_5_met': 3.0 <= gain <= 5.0}
    (OUT / 'official_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report('done', 3, summary=summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
