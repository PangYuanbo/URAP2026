from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\aaron\Desktop\URAP')
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from tools.run_tvd_frame_budget_v146 import VAL, TEST, load, budget
from tools.run_tvd_oof_stack_v130 import metrics, sigmoid, logit
from tools.sweep_tvd_predictionsgt_score_fusion import fuse_score, load_row_scores

RUN = ROOT / 'artifacts' / 'detached_tvd_v146_hard_blend_v152'
OUT = Path(r'D:\URAP_vatd_rank_results\tvd_v146_hard_blend_v152')
TVD = Path(r'D:\urap_modal_stage\TransVisDrone')
HARD = Path(r'D:\URAP_vatd_rank_results\tvd_hard_domain_action_v142')
VATD = 0.93844
FIELD = 'tvd_hard_domain_action_score'


def report(stage: str, done: int, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {'stage': stage, 'done': done, 'total': 3, 'updated': datetime.now().astimezone().isoformat(), **extra}
    (RUN / 'progress.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload), flush=True)


def hard_fused(locations, split: str) -> np.ndarray:
    score_path = HARD / ('val_scores.jsonl' if split == 'val' else 'test_scores.jsonl')
    score_map, _ = load_row_scores(score_path, FIELD, 1)
    config = json.loads((HARD / 'official_summary.json').read_text(encoding='utf-8'))['validation_selection']
    return np.asarray([
        fuse_score(float(raw), float(score_map.get((sequence, frame_id, index), raw)), float(config['alpha']), str(config['mode']))
        for sequence, frame_id, index, _image_id, raw in locations
    ])


def mix(primary: np.ndarray, auxiliary: np.ndarray, weight: float, mode: str) -> np.ndarray:
    if mode == 'linear':
        return np.clip((1.0 - weight) * primary + weight * auxiliary, 0.0, 1.0)
    if mode == 'logit':
        return sigmoid((1.0 - weight) * logit(primary) + weight * logit(auxiliary))
    if mode == 'max':
        return np.maximum(primary, weight * auxiliary + (1.0 - weight) * primary)
    raise ValueError(mode)


def v146(split: str, source: Path):
    correct, pred_cls, target_cls, locations, labels, base = load(split, source)
    selected = json.loads(Path(r'D:\URAP_vatd_rank_results\tvd_frame_budget_v146\official_summary.json').read_text(encoding='utf-8'))['validation_selection']
    score, changed = budget(base, locations, int(selected['top_k']), float(selected['suppression_factor']), float(selected['score_gate']))
    return correct, pred_cls, target_cls, locations, labels, score, changed


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report('select_validation', 0)
    correct, pred_cls, target_cls, locations, labels, primary, changed = v146('val', VAL)
    auxiliary = hard_fused(locations, 'val')
    rows = []
    for mode in ('linear', 'logit', 'max'):
        for weight in (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
            rows.append({'mode': mode, 'hard_weight': weight, **metrics(correct, mix(primary, auxiliary, weight, mode), pred_cls, target_cls, TVD)})
    best = max(rows, key=lambda row: float(row['map50']))
    (OUT / 'val_sweep.json').write_text(json.dumps({'best': best, 'rows': rows, 'labels': labels, 'v146_changed': changed}, indent=2), encoding='utf-8')
    report('fixed_test', 2, validation_selection=best)
    qcorrect, qpred_cls, qtarget_cls, qlocations, qlabels, qprimary, qchanged = v146('test', TEST)
    qauxiliary = hard_fused(qlocations, 'test')
    test = {**metrics(qcorrect, mix(qprimary, qauxiliary, float(best['hard_weight']), str(best['mode'])), qpred_cls, qtarget_cls, TVD), 'labels': qlabels, 'detections': len(qlocations), 'v146_changed': qchanged}
    gain = 100.0 * (float(test['map50']) - VATD)
    summary = {'protocol': 'validation-selected blend of V146 frame-budget Action Memory and train-only hard-domain Action Bank; fixed test', 'validation_selection': best, 'test_fixed': test, 'vatd_map50': VATD, 'gain_over_vatd_points': gain, 'target_3_to_5_met': 3.0 <= gain <= 5.0}
    (OUT / 'official_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report('done', 3, summary=summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
