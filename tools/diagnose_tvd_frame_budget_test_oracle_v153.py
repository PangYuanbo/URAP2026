from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r'C:\Users\aaron\Desktop\URAP')
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from tools.run_tvd_frame_budget_v146 import TEST, load, budget
from tools.run_tvd_oof_stack_v130 import metrics

OUT = Path(r'D:\URAP_vatd_rank_results\tvd_frame_budget_oracle_v153')
TVD = Path(r'D:\urap_modal_stage\TransVisDrone')
VATD = 0.93844


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    correct, pred_cls, target_cls, locations, labels, base = load('test', TEST)
    rows = []
    for top_k in (1, 2, 3, 4, 5, 8, 10, 15):
        for factor in (0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8):
            for gate in (0.05, 0.1, 0.2, 0.4, 1.0):
                score, changed = budget(base, locations, top_k, factor, gate)
                rows.append({'top_k': top_k, 'suppression_factor': factor, 'score_gate': gate, 'changed_rows': changed, **metrics(correct, score, pred_cls, target_cls, TVD)})
    best = max(rows, key=lambda row: float(row['map50']))
    official = next(row for row in rows if row['top_k'] == 4 and abs(row['suppression_factor'] - 0.6) < 1e-12 and abs(row['score_gate'] - 0.1) < 1e-12)
    result = {
        'warning': 'test-label diagnostic oracle only; not a valid selected result',
        'best_test_oracle': best,
        'official_v146_config': official,
        'oracle_gain_over_vatd_points': 100.0 * (float(best['map50']) - VATD),
        'family_can_reach_target': float(best['map50']) >= VATD + 0.03,
        'labels': labels,
        'rows': rows,
    }
    (OUT / 'diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
