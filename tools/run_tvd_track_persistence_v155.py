from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\aaron\Desktop\URAP')
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from tools.run_tvd_frame_budget_v146 import VAL, TEST, load, budget
from tools.run_tvd_oof_stack_v130 import metrics
from tools.run_tvd_track_memory_v144 import TRACKS, load_track_indices

RUN = ROOT / 'artifacts' / 'detached_tvd_track_persistence_v155'
OUT = Path(r'D:\URAP_vatd_rank_results\tvd_track_persistence_v155')
TVD = Path(r'D:\urap_modal_stage\TransVisDrone')
VATD = 0.93844


def report(stage: str, done: int, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {'stage': stage, 'done': done, 'total': 3, 'updated': datetime.now().astimezone().isoformat(), **extra}
    (RUN / 'progress.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload), flush=True)


def base(split: str, source: Path):
    correct, pred_cls, target_cls, locations, labels, score = load(split, source)
    selected = json.loads(Path(r'D:\URAP_vatd_rank_results\tvd_frame_budget_v146\official_summary.json').read_text(encoding='utf-8'))['validation_selection']
    score, _ = budget(score, locations, int(selected['top_k']), float(selected['suppression_factor']), float(selected['score_gate']))
    lookup = {(sequence, frame_id, index): row for row, (sequence, frame_id, index, _image_id, _raw) in enumerate(locations)}
    tracks, mapped, source_rows = load_track_indices(TRACKS[split], lookup)
    track_length = np.ones(len(locations), dtype=np.int32)
    track_span = np.ones(len(locations), dtype=np.int32)
    for track in tracks:
        frame_ids = np.asarray([item[0] for item in track], dtype=np.int64)
        row_ids = np.asarray([item[1] for item in track], dtype=np.int64)
        length = len(row_ids)
        span = int(frame_ids.max() - frame_ids.min() + 1)
        track_length[row_ids] = np.maximum(track_length[row_ids], length)
        track_span[row_ids] = np.maximum(track_span[row_ids], span)
    return correct, pred_cls, target_cls, locations, labels, score, track_length, track_span, len(tracks), mapped, source_rows


def suppress_short_tracks(score: np.ndarray, track_length: np.ndarray, minimum_rows: int, factor: float, score_gate: float):
    output = score.copy()
    mask = (track_length < minimum_rows) & (score < score_gate)
    output[mask] *= factor
    return output, int(mask.sum())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report('select_validation', 0)
    correct, pred_cls, target_cls, locations, labels, score, lengths, spans, tracks, mapped, source_rows = base('val', VAL)
    rows = []
    for minimum_rows in (2, 3, 5, 8, 12, 16, 20):
        for factor in (0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8):
            for score_gate in (0.01, 0.03, 0.05, 0.1, 0.2, 1.0):
                candidate, changed = suppress_short_tracks(score, lengths, minimum_rows, factor, score_gate)
                rows.append({'minimum_track_rows': minimum_rows, 'suppression_factor': factor, 'score_gate': score_gate, 'changed_rows': changed, **metrics(correct, candidate, pred_cls, target_cls, TVD)})
    best = max(rows, key=lambda row: float(row['map50']))
    (OUT / 'val_sweep.json').write_text(json.dumps({'best': best, 'top': sorted(rows, key=lambda row: -float(row['map50']))[:50], 'labels': labels, 'tracks': tracks, 'mapped': mapped, 'source_rows': source_rows}, indent=2), encoding='utf-8')
    report('fixed_test', 2, validation_selection=best)
    qcorrect, qpred_cls, qtarget_cls, qlocations, qlabels, qscore, qlengths, qspans, qtracks, qmapped, qsource_rows = base('test', TEST)
    qcandidate, qchanged = suppress_short_tracks(qscore, qlengths, int(best['minimum_track_rows']), float(best['suppression_factor']), float(best['score_gate']))
    test = {**metrics(qcorrect, qcandidate, qpred_cls, qtarget_cls, TVD), 'labels': qlabels, 'detections': len(qlocations), 'tracks': qtracks, 'mapped': qmapped, 'source_rows': qsource_rows, 'changed_rows': qchanged}
    gain = 100.0 * (float(test['map50']) - VATD)
    summary = {'protocol': 'validation-selected short-track suppression after V146; fixed test', 'validation_selection': best, 'test_fixed': test, 'vatd_map50': VATD, 'gain_over_vatd_points': gain, 'target_3_to_5_met': 3.0 <= gain <= 5.0}
    (OUT / 'official_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report('done', 3, summary=summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
