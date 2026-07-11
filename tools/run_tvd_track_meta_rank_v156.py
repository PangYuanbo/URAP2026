from __future__ import annotations

import gc
import json
import math
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(r'C:\Users\aaron\Desktop\URAP')
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from tools.run_tvd_frame_budget_v146 import VAL, TEST, load, budget
from tools.run_tvd_oof_stack_v130 import flat_stats, load_predictionsgt, metrics
from tools.run_tvd_temporal_nms_v154 import boxes_for, suppress

TRAIN = Path(r'D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\official_train_dense\predictionsgt\predictionsgt_split_0.pkl')
TRACKS = {
    'train': Path(r'D:\URAP_nps_train_tvd\route_b_official\tracklets\proposal_tracklets.jsonl'),
    'val': Path(r'D:\URAP_nps_val_tvd\route_b_official\tracklets\proposal_tracklets.jsonl'),
    'test': Path(r'D:\URAP_vatd_rank_inputs\nps_tracklets_with_vatd.jsonl'),
}
SIZE_MAP = json.loads((ROOT / 'data_templates' / 'nps_sequence_sizes_actual.json').read_text(encoding='utf-8'))
RUN = ROOT / 'artifacts' / 'detached_tvd_track_meta_rank_v156'
OUT = Path(r'D:\URAP_vatd_rank_results\tvd_track_meta_rank_v156')
TVD = Path(r'D:\urap_modal_stage\TransVisDrone')
VATD = 0.93844


def report(stage: str, done: int, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {'stage': stage, 'done': done, 'total': 5, 'updated': datetime.now().astimezone().isoformat(), **extra}
    (RUN / 'progress.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload), flush=True)


def finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def candidate_features(data, locations) -> np.ndarray:
    features = np.zeros((len(locations), 15), dtype=np.float32)
    frames: dict[str, list[int]] = {}
    for row, (sequence, _frame_id, index, image_id, raw) in enumerate(locations):
        frames.setdefault(image_id, []).append(row)
        detection = data[image_id]['detections'][index]
        x1, y1, x2, y2 = [finite(value) for value in detection['bbox']]
        width_image, height_image = SIZE_MAP.get(sequence, (1920, 1280))
        width = max(1e-3, x2 - x1)
        height = max(1e-3, y2 - y1)
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        clipped = np.clip(float(raw), 1e-6, 1.0 - 1e-6)
        features[row, :11] = [
            raw,
            math.log(clipped / (1.0 - clipped)),
            center_x / width_image,
            center_y / height_image,
            width / width_image,
            height / height_image,
            width * height / (width_image * height_image),
            math.log(width / height),
            min(center_x, width_image - center_x, center_y, height_image - center_y) / min(width_image, height_image),
            math.log1p(len(data[image_id].get('detections', []))) / 6.0,
            math.log1p(max(width, height)) / 6.0,
        ]
    for row_ids in frames.values():
        ids = np.asarray(row_ids, dtype=np.int64)
        scores = features[ids, 0]
        order = np.argsort(scores)
        ranks = np.empty(len(ids), dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, len(ids), dtype=np.float32) if len(ids) > 1 else 1.0
        features[ids, 11] = ranks
        features[ids, 12] = float(scores.max()) - scores
        features[ids, 13] = np.asarray([np.sum(scores > score) for score in scores], dtype=np.float32) / max(1, len(scores))
        features[ids, 14] = scores / max(1e-6, float(scores.max()))
    return features


def track_vector(meta: dict[str, object]) -> np.ndarray:
    rows = max(1.0, finite(meta.get('num_rows'), finite(meta.get('num_rows_raw'), 1.0)))
    span = max(1.0, finite(meta.get('track_span_frames'), rows))
    side = max(1e-3, finite(meta.get('mean_box_side'), 1.0))
    return np.asarray([
        math.log1p(rows) / 6.0,
        math.log1p(span) / 6.0,
        finite(meta.get('frame_density')),
        finite(meta.get('mean_final_score')),
        finite(meta.get('max_final_score')),
        finite(meta.get('mean_objectness')),
        finite(meta.get('max_objectness')),
        finite(meta.get('score_slope')),
        finite(meta.get('score_above_02_rate')),
        finite(meta.get('score_above_02_longest_streak')) / rows,
        finite(meta.get('gap_rate')),
        finite(meta.get('max_frame_gap')) / span,
        finite(meta.get('mean_frame_gap')) / span,
        finite(meta.get('mean_center_step')) / side,
        finite(meta.get('max_center_step')) / side,
        finite(meta.get('std_center_step')) / side,
        math.log1p(side) / 6.0,
        finite(meta.get('first_final_score')),
        finite(meta.get('last_final_score')),
        finite(meta.get('last_final_score')) - finite(meta.get('first_final_score')),
        finite(meta.get('background_dominance_rate')),
        finite(meta.get('background_dominance_longest_streak')) / rows,
        finite(meta.get('final_margin_mean')),
        finite(meta.get('final_margin_min')),
        finite(meta.get('final_margin_slope')),
    ], dtype=np.float32)


def add_track_features(features: np.ndarray, locations, track_path: Path) -> tuple[np.ndarray, int, int]:
    lookup = {(sequence, frame_id, index): row for row, (sequence, frame_id, index, _image_id, _raw) in enumerate(locations)}
    track_features = np.zeros((len(locations), 29), dtype=np.float32)
    assigned_length = np.zeros(len(locations), dtype=np.float32)
    mapped = 0
    tracks = 0
    with track_path.open(encoding='utf-8-sig') as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            rows = item.get('rows') or []
            members = []
            for position, row in enumerate(rows):
                key = (str(row.get('seq') or ''), int(row.get('frame_id', 0)), int(row.get('prediction_index', -1)))
                index = lookup.get(key)
                if index is not None:
                    members.append((position, index))
            if not members:
                continue
            tracks += 1
            meta_vector = track_vector(item.get('meta') or {})
            length = float(len(rows))
            denominator = max(1, len(rows) - 1)
            for position, index in members:
                if length < assigned_length[index]:
                    continue
                assigned_length[index] = length
                relative = position / denominator
                track_features[index, :25] = meta_vector
                track_features[index, 25:] = [relative, min(relative, 1.0 - relative), features[index, 0] - meta_vector[3], features[index, 0] / max(1e-6, meta_vector[4])]
                mapped += 1
    return np.concatenate((features, track_features), axis=1), mapped, tracks


def dataset(source: Path, track_path: Path, labels: bool):
    data = load_predictionsgt(source)
    correct, pred_cls, target_cls, locations, label_count = flat_stats(data)
    features = candidate_features(data, locations)
    features, mapped, tracks = add_track_features(features, locations, track_path)
    y = correct[:, 0].astype(np.int32) if labels else None
    del data
    gc.collect()
    return correct, pred_cls, target_cls, locations, label_count, features, y, mapped, tracks


def hard_training_rows(features: np.ndarray, y: np.ndarray, locations) -> np.ndarray:
    frames: dict[str, list[int]] = {}
    for row, location in enumerate(locations):
        frames.setdefault(location[3], []).append(row)
    keep = []
    for row_ids in frames.values():
        ids = np.asarray(row_ids, dtype=np.int64)
        positives = ids[y[ids] > 0]
        negatives = ids[y[ids] == 0]
        keep.extend(positives.tolist())
        if len(negatives):
            order = negatives[np.argsort(features[negatives, 0])[::-1]]
            keep.extend(order[:24].tolist())
    return np.asarray(sorted(set(keep)), dtype=np.int64)


def wait_for_ard_gpu() -> None:
    ard_run = ROOT / 'artifacts' / 'detached_ard100_yolomg_vatd_v2'
    comparison = Path(r'D:\URAP_vatd_rank_results\ard100_yolomg_generalization_v2\official_comparison.json')
    while not comparison.exists():
        pid_path = ard_run / 'pid.txt'
        if not pid_path.exists():
            return
        pid = int(pid_path.read_text(encoding='ascii').strip())
        status = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'], capture_output=True, text=True, check=False)
        alive = str(pid) in status.stdout and 'No tasks are running' not in status.stdout
        if not alive:
            return
        report('waiting_for_ard_gpu', 1)
        time.sleep(30)

def fit_model(features: np.ndarray, y: np.ndarray, locations) -> tuple[xgb.XGBClassifier, int]:
    keep = hard_training_rows(features, y, locations)
    binary = y[keep]
    positives = max(1, int(binary.sum()))
    negatives = max(1, len(binary) - positives)
    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.035,
        min_child_weight=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=10,
        reg_alpha=0.15,
        gamma=0.02,
        objective='binary:logistic',
        eval_metric='aucpr',
        tree_method='hist',
        device='cuda',
        max_bin=256,
        scale_pos_weight=min(12.0, negatives / positives),
        n_jobs=8,
        random_state=2026,
    )
    model.fit(features[keep], binary, verbose=False)
    return model, len(keep)


def structural_base(split: str, source: Path):
    correct, pred_cls, target_cls, locations, labels, score = load(split, source)
    budget_selection = json.loads(Path(r'D:\URAP_vatd_rank_results\tvd_frame_budget_v146\official_summary.json').read_text(encoding='utf-8'))['validation_selection']
    score, _ = budget(score, locations, int(budget_selection['top_k']), float(budget_selection['suppression_factor']), float(budget_selection['score_gate']))
    nms_selection = json.loads(Path(r'D:\URAP_vatd_rank_results\tvd_temporal_nms_v154\official_summary.json').read_text(encoding='utf-8'))['validation_selection']
    data = load_predictionsgt(source)
    boxes = boxes_for(data, locations)
    score, _ = suppress(score, locations, boxes, float(nms_selection['iou_threshold']), float(nms_selection['suppression_factor']), float(nms_selection['minimum_score']))
    return correct, pred_cls, target_cls, locations, labels, score


def fuse(base: np.ndarray, learned: np.ndarray, alpha: float, mode: str) -> np.ndarray:
    base = np.clip(base, 1e-6, 1.0 - 1e-6)
    learned = np.clip(learned, 1e-6, 1.0 - 1e-6)
    if mode == 'logit':
        logit_base = np.log(base / (1.0 - base))
        logit_learned = np.log(learned / (1.0 - learned))
        return 1.0 / (1.0 + np.exp(-((1.0 - alpha) * logit_base + alpha * logit_learned)))
    if mode == 'geom':
        return np.exp((1.0 - alpha) * np.log(base) + alpha * np.log(learned))
    if mode == 'fp_suppress':
        return base * ((1.0 - alpha) + alpha * learned)
    return (1.0 - alpha) * base + alpha * learned


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report('load_train_track_features', 0)
    _correct, _pred, _target, train_locations, _labels, train_x, train_y, train_mapped, train_tracks = dataset(TRAIN, TRACKS['train'], True)
    report('fit_train_only_model', 1, train_rows=len(train_x), mapped=train_mapped, tracks=train_tracks, features=train_x.shape[1])
    wait_for_ard_gpu()
    model, hard_rows = fit_model(train_x, train_y, train_locations)
    model.save_model(OUT / 'track_meta_rank.ubj')
    del train_x, train_y, train_locations
    gc.collect()

    report('select_validation', 2, hard_rows=hard_rows)
    _vc, _vp, _vt, val_locations, _vl, val_x, _vy, val_mapped, val_tracks = dataset(VAL, TRACKS['val'], True)
    learned_val = model.predict_proba(val_x)[:, 1].astype(np.float64)
    val_correct, val_pred, val_target, base_locations, val_labels, base_val = structural_base('val', VAL)
    if val_locations != base_locations:
        raise RuntimeError('validation candidate order mismatch')
    rows = []
    for mode in ('logit', 'geom', 'linear', 'fp_suppress'):
        for alpha in (0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.14, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0):
            rows.append({'mode': mode, 'alpha': alpha, **metrics(val_correct, fuse(base_val, learned_val, alpha, mode), val_pred, val_target, TVD)})
    best = max(rows, key=lambda row: float(row['map50']))
    (OUT / 'val_sweep.json').write_text(json.dumps({'best': best, 'top': sorted(rows, key=lambda row: -float(row['map50']))[:30], 'mapped': val_mapped, 'tracks': val_tracks, 'labels': val_labels}, indent=2), encoding='utf-8')
    del val_x, learned_val, base_val
    gc.collect()

    report('fixed_test', 4, validation_selection=best)
    _qc, _qp, _qt, test_locations, _ql, test_x, _qy, test_mapped, test_tracks = dataset(TEST, TRACKS['test'], False)
    learned_test = model.predict_proba(test_x)[:, 1].astype(np.float64)
    test_correct, test_pred, test_target, base_test_locations, test_labels, base_test = structural_base('test', TEST)
    if test_locations != base_test_locations:
        raise RuntimeError('test candidate order mismatch')
    score = fuse(base_test, learned_test, float(best['alpha']), str(best['mode']))
    test = {**metrics(test_correct, score, test_pred, test_target, TVD), 'labels': test_labels, 'detections': len(test_locations), 'mapped': test_mapped, 'tracks': test_tracks}
    gain = 100.0 * (float(test['map50']) - VATD)
    summary = {'protocol': 'train-only track-motion metadata ranker; validation-selected fusion on V154; fixed test', 'features': int(test_x.shape[1]), 'hard_train_rows': hard_rows, 'validation_selection': best, 'test_fixed': test, 'vatd_map50': VATD, 'gain_over_vatd_points': gain, 'target_3_to_5_met': 3.0 <= gain <= 5.0}
    (OUT / 'official_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report('done', 5, summary=summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())



