from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor


ROOT = Path(r"C:\Users\aaron\Desktop\URAP")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from tools.run_tvd_grouped_track_rank_v170 import fuse, selected_v162
from tools.run_tvd_oof_stack_v130 import metrics
from tools.run_tvd_track_meta_rank_v156 import TRACKS, finite, track_vector
import tools.run_tvd_track_supported_budget_v162 as v162


RUN = ROOT / "artifacts/detached_tvd_samurai_track_quality_v172"
OUTPUT = Path(r"D:\URAP_vatd_rank_results\tvd_samurai_track_quality_v172")
TVD = Path(r"D:\urap_modal_stage\TransVisDrone")
VATD_MAP50 = 0.93844


def report(stage: str, done: int, total: int = 5, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now().astimezone().isoformat(), **extra}
    (RUN / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def track_dataset(split: str, source: Path, track_path: Path, with_targets: bool):
    result = selected_v162(split, source)
    correct, pred, target, locations, labels, calibrated, support, length, base = result[:9]
    lookup = {(sequence, frame_id, index): row for row, (sequence, frame_id, index, _image_id, _raw) in enumerate(locations)}
    track_features: list[np.ndarray] = []
    track_targets: list[float] = []
    track_groups: list[str] = []
    track_members: list[np.ndarray] = []
    with track_path.open(encoding="utf-8-sig") as source_file:
        for line in source_file:
            if not line.strip():
                continue
            item = json.loads(line)
            members: list[int] = []
            sequences: list[str] = []
            for row in item.get("rows") or []:
                sequence = str(row.get("seq") or "")
                key = (sequence, int(row.get("frame_id", 0)), int(row.get("prediction_index", -1)))
                index = lookup.get(key)
                if index is not None:
                    members.append(index)
                    sequences.append(sequence)
            if len(members) < 2:
                continue
            ids = np.asarray(members, dtype=np.int64)
            raw = np.asarray([locations[index][4] for index in ids], dtype=np.float64)
            final = base[ids]
            track_support = support[ids]
            steps = np.diff(np.asarray([locations[index][1] for index in ids], dtype=np.float64))
            meta = track_vector(item.get("meta") or {})
            aggregate = np.asarray([
                float(raw.mean()), float(raw.max()), float(np.quantile(raw, 0.75)), float(np.quantile(raw, 0.9)),
                float(final.mean()), float(final.max()), float(np.quantile(final, 0.75)), float(np.quantile(final, 0.9)),
                float(track_support.mean()), float(track_support.max()), math.log1p(len(ids)) / 6.0,
                float(np.mean(steps == 1)) if len(steps) else 1.0,
                float(np.mean(steps)) if len(steps) else 1.0,
                float(np.std(raw)), float(np.std(final)),
            ], dtype=np.float32)
            track_features.append(np.concatenate((meta, aggregate)))
            track_members.append(ids)
            track_groups.append(sequences[0])
            if with_targets:
                track_targets.append(float(correct[ids, 0].mean()))
    return result, np.asarray(track_features, dtype=np.float32), np.asarray(track_targets, dtype=np.float32) if with_targets else None, np.asarray(track_groups, dtype=object), track_members


def factories():
    return {
        "xgb_stump": lambda: XGBRegressor(n_estimators=160, max_depth=1, learning_rate=0.035, min_child_weight=10, subsample=0.85, colsample_bytree=0.85, reg_lambda=30.0, objective="reg:squarederror", tree_method="hist", device="cuda", n_jobs=4, random_state=2026172),
        "xgb_depth2": lambda: XGBRegressor(n_estimators=128, max_depth=2, learning_rate=0.03, min_child_weight=20, subsample=0.85, colsample_bytree=0.85, reg_lambda=40.0, objective="reg:squarederror", tree_method="hist", device="cuda", n_jobs=4, random_state=2026173),
        "extra_trees": lambda: ExtraTreesRegressor(n_estimators=160, max_depth=6, min_samples_leaf=12, max_features=0.7, n_jobs=6, random_state=2026174),
        "random_forest": lambda: RandomForestRegressor(n_estimators=120, max_depth=6, min_samples_leaf=12, max_features=0.7, n_jobs=6, random_state=2026175),
    }


def row_quality(base: np.ndarray, members: list[np.ndarray], track_scores: np.ndarray, aggregation: str) -> tuple[np.ndarray, int]:
    if aggregation == "mean":
        total = np.zeros(len(base), dtype=np.float64)
        count = np.zeros(len(base), dtype=np.int32)
        for ids, score in zip(members, track_scores):
            total[ids] += float(score)
            count[ids] += 1
        output = base.copy()
        valid = count > 0
        output[valid] = total[valid] / count[valid]
        return output, int(valid.sum())
    output = np.zeros(len(base), dtype=np.float64)
    valid = np.zeros(len(base), dtype=bool)
    for ids, score in zip(members, track_scores):
        if aggregation == "longest":
            effective = float(score) * (1 - math.exp(-len(ids) / 8.0))
        else:
            effective = float(score)
        output[ids] = np.maximum(output[ids], effective)
        valid[ids] = True
    output[~valid] = base[~valid]
    return output, int(valid.sum())


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report("load_validation_tracks", 0)
    val_result, features, targets, groups, members = track_dataset("val", v162.VAL, TRACKS["val"], True)
    correct, pred, target, locations, labels, _calibrated, _support, _length, base = val_result[:9]
    unique_groups = sorted(set(groups.tolist()))
    splitter = GroupKFold(n_splits=len(unique_groups))
    rows: list[dict[str, object]] = []
    for model_index, (name, factory) in enumerate(factories().items(), 1):
        report("fit_track_quality_oof", 1, model=name, model_index=model_index, models_total=len(factories()), tracks=len(features), groups=len(unique_groups))
        oof = np.zeros(len(targets), dtype=np.float64)
        for fold, (train_indices, held_indices) in enumerate(splitter.split(features, targets, groups), 1):
            model = factory()
            sample_weight = 0.25 + 3.0 * targets[train_indices]
            model.fit(features[train_indices], targets[train_indices], sample_weight=sample_weight)
            oof[held_indices] = np.clip(model.predict(features[held_indices]), 0.0, 1.0)
            report("fit_track_quality_oof", 1, model=name, fold=fold, folds=len(unique_groups))
        for aggregation in ("max", "mean", "longest"):
            quality, mapped = row_quality(base, members, oof, aggregation)
            for mode in ("logit", "geom", "linear"):
                for alpha in (0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.14, 0.2, 0.3, 0.4, 0.55):
                    rows.append({"model": name, "aggregation": aggregation, "mode": mode, "alpha": alpha, "mapped_rows": mapped, **metrics(correct, fuse(base, quality, alpha, mode), pred, target, TVD)})
    best = max(rows, key=lambda row: float(row["map50"]))
    baseline = metrics(correct, base, pred, target, TVD)
    (OUTPUT / "oof_validation.json").write_text(json.dumps({"best": best, "baseline_v162": baseline, "top": sorted(rows, key=lambda row: -float(row["map50"]))[:80], "tracks": len(features), "groups": unique_groups, "features": features.shape[1], "target_mean": float(targets.mean()), "positive_tracks": int((targets > 0).sum())}, indent=2), encoding="utf-8")
    report("refit_validation_tracks", 3, validation_selection=best)
    model = factories()[str(best["model"])]()
    model.fit(features, targets, sample_weight=0.25 + 3.0 * targets)
    report("load_fixed_test", 4)
    test_result, test_features, _test_targets, _test_groups, test_members = track_dataset("test", v162.TEST, TRACKS["test"], False)
    test_correct, test_pred, test_target, test_locations, test_labels, _tc, _ts, _tl, test_base = test_result[:9]
    predicted_tracks = np.clip(model.predict(test_features), 0.0, 1.0)
    test_quality, mapped = row_quality(test_base, test_members, predicted_tracks, str(best["aggregation"]))
    test_score = fuse(test_base, test_quality, float(best["alpha"]), str(best["mode"]))
    test = {**metrics(test_correct, test_score, test_pred, test_target, TVD), "labels": test_labels, "detections": len(test_locations), "tracks": len(test_features), "mapped_rows": mapped}
    gain = 100 * (float(test["map50"]) - VATD_MAP50)
    summary = {
        "protocol": "SAMURAI-style whole-track quality memory: regress TP fraction per track using track dynamics and score history; leave-one-validation-video-out OOF model/fusion selection; refit on validation; fixed test",
        "validation_selection": best,
        "validation_v162": baseline,
        "test_fixed": test,
        "vatd_map50": VATD_MAP50,
        "gain_over_vatd_points": gain,
        "target_3_to_5_met": 3 <= gain <= 5,
        "target_at_least_3_met": gain >= 3,
    }
    summary_path = OUTPUT / "official_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", 5, summary=summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        report("failed", 0, error=repr(error))
        raise
