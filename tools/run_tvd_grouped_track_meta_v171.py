from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(r"C:\Users\aaron\Desktop\URAP")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from tools.run_tvd_grouped_track_rank_v170 import fuse, selected_v162
from tools.run_tvd_oof_stack_v130 import metrics
from tools.run_tvd_track_meta_rank_v156 import TRACKS, dataset, hard_training_rows
import tools.run_tvd_track_supported_budget_v162 as v162


RUN = ROOT / "artifacts/detached_tvd_grouped_track_meta_v171"
OUTPUT = Path(r"D:\URAP_vatd_rank_results\tvd_grouped_track_meta_v171")
TVD = Path(r"D:\urap_modal_stage\TransVisDrone")
VATD_MAP50 = 0.93844


def report(stage: str, done: int, total: int = 5, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now().astimezone().isoformat(), **extra}
    (RUN / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def factories():
    return {
        "logistic_c01": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=300, solver="lbfgs", class_weight={0: 1.0, 1: 10.0})),
        "xgb_stump": lambda: XGBClassifier(n_estimators=128, max_depth=1, learning_rate=0.04, min_child_weight=30, subsample=0.85, colsample_bytree=0.8, reg_lambda=30.0, scale_pos_weight=10.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", n_jobs=4, random_state=2026171),
        "xgb_depth2": lambda: XGBClassifier(n_estimators=96, max_depth=2, learning_rate=0.035, min_child_weight=50, subsample=0.85, colsample_bytree=0.8, reg_lambda=50.0, scale_pos_weight=10.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", n_jobs=4, random_state=2026172),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report("load_validation_track_metadata", 0)
    _c, _p, _t, locations, _labels, features, y, mapped, tracks = dataset(v162.VAL, TRACKS["val"], True)
    correct, pred, target, base_locations, labels, _calibrated, _support, _length, base = selected_v162("val", v162.VAL)[:9]
    if locations != base_locations:
        raise RuntimeError("validation candidate order mismatch")
    groups = np.asarray([location[0] for location in locations], dtype=object)
    unique_groups = sorted(set(groups.tolist()))
    splitter = GroupKFold(n_splits=len(unique_groups))
    rows: list[dict[str, object]] = []
    for model_index, (name, factory) in enumerate(factories().items(), 1):
        report("fit_grouped_oof", 1, model=name, model_index=model_index, models_total=len(factories()), rows=len(features), mapped=mapped, tracks=tracks)
        oof = np.zeros(len(y), dtype=np.float64)
        for fold, (train_indices, held_indices) in enumerate(splitter.split(features, y, groups), 1):
            local_keep = hard_training_rows(features[train_indices], y[train_indices], [locations[index] for index in train_indices])
            fit_indices = train_indices[local_keep]
            model = factory()
            model.fit(features[fit_indices], y[fit_indices])
            oof[held_indices] = model.predict_proba(features[held_indices])[:, 1]
            report("fit_grouped_oof", 1, model=name, fold=fold, folds=len(unique_groups), fit_rows=len(fit_indices))
        for mode in ("logit", "geom", "linear"):
            for alpha in (0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.14, 0.2, 0.3, 0.4, 0.55, 0.7):
                rows.append({"model": name, "mode": mode, "alpha": alpha, **metrics(correct, fuse(base, oof, alpha, mode), pred, target, TVD)})
    best = max(rows, key=lambda row: float(row["map50"]))
    baseline = metrics(correct, base, pred, target, TVD)
    (OUTPUT / "oof_validation.json").write_text(json.dumps({"best": best, "baseline_v162": baseline, "top": sorted(rows, key=lambda row: -float(row["map50"]))[:60], "groups": unique_groups, "features": features.shape[1], "mapped": mapped, "tracks": tracks}, indent=2), encoding="utf-8")
    report("refit_validation", 3, validation_selection=best)
    keep = hard_training_rows(features, y, locations)
    model = factories()[str(best["model"])]()
    model.fit(features[keep], y[keep])
    report("load_fixed_test", 4, fit_rows=len(keep))
    _qc, _qp, _qt, test_locations, _test_labels0, test_features, _test_y, test_mapped, test_tracks = dataset(v162.TEST, TRACKS["test"], False)
    test_correct, test_pred, test_target, test_base_locations, test_labels, _tc, _ts, _tl, test_base = selected_v162("test", v162.TEST)[:9]
    if test_locations != test_base_locations:
        raise RuntimeError("test candidate order mismatch")
    learned = model.predict_proba(test_features)[:, 1]
    test_score = fuse(test_base, learned, float(best["alpha"]), str(best["mode"]))
    test = {**metrics(test_correct, test_score, test_pred, test_target, TVD), "labels": test_labels, "detections": len(test_locations), "mapped": test_mapped, "tracks": test_tracks}
    gain = 100 * (float(test["map50"]) - VATD_MAP50)
    summary = {
        "protocol": "44-dimensional candidate plus whole-track metadata; leave-one-validation-video-out OOF model/fusion selection; refit on all validation videos; fixed test",
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
