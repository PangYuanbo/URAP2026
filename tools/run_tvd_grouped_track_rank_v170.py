from __future__ import annotations

import json
import pickle
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
from tools.run_tvd_oof_stack_v130 import metrics
from tools.run_tvd_track_supported_budget_v162 import OUT as V162_OUTPUT
import tools.run_tvd_track_supported_budget_v162 as v162


RUN = ROOT / "artifacts/detached_tvd_grouped_track_rank_v170"
OUTPUT = Path(r"D:\URAP_vatd_rank_results\tvd_grouped_track_rank_v170")
TVD = Path(r"D:\urap_modal_stage\TransVisDrone")
VATD_MAP50 = 0.93844


def report(stage: str, done: int, total: int = 5, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now().astimezone().isoformat(), **extra}
    (RUN / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -60, 60)))


def percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    output = np.empty(len(values), dtype=np.float32)
    output[order] = (np.arange(len(values), dtype=np.float32) + 0.5) / max(len(values), 1)
    return output


def selected_v162(split: str, source: Path):
    correct, pred, target, locations, labels, calibrated, support, length, tracks, mapped, source_rows = v162.base(split, source)
    selected = json.loads((V162_OUTPUT / "official_summary.json").read_text(encoding="utf-8"))["validation_selection"]
    v162.length_global = length
    final, changed, promoted = v162.apply(
        calibrated,
        support,
        locations,
        int(selected["top_k"]),
        float(selected["suppression_factor"]),
        float(selected["score_gate"]),
        float(selected["promotion_alpha"]),
        int(selected["minimum_track_rows"]),
    )
    return correct, pred, target, locations, labels, calibrated, support, length, final, tracks, mapped, source_rows, changed, promoted


def load_data(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_features(split: str, source: Path):
    result = selected_v162(split, source)
    correct, pred, target, locations, labels, calibrated, support, length, final = result[:9]
    data = load_data(source)
    rows = len(locations)
    features = np.zeros((rows, 15), dtype=np.float32)
    groups = np.empty(rows, dtype=object)
    frames: dict[str, list[int]] = {}
    for row, (sequence, _frame_id, prediction_index, image_id, raw_score) in enumerate(locations):
        detection = data[str(image_id)]["detections"][int(prediction_index)]
        x1, y1, x2, y2 = [float(value) for value in detection["bbox"]]
        width = max(x2 - x1, 1e-3)
        height = max(y2 - y1, 1e-3)
        raw = float(raw_score)
        features[row, 0] = logit(np.asarray([raw]))[0]
        features[row, 1] = logit(np.asarray([calibrated[row]]))[0]
        features[row, 2] = logit(np.asarray([support[row]]))[0]
        features[row, 3] = logit(np.asarray([final[row]]))[0]
        features[row, 4] = np.log1p(float(length[row]))
        features[row, 8] = np.log(width * height)
        features[row, 9] = np.log(width / height)
        features[row, 10] = ((x1 + x2) * 0.5) / 1920.0
        features[row, 11] = ((y1 + y2) * 0.5) / 1280.0
        features[row, 12] = width / 1920.0
        features[row, 13] = height / 1280.0
        features[row, 14] = features[row, 3] - features[row, 0]
        groups[row] = str(sequence)
        frames.setdefault(str(image_id), []).append(row)
    for frame_rows in frames.values():
        ids = np.asarray(frame_rows, dtype=np.int64)
        features[ids, 5] = percentile_rank(np.asarray([locations[index][4] for index in ids], dtype=np.float64))
        features[ids, 6] = percentile_rank(support[ids])
        features[ids, 7] = np.log1p(len(ids))
    labels_binary = correct[:, 0].astype(np.int8)
    return result, features, labels_binary, groups


def model_specs():
    return [
        ("logistic_c01", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=300, solver="lbfgs", class_weight={0: 1.0, 1: 12.0}))),
        ("logistic_c1", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", class_weight={0: 1.0, 1: 12.0}))),
        ("xgb_stump", lambda: XGBClassifier(n_estimators=96, max_depth=1, learning_rate=0.05, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=20.0, scale_pos_weight=12.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", n_jobs=4, random_state=2026170)),
        ("xgb_depth2", lambda: XGBClassifier(n_estimators=64, max_depth=2, learning_rate=0.04, min_child_weight=40, subsample=0.8, colsample_bytree=0.8, reg_lambda=40.0, scale_pos_weight=12.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", n_jobs=4, random_state=2026171)),
    ]


def fuse(base: np.ndarray, learned: np.ndarray, alpha: float, mode: str) -> np.ndarray:
    base = np.clip(base, 1e-7, 1 - 1e-7)
    learned = np.clip(learned, 1e-7, 1 - 1e-7)
    if mode == "logit":
        return sigmoid((1 - alpha) * logit(base) + alpha * logit(learned))
    if mode == "geom":
        return np.exp((1 - alpha) * np.log(base) + alpha * np.log(learned))
    return (1 - alpha) * base + alpha * learned


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report("load_validation", 0)
    val_result, val_x, val_y, groups = build_features("val", v162.VAL)
    correct, pred, target, locations, labels, _calibrated, _support, _length, base = val_result[:9]
    unique_groups = sorted(set(groups.tolist()))
    folds = min(5, len(unique_groups))
    splitter = GroupKFold(n_splits=folds)
    model_rows: list[dict[str, object]] = []
    oof_by_model: dict[str, np.ndarray] = {}
    for model_index, (name, factory) in enumerate(model_specs()):
        report("fit_oof", 1, model=name, model_index=model_index + 1, models_total=len(model_specs()), rows=len(val_x), positive_rows=int(val_y.sum()), groups=len(unique_groups))
        oof = np.zeros(len(val_y), dtype=np.float64)
        for fold, (train_indices, held_indices) in enumerate(splitter.split(val_x, val_y, groups), 1):
            model = factory()
            model.fit(val_x[train_indices], val_y[train_indices])
            oof[held_indices] = model.predict_proba(val_x[held_indices])[:, 1]
            report("fit_oof", 1, model=name, fold=fold, folds=folds)
        oof_by_model[name] = oof
        for mode in ("logit", "geom", "linear"):
            for alpha in (0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.14, 0.2, 0.3, 0.4, 0.55, 0.7):
                candidate = fuse(base, oof, alpha, mode)
                model_rows.append({"model": name, "mode": mode, "alpha": alpha, **metrics(correct, candidate, pred, target, TVD)})
    best = max(model_rows, key=lambda row: float(row["map50"]))
    baseline = metrics(correct, base, pred, target, TVD)
    (OUTPUT / "oof_validation.json").write_text(json.dumps({"best": best, "baseline_v162": baseline, "top": sorted(model_rows, key=lambda row: -float(row["map50"]))[:80], "groups": unique_groups, "folds": folds, "features": val_x.shape[1]}, indent=2), encoding="utf-8")
    report("refit_validation", 3, validation_selection=best, baseline=baseline)
    factory = dict(model_specs())[str(best["model"])]
    model = factory()
    model.fit(val_x, val_y)
    del val_x, val_y, groups, oof_by_model
    report("load_fixed_test", 4)
    test_result, test_x, _test_y, _test_groups = build_features("test", v162.TEST)
    test_correct, test_pred, test_target, test_locations, test_labels, _tc, _ts, _tl, test_base = test_result[:9]
    learned_test = model.predict_proba(test_x)[:, 1]
    test_score = fuse(test_base, learned_test, float(best["alpha"]), str(best["mode"]))
    test = {**metrics(test_correct, test_score, test_pred, test_target, TVD), "labels": test_labels, "detections": len(test_locations)}
    gain = 100 * (float(test["map50"]) - VATD_MAP50)
    summary = {
        "protocol": "low-capacity 15-feature candidate/track ranker; GroupKFold by validation video; model and fusion selected on OOF validation; refit on all validation videos; fixed untouched test",
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
