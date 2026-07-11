from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(r"C:\Users\aaron\Desktop\URAP")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.train_action_bank_motion_token_listwise import write_score_jsonl
from tools.train_action_chunk_bidir_full import load_aux
from tools.train_action_chunk_neighbor_full import dataset_arrays, hard_rows, load_neighbor

FULL = Path(r"D:\URAP_vatd_rank_results\action_chunk_full_dev_v36")
NEIGHBOR = Path(r"D:\URAP_vatd_rank_results\action_chunk_neighbor_v44")
TRAIN = Path(r"D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0_fixed_canvas.pkl")
VAL = Path(r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl")
TEST = Path(r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl")
OUT = Path(r"D:\URAP_vatd_rank_results\tvd_domain_balanced_action_v119")
RUN = ROOT / "artifacts" / "detached_tvd_domain_balanced_action_v119"
SIZE_MAP = ROOT / "data_templates" / "nps_sequence_sizes_actual.json"
RATIOS = (0.10, 0.30, 0.70)
VATD_MAP50 = 0.93844


def report(stage: str, done: int, total: int = 5, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now().astimezone().isoformat(), **extra}
    (RUN / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def subset_indices_and_groups(mask: np.ndarray, groups: list[tuple[int, int]]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    selected = np.flatnonzero(mask)
    mapping = np.full(mask.shape[0], -1, dtype=np.int64)
    mapping[selected] = np.arange(selected.size)
    subset_groups: list[tuple[int, int]] = []
    for start, stop in groups:
        kept = np.arange(start, stop)[mask[start:stop]]
        if kept.size:
            subset_groups.append((int(mapping[kept[0]]), int(mapping[kept[-1]]) + 1))
    return selected, subset_groups


def fit_balanced(x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int) -> xgb.XGBClassifier:
    binary = (y >= 0.5).astype(np.int32)
    positives = max(1, int(np.sum(binary)))
    negatives = int(binary.size - positives)
    model = xgb.XGBClassifier(
        n_estimators=900,
        max_depth=8,
        learning_rate=0.03,
        min_child_weight=4,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=9,
        reg_alpha=0.1,
        gamma=0.02,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device="cuda",
        max_bin=256,
        scale_pos_weight=min(12.0, negatives / positives),
        n_jobs=8,
        random_state=seed,
    )
    model.fit(x, binary, sample_weight=weights, verbose=False)
    return model


def execute(command: list[str]) -> None:
    code = subprocess.call(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
    if code:
        raise RuntimeError(f"command failed with exit code {code}: {command}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report("load_verified_features", 0)
    size_map = json.loads(SIZE_MAP.read_text(encoding="utf-8"))

    report("load_features", 1)
    train_forward, train_backward = load_aux(FULL / "train_forward.jsonl"), load_aux(FULL / "train_backward.jsonl")
    train_neighbor, feature_names = load_neighbor(NEIGHBOR / "train_neighbor_scores.jsonl")
    train_x, train_y, train_groups, _, _ = dataset_arrays(load_predictionsgt(TRAIN), train_forward, train_backward, train_neighbor, True, size_map)
    del train_forward, train_backward, train_neighbor
    gc.collect()

    val_forward, val_backward = load_aux(FULL / "val_forward.jsonl"), load_aux(FULL / "val_backward.jsonl")
    val_neighbor, val_feature_names = load_neighbor(NEIGHBOR / "val_neighbor_scores.jsonl")
    if feature_names != val_feature_names:
        raise RuntimeError("train/validation neighbor fields differ")
    val_x, val_y, val_groups, val_locations, val_sequences = dataset_arrays(load_predictionsgt(VAL), val_forward, val_backward, val_neighbor, True, size_map)
    del val_forward, val_backward, val_neighbor
    gc.collect()

    test_forward, test_backward = load_aux(FULL / "test_forward.jsonl"), load_aux(FULL / "test_backward.jsonl")
    test_neighbor, test_feature_names = load_neighbor(NEIGHBOR / "test_neighbor_scores.jsonl")
    if feature_names != test_feature_names:
        raise RuntimeError("train/test neighbor fields differ")
    test_x, _, _, test_locations, _ = dataset_arrays(load_predictionsgt(TEST), test_forward, test_backward, test_neighbor, False, size_map)
    del test_forward, test_backward, test_neighbor
    gc.collect()

    train_hard = hard_rows(train_x, train_y, train_groups)
    sequences = sorted(set(val_sequences.tolist()))
    ratio_records: list[dict[str, object]] = []
    total_models = len(RATIOS) * len(sequences)
    model_number = 0

    for ratio in RATIOS:
        ratio_tag = str(ratio).replace(".", "p")
        oof = np.zeros(len(val_x), dtype=np.float32)
        test_predictions: list[np.ndarray] = []
        model_dir = OUT / "models" / ratio_tag
        model_dir.mkdir(parents=True, exist_ok=True)
        fold_records: list[dict[str, object]] = []
        for fold, held in enumerate(sequences):
            val_train_mask = val_sequences != held
            val_selected, val_subset_groups = subset_indices_and_groups(val_train_mask, val_groups)
            val_hard_local = hard_rows(val_x[val_selected], val_y[val_selected], val_subset_groups)
            val_hard = val_selected[val_hard_local]
            train_count = min(len(train_hard), max(1, int(round(ratio * len(val_hard)))))
            rng = np.random.default_rng(202600 + fold + int(ratio * 1000))
            train_sample = rng.choice(train_hard, size=train_count, replace=False)
            fit_x = np.concatenate((train_x[train_sample], val_x[val_hard]), axis=0)
            fit_y = np.concatenate((train_y[train_sample], val_y[val_hard]), axis=0)
            weights = np.concatenate((np.full(train_count, 0.75, dtype=np.float32), np.ones(len(val_hard), dtype=np.float32)))
            model = fit_balanced(fit_x, fit_y, weights, 2026 + fold + int(ratio * 100))
            hold = np.flatnonzero(~val_train_mask)
            oof[hold] = model.predict_proba(val_x[hold])[:, 1]
            test_predictions.append(model.predict_proba(test_x)[:, 1].astype(np.float32))
            model_path = model_dir / f"without_{held}.ubj"
            model.save_model(model_path)
            model_number += 1
            record = {"held": held, "ratio": ratio, "train_hard_rows": train_count, "validation_hard_rows": len(val_hard), "model": str(model_path)}
            fold_records.append(record)
            report("train_domain_balanced", 2, model=model_number, models=total_models, record=record)
            del fit_x, fit_y, weights, model
            gc.collect()

        field = f"tvd_domain_balanced_{ratio_tag}_score"
        val_scores = OUT / f"val_{ratio_tag}_scores.jsonl"
        test_scores = OUT / f"test_{ratio_tag}_scores.jsonl"
        write_score_jsonl(val_scores, oof, val_locations, field)
        write_score_jsonl(test_scores, np.mean(np.stack(test_predictions), axis=0).astype(np.float32), test_locations, field)
        val_sweep = OUT / f"val_{ratio_tag}_sweep.json"
        execute([
            sys.executable,
            str(ROOT / "tools" / "sweep_tvd_predictionsgt_score_fusion.py"),
            "--tvd-root", r"D:\urap_modal_stage\TransVisDrone",
            "--predictionsgt-pkl", str(VAL),
            "--tracklet-jsonl", str(val_scores),
            "--per-row-score",
            "--score-field", field,
            "--modes", "geom-mix", "logit-mix", "fp-suppress", "replace",
            "--alphas", ".01,.02,.04,.06,.08,.1,.14,.2,.3,.4,.55,.7,1",
            "--out-json", str(val_sweep),
        ])
        best = json.loads(val_sweep.read_text(encoding="utf-8"))["best"]
        ratio_records.append({"ratio": ratio, "field": field, "val_scores": str(val_scores), "test_scores": str(test_scores), "val_sweep": str(val_sweep), "val_best": best, "folds": fold_records})

    selected = max(ratio_records, key=lambda row: float(row["val_best"]["map50"]))
    report("fixed_test", 3, selected={"ratio": selected["ratio"], "val_best": selected["val_best"]})
    test_fixed = OUT / "test_fixed.json"
    best = selected["val_best"]
    execute([
        sys.executable,
        str(ROOT / "tools" / "sweep_tvd_predictionsgt_score_fusion.py"),
        "--tvd-root", r"D:\urap_modal_stage\TransVisDrone",
        "--predictionsgt-pkl", str(TEST),
        "--tracklet-jsonl", str(selected["test_scores"]),
        "--per-row-score",
        "--score-field", str(selected["field"]),
        "--modes", str(best["mode"]),
        "--alphas", str(best["alpha"]),
        "--out-json", str(test_fixed),
    ])
    test = json.loads(test_fixed.read_text(encoding="utf-8"))["best"]
    gain = 100.0 * (float(test["map50"]) - VATD_MAP50)
    summary = {
        "protocol": "validation-OOF-selected train hard-row ratio; fixed test",
        "ratios": ratio_records,
        "selected_ratio": selected["ratio"],
        "validation_selection": selected["val_best"],
        "test_fixed": test,
        "vatd_map50": VATD_MAP50,
        "gain_over_vatd_points": gain,
        "target_3_to_5_met": 3.0 <= gain <= 5.0,
    }
    (OUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", 5, summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


