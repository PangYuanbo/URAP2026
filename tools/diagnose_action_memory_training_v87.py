from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for entry in (REPO, REPO / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_action_chunk_temporal_multiplicity import temporal_gate_map
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.sweep_tvd_predictionsgt_score_fusion import fuse_score, load_row_scores
from tools.train_action_bank_motion_token_listwise import (
    ACTION_QUERY_NAMES,
    FEATURE_NAMES,
    dataset_arrays,
    greedy_match_qualities,
    load_auxiliary,
)


ARD_SOURCE = Path(r"D:\URAP_vatd_rank_results\ard100_yolomg_generalization_v2")
ARD_TARGET = Path(r"D:\URAP_vatd_rank_results\ard100_action_memory_target_v86")
ARD_FROZEN = Path(r"D:\URAP_vatd_rank_results\ard100_action_memory_action_only_v84")
AOT_TARGET = REPO / "artifacts" / "route_b_official" / "aot_action_memory_target_v86"
AOT_FROZEN = REPO / "artifacts" / "route_b_official" / "aot_action_memory_frozen_corrected_v86"
OUT = REPO / "artifacts" / "route_b_official" / "action_memory_training_diagnosis_v87.json"


def candidate_labels(data: dict[str, object]) -> tuple[np.ndarray, list[tuple[str, int]], list[tuple[int, int]]]:
    labels: list[np.ndarray] = []
    locations: list[tuple[str, int]] = []
    groups: list[tuple[int, int]] = []
    cursor = 0
    for image_id, item in data.items():
        detections = list(item.get("detections") or [])
        gt = np.asarray([row["bbox"] for row in item.get("labels") or []], dtype=np.float32).reshape(-1, 4)
        matched = greedy_match_qualities([row["bbox"] for row in detections], gt)
        labels.append((matched >= 0.5).astype(np.int8))
        locations.extend((str(image_id), index) for index in range(len(detections)))
        groups.append((cursor, cursor + len(detections)))
        cursor += len(detections)
    return np.concatenate(labels), locations, groups


def raw_scores(data: dict[str, object]) -> np.ndarray:
    return np.asarray([float(row.get("score", 0.0)) for item in data.values() for row in item.get("detections") or []], dtype=np.float64)


def aligned_scores(path: Path, field: str, locations: list[tuple[str, int]]) -> np.ndarray:
    values, _ = load_row_scores(path, field, 1)
    return np.asarray([values.get(image_key(image_id, index), np.nan) for image_id, index in locations], dtype=np.float64)


def ranking_metrics(labels: np.ndarray, scores: np.ndarray, groups: list[tuple[int, int]], raw: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(scores)
    y = labels[valid]
    s = scores[valid]
    positives = s[y == 1]
    negatives = s[y == 0]
    top1_hits = top1_total = 0
    reciprocal_ranks = []
    for start, stop in groups:
        local_labels = labels[start:stop]
        local_scores = scores[start:stop]
        if not local_labels.any() or not np.isfinite(local_scores).any():
            continue
        order = np.argsort(-np.nan_to_num(local_scores, nan=-np.inf), kind="stable")
        top1_hits += int(local_labels[order[0]] == 1)
        top1_total += 1
        positive_ranks = np.flatnonzero(local_labels[order] == 1)
        reciprocal_ranks.append(1.0 / float(positive_ranks[0] + 1))
    return {
        "rows": int(valid.sum()),
        "positive_rows": int(y.sum()),
        "candidate_ap": float(average_precision_score(y, s)),
        "candidate_auc": float(roc_auc_score(y, s)),
        "positive_mean": float(positives.mean()),
        "negative_mean": float(negatives.mean()),
        "separation": float(positives.mean() - negatives.mean()),
        "top1_positive_rate": top1_hits / max(top1_total, 1),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "raw_score_correlation": float(np.corrcoef(raw[valid], s)[0, 1]),
    }


def old_ard_final_scores(data: dict[str, object], locations: list[tuple[str, int]]) -> np.ndarray:
    base = aligned_scores(ARD_SOURCE / "v46_scores.jsonl", "action_chunk_neighbor_score", locations)
    expert = aligned_scores(ARD_SOURCE / "v52_scores.jsonl", "action_chunk_multi_expert_score", locations)
    config = json.loads((ARD_SOURCE / "action_bank_summary.json").read_text(encoding="utf-8"))["action_bank"]
    fps = json.loads((REPO / "data_templates" / "ard100_sequence_fps.json").read_text(encoding="utf-8"))
    gates = temporal_gate_map(data, config["threshold"], config["window_seconds"], config["min_fraction"], fps)
    output = []
    cursor = 0
    for image_id, item in data.items():
        gate = gates.get(str(image_id), False)
        for row in item.get("detections") or []:
            auxiliary = math.exp((1.0 - config["expert_weight"]) * math.log(max(base[cursor], 1e-9)) + config["expert_weight"] * math.log(max(expert[cursor], 1e-9))) if gate else base[cursor]
            output.append(fuse_score(float(row.get("score", 0.0)), auxiliary, config["alpha"], "geom-mix"))
            cursor += 1
    return np.asarray(output, dtype=np.float64)


def normalization_shift(predictions: Path, auxiliary: Path, model: Path) -> dict[str, object]:
    aux, sizes = load_auxiliary(auxiliary)
    features, _, _, _, _ = dataset_arrays(load_predictionsgt(predictions), aux, sizes, {}, False)
    checkpoint = torch.load(model, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    indices = np.asarray([FEATURE_NAMES.index(name) for name in ACTION_QUERY_NAMES], dtype=np.int64)
    z = np.abs((features[:, indices] - mean[indices]) / std[indices])
    per_feature = []
    for column, name in enumerate(ACTION_QUERY_NAMES):
        per_feature.append({
            "feature": name,
            "mean_abs_z": float(z[:, column].mean()),
            "p95_abs_z": float(np.quantile(z[:, column], 0.95)),
            "fraction_abs_z_gt_5": float((z[:, column] > 5).mean()),
        })
    return {
        "rows": len(features),
        "mean_abs_z": float(z.mean()),
        "fraction_abs_z_gt_5": float((z > 5).mean()),
        "largest_shifts": sorted(per_feature, key=lambda row: row["mean_abs_z"], reverse=True)[:8],
    }


def main() -> int:
    ard = load_predictionsgt(ARD_SOURCE / "ard100_yolomg_predictionsgt.pkl")
    ard_labels, ard_locations, ard_groups = candidate_labels(ard)
    ard_raw = raw_scores(ard)
    ard_methods = {
        "detector_raw": ard_raw,
        "legacy_action_bank_final": old_ard_final_scores(ard, ard_locations),
        "frozen_action_memory": aligned_scores(ARD_FROZEN / "test_scores.jsonl", "action_memory_action_only_score", ard_locations),
        "target_trained_action_memory": aligned_scores(ARD_TARGET / "test_scores.jsonl", "ard100_target_action_memory_score", ard_locations),
    }
    ard_metrics = {name: ranking_metrics(ard_labels, scores, ard_groups, ard_raw) for name, scores in ard_methods.items()}

    aot = load_predictionsgt(AOT_TARGET / "data" / "val_predictionsgt.pkl")
    aot_labels, aot_locations, aot_groups = candidate_labels(aot)
    aot_raw = raw_scores(aot)
    aot_methods = {
        "detector_raw": aot_raw,
        "frozen_action_memory": aligned_scores(AOT_FROZEN / "val_scores.jsonl", "aot_frozen_corrected_score", aot_locations),
        "target_trained_action_memory": aligned_scores(AOT_TARGET / "val_scores.jsonl", "aot_target_action_memory_score", aot_locations),
    }
    aot_metrics = {name: ranking_metrics(aot_labels, scores, aot_groups, aot_raw) for name, scores in aot_methods.items()}

    ard_counts = np.asarray([len(item.get("detections") or []) for item in ard.values()])
    train_summary = json.loads((ARD_TARGET / "train_summary.json").read_text(encoding="utf-8"))
    report = {
        "ard100": {
            "candidate_ranking": ard_metrics,
            "test_candidate_count": {"mean": float(ard_counts.mean()), "p95": float(np.quantile(ard_counts, 0.95)), "max": int(ard_counts.max())},
            "training": {
                "rows_before_cap": train_summary["uncapped_train_rows"],
                "rows_after_cap": train_summary["train_rows"],
                "positive_rows": train_summary["train_positive_rows"],
                "background_frames": train_summary["background_frames"],
                "future_consistency_mean": train_summary["train_future_consistency_mean"],
                "epochs": len(train_summary["history"]),
            },
            "nps_normalization_shift": normalization_shift(
                ARD_SOURCE / "ard100_yolomg_val_predictionsgt.pkl",
                Path(r"D:\URAP_vatd_rank_results\ard100_action_memory_cross_attention_v83\val_aux.jsonl"),
                Path(r"D:\URAP_vatd_rank_results\nps_action_memory_action_only_v84\model.pt"),
            ),
        },
        "aot": {
            "validation_candidate_ranking": aot_metrics,
            "nps_normalization_shift": normalization_shift(
                AOT_TARGET / "data" / "val_predictionsgt.pkl",
                AOT_TARGET / "data" / "val_aux.jsonl",
                Path(r"D:\URAP_vatd_rank_results\nps_action_memory_action_only_v84\model.pt"),
            ),
        },
        "architectural_differences": {
            "legacy_action_bank": ["forward history", "backward/future history", "bidirectional neighbor features", "current detector score", "candidate-to-candidate context", "XGBoost ensemble", "explicit detector-score geometric fusion"],
            "automatic_action_memory": ["past-only causal 1s/3s memory", "action-only current query", "no direct current detector score in query", "no candidate-to-candidate context", "one neural cross-attention head", "listwise IoU supervision"],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

