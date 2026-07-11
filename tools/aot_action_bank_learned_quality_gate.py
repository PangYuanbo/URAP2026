import argparse
import copy
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold


IMAGE_RE = re.compile(r"^(?P<seq>Clip_\d+)_(?P<frame>\d+)\.png$")
SOURCES = (
    "action_bank_track_memory_promotion",
    "action_bank_cross_segment_interpolation",
    "action_bank_edge_extrapolation",
)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def find_match_csv(folder: Path) -> Path:
    matches = sorted(folder.glob("gt_det_matches_extended*.csv"))
    if not matches:
        raise FileNotFoundError(f"No match CSV in {folder}")
    return matches[0]


def incremental_labels(candidate_matches: Path, base_matches: Path) -> dict[int, int]:
    keys = ["flight_id", "time", "id"]
    candidate = pd.read_csv(candidate_matches, usecols=["index", "gt_det_match", *keys])
    base = pd.read_csv(base_matches, usecols=["gt_det_match", *keys])
    base_hits = pd.MultiIndex.from_frame(base.loc[base.gt_det_match.eq(1), keys].drop_duplicates())
    positives = candidate.loc[candidate.gt_det_match.eq(1) & candidate["index"].notna(), ["index", *keys]].copy()
    positives["useful"] = (~pd.MultiIndex.from_frame(positives[keys]).isin(base_hits)).astype("int8")
    return {int(index): int(value) for index, value in positives.groupby("index")["useful"].max().items()}


def parse_image(name: str) -> tuple[str, int]:
    match = IMAGE_RE.match(name)
    if not match:
        return "", 0
    return match.group("seq"), int(match.group("frame"))


def value(detection: dict, key: str, fallback: float = 0.0) -> float:
    try:
        return float(detection.get(key, fallback) or fallback)
    except (TypeError, ValueError):
        return fallback


def box_iou(left: dict, right: dict) -> float:
    left_x1 = value(left, "x") - value(left, "w") / 2
    left_y1 = value(left, "y") - value(left, "h") / 2
    left_x2 = left_x1 + value(left, "w")
    left_y2 = left_y1 + value(left, "h")
    right_x1 = value(right, "x") - value(right, "w") / 2
    right_y1 = value(right, "y") - value(right, "h") / 2
    right_x2 = right_x1 + value(right, "w")
    right_y2 = right_y1 + value(right, "h")
    intersection = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1)) * max(
        0.0, min(left_y2, right_y2) - max(left_y1, right_y1)
    )
    union = value(left, "w") * value(left, "h") + value(right, "w") * value(right, "h") - intersection
    return intersection / union if union > 0 else 0.0


def build_rows(records: list[dict], labels: dict[int, int], validation_clips: set[str]):
    track_rows = defaultdict(list)
    additions = []
    include_flow_features = any(source.startswith("action_bank_camera_compensated") for source in SOURCES)
    flat_index = 0
    for record_position, record in enumerate(records):
        seq, frame = parse_image(str(record.get("img_name") or ""))
        clip = seq.removeprefix("Clip_")
        for detection_position, detection in enumerate(record.get("detections") or []):
            source = str(detection.get("source") or "base")
            track_id = int(detection.get("track_id", -1))
            row = {
                "flat_index": flat_index,
                "record_position": record_position,
                "detection_position": detection_position,
                "record": record,
                "detection": detection,
                "seq": seq,
                "clip": clip,
                "frame": frame,
                "source": source,
                "track_key": (seq, track_id),
                "score": value(detection, "s"),
                "raw": value(detection, "tracklet_rescore_raw_s", value(detection, "s")),
                "action": value(detection, "video_action_model_fusion_score_tracklet_score"),
            }
            track_rows[row["track_key"]].append(row)
            if source in SOURCES and row["score"] >= 0.2:
                row["label"] = int(labels.get(flat_index, 0))
                row["is_validation"] = clip in validation_clips
                additions.append(row)
            flat_index += 1

    track_stats = {}
    neighbor_stats = {}
    for key, rows in track_rows.items():
        rows.sort(key=lambda item: item["frame"])
        scores = np.asarray([item["score"] for item in rows], dtype=np.float32)
        raw_scores = np.asarray([item["raw"] for item in rows], dtype=np.float32)
        actions = np.asarray([item["action"] for item in rows], dtype=np.float32)
        frames = np.asarray([item["frame"] for item in rows], dtype=np.int32)
        span = int(frames[-1] - frames[0] + 1) if len(frames) else 1
        track_stats[key] = (
            len(rows),
            span,
            float(scores.mean()),
            float(scores.max()),
            float(scores.std()),
            float(raw_scores.mean()),
            float(raw_scores.max()),
            float(actions.mean()),
            float(actions.max()),
            float(actions.std()),
        )
        for position, row in enumerate(rows):
            previous_gap = row["frame"] - rows[position - 1]["frame"] if position else span + 1
            next_gap = rows[position + 1]["frame"] - row["frame"] if position + 1 < len(rows) else span + 1
            neighbor_stats[row["flat_index"]] = (previous_gap, next_gap)

    features = []
    for row in additions:
        detection = row["detection"]
        width = max(value(detection, "w"), 1e-3)
        height = max(value(detection, "h"), 1e-3)
        x = value(detection, "x")
        y = value(detection, "y")
        area = width * height
        edge = min(x, 2448.0 - x, y, 2048.0 - y)
        count, span, score_mean, score_max, score_std, raw_mean, raw_max, action_mean, action_max, action_std = track_stats[row["track_key"]]
        previous_gap, next_gap = neighbor_stats[row["flat_index"]]
        source_flags = [float(row["source"] == source) for source in SOURCES]
        feature_row = [
                row["score"],
                row["raw"],
                row["action"],
                row["score"] - row["raw"],
                x / 2448.0,
                y / 2048.0,
                width / 2448.0,
                height / 2048.0,
                math.log1p(area),
                math.log(width / height),
                edge / 2048.0,
                math.log1p(count),
                math.log1p(span),
                count / max(span, 1),
                score_mean,
                score_max,
                score_std,
                raw_mean,
                raw_max,
                action_mean,
                action_max,
                action_std,
                math.log1p(previous_gap),
                math.log1p(next_gap),
                *source_flags,
            ]
        if include_flow_features:
            feature_row.extend(
                [
                value(detection, "flow_endpoint_iou"),
                math.log1p(value(detection, "flow_center_disagreement")),
                value(detection, "flow_width_ratio"),
                value(detection, "flow_height_ratio"),
                value(detection, "flow_area_ratio"),
                math.log1p(value(detection, "flow_gap")),
                value(detection, "flow_alpha"),
                abs(value(detection, "flow_alpha") - 0.5),
                value(detection, "flow_action_score"),
                value(detection, "flow_left_score"),
                value(detection, "flow_right_score"),
                min(value(detection, "flow_left_score"), value(detection, "flow_right_score")),
                value(detection, "appearance_left_ncc", -1.0),
                value(detection, "appearance_right_ncc", -1.0),
                value(detection, "appearance_mean_ncc", -1.0),
                value(detection, "appearance_min_ncc", -1.0),
                value(detection, "appearance_endpoint_ncc", -1.0),
                value(detection, "appearance_search_shift"),
                value(detection, "appearance_patch_std"),
                ]
            )
        features.append(feature_row)
    return additions, np.asarray(features, dtype=np.float32)


def make_model(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=8,
        max_features=0.8,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def oof_predictions(features: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    unique_groups = np.unique(groups)
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    probabilities = np.zeros(len(labels), dtype=np.float32)
    for train_indices, validation_indices in splitter.split(features, labels, groups):
        model = make_model(seed)
        model.fit(features[train_indices], labels[train_indices])
        probabilities[validation_indices] = model.predict_proba(features[validation_indices])[:, 1]
    return probabilities


def threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        selected = probabilities >= threshold
        true_positive = int(labels[selected].sum())
        false_positive = int(selected.sum() - true_positive)
        rows.append(
            {
                "threshold": threshold,
                "selected": int(selected.sum()),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "precision": true_positive / max(true_positive + false_positive, 1),
                "recall": true_positive / max(int(labels.sum()), 1),
            }
        )
    return rows


def merge_candidates(base_records: list[dict], additions: list[dict], probabilities: np.ndarray, threshold: float):
    output = copy.deepcopy(base_records)
    by_name = {str(record.get("img_name") or ""): record for record in output}
    counters = defaultdict(int)
    for row, probability in zip(additions, probabilities):
        if probability < threshold:
            continue
        name = str(row["record"].get("img_name") or "")
        record = by_name.get(name)
        if record is None:
            record = {"img_name": name, "detections": []}
            output.append(record)
            by_name[name] = record
            counters["created_missing_record"] += 1
        candidate = copy.deepcopy(row["detection"])
        candidate["action_bank_quality_probability"] = float(probability)
        detections = record.setdefault("detections", [])
        best = max(detections, key=lambda detection: box_iou(detection, candidate), default=None)
        best_iou = box_iou(best, candidate) if best is not None else 0.0
        if best is not None and best_iou >= 0.7:
            if value(candidate, "s") > value(best, "s"):
                best["s"] = value(candidate, "s")
                best["action_bank_quality_probability"] = float(probability)
                best["source"] = row["source"] + "_learned_promotion"
                counters["promoted_existing"] += 1
            else:
                counters["duplicate_unchanged"] += 1
            continue
        candidate["source"] = row["source"] + "_learned_gate"
        detections.append(candidate)
        counters["added_new"] += 1
    counters["remaining_detections"] = sum(len(record.get("detections") or []) for record in output)
    return output, dict(counters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-predictions", required=True, type=Path)
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--candidate-match-folder", required=True, type=Path)
    parser.add_argument("--base-match-folder", required=True, type=Path)
    parser.add_argument("--validation-predictions", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.15, 0.25, 0.4, 0.6])
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    candidate_records = load_pickle(args.candidate_predictions)
    base_records = load_pickle(args.base_predictions)
    validation_records = load_pickle(args.validation_predictions)
    validation_clips = {parse_image(str(record.get("img_name") or ""))[0].removeprefix("Clip_") for record in validation_records}
    labels = incremental_labels(find_match_csv(args.candidate_match_folder), find_match_csv(args.base_match_folder))
    additions, features = build_rows(candidate_records, labels, validation_clips)
    train_indices = np.asarray([index for index, row in enumerate(additions) if row["is_validation"]], dtype=np.int64)
    train_features = features[train_indices]
    train_labels = np.asarray([additions[index]["label"] for index in train_indices], dtype=np.int8)
    groups = np.asarray([additions[index]["clip"] for index in train_indices])
    oof = oof_predictions(train_features, train_labels, groups, args.seed)
    model = make_model(args.seed)
    model.fit(train_features, train_labels)
    probabilities = model.predict_proba(features)[:, 1]

    oof_precision, oof_recall, oof_thresholds = precision_recall_curve(train_labels, oof)
    best_f1_index = int(np.nanargmax(2 * oof_precision * oof_recall / np.maximum(oof_precision + oof_recall, 1e-9)))
    best_f1_threshold = float(oof_thresholds[min(best_f1_index, len(oof_thresholds) - 1)]) if len(oof_thresholds) else 0.5
    thresholds = sorted(set([*args.thresholds, best_f1_threshold]))
    summary = {
        "protocol": "part0 clip-group OOF learned incremental-recall gate; frozen full-AOT application",
        "compute": "CPU; training set is fewer than 2,000 candidate rows, so GPU launch overhead would dominate",
        "validation_clips": sorted(validation_clips),
        "training_rows": int(len(train_labels)),
        "training_positive": int(train_labels.sum()),
        "oof_average_precision": float(average_precision_score(train_labels, oof)),
        "oof_roc_auc": float(roc_auc_score(train_labels, oof)),
        "oof_best_f1_threshold": best_f1_threshold,
        "oof_thresholds": threshold_metrics(train_labels, oof, thresholds),
        "variants": [],
        "uses_full_test_labels_for_training": False,
        "uses_part0_labels_for_training": True,
    }

    for threshold in thresholds:
        tag = str(round(threshold, 6)).replace(".", "p")
        variant_root = args.out_root / f"gate_{tag}"
        prediction_dir = variant_root / "aotpredictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        output, counters = merge_candidates(base_records, additions, probabilities, threshold)
        output_path = prediction_dir / "predictions_split_0.pkl"
        with output_path.open("wb") as handle:
            pickle.dump(output, handle)
        variant = {
            "threshold": threshold,
            "output": str(output_path),
            "selected_candidates": int((probabilities >= threshold).sum()),
            "counters": counters,
        }
        summary["variants"].append(variant)
        (variant_root / "quality_gate_summary.json").write_text(json.dumps(variant, indent=2), encoding="utf-8")

    (args.out_root / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()




