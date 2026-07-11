from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO, REPO / "tools"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt, process_batch, row_to_det, row_to_label
from tools.sweep_action_chunk_context_temporal_gate import gates, logit, sigmoid
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.sweep_tvd_predictionsgt_score_fusion import fuse_score, load_row_scores


def score_v53(raw: float, neighbor: float, expert: float, enabled: bool) -> float:
    auxiliary = math.sqrt(max(1e-9, neighbor) * max(1e-9, expert)) if enabled else neighbor
    return fuse_score(raw, auxiliary, 0.4, "geom-mix")


def score_v71(raw: float, neighbor: float, context: float) -> float:
    return sigmoid(0.8 * logit(raw) + 0.1 * logit(neighbor) + 0.1 * logit(context))


def metrics(correct, confidence, predicted_classes, target_classes, ap_per_class):
    if not target_classes:
        return {"map50": 0.0, "map5095": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    correct_array = np.concatenate(correct) if correct else np.zeros((0, 10), dtype=bool)
    confidence_array = np.asarray(confidence, dtype=np.float64)
    predicted_array = np.asarray(predicted_classes, dtype=np.float32)
    target_array = np.asarray(target_classes, dtype=np.float32)
    precision, recall, ap, f1, _ = ap_per_class(correct_array, confidence_array, predicted_array, target_array, plot=False, names={0: "drone"})
    return {"map50": float(ap[:, 0].mean()), "map5095": float(ap.mean(1).mean()), "precision": float(precision.mean()), "recall": float(recall.mean()), "f1": float(f1.mean())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tvd-root", type=Path, required=True)
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--v46", type=Path, required=True)
    parser.add_argument("--v51", type=Path, required=True)
    parser.add_argument("--v52", type=Path, required=True)
    parser.add_argument("--fps-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tvd_root.resolve()))
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid
    from utils.metrics import ap_per_class

    data = load_predictionsgt(args.predictionsgt_pkl)
    v46, _ = load_row_scores(args.v46, "action_chunk_neighbor_score", 1)
    v51, _ = load_row_scores(args.v51, "action_chunk_candidate_context_score", 1)
    v52, _ = load_row_scores(args.v52, "action_chunk_multi_expert_score", 1)
    fps_map = json.loads(args.fps_json.read_text(encoding="utf8"))
    gate = gates(data, 0.3, 3.0, 0.75, fps_map)
    iouv = torch.linspace(0.5, 0.95, 10)
    grouped = defaultdict(lambda: {"correct": [], "raw": [], "v53": [], "v71": [], "pc": [], "tc": [], "images": 0, "detections": 0, "labels": 0, "gated": 0})
    for image_id in sorted(data):
        item = data[image_id]
        sequence, _, _ = image_key(str(image_id), 0)
        bucket = grouped[sequence]
        bucket["images"] += 1
        bucket["gated"] += int(gate.get(str(image_id), False))
        detection_rows = []
        indices = []
        for index, row in enumerate(item.get("detections") or []):
            value = row_to_det(row)
            if value is not None:
                detection_rows.append(value)
                indices.append(index)
        label_rows = [value for row in item.get("labels") or [] if (value := row_to_label(row)) is not None]
        detections = torch.tensor(detection_rows, dtype=torch.float32) if detection_rows else torch.zeros((0, 6))
        labels = torch.tensor(label_rows, dtype=torch.float32) if label_rows else torch.zeros((0, 5))
        bucket["correct"].append(process_batch(detections, labels, iouv).numpy())
        bucket["tc"].extend(labels[:, 0].tolist() if labels.numel() else [])
        bucket["detections"] += len(detection_rows)
        bucket["labels"] += len(label_rows)
        enabled = gate.get(str(image_id), False)
        for index, detection in zip(indices, detection_rows):
            key = image_key(str(image_id), index)
            raw = float(detection[4])
            neighbor = float(v46.get(key, raw))
            context = float(v51.get(key, raw))
            expert = float(v52.get(key, raw))
            bucket["raw"].append(raw)
            bucket["v53"].append(score_v53(raw, neighbor, expert, enabled))
            bucket["v71"].append(score_v71(raw, neighbor, context))
            bucket["pc"].append(float(detection[5]))
    rows = []
    for sequence, bucket in sorted(grouped.items()):
        raw_metrics = metrics(bucket["correct"], bucket["raw"], bucket["pc"], bucket["tc"], ap_per_class)
        v53_metrics = metrics(bucket["correct"], bucket["v53"], bucket["pc"], bucket["tc"], ap_per_class)
        v71_metrics = metrics(bucket["correct"], bucket["v71"], bucket["pc"], bucket["tc"], ap_per_class)
        rows.append({"sequence": sequence, "fps": float(fps_map.get(sequence, 30.0)), "images": bucket["images"], "labels": bucket["labels"], "detections": bucket["detections"], "detections_per_image": bucket["detections"] / max(1, bucket["images"]), "positive_fraction": bucket["labels"] / max(1, bucket["images"]), "gated_fraction": bucket["gated"] / max(1, bucket["images"]), "raw_map50": raw_metrics["map50"], "v53_map50": v53_metrics["map50"], "v71_map50": v71_metrics["map50"], "v53_minus_v71": v53_metrics["map50"] - v71_metrics["map50"]})
    summary = {"rows": rows, "v53_wins": sum(row["v53_minus_v71"] > 0 for row in rows), "v71_wins": sum(row["v53_minus_v71"] < 0 for row in rows)}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({"sequences": len(rows), "v53_wins": summary["v53_wins"], "v71_wins": summary["v71_wins"], "out": str(args.out_json)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
