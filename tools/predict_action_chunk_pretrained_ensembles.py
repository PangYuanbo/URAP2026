from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

REPO = Path(__file__).resolve().parents[1]
for entry in (REPO, REPO / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from tools.action_chunk_candidate_context import candidate_context_features
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.train_action_chunk_bidir_full import load_aux
from tools.train_action_chunk_neighbor_full import dataset_arrays, load_neighbor


def load_models(model_dir: Path, pattern: str) -> list[xgb.XGBClassifier]:
    paths = sorted(model_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no models matching {pattern} under {model_dir}")
    models = []
    for model_path in paths:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        model.set_params(device="cuda")
        models.append(model)
    return models


def predict_ensemble(models: list[xgb.XGBClassifier], features: np.ndarray) -> np.ndarray:
    if not len(features):
        return np.zeros((0,), dtype=np.float32)
    predictions = [model.predict_proba(features)[:, 1].astype(np.float32) for model in models]
    return np.mean(np.stack(predictions), axis=0).astype(np.float32)


def write_sequence(target, locations: list[tuple[str, int]], scores: np.ndarray, field: str) -> int:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (image_id, prediction_index), score in zip(locations, scores):
        grouped[image_id].append((prediction_index, float(score)))
    rows_written = 0
    for image_id in sorted(grouped):
        sequence, frame_id, _ = image_key(image_id, 0)
        rows = [
            {"seq": sequence, "frame_id": frame_id, "prediction_index": index, field: score}
            for index, score in sorted(grouped[image_id])
        ]
        target.write(json.dumps({"meta": {"seq": sequence, "image_id": image_id}, "rows": rows}, separators=(",", ":")) + "\n")
        rows_written += len(rows)
    return rows_written


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the frozen NPS Action Bank ensembles to another dataset without tuning.")
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--forward-jsonl", type=Path, required=True)
    parser.add_argument("--backward-jsonl", type=Path, required=True)
    parser.add_argument("--neighbor-jsonl", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--expert-model-dir", type=Path, required=True)
    parser.add_argument("--out-base-jsonl", type=Path, required=True)
    parser.add_argument("--out-expert-jsonl", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    predictions = load_predictionsgt(args.predictionsgt_pkl)
    forward = load_aux(args.forward_jsonl)
    backward = load_aux(args.backward_jsonl)
    neighbor, neighbor_fields = load_neighbor(args.neighbor_jsonl)
    base_models = load_models(args.base_model_dir, "action_chunk_neighbor_without_*.ubj")
    expert_models = load_models(args.expert_model_dir, "action_chunk_multi_expert_without_*.ubj")
    grouped: dict[str, dict[str, object]] = defaultdict(dict)
    for image_id, item in predictions.items():
        sequence, _, _ = image_key(str(image_id), 0)
        grouped[sequence][str(image_id)] = item

    args.out_base_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_expert_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    sequence_summaries = []
    with args.out_base_jsonl.open("w", encoding="utf-8") as base_target, args.out_expert_jsonl.open("w", encoding="utf-8") as expert_target:
        for sequence, sequence_predictions in sorted(grouped.items()):
            base_features, _, _, locations, _ = dataset_arrays(sequence_predictions, forward, backward, neighbor, False)
            context_chunks = []
            for item in sequence_predictions.values():
                values = candidate_context_features(list(item.get("detections") or []))
                if len(values):
                    context_chunks.append(values)
            context = np.concatenate(context_chunks) if context_chunks else np.zeros((0, 23), dtype=np.float32)
            if len(context) != len(base_features):
                raise RuntimeError(f"candidate context alignment mismatch for {sequence}: {len(context)} != {len(base_features)}")
            expert_features = np.concatenate((base_features, context), axis=1)
            base_scores = predict_ensemble(base_models, base_features)
            expert_scores = predict_ensemble(expert_models, expert_features)
            base_rows = write_sequence(base_target, locations, base_scores, "action_chunk_neighbor_score")
            expert_rows = write_sequence(expert_target, locations, expert_scores, "action_chunk_multi_expert_score")
            if base_rows != expert_rows:
                raise RuntimeError(f"score row mismatch for {sequence}: {base_rows} != {expert_rows}")
            total_rows += base_rows
            sequence_summaries.append({"sequence": sequence, "rows": base_rows})
            print(json.dumps({"kind": "pretrained_action_chunk_sequence", "sequence": sequence, "rows": base_rows}), flush=True)

    summary = {
        "protocol": "frozen NPS V46/V52 Action Bank ensembles; no target-dataset fitting",
        "rows": total_rows,
        "base_models": len(base_models),
        "expert_models": len(expert_models),
        "neighbor_features": neighbor_fields,
        "sequences": sequence_summaries,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
