from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO, REPO / "tools"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_action_chunk_context_temporal_gate import gates, logit, sigmoid
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data, image_key, parse_csv_floats
from tools.sweep_tvd_predictionsgt_score_fusion import fuse_score, load_row_scores


def v71_score(raw: float, neighbor: float, context: float) -> float:
    return sigmoid(0.8 * logit(raw) + 0.1 * logit(neighbor) + 0.1 * logit(context))


def v53_score(raw: float, neighbor: float, expert: float) -> float:
    auxiliary = math.sqrt(max(1e-9, neighbor) * max(1e-9, expert))
    return fuse_score(raw, auxiliary, 0.4, "geom-mix")


def evaluate(data, neighbor_scores, context_scores, expert_scores, gate, route_strength: float, tvd_root: Path, out_dir: Path):
    output = {}
    for image_id, item in data.items():
        enabled = gate.get(str(image_id), False)
        rows = []
        for index, row in enumerate(item.get("detections") or []):
            key = image_key(str(image_id), index)
            raw = float(row.get("score", 0.0))
            neighbor = float(neighbor_scores.get(key, raw))
            context = float(context_scores.get(key, raw))
            expert = float(expert_scores.get(key, raw))
            context_score = v71_score(raw, neighbor, context)
            if enabled:
                temporal_score = v53_score(raw, neighbor, expert)
                score = sigmoid((1.0 - route_strength) * logit(context_score) + route_strength * logit(temporal_score))
            else:
                score = context_score
            updated = dict(row)
            updated["score"] = score
            rows.append(updated)
        output[image_id] = {"labels": item.get("labels", []), "detections": rows}
    return evaluate_data(output, tvd_root, out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Route between context and temporal Action Chunk experts.")
    parser.add_argument("--tvd-root", type=Path, required=True)
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--v46", type=Path, required=True)
    parser.add_argument("--v51", type=Path, required=True)
    parser.add_argument("--v52", type=Path, required=True)
    parser.add_argument("--fps-json", type=Path, required=True)
    parser.add_argument("--thresholds", default=".2,.3,.4")
    parser.add_argument("--windows", default="1,3")
    parser.add_argument("--fractions", default=".5,.75")
    parser.add_argument("--route-strengths", default=".25,.5,.75,1")
    parser.add_argument("--fixed-config-json", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    data = load_predictionsgt(args.predictionsgt_pkl)
    v46, _ = load_row_scores(args.v46, "action_chunk_neighbor_score", 1)
    v51, _ = load_row_scores(args.v51, "action_chunk_candidate_context_score", 1)
    v52, _ = load_row_scores(args.v52, "action_chunk_multi_expert_score", 1)
    fps_map = json.loads(args.fps_json.read_text(encoding="utf8"))
    if args.fixed_config_json:
        best = json.loads(args.fixed_config_json.read_text(encoding="utf8"))["best"]
        configs = [(best["threshold"], best["window_seconds"], best["min_fraction"], best["route_strength"])]
    else:
        configs = [(threshold, window, fraction, route_strength) for threshold in parse_csv_floats(args.thresholds) for window in parse_csv_floats(args.windows) for fraction in parse_csv_floats(args.fractions) for route_strength in parse_csv_floats(args.route_strengths)]
    gate_cache = {}
    rows = []
    for threshold, window, fraction, route_strength in configs:
        gate_key = (threshold, window, fraction)
        gate = gate_cache.setdefault(gate_key, gates(data, threshold, window, fraction, fps_map))
        metrics = evaluate(data, v46, v51, v52, gate, route_strength, args.tvd_root, args.out_json.parent)
        record = {"threshold": threshold, "window_seconds": window, "min_fraction": fraction, "route_strength": route_strength, "gated_images": sum(gate.values()), **metrics}
        rows.append(record)
        print(json.dumps(record), flush=True)
    best = max(rows, key=lambda row: row["map50"])
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"best": best, "rows": rows}, indent=2), encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
