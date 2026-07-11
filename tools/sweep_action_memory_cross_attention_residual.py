from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO, REPO / "tools"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_action_chunk_bounded_residual import bounded_aux
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data, image_key, parse_csv_floats
from tools.sweep_tvd_predictionsgt_score_fusion import fuse_score, load_row_scores


def evaluate_config(
    data: dict[str, Any],
    base: dict,
    incumbent: dict,
    cross_attention: dict,
    incumbent_cap: float,
    incumbent_weight: float,
    new_mode: str,
    new_cap: float,
    new_weight: float,
    alpha: float,
    tvd_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    output = {}
    changed_from_incumbent = 0
    for image_id, item in data.items():
        detections = []
        for index, row in enumerate(item.get("detections") or []):
            key = image_key(str(image_id), index)
            raw = float(row.get("score", 0.0))
            base_score = float(base.get(key, raw))
            incumbent_score = float(incumbent.get(key, base_score))
            incumbent_aux = bounded_aux(base_score, incumbent_score, "symmetric", incumbent_cap, incumbent_weight)
            attention_score = float(cross_attention.get(key, incumbent_aux))
            guarded_aux = bounded_aux(incumbent_aux, attention_score, new_mode, new_cap, new_weight)
            changed_from_incumbent += int(abs(guarded_aux - incumbent_aux) > 1e-12)
            new_row = dict(row)
            new_row["score"] = fuse_score(raw, guarded_aux, alpha, "geom-mix")
            detections.append(new_row)
        output[image_id] = {"labels": item.get("labels", []), "detections": detections}
    return {"changed_from_incumbent": changed_from_incumbent, **evaluate_data(output, tvd_root, out_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Guarded Cross-Attention residual on top of the strict causal Action Chunk champion.")
    parser.add_argument("--tvd-root", type=Path, required=True)
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--base-field", required=True)
    parser.add_argument("--incumbent-jsonl", type=Path, required=True)
    parser.add_argument("--incumbent-field", required=True)
    parser.add_argument("--cross-attention-jsonl", type=Path, required=True)
    parser.add_argument("--cross-attention-field", required=True)
    parser.add_argument("--incumbent-cap", type=float, default=0.5)
    parser.add_argument("--incumbent-weight", type=float, default=0.5)
    parser.add_argument("--modes", default="boost-only,symmetric")
    parser.add_argument("--caps", default=".05,.1,.25,.5")
    parser.add_argument("--weights", default="0,.1,.25,.5")
    parser.add_argument("--alphas", default=".2")
    parser.add_argument("--fixed-config-json", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    data = load_predictionsgt(args.predictionsgt_pkl)
    base, base_summary = load_row_scores(args.base_jsonl, args.base_field, 1)
    incumbent, incumbent_summary = load_row_scores(args.incumbent_jsonl, args.incumbent_field, 1)
    cross_attention, cross_attention_summary = load_row_scores(args.cross_attention_jsonl, args.cross_attention_field, 1)
    if args.fixed_config_json:
        best = json.loads(args.fixed_config_json.read_text(encoding="utf8"))["best"]
        configs = [(str(best["new_mode"]), float(best["new_cap"]), float(best["new_weight"]), float(best["alpha"]))]
    else:
        modes = [value.strip() for value in args.modes.split(",") if value.strip()]
        configs = [
            (mode, cap, weight, alpha)
            for mode in modes
            for cap in parse_csv_floats(args.caps)
            for weight in parse_csv_floats(args.weights)
            for alpha in parse_csv_floats(args.alphas)
        ]
    rows = []
    for mode, cap, weight, alpha in configs:
        metrics = evaluate_config(
            data, base, incumbent, cross_attention,
            args.incumbent_cap, args.incumbent_weight,
            mode, cap, weight, alpha, args.tvd_root, args.out_json.parent,
        )
        row = {"new_mode": mode, "new_cap": cap, "new_weight": weight, "alpha": alpha, **metrics}
        rows.append(row)
        print(json.dumps(row), flush=True)
    best = max(rows, key=lambda row: float(row["map50"]))
    summary = {
        "guardrail": "new_weight=0 exactly reproduces the incumbent strict-causal V60 scoring path",
        "base": base_summary,
        "incumbent": incumbent_summary,
        "cross_attention": cross_attention_summary,
        "best": best,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
