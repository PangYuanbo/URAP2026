from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO, REPO / "tools"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_action_chunk_bounded_residual import bounded_aux
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data, image_key
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a fixed bounded Action Memory residual directly to an already-rescored predictionsgt file.")
    parser.add_argument("--tvd-root", type=Path, required=True)
    parser.add_argument("--base-pkl", type=Path, required=True)
    parser.add_argument("--residual-jsonl", type=Path, required=True)
    parser.add_argument("--residual-field", required=True)
    parser.add_argument("--mode", choices=("boost-only", "suppress-only", "symmetric"), required=True)
    parser.add_argument("--cap", type=float, required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    data = load_predictionsgt(args.base_pkl)
    residual, residual_summary = load_row_scores(args.residual_jsonl, args.residual_field, 1)
    output = {}
    changed = 0
    for image_id, item in data.items():
        detections = []
        for index, row in enumerate(item.get("detections") or []):
            key = image_key(str(image_id), index)
            base_score = float(row.get("score", 0.0))
            residual_score = float(residual.get(key, base_score))
            new_score = bounded_aux(base_score, residual_score, args.mode, args.cap, args.weight)
            changed += int(abs(new_score - base_score) > 1e-12)
            new_row = dict(row)
            new_row["score"] = new_score
            detections.append(new_row)
        output[image_id] = {"labels": item.get("labels", []), "detections": detections}
    metrics = evaluate_data(output, args.tvd_root, args.out_json.parent)
    summary = {"mode": args.mode, "cap": args.cap, "weight": args.weight, "changed_rows": changed, "residual": residual_summary, **metrics}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
