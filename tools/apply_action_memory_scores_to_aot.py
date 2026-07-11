from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO, REPO / "tools"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.sweep_action_chunk_bounded_residual import bounded_aux
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.sweep_tvd_predictionsgt_score_fusion import fuse_score, load_row_scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply fixed bounded Action Memory row scores to AOT list-format predictions.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--score-field", required=True)
    parser.add_argument("--mode", choices=("boost-only", "suppress-only", "symmetric"))
    parser.add_argument("--cap", type=float)
    parser.add_argument("--weight", type=float)
    parser.add_argument("--fusion-mode", choices=("replace", "linear-mix", "logit-add", "logit-mix", "geom-mix", "fp-suppress", "tp-boost"))
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.fusion_mode) == bool(args.mode):
        parser.error("choose exactly one of --mode or --fusion-mode")
    if args.mode and (args.cap is None or args.weight is None):
        parser.error("bounded residual mode requires --cap and --weight")
    if args.fusion_mode and args.alpha is None:
        parser.error("fusion mode requires --alpha")
    with args.input.open("rb") as source:
        records = pickle.load(source)
    scores, score_summary = load_row_scores(args.scores, args.score_field, 1)
    changed = 0
    for record in records:
        image_id = Path(str(record["img_name"])).stem
        for index, detection in enumerate(record.get("detections") or []):
            base = float(detection.get("s", 0.0))
            residual = float(scores.get(image_key(image_id, index), base))
            updated = fuse_score(base, residual, args.alpha, args.fusion_mode) if args.fusion_mode else bounded_aux(base, residual, args.mode, args.cap, args.weight)
            changed += int(abs(updated - base) > 1e-12)
            detection["s"] = updated
            detection["action_memory_cross_attention_score"] = residual
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "predictions_split_0.pkl"
    with output.open("wb") as target:
        pickle.dump(records, target, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {"input": str(args.input), "output": str(output), "mode": args.mode, "cap": args.cap, "weight": args.weight, "fusion_mode": args.fusion_mode, "alpha": args.alpha, "changed": changed, "scores": score_summary}
    (args.out_dir.parent / "rescore_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
