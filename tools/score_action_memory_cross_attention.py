from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO, REPO / "tools"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from qstr_dronedet.tracking.action_memory_attention import ActionMemoryCrossAttentionRanker
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.train_action_bank_motion_token_listwise import (
    FEATURE_NAMES,
    ACTION_QUERY_NAMES,
    LONG_TOKEN_COUNT,
    SHORT_TOKEN_COUNT,
    TOKEN_DIM,
    TOKEN_FEATURE_NAMES,
    dataset_arrays,
    load_auxiliary,
    load_native,
    predict,
    write_score_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score candidates with a frozen current-Action Query to historical Action Memory Cross-Attention model.")
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--aux-jsonl", type=Path, required=True)
    parser.add_argument("--native-jsonl", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-scores", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--score-field", default="action_memory_cross_attention_score")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") not in {"action_memory_cross_attention", "action_memory_cross_attention_action_only"}:
        raise ValueError(f"expected Action Memory Cross-Attention checkpoint, got {checkpoint.get('model_type')!r}")
    if list(checkpoint.get("features") or []) != FEATURE_NAMES:
        raise ValueError("checkpoint feature schema does not match the current Action Memory scorer")
    auxiliary, sequence_sizes = load_auxiliary(args.aux_jsonl)
    native = load_native(args.native_jsonl)
    features, _, _, _, locations = dataset_arrays(
        load_predictionsgt(args.predictionsgt_pkl), auxiliary, sequence_sizes, native, False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionMemoryCrossAttentionRanker(
        features.shape[1],
        token_start=FEATURE_NAMES.index(TOKEN_FEATURE_NAMES[0]),
        token_dim=TOKEN_DIM,
        short_tokens=SHORT_TOKEN_COUNT,
        long_tokens=LONG_TOKEN_COUNT,
        hidden=int(checkpoint["hidden"]),
        heads=int(checkpoint.get("attention_heads", 4)),
        memory_layers=int(checkpoint.get("memory_layers", 1)),
        query_indices=[FEATURE_NAMES.index(name) for name in ACTION_QUERY_NAMES] if checkpoint.get("action_only_query") else None,
        use_static_head=not bool(checkpoint.get("action_only_query")),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    scores = predict(model, features, mean, std, args.batch_size, device)
    write_score_jsonl(args.out_scores, scores, locations, args.score_field)
    summary = {
        "model": str(args.model),
        "model_type": checkpoint["model_type"],
        "predictionsgt_pkl": str(args.predictionsgt_pkl),
        "aux_jsonl": str(args.aux_jsonl),
        "rows": len(scores),
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "score_std": float(scores.std()) if len(scores) else 0.0,
        "device": str(device),
        "architecture_modified_for_dataset": False,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
