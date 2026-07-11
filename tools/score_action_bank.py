from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qstr_dronedet.tracking.action_bank import ActionBankConfig, DualTimeActionBankTransformer, action_token, build_action_bank, row_timestamp


def score_items(
    items: list[dict[str, Any]],
    model: DualTimeActionBankTransformer,
    config: ActionBankConfig,
    device: torch.device,
    batch_size: int,
) -> int:
    samples = []
    for item_index, item in enumerate(items):
        rows = sorted(
            list(item.get("rows") or []),
            key=lambda row: row_timestamp(row, config.fps_fallback, config.sequence_fps),
        )
        item["rows"] = rows
        for row_index in range(3, len(rows)):
            snapshot = build_action_bank(rows[:row_index], config=config)
            candidate = action_token(rows[row_index - 1], rows[row_index], config).values[[3, 4, 7, 8]].astype(np.float32)
            samples.append((item_index, row_index, snapshot, candidate))
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            short = torch.from_numpy(np.stack([sample[2].short_tokens for sample in batch])).to(device, non_blocking=True)
            short_mask = torch.from_numpy(np.stack([sample[2].short_mask for sample in batch])).to(device, non_blocking=True)
            long = torch.from_numpy(np.stack([sample[2].long_tokens for sample in batch])).to(device, non_blocking=True)
            long_mask = torch.from_numpy(np.stack([sample[2].long_mask for sample in batch])).to(device, non_blocking=True)
            motion_logits, future_motion, reliability = model(short, short_mask, long, long_mask)
            motion = torch.sigmoid(motion_logits)
            candidates = torch.from_numpy(np.stack([sample[3] for sample in batch])).to(device, non_blocking=True)
            error = torch.nn.functional.smooth_l1_loss(future_motion[:, 0], candidates, reduction="none").mean(dim=1)
            consistency = torch.exp(-8.0 * error)
            score = motion * (0.35 + 0.65 * consistency) * (0.5 + 0.5 * reliability)
            score_values = score.cpu().tolist()
            motion_values = motion.cpu().tolist()
            consistency_values = consistency.cpu().tolist()
            reliability_values = reliability.cpu().tolist()
            for offset, (item_index, row_index, _, _) in enumerate(batch):
                items[item_index]["rows"][row_index].update({
                    "action_bank_learned_score": float(score_values[offset]),
                    "action_bank_motion_probability": float(motion_values[offset]),
                    "action_bank_future_consistency": float(consistency_values[offset]),
                    "action_bank_reliability": float(reliability_values[offset]),
                })
    for item in items:
        values = [float(row["action_bank_learned_score"]) for row in item["rows"] if "action_bank_learned_score" in row]
        meta = dict(item.get("meta") or {})
        meta["action_bank_learned_score"] = float(np.mean(values)) if values else 0.0
        meta["num_action_bank_learned_windows"] = len(values)
        item["meta"] = meta
    return len(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description="Causally score candidates with a trained 1s/3s Action Bank checkpoint.")
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--chunk-tracklets", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    config = ActionBankConfig(**checkpoint["config"])
    model = DualTimeActionBankTransformer(short_tokens=config.short_tokens, long_tokens=config.long_tokens)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    input_path = Path(args.tracklets)
    total_tracklets = sum(1 for line in input_path.open("r", encoding="utf-8-sig") if line.strip())
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    tracklets_done = samples_done = 0
    chunk: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8-sig") as source, output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            chunk.append(json.loads(line))
            if len(chunk) < args.chunk_tracklets:
                continue
            samples_done += score_items(chunk, model, config, device, args.batch_size)
            for item in chunk:
                target.write(json.dumps(item, ensure_ascii=False) + "\n")
            tracklets_done += len(chunk)
            payload = {"kind": "action_bank_score_progress", "done": tracklets_done, "total": total_tracklets, "samples": samples_done, "device": str(device)}
            if device.type == "cuda":
                payload["cuda_memory_allocated_mb"] = round(torch.cuda.memory_allocated(device) / 1048576, 3)
            print(json.dumps(payload), flush=True)
            chunk.clear()
        if chunk:
            samples_done += score_items(chunk, model, config, device, args.batch_size)
            for item in chunk:
                target.write(json.dumps(item, ensure_ascii=False) + "\n")
            tracklets_done += len(chunk)
            payload = {"kind": "action_bank_score_progress", "done": tracklets_done, "total": total_tracklets, "samples": samples_done, "device": str(device)}
            if device.type == "cuda":
                payload["cuda_memory_allocated_mb"] = round(torch.cuda.memory_allocated(device) / 1048576, 3)
            print(json.dumps(payload), flush=True)

    print(json.dumps({"kind": "action_bank_score_done", "tracklets": tracklets_done, "samples": samples_done, "out": str(output), "device": str(device)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
