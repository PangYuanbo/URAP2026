from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qstr_dronedet.tracking.action_bank import ActionBankConfig, ActionBankWindowDataset, DualTimeActionBankTransformer


def main() -> int:
    parser = argparse.ArgumentParser(description="Train causal 1s/3s Action Bank with future-only supervision.")
    parser.add_argument("--train-tracklets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--short-tokens", type=int, default=12)
    parser.add_argument("--long-tokens", type=int, default=18)
    parser.add_argument("--fps-fallback", type=float, default=29.97)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence-fps-json")
    parser.add_argument("--positive-weight", type=float, default=0.0, help="0 selects automatic negative/positive window ratio")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    sequence_fps = json.loads(Path(args.sequence_fps_json).read_text(encoding="utf-8")) if args.sequence_fps_json else {}
    config = ActionBankConfig(short_tokens=args.short_tokens, long_tokens=args.long_tokens, fps_fallback=args.fps_fallback, sequence_fps={str(key): float(value) for key, value in sequence_fps.items()})
    dataset = ActionBankWindowDataset(args.train_tracklets, config=config, max_samples=args.max_samples)
    if not dataset:
        raise RuntimeError("no Action Bank training windows found")
    dataset.materialize(args.cache_dir)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    model = DualTimeActionBankTransformer(short_tokens=config.short_tokens, long_tokens=config.long_tokens).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    positive_windows, negative_windows = dataset.label_counts()
    positive_weight = float(args.positive_weight) if args.positive_weight > 0 else negative_windows / max(1, positive_windows)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "motion": 0.0, "future": 0.0, "reliability": 0.0}
        samples = 0
        for batch in loader:
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            motion_logits, future_motion, reliability = model(batch["short_tokens"], batch["short_mask"], batch["long_tokens"], batch["long_mask"])
            motion_loss = bce(motion_logits, batch["target_motion"])
            future_per_sample = torch.nn.functional.smooth_l1_loss(future_motion, batch["future_motion"], reduction="none").mean(dim=(1, 2))
            future_weights = batch["future_reliable"] * batch["target_motion"]
            future_loss = (future_per_sample * future_weights).sum() / future_weights.sum().clamp_min(1.0)
            reliability_target = batch["future_reliable"] * batch["target_motion"]
            reliability_loss = torch.nn.functional.binary_cross_entropy(reliability, reliability_target)
            loss = motion_loss + 0.75 * future_loss + 0.15 * reliability_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = int(batch["target_motion"].shape[0])
            samples += count
            for name, value in (("loss", loss), ("motion", motion_loss), ("future", future_loss), ("reliability", reliability_loss)):
                totals[name] += float(value.detach()) * count
        row = {"epoch": epoch, "epochs": args.epochs, "samples": samples, "device": str(device), **{name: value / max(1, samples) for name, value in totals.items()}}
        if device.type == "cuda":
            row["cuda_memory_allocated_mb"] = round(torch.cuda.memory_allocated(device) / 1048576, 3)
        history.append(row)
        print(json.dumps({"kind": "action_bank_train_progress", **row}), flush=True)
        torch.save({"model": model.state_dict(), "config": config.__dict__, "epoch": epoch, "history": history}, output)
    print(json.dumps({"kind": "action_bank_train_done", "out": str(output), "epochs": args.epochs, "samples": len(dataset), "positive_windows": positive_windows, "negative_windows": negative_windows, "positive_weight": positive_weight, "device": str(device)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
