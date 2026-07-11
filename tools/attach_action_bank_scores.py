from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qstr_dronedet.tracking.action_bank import ActionBankConfig, attach_action_bank_scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Attach real-time 1s/3s Action Bank scores to VATD tracklets.")
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--short-seconds", type=float, default=1.0)
    parser.add_argument("--long-seconds", type=float, default=3.0)
    parser.add_argument("--short-tokens", type=int, default=12)
    parser.add_argument("--long-tokens", type=int, default=18)
    parser.add_argument("--fps-fallback", type=float, default=29.97)
    parser.add_argument("--max-gap-seconds", type=float, default=0.5)
    parser.add_argument("--sequence-fps-json")
    parser.add_argument("--min-history-actions", type=int, default=2)
    args = parser.parse_args()
    sequence_fps = json.loads(Path(args.sequence_fps_json).read_text(encoding="utf-8-sig")) if args.sequence_fps_json else {}
    config = ActionBankConfig(
        short_seconds=args.short_seconds,
        long_seconds=args.long_seconds,
        short_tokens=args.short_tokens,
        long_tokens=args.long_tokens,
        fps_fallback=args.fps_fallback,
        max_gap_seconds=args.max_gap_seconds,
        sequence_fps={str(key): float(value) for key, value in sequence_fps.items()},
    )
    summary = attach_action_bank_scores(args.tracklets, args.out, config=config, min_history_actions=args.min_history_actions)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
