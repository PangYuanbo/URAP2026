from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def sigmoid(value: float, temperature: float) -> float:
    scaled = max(-60.0, min(60.0, value / temperature))
    return 1.0 / (1.0 + math.exp(-scaled))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-field", default="lambda_rank_score")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tracklets = 0
    rows = 0
    with args.input.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            meta = item.get("meta") or {}
            for temperature in args.temperatures:
                suffix = str(temperature).replace(".", "p")
                value = meta.get(args.source_field)
                if value is not None:
                    meta[f"{args.source_field}_sigmoid_t{suffix}"] = sigmoid(float(value), temperature)
            for row in item.get("rows") or []:
                value = row.get(args.source_field)
                if value is None:
                    continue
                for temperature in args.temperatures:
                    suffix = str(temperature).replace(".", "p")
                    row[f"{args.source_field}_sigmoid_t{suffix}"] = sigmoid(float(value), temperature)
                rows += 1
            target.write(json.dumps(item, separators=(",", ":")) + "\n")
            tracklets += 1
    print(json.dumps({"tracklets": tracklets, "rows": rows, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
