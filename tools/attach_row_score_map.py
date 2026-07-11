from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tracklets", type=Path, required=True)
    parser.add_argument("--score-map-pkl", type=Path, required=True)
    parser.add_argument("--output-tracklets", type=Path, required=True)
    parser.add_argument("--score-field", required=True)
    args = parser.parse_args()

    with args.score_map_pkl.open("rb") as handle:
        score_map = pickle.load(handle)
    args.output_tracklets.parent.mkdir(parents=True, exist_ok=True)
    tracklets = 0
    rows = 0
    matched = 0
    with args.input_tracklets.open("r", encoding="utf-8-sig") as source, args.output_tracklets.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            meta = item.get("meta") or {}
            for row in item.get("rows") or []:
                key = (
                    str(row.get("seq") or meta.get("seq") or ""),
                    int(float(row.get("frame_id", 0))),
                    int(float(row.get("prediction_index", 0))),
                )
                value = score_map.get(key)
                if value is not None:
                    row[args.score_field] = float(value)
                    matched += 1
                rows += 1
            target.write(json.dumps(item, separators=(",", ":")) + "\n")
            tracklets += 1
    print(json.dumps({"tracklets": tracklets, "rows": rows, "matched": matched, "score_field": args.score_field, "output": str(args.output_tracklets)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
