import argparse
import json
from collections import defaultdict
from pathlib import Path


def frame_bounds(item):
    frames = [int(float(row.get("frame_id", 0) or 0)) for row in item.get("rows") or []]
    return min(frames), max(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracklets", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--score-field", default="vatd_score")
    parser.add_argument("--max-gap", type=int, default=30)
    parser.add_argument("--short-max-rows", type=int, default=3)
    parser.add_argument("--strong-min-rows", type=int, default=10)
    parser.add_argument("--strong-min-score", type=float, default=0.7)
    parser.add_argument("--neutral-score", type=float, default=0.5)
    args = parser.parse_args()

    items = [json.loads(line) for line in args.tracklets.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    groups = defaultdict(list)
    for index, item in enumerate(items):
        meta = item.get("meta") or {}
        rows = item.get("rows") or []
        if not rows:
            continue
        seq = str(meta.get("seq") or rows[0].get("seq") or "")
        raw_track = str(meta.get("raw_track_id") or rows[0].get("raw_track_id") or meta.get("track_id") or "")
        first, last = frame_bounds(item)
        groups[(seq, raw_track)].append((first, last, index))

    protected = 0
    protected_rows = 0
    examples = []
    for segments in groups.values():
        segments.sort()
        for position, (first, last, index) in enumerate(segments):
            item = items[index]
            meta = item.get("meta") or {}
            rows = item.get("rows") or []
            score = float(meta.get(args.score_field, 0.0) or 0.0)
            if len(rows) > args.short_max_rows or score >= args.neutral_score:
                continue
            neighbors = []
            if position > 0:
                neighbors.append(segments[position - 1])
            if position + 1 < len(segments):
                neighbors.append(segments[position + 1])
            bridge = None
            for neighbor_first, neighbor_last, neighbor_index in neighbors:
                gap = neighbor_first - last if neighbor_first > last else first - neighbor_last
                neighbor = items[neighbor_index]
                neighbor_meta = neighbor.get("meta") or {}
                neighbor_rows = neighbor.get("rows") or []
                neighbor_score = float(neighbor_meta.get(args.score_field, 0.0) or 0.0)
                if 0 <= gap <= args.max_gap and (len(neighbor_rows) >= args.strong_min_rows or neighbor_score >= args.strong_min_score):
                    bridge = (gap, neighbor_meta.get("track_id"), len(neighbor_rows), neighbor_score)
                    break
            if bridge is None:
                continue
            old_score = score
            meta[args.score_field] = max(score, args.neutral_score)
            meta["action_chunk_bridge_protected"] = True
            meta["action_chunk_bridge_from_score"] = old_score
            for row in rows:
                row[args.score_field] = meta[args.score_field]
                row["action_chunk_bridge_protected"] = True
            protected += 1
            protected_rows += len(rows)
            if len(examples) < 25:
                examples.append({"seq": meta.get("seq"), "track_id": meta.get("track_id"), "rows": len(rows), "frames": [first, last], "old_score": old_score, "new_score": meta[args.score_field], "neighbor": bridge})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as target:
        for item in items:
            target.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {"input": str(args.tracklets), "out": str(args.out), "groups": len(groups), "tracklets": len(items), "protected_tracklets": protected, "protected_rows": protected_rows, "uses_labels": False, "rule": "protect short segments that bridge to a strong segment on the same raw track within max_gap", "parameters": vars(args), "examples": examples}
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
