import argparse
import json
import math
from pathlib import Path
from statistics import fmean, pstdev


def features(rows, short_frames, long_frames):
    rows = sorted(rows, key=lambda row: int(float(row.get("frame_id", 0) or 0)))
    frames = [int(float(row.get("frame_id", 0) or 0)) for row in rows]
    raw = [float(row.get("final_drone_score", row.get("objectness", 0)) or 0) for row in rows]
    count = len(rows)
    span = max(1, frames[-1] - frames[0] + 1)
    velocities = []
    for left, right in zip(rows, rows[1:]):
        lx1, ly1, lx2, ly2 = map(float, left["bbox"][:4])
        rx1, ry1, rx2, ry2 = map(float, right["bbox"][:4])
        delta = max(1, int(float(right["frame_id"])) - int(float(left["frame_id"])))
        left_scale = math.sqrt(max(1, lx2 - lx1) * max(1, ly2 - ly1))
        right_scale = math.sqrt(max(1, rx2 - rx1) * max(1, ry2 - ry1))
        scale = max(1.0, left_scale, right_scale)
        velocities.append((((rx1 + rx2 - lx1 - lx2) / 2) / delta / scale, ((ry1 + ry2 - ly1 - ly2) / 2) / delta / scale))
    speeds = [math.hypot(*velocity) for velocity in velocities]
    speed_mean = fmean(speeds) if speeds else 0.0
    speed_std = pstdev(speeds) if len(speeds) > 1 else 0.0
    acceleration = [math.hypot(right[0] - left[0], right[1] - left[1]) for left, right in zip(velocities, velocities[1:])]
    return {
        "rows": count,
        "continuity": count / span,
        "short_support": 1 - math.exp(-count / max(1, short_frames)),
        "long_support": 1 - math.exp(-count / max(1, long_frames)),
        "speed_consistency": math.exp(-speed_std / max(0.05, speed_mean + 0.05)) if speeds else 0.5,
        "action_consistency": math.exp(-(fmean(acceleration) if acceleration else 0.0) / 0.75),
        "confidence_stability": math.exp(-(pstdev(raw) if len(raw) > 1 else 0.0) / 0.12),
        "confidence_evidence": 1 / (1 + math.exp(-max(-30, min(30, (fmean(raw) - 0.20) / 0.055)))),
    }


def action_score(values):
    score = 0.22 * values["short_support"] + 0.18 * values["long_support"]
    score += 0.20 * values["continuity"] + 0.15 * values["speed_consistency"]
    score += 0.10 * values["action_consistency"] + 0.10 * values["confidence_stability"]
    score += 0.05 * values["confidence_evidence"]
    return max(0.0, min(1.0, score * {1: 0.52, 2: 0.72, 3: 0.86}.get(values["rows"], 1.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracklet-jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--short-frames", type=int, default=10)
    parser.add_argument("--long-frames", type=int, default=30)
    parser.add_argument("--score-field", default="action_chunk_transfer_score")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores = []
    rows_total = 0
    with args.tracklet_jsonl.open(encoding="utf-8-sig") as source, args.out.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            meta = dict(item.get("meta") or {})
            rows = [dict(row) for row in item.get("rows") or []]
            if not rows:
                continue
            values = features(rows, args.short_frames, args.long_frames)
            score = action_score(values)
            output = {
                "seq": str(meta.get("seq") or rows[0].get("seq") or ""),
                "track_id": str(meta.get("track_id") or rows[0].get("track_id") or ""),
                "raw_track_id": str(meta.get("raw_track_id") or rows[0].get("raw_track_id") or ""),
                args.score_field: score,
                "score": score,
                "diagnostics": values,
            }
            target.write(json.dumps(output, ensure_ascii=False) + "\n")
            scores.append(score)
            rows_total += len(rows)
    summary = {"tracklet_jsonl": str(args.tracklet_jsonl), "out": str(args.out), "score_field": args.score_field}
    summary.update({"short_frames": args.short_frames, "long_frames": args.long_frames, "tracklets": len(scores), "rows": rows_total})
    summary.update({"score_min": min(scores), "score_mean": fmean(scores), "score_max": max(scores)})
    summary.update({"uses_aot_labels": False, "protocol": "fixed zero-shot dual-timescale action chunk transfer"})
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
