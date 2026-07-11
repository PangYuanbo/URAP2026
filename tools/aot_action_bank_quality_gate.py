import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path


IMAGE_RE = re.compile(r"^(?P<seq>Clip_\d+)_(?P<frame>\d+)\.png$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--tracklets", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--promotion-min-memory", type=float, default=0.6)
    parser.add_argument("--promotion-min-own-action", type=float, default=0.2)
    parser.add_argument("--promotion-min-raw-score", type=float, default=0.1)
    parser.add_argument("--suppress-max-own-action", type=float, default=0.15)
    parser.add_argument("--suppress-max-score", type=float, default=0.3)
    parser.add_argument("--drop-interpolation", action="store_true")
    parser.add_argument("--drop-edge-extrapolation", action="store_true")
    args = parser.parse_args()

    memory_score = defaultdict(float)
    for line in args.tracklets.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows = item.get("rows") or []
        meta = item.get("meta") or {}
        if not rows:
            continue
        seq = str(meta.get("seq") or rows[0].get("seq") or "")
        raw_track = int(str(meta.get("raw_track_id") or rows[0].get("raw_track_id") or meta.get("track_id")))
        memory_score[(seq, raw_track)] = max(memory_score[(seq, raw_track)], float(meta.get("vatd_score", 0) or 0))

    records = pickle.load(args.predictions.open("rb"))
    counters = defaultdict(int)
    for record in records:
        match = IMAGE_RE.match(str(record.get("img_name") or ""))
        seq = match.group("seq") if match else ""
        kept = []
        for detection in record.get("detections") or []:
            source = str(detection.get("source") or "base")
            score = float(detection.get("s", 0) or 0)
            has_own_action = "video_action_model_fusion_score_tracklet_score" in detection
            own_action = float(detection.get("video_action_model_fusion_score_tracklet_score", 0) or 0)
            raw_score = float(detection.get("tracklet_rescore_raw_s", score) or 0)
            track_id = int(detection.get("track_id", -1))
            memory = memory_score[(seq, track_id)]

            if source == "action_bank_edge_extrapolation" and args.drop_edge_extrapolation:
                counters["dropped_edge_extrapolation"] += 1
                continue
            if source == "action_bank_cross_segment_interpolation" and args.drop_interpolation:
                counters["dropped_interpolation"] += 1
                continue
            if source == "action_bank_track_memory_promotion":
                if memory < args.promotion_min_memory or own_action < args.promotion_min_own_action or raw_score < args.promotion_min_raw_score:
                    counters["dropped_promotions"] += 1
                    continue
                counters["kept_promotions"] += 1
            elif has_own_action and own_action <= args.suppress_max_own_action and score <= args.suppress_max_score:
                counters["suppressed_base"] += 1
                continue
            kept.append(detection)
        record["detections"] = kept

    prediction_dir = args.out_dir / "aotpredictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    output = prediction_dir / "predictions_split_0.pkl"
    pickle.dump(records, output.open("wb"))
    summary = {
        "input": str(args.predictions),
        "output": str(output),
        "parameters": vars(args) | {"predictions": str(args.predictions), "tracklets": str(args.tracklets), "out_dir": str(args.out_dir)},
        "counters": dict(counters),
        "remaining_detections": sum(len(record.get("detections") or []) for record in records),
        "uses_labels": False,
    }
    (args.out_dir / "quality_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

