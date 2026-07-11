from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qstr_dronedet.pipelines.temporal_recovery import MotionMemoryTrack, TemporalRecoveryConfig, score_candidates_with_motion_memory
from qstr_dronedet.types import DetectionCandidate


FRAME_RE = re.compile(r"^(?P<clip>.+)_(?P<frame>\d+)$")


def parse_key(key: str) -> tuple[str, int]:
    match = FRAME_RE.match(key)
    if not match:
        return key, 0
    return match.group("clip"), int(match.group("frame"))


def row_to_candidate(row: dict[str, Any]) -> DetectionCandidate | None:
    bbox = row.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None
    return DetectionCandidate((x1, y1, x2, y2), float(row.get("score", 0.0)), "tvd_detector", extra={"category_id": int(row.get("category_id", 0)), "raw_row": row})


def candidate_to_row(cand: DetectionCandidate, no_score_decay: bool = True) -> dict[str, Any]:
    raw = deepcopy(cand.extra.get("raw_row", {}))
    raw_score = float(cand.extra.get("raw_objectness", cand.objectness))
    final_score = max(raw_score, float(cand.objectness)) if no_score_decay else float(cand.objectness)
    raw["bbox"] = [float(v) for v in cand.bbox_xyxy]
    raw["score"] = final_score
    raw["category_id"] = int(cand.extra.get("category_id", raw.get("category_id", 0)))
    raw["temporal_recovery"] = {
        "raw_objectness": raw_score,
        "motion_memory_score": float(cand.extra.get("motion_memory_score", cand.motion_score)),
        "temporal_score": float(cand.extra.get("temporal_score", cand.objectness)),
        "final_score": final_score,
        "no_score_decay": no_score_decay,
    }
    return raw


def rescore(data: dict[str, Any], cfg: TemporalRecoveryConfig, keep_top_k: int = 300, no_score_decay: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    out = deepcopy(data)
    by_clip: dict[str, list[tuple[int, str]]] = {}
    for key in data:
        clip, frame = parse_key(str(key))
        by_clip.setdefault(clip, []).append((frame, str(key)))

    frames = 0
    detections_in = 0
    detections_out = 0
    boosted = 0
    for clip, entries in sorted(by_clip.items()):
        memory: MotionMemoryTrack | None = None
        for _, key in sorted(entries):
            item = out[key]
            cands = [row_to_candidate(row) for row in item.get("detections", [])]
            cands = [cand for cand in cands if cand is not None]
            detections_in += len(cands)
            scored = score_candidates_with_motion_memory(cands, memory, (10_000, 10_000, 3), cfg)
            if scored:
                best = scored[0]
                if memory is None:
                    memory = MotionMemoryTrack(best.bbox_xyxy, score=best.objectness, history=[best.bbox_xyxy])
                else:
                    memory.update(best, (10_000, 10_000, 3))
                boosted += sum(1 for cand in scored if float(cand.extra.get("temporal_score", cand.objectness)) > float(cand.extra.get("raw_objectness", cand.objectness)))
            elif memory is not None:
                memory.mark_miss()
                if memory.misses > cfg.miss_patience:
                    memory = None
            selected = scored[:keep_top_k] if keep_top_k > 0 else scored
            item["detections"] = [candidate_to_row(cand, no_score_decay=no_score_decay) for cand in selected]
            detections_out += len(selected)
            frames += 1

    summary = {
        "clips": len(by_clip),
        "frames": frames,
        "detections_in": detections_in,
        "detections_out": detections_out,
        "boosted_candidates": boosted,
        "config": {
            "top_k": cfg.top_k,
            "max_center_distance": cfg.max_center_distance,
            "motion_weight": cfg.motion_weight,
            "detector_weight": cfg.detector_weight,
            "memory_iou_bonus": cfg.memory_iou_bonus,
            "miss_patience": cfg.miss_patience,
        },
        "keep_top_k": keep_top_k,
        "no_score_decay": no_score_decay,
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore TransVisDrone predictionsgt pkl with detector-first temporal recovery prior.")
    parser.add_argument("--input-pkl", type=Path, required=True)
    parser.add_argument("--out-pkl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--keep-top-k", type=int, default=300)
    parser.add_argument("--motion-weight", type=float, default=0.25)
    parser.add_argument("--detector-weight", type=float, default=0.75)
    parser.add_argument("--max-center-distance", type=float, default=96.0)
    parser.add_argument("--memory-iou-bonus", type=float, default=0.15)
    parser.add_argument("--allow-score-decay", action="store_true")
    args = parser.parse_args()

    with args.input_pkl.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{args.input_pkl}: expected dict predictionsgt pkl, got {type(data)}")
    cfg = TemporalRecoveryConfig(
        top_k=args.top_k,
        motion_weight=args.motion_weight,
        detector_weight=args.detector_weight,
        max_center_distance=args.max_center_distance,
        memory_iou_bonus=args.memory_iou_bonus,
    )
    rescored, summary = rescore(data, cfg, keep_top_k=args.keep_top_k, no_score_decay=not args.allow_score_decay)
    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pkl.open("wb") as f:
        pickle.dump(rescored, f)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps({"input_pkl": str(args.input_pkl), "out_pkl": str(args.out_pkl), **summary}, indent=2), encoding="utf-8")
    print(json.dumps({"input_pkl": str(args.input_pkl), "out_pkl": str(args.out_pkl), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
