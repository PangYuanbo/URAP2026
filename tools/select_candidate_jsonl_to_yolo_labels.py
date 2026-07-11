from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = min(max(0.0, float(x1)), float(width))
    x2 = min(max(0.0, float(x2)), float(width))
    y1 = min(max(0.0, float(y1)), float(height))
    y2 = min(max(0.0, float(y2)), float(height))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return (x1 + bw / 2.0) / width, (y1 + bh / 2.0) / height, bw / width, bh / height


def _source_parts(source: str) -> set[str]:
    return {part for part in str(source).split("+") if part}


def _has_detector(source: str) -> bool:
    parts = _source_parts(source)
    return any(part in {"yolo", "yolo_tile", "yolov5_dual", "zoom_redetect", "crop_yolo"} or part.startswith("yolo") for part in parts)


def _has_support(source: str) -> bool:
    parts = _source_parts(source)
    return any(part in {"gray_ncc", "ncc", "tracker", "motion"} for part in parts)


def _is_pure_detector(source: str) -> bool:
    return _has_detector(source) and not _has_support(source)


def _candidate_score(candidate: dict[str, Any], args: argparse.Namespace) -> float:
    raw = float(candidate.get("raw_objectness") or 0.0)
    score = float(candidate.get("score") or 0.0)
    motion = float(candidate.get("motion_memory_score") or 0.0)
    source = str(candidate.get("source") or "")
    rank = int(candidate.get("rank") or 0)
    final = (
        args.raw_weight * raw
        + args.score_weight * score
        + args.motion_weight * motion
        + (args.detector_bonus if _has_detector(source) else 0.0)
        - (args.support_penalty if _has_support(source) else 0.0)
        - args.rank_penalty * rank
    )
    if args.clip_score:
        final = min(1.0, max(0.0, final))
    return float(final)


def _select_candidate(candidates: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any] | None, float | None, str]:
    if not candidates:
        return None, None, "no_candidates"

    pool = list(candidates)
    if args.max_rank is not None:
        pool = [cand for cand in pool if int(cand.get("rank") or 0) <= args.max_rank]

    if args.detector_only:
        filtered = [cand for cand in pool if _is_pure_detector(str(cand.get("source") or ""))]
        if filtered:
            pool = filtered
        else:
            return None, None, "detector_only_no_candidate"

    if args.prefer_detector_min_raw is not None:
        detector_pool = [
            cand
            for cand in pool
            if _is_pure_detector(str(cand.get("source") or ""))
            and float(cand.get("raw_objectness") or 0.0) >= args.prefer_detector_min_raw
        ]
        if detector_pool:
            pool = detector_pool

    if not pool:
        return None, None, "filtered_empty"

    selected = max(pool, key=lambda cand: (_candidate_score(cand, args), -int(cand.get("rank") or 0)))
    return selected, _candidate_score(selected, args), "selected"


def select_candidate_jsonl_to_yolo_labels(
    candidate_jsonl: Path,
    out_label_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_label_dir.mkdir(parents=True, exist_ok=True)
    frames = 0
    written = 0
    skipped = 0
    reasons: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    with candidate_jsonl.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            item = json.loads(line)
            frames += 1
            image_path = Path(str(item["image_path"]))
            width = int(item.get("width") or args.image_width)
            height = int(item.get("height") or args.image_height)
            selected, final_score, reason = _select_candidate(list(item.get("candidates") or []), args)
            reasons[reason] = reasons.get(reason, 0) + 1
            label_path = out_label_dir / f"{image_path.stem}.txt"
            if selected is None or final_score is None or final_score < args.min_score:
                label_path.write_text("", encoding="utf-8")
                skipped += 1
                continue
            cx, cy, bw, bh = _xyxy_to_yolo(
                float(selected["x1"]),
                float(selected["y1"]),
                float(selected["x2"]),
                float(selected["y2"]),
                width,
                height,
            )
            if bw <= 0.0 or bh <= 0.0:
                label_path.write_text("", encoding="utf-8")
                skipped += 1
                reasons["invalid_box"] = reasons.get("invalid_box", 0) + 1
                continue
            label_path.write_text(f"{args.class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f} {final_score:.8f}\n", encoding="utf-8")
            written += 1
            source = str(selected.get("source") or "")
            source_counts[source] = source_counts.get(source, 0) + 1

    summary = {
        "candidate_jsonl": str(candidate_jsonl),
        "out_label_dir": str(out_label_dir),
        "frames": frames,
        "written": written,
        "skipped": skipped,
        "selection_reasons": reasons,
        "selected_sources": source_counts,
        "params": {
            "raw_weight": args.raw_weight,
            "score_weight": args.score_weight,
            "motion_weight": args.motion_weight,
            "detector_bonus": args.detector_bonus,
            "support_penalty": args.support_penalty,
            "rank_penalty": args.rank_penalty,
            "prefer_detector_min_raw": args.prefer_detector_min_raw,
            "detector_only": args.detector_only,
            "max_rank": args.max_rank,
            "min_score": args.min_score,
            "clip_score": args.clip_score,
        },
    }
    (out_label_dir.parent / f"{out_label_dir.name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Select one YOLO label per frame from temporal-recovery candidate JSONL.")
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--out-label-dir", type=Path, required=True)
    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--raw-weight", type=float, default=1.0)
    parser.add_argument("--score-weight", type=float, default=0.0)
    parser.add_argument("--motion-weight", type=float, default=0.0)
    parser.add_argument("--detector-bonus", type=float, default=0.0)
    parser.add_argument("--support-penalty", type=float, default=0.0)
    parser.add_argument("--rank-penalty", type=float, default=0.0)
    parser.add_argument("--prefer-detector-min-raw", type=float)
    parser.add_argument("--detector-only", action="store_true")
    parser.add_argument("--max-rank", type=int)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--clip-score", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    summary = select_candidate_jsonl_to_yolo_labels(args.candidate_jsonl, args.out_label_dir, args)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
