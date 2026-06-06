from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_tvd_coco_pkl_on_li_tetc import Box, Counters, add_fn, add_tp, iou, load_gt


def load_diagnostics(run_root: Path, profile: str, diagnostics_name: str, videos: set[int]) -> dict[tuple[int, int], list[tuple[float, Box]]]:
    preds: dict[tuple[int, int], list[tuple[float, Box]]] = {}
    profile_root = run_root / profile
    for seq_dir in sorted(path for path in profile_root.glob("Clip_*") if path.is_dir()):
        try:
            video_id = int(seq_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if video_id not in videos:
            continue
        diag = seq_dir / diagnostics_name
        if not diag.exists():
            continue
        with diag.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                bbox = row.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                frame_id = int(float(row.get("frame_id", 0) or 0))
                score = float(max(row.get("objectness", 0.0), row.get("final_drone_score", 0.0), row.get("score", 0.0)))
                x1, y1, x2, y2 = [float(v) for v in bbox]
                if x2 > x1 and y2 > y1:
                    preds.setdefault((video_id, frame_id), []).append((score, Box(x1=x1, y1=y1, x2=x2, y2=y2)))
    return preds


def size_bin(box: Box) -> str:
    area = box.area()
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def evaluate(
    gt: dict[tuple[int, int], list[Box]],
    preds: dict[tuple[int, int], list[tuple[float, Box]]],
    scores: list[float],
    iou_thr: float,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    frames = len(gt)
    for score_thr in scores:
        c = Counters()
        for key, gt_boxes in gt.items():
            frame_preds = [(s, b) for s, b in preds.get(key, []) if s >= score_thr]
            frame_preds.sort(key=lambda item: item[0], reverse=True)
            matched = [False] * len(gt_boxes)
            for _score, pred_box in frame_preds:
                best_iou = -1.0
                best_idx = -1
                for idx, gt_box in enumerate(gt_boxes):
                    if matched[idx]:
                        continue
                    value = iou(pred_box, gt_box)
                    if value > best_iou:
                        best_iou = value
                        best_idx = idx
                if best_iou >= iou_thr and best_idx >= 0:
                    matched[best_idx] = True
                    add_tp(c, gt_boxes[best_idx])
                else:
                    c.fp += 1
            for idx, gt_box in enumerate(gt_boxes):
                if not matched[idx]:
                    add_fn(c, gt_box)
        results[f"{score_thr:g}"] = c.as_dict(frames=frames)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Route-B diagnostics proposals on Li-TETC time_layer GT.")
    parser.add_argument("--repo-root", type=Path, default=Path("Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--profile", default="hard_recovery")
    parser.add_argument("--diagnostics-name", default="diagnostics_raw.jsonl")
    parser.add_argument("--videos", type=int, nargs="*", default=list(range(41, 51)))
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--scores", type=float, nargs="*", default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--match-pt-pipeline-sampling", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--empty-stride", type=int, default=10)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    gt = load_gt(repo_root, args.videos)
    if args.max_frames > 0:
        gt = {key: boxes for key, boxes in gt.items() if key[1] <= int(args.max_frames)}
    if args.match_pt_pipeline_sampling:
        frame_stride = max(1, int(args.frame_stride))
        empty_stride = max(1, int(args.empty_stride))
        gt = {
            key: boxes
            for key, boxes in gt.items()
            if ((key[1] - 1) % frame_stride == 0) and (boxes or (key[1] % empty_stride == 0))
        }
    preds = load_diagnostics(args.run_root.resolve(), args.profile, args.diagnostics_name, set(args.videos))
    summary: dict[str, Any] = {
        "repo_root": str(repo_root),
        "run_root": str(args.run_root.resolve()),
        "profile": args.profile,
        "diagnostics_name": args.diagnostics_name,
        "videos": list(args.videos),
        "iou": float(args.iou),
        "frames": len(gt),
        "gt_boxes": int(sum(len(v) for v in gt.values())),
        "prediction_frames": len(preds),
        "prediction_boxes": int(sum(len(v) for v in preds.values())),
        "scores": evaluate(gt, preds, args.scores, float(args.iou)),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
