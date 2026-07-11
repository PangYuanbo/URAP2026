from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path
from typing import Any


IMAGE_RE = re.compile(r"^(Clip_\d+)_(\d+)$")


def parse_image_id(image_id: str) -> tuple[str, int]:
    match = IMAGE_RE.match(str(image_id))
    if match is None:
        return str(image_id), 0
    return match.group(1), int(match.group(2))


def image_path_for(frame_root: Path | None, image_id: str) -> str:
    if frame_root is None:
        return f"{image_id}.png"
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = frame_root / f"{image_id}{suffix}"
        if candidate.is_file():
            return str(candidate.resolve())
    return str((frame_root / f"{image_id}.png").resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Export TransVisDrone predictionsgt pkl to Route-B diagnostics and GT CSV.")
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--out-run-root", type=Path, required=True)
    parser.add_argument("--out-gt-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, default=None)
    parser.add_argument("--profile", default="hard_recovery")
    parser.add_argument("--diagnostics-name", default="diagnostics_raw.jsonl")
    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1280)
    args = parser.parse_args()

    with args.predictionsgt_pkl.resolve().open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{args.predictionsgt_pkl}: expected dict, got {type(data)}")

    profile_root = args.out_run_root.resolve() / args.profile
    profile_root.mkdir(parents=True, exist_ok=True)
    args.out_gt_csv.parent.mkdir(parents=True, exist_ok=True)
    handles: dict[str, Any] = {}
    frames = 0
    detections = 0
    labels = 0
    per_seq: dict[str, dict[str, int]] = {}
    try:
        with args.out_gt_csv.open("w", encoding="utf-8", newline="") as gt_f:
            writer = csv.DictWriter(gt_f, fieldnames=["seq", "frame_id", "x1", "y1", "x2", "y2", "video_path"])
            writer.writeheader()
            for image_id in sorted(data):
                frames += 1
                seq, frame_id = parse_image_id(str(image_id))
                per = per_seq.setdefault(seq, {"frames": 0, "detections": 0, "labels": 0})
                per["frames"] += 1
                seq_dir = profile_root / seq
                seq_dir.mkdir(parents=True, exist_ok=True)
                if seq not in handles:
                    handles[seq] = (seq_dir / args.diagnostics_name).open("w", encoding="utf-8")
                img_path = image_path_for(args.frame_root, str(image_id))
                gt_video_path = f"{seq}/{image_id}.png"
                item = data[image_id]
                for pred_index, row in enumerate(item.get("detections", [])):
                    bbox = row.get("bbox")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    score = float(row.get("score", 0.0))
                    out = {
                        "seq": seq,
                        "frame_id": frame_id,
                        "bbox": [x1, y1, x2, y2],
                        "objectness": score,
                        "final_drone_score": score,
                        "score": score,
                        "source": "transvisdrone_predictionsgt",
                        "class_id": int(row.get("category_id", 0)),
                        "prediction_index": pred_index,
                        "image_path": img_path,
                        "frame_path": img_path,
                        "visible": True,
                        "image_width": int(args.image_width),
                        "image_height": int(args.image_height),
                        "predicted_class": "drone",
                        "final_probs": {"drone": score, "background": max(0.0, 1.0 - score)},
                    }
                    handles[seq].write(json.dumps(out, ensure_ascii=False) + "\n")
                    detections += 1
                    per["detections"] += 1
                for label in item.get("labels", []):
                    bbox = label.get("bbox")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    writer.writerow({"seq": seq, "frame_id": frame_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "video_path": gt_video_path})
                    labels += 1
                    per["labels"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    summary = {
        "predictionsgt_pkl": str(args.predictionsgt_pkl.resolve()),
        "out_run_root": str(args.out_run_root.resolve()),
        "out_gt_csv": str(args.out_gt_csv.resolve()),
        "profile": args.profile,
        "diagnostics_name": args.diagnostics_name,
        "frame_root": str(args.frame_root.resolve()) if args.frame_root is not None else None,
        "frames": frames,
        "detections": detections,
        "labels": labels,
        "per_seq": per_seq,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
