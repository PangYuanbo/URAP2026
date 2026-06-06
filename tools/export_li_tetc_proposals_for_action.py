from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torchvision.ops import nms
from tqdm import tqdm


def add_pt_pipeline_to_path(repo_root: Path) -> Path:
    pipeline = repo_root / "pt_pipeline"
    if not pipeline.exists():
        pipeline = repo_root
    sys.path.insert(0, str(pipeline))
    return pipeline


def yxyx_to_xyxy(box: tuple[int, int, int, int]) -> list[float]:
    y1, x1, y2, x2 = box
    return [float(x1), float(y1), float(x2), float(y2)]


def write_gt_csv(repo_root: Path, videos: list[int], out_csv: Path, max_frames: int = 0, frame_stride: int = 1, include_empty: bool = True, empty_stride: int = 10) -> dict[str, Any]:
    from uav_annotations import load_video_annotations  # type: ignore

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    gt_boxes = 0
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq", "frame_id", "x1", "y1", "x2", "y2", "video_path"])
        writer.writeheader()
        for video_id in videos:
            ann_path = repo_root / "Data" / "Annotation_update_180925" / f"Video_{video_id}_gt.txt"
            if not ann_path.exists():
                continue
            anns = load_video_annotations(ann_path)
            for ann in anns:
                frame_id = int(ann.frame_id)
                if max_frames and frame_id > max_frames:
                    break
                if frame_stride > 1 and ((frame_id - 1) % frame_stride != 0):
                    continue
                if not ann.boxes_yxyx and (not include_empty or frame_id % empty_stride != 0):
                    continue
                for box in ann.boxes_yxyx:
                    x1, y1, x2, y2 = yxyx_to_xyxy(box)
                    writer.writerow(
                        {
                            "seq": f"Clip_{video_id}",
                            "frame_id": frame_id,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "video_path": f"Clip_{video_id}/Clip_{video_id}_{frame_id:05d}.jpg",
                        }
                    )
                    rows += 1
                    gt_boxes += 1
    return {"gt_csv": str(out_csv), "gt_rows": rows, "gt_boxes": gt_boxes}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Li-TETC NPS scored proposals into Route-B diagnostics JSONL.")
    parser.add_argument("--repo-root", type=Path, default=Path("Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking"))
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--videos", type=int, nargs="*", default=list(range(41, 51)))
    parser.add_argument("--out-run-root", type=Path, required=True)
    parser.add_argument("--profile", default="hard_recovery")
    parser.add_argument("--diagnostics-name", default="diagnostics_raw.jsonl")
    parser.add_argument("--out-gt-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-dt-dir", type=Path, default=None)
    parser.add_argument("--score", type=float, default=0.02)
    parser.add_argument("--nms", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--include-empty", action="store_true", default=True)
    parser.add_argument("--empty-stride", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/frames"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-size", type=int, default=1080)
    parser.add_argument("--max-size", type=int, default=1920)
    parser.add_argument("--anchor-preset", choices=["default", "small", "tiny"], default="tiny")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    pipeline = add_pt_pipeline_to_path(repo_root)
    from modeling import FasterRcnnCfg, build_fasterrcnn  # type: ignore
    from uav_dataset import UavVideoFrameDataset, collate_fn  # type: ignore

    out_run_root = args.out_run_root.resolve()
    profile_root = out_run_root / args.profile
    profile_root.mkdir(parents=True, exist_ok=True)
    if args.out_dt_dir is not None:
        args.out_dt_dir.mkdir(parents=True, exist_ok=True)

    gt_summary = write_gt_csv(
        repo_root,
        list(args.videos),
        args.out_gt_csv.resolve(),
        max_frames=int(args.max_frames),
        frame_stride=max(1, int(args.frame_stride)),
        include_empty=bool(args.include_empty),
        empty_stride=max(1, int(args.empty_stride)),
    )

    ds = UavVideoFrameDataset(
        repo_root=repo_root,
        video_ids=list(args.videos),
        max_frames_per_video=int(args.max_frames),
        frame_stride=max(1, int(args.frame_stride)),
        include_empty=bool(args.include_empty),
        empty_stride=max(1, int(args.empty_stride)),
        cache_dir=args.cache_dir,
        augment=False,
    )
    if len(ds) == 0:
        raise RuntimeError("Li-TETC dataset is empty")

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt.resolve(), map_location="cpu")
    model = build_fasterrcnn(
        FasterRcnnCfg(weights=None, min_size=int(args.min_size), max_size=int(args.max_size), anchor_preset=args.anchor_preset)
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    handles: dict[str, Any] = {}
    dt_lines: dict[int, list[str]] = {}
    frames = 0
    proposal_rows = 0
    frames_with_proposals = 0
    per_video: dict[str, dict[str, int]] = {}

    try:
        with torch.no_grad():
            for imgs, targets in tqdm(dl, desc="export-li-tetc-proposals"):
                outs = model([img.to(device) for img in imgs])
                for out, target in zip(outs, targets):
                    frames += 1
                    video_id = int(target["video_id"])
                    frame_id = int(target["frame_id"])
                    seq = f"Clip_{video_id}"
                    seq_dir = profile_root / seq
                    seq_dir.mkdir(parents=True, exist_ok=True)
                    if seq not in handles:
                        handles[seq] = (seq_dir / args.diagnostics_name).open("w", encoding="utf-8")
                    per = per_video.setdefault(seq, {"frames": 0, "proposal_rows": 0})
                    per["frames"] += 1

                    boxes = out["boxes"].detach().cpu()
                    scores = out["scores"].detach().cpu()
                    keep = scores >= float(args.score)
                    boxes = boxes[keep]
                    scores = scores[keep]
                    if boxes.numel() > 0:
                        keep_nms = nms(boxes, scores, float(args.nms))
                        if args.max_detections > 0:
                            keep_nms = keep_nms[: int(args.max_detections)]
                        boxes = boxes[keep_nms]
                        scores = scores[keep_nms]

                    if int(boxes.shape[0]) > 0:
                        frames_with_proposals += 1
                    det_parts: list[str] = []
                    image_path = f"{seq}/{seq}_{frame_id:05d}.jpg"
                    for pred_index, (box_t, score_t) in enumerate(zip(boxes, scores)):
                        x1, y1, x2, y2 = [float(v) for v in box_t.tolist()]
                        score = float(score_t.item())
                        row = {
                            "seq": seq,
                            "frame_id": frame_id,
                            "bbox": [x1, y1, x2, y2],
                            "objectness": score,
                            "final_drone_score": score,
                            "score": score,
                            "source": "li_tetc_fasterrcnn_lowconf",
                            "class_id": 0,
                            "prediction_index": pred_index,
                            "image_path": image_path,
                            "frame_path": image_path,
                            "visible": True,
                            "image_width": int(imgs[0].shape[-1]),
                            "image_height": int(imgs[0].shape[-2]),
                            "predicted_class": "drone",
                            "final_probs": {"drone": score, "background": max(0.0, 1.0 - score)},
                        }
                        handles[seq].write(json.dumps(row, ensure_ascii=False) + "\n")
                        proposal_rows += 1
                        per["proposal_rows"] += 1
                        det_parts.append(f"({int(y1)}, {int(x1)}, {int(y2)}, {int(x2)}),")
                    dt_lines.setdefault(video_id, []).append(f"time_layer: {frame_id} detections: {' '.join(det_parts)}".rstrip())
    finally:
        for handle in handles.values():
            handle.close()

    if args.out_dt_dir is not None:
        for video_id, lines in dt_lines.items():
            (args.out_dt_dir / f"Video_{video_id}_dt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "kind": "li_tetc_scored_proposal_export",
        "repo_root": str(repo_root),
        "pt_pipeline": str(pipeline),
        "ckpt": str(args.ckpt.resolve()),
        "videos": list(args.videos),
        "out_run_root": str(out_run_root),
        "profile": args.profile,
        "diagnostics_name": args.diagnostics_name,
        "score": float(args.score),
        "nms": float(args.nms),
        "max_detections": int(args.max_detections),
        "frame_stride": int(args.frame_stride),
        "include_empty": bool(args.include_empty),
        "empty_stride": int(args.empty_stride),
        "frames": frames,
        "frames_with_proposals": frames_with_proposals,
        "proposal_rows": proposal_rows,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "per_video": per_video,
        **gt_summary,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
