import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import yolomg_parallax_robust_difference as pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate near-texture suppression and annotated target preservation."
    )
    parser.add_argument("--grass-video", type=Path, required=True)
    parser.add_argument("--grass-frame", type=int, required=True)
    parser.add_argument("--target-video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-samples", type=int, default=24)
    return parser.parse_args()


def pipeline_args():
    return SimpleNamespace(
        local_grid_cols=36,
        local_grid_rows=21,
        local_neighbors=18,
        local_radius=190.0,
        local_max_shift=18.0,
        local_apply_threshold=0.18,
        track_scale=0.5,
        adaptive_strength=1.2,
        residual_floor=2.0,
        residual_gain=1.2,
        color_floor=32.0,
        color_gamma=0.72,
    )


def read_triplet(capture, reference_frame):
    capture.set(cv2.CAP_PROP_POS_FRAMES, reference_frame - 1)
    frames = []
    for _ in range(3):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read triplet around frame {reference_frame}")
        frames.append(pipeline.resize_1080p(frame))
    return frames


def motion_maps(frames, args):
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    old_difference, old_valid, _ = pipeline.compensated_difference(
        gray[0], gray[1], gray[2], args, local=False
    )
    new_difference, new_valid, new_stats = pipeline.compensated_difference(
        gray[0], gray[1], gray[2], args, local=True
    )
    old_map = np.uint8(np.clip(old_difference, 0.0, 255.0))
    old_map[old_valid == 0] = 0
    new_map = pipeline.adaptive_residual(
        new_difference,
        new_valid,
        args.adaptive_strength,
        args.residual_floor,
        args.residual_gain,
    )
    return old_map, new_map, new_stats


def values_summary(values):
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "bright10_fraction": float(np.mean(values >= 10)),
        "bright16_fraction": float(np.mean(values >= 16)),
    }


def grass_validation(video, frame_id, args, output_dir):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open grass video: {video}")
    frames = read_triplet(capture, frame_id)
    capture.release()
    old_map, new_map, stats = motion_maps(frames, args)
    height, width = old_map.shape
    near = np.s_[
        int(height * 0.72) : int(height * 0.96),
        int(width * 0.05) : int(width * 0.95),
    ]
    far = np.s_[
        int(height * 0.50) : int(height * 0.68),
        int(width * 0.05) : int(width * 0.95),
    ]
    old_near = values_summary(old_map[near])
    new_near = values_summary(new_map[near])
    old_far = values_summary(old_map[far])
    new_far = values_summary(new_map[far])
    old_color = pipeline.colorize_difference(old_map, 1.0, 0.55)
    new_color = pipeline.colorize_difference(
        new_map, args.color_floor, args.color_gamma
    )
    cv2.imwrite(str(output_dir / "grass_rgb.jpg"), frames[1])
    cv2.imwrite(str(output_dir / "grass_old.jpg"), old_color)
    cv2.imwrite(str(output_dir / "grass_new.jpg"), new_color)
    cv2.imwrite(
        str(output_dir / "grass_old_vs_new.jpg"),
        pipeline.comparison_frame(old_color, new_color),
    )
    return {
        "video": str(video),
        "reference_frame": frame_id,
        "old_near": old_near,
        "new_near": new_near,
        "old_far": old_far,
        "new_far": new_far,
        "near_mean_reduction": 1.0 - new_near["mean"] / max(old_near["mean"], 1e-6),
        "near_bright10_reduction": (
            1.0
            - new_near["bright10_fraction"]
            / max(old_near["bright10_fraction"], 1e-9)
        ),
        "alignment_stats": stats,
    }


def box_masks(shape, box, pad):
    height, width = shape
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, int(np.floor(x1))))
    y1 = max(0, min(height, int(np.floor(y1))))
    x2 = max(0, min(width, int(np.ceil(x2))))
    y2 = max(0, min(height, int(np.ceil(y2))))
    target = np.zeros(shape, dtype=bool)
    target[y1:y2, x1:x2] = True
    outer = np.zeros(shape, dtype=bool)
    outer[
        max(0, y1 - pad) : min(height, y2 + pad),
        max(0, x1 - pad) : min(width, x2 + pad),
    ] = True
    return target, ~outer


def target_validation(video, annotations_path, args, sample_count):
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    annotations = [
        item
        for item in data["annotations"]
        if item.get("boxes")
        and 0 < int(item["frame_id"]) < int(data["total_frames"]) - 1
    ]
    if not annotations:
        raise RuntimeError("No usable target annotations")
    indices = np.linspace(
        0, len(annotations) - 1, min(sample_count, len(annotations)), dtype=int
    )
    selected = [annotations[index] for index in indices]
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open target video: {video}")
    rows = []
    for index, annotation in enumerate(selected, 1):
        frame_id = int(annotation["frame_id"])
        frames = read_triplet(capture, frame_id)
        old_map, new_map, _ = motion_maps(frames, args)
        box = annotation["boxes"][0]
        scale_x = old_map.shape[1] / float(annotation["width"])
        scale_y = old_map.shape[0] / float(annotation["height"])
        scaled_box = (
            float(box["x1"]) * scale_x,
            float(box["y1"]) * scale_y,
            float(box["x2"]) * scale_x,
            float(box["y2"]) * scale_y,
        )
        target, background = box_masks(old_map.shape, scaled_box, 12)
        old_target = old_map[target]
        new_target = new_map[target]
        old_background = old_map[background]
        new_background = new_map[background]
        old_p99 = float(np.percentile(old_target, 99.0))
        new_p99 = float(np.percentile(new_target, 99.0))
        rows.append(
            {
                "frame_id": frame_id,
                "old_target_p99": old_p99,
                "new_target_p99": new_p99,
                "target_p99_ratio": new_p99 / max(old_p99, 1.0),
                "old_background_bright10": float(np.mean(old_background >= 10)),
                "new_background_bright10": float(np.mean(new_background >= 10)),
            }
        )
        print(f"[target] {index}/{len(selected)} frame={frame_id}", flush=True)
    capture.release()
    return {
        "video": str(video),
        "annotations": str(annotations_path),
        "frames_evaluated": len(rows),
        "frame_ids": [row["frame_id"] for row in rows],
        "mean_old_target_p99": float(
            np.mean([row["old_target_p99"] for row in rows])
        ),
        "mean_new_target_p99": float(
            np.mean([row["new_target_p99"] for row in rows])
        ),
        "median_target_p99_ratio": float(
            np.median([row["target_p99_ratio"] for row in rows])
        ),
        "target_p99_retained_80pct_rate": float(
            np.mean([row["target_p99_ratio"] >= 0.8 for row in rows])
        ),
        "mean_old_background_bright10": float(
            np.mean([row["old_background_bright10"] for row in rows])
        ),
        "mean_new_background_bright10": float(
            np.mean([row["new_background_bright10"] for row in rows])
        ),
        "rows": rows,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = pipeline_args()
    grass = grass_validation(
        args.grass_video, args.grass_frame, settings, args.output_dir
    )
    target = target_validation(
        args.target_video, args.annotations, settings, args.target_samples
    )
    report = {
        "method": (
            "global H/RANSAC + optional robust local background warp + "
            "local residual noise calibration"
        ),
        "settings": vars(settings),
        "grass": grass,
        "target": target,
    }
    report_path = args.output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] {report_path}", flush=True)


if __name__ == "__main__":
    main()
