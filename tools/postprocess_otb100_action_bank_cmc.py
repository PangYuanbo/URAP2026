from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qstr_dronedet.camera_motion import estimate_background_homography
from qstr_dronedet.tracking.online_action_bank import OnlineActionTrack
from tools.evaluate_samurai_otb100 import (
    blend_xyxy,
    load_groundtruth,
    sequence_metrics,
    summarize,
    valid_box,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def load_boxes(path: Path) -> list[list[float]]:
    return [
        [float(value) for value in line.split(",")]
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply optical-flow camera compensation and a causal time Action Bank to existing OTB100 SAMURAI boxes.")
    parser.add_argument("--source-dir", type=Path, default=Path(r"D:\URAP_vatd_rank_results\otb100_samurai_cmc_timebank_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\URAP_vatd_rank_results\otb100_samurai_optflow_action_bank_v3"))
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\URAP_local_datasets\OTB100"))
    parser.add_argument("--metadata", type=Path, default=REPO / "data_templates/otb100_sequences.json")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--short-seconds", type=float, default=1.0)
    parser.add_argument("--long-seconds", type=float, default=3.0)
    parser.add_argument("--min-motion-score", type=float, default=0.42)
    parser.add_argument("--blend-weight", type=float, default=0.20)
    parser.add_argument("--camera-max-size", type=int, default=512)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--target", type=float, default=70.0)
    args = parser.parse_args()

    metadata: dict[str, dict[str, Any]] = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    selected = list(metadata.items())[: args.max_sequences]
    result_dir = args.output_dir / "sequence_results"
    prediction_dir = args.output_dir / "predictions"
    result_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    available = [item for item in selected if (args.source_dir / "predictions" / f"{item[0]}_raw_samurai.txt").is_file()]
    for sequence_index, (name, attributes) in enumerate(available, start=1):
        result_path = result_dir / f"{name}.json"
        if result_path.is_file():
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if int(cached.get("frames", 0)) > 0:
                results.append(cached)
                continue
        base = str(attributes.get("base", name))
        sequence_root = args.dataset_root / base
        groundtruth_path = sequence_root / str(attributes.get("groundtruth", "groundtruth_rect.txt"))
        start = int(attributes.get("start", 1))
        stop = int(attributes["stop"]) if attributes.get("stop") is not None else None
        groundtruth = load_groundtruth(groundtruth_path, start, stop)
        raw_predictions = load_boxes(args.source_dir / "predictions" / f"{name}_raw_samurai.txt")
        if len(raw_predictions) != len(groundtruth):
            raise RuntimeError(f"{name}: raw prediction length {len(raw_predictions)} != groundtruth {len(groundtruth)}")

        initial_box = xywh_to_xyxy(groundtruth[0])
        bank = OnlineActionTrack(0, 0.0, initial_box, 1.0)
        output_predictions = [list(groundtruth[0])]
        identity = np.eye(3, dtype=np.float64)
        previous_image = cv2.imread(str(args.source_dir / "frame_views" / name / "00000000.jpg"), cv2.IMREAD_COLOR)
        if previous_image is None:
            raise FileNotFoundError(args.source_dir / "frame_views" / name / "00000000.jpg")
        valid_camera_frames = 0

        for frame_index in range(1, len(raw_predictions)):
            current_path = args.source_dir / "frame_views" / name / f"{frame_index:08d}.jpg"
            current_image = cv2.imread(str(current_path), cv2.IMREAD_COLOR)
            if current_image is None:
                raise FileNotFoundError(current_path)
            camera = estimate_background_homography(previous_image, current_image, max_size=args.camera_max_size)
            camera_transform = camera.matrix if camera.valid else identity
            camera_validity = camera.inlier_ratio if camera.valid else 0.0
            valid_camera_frames += int(camera.valid)
            previous_image = current_image

            timestamp = frame_index / args.fps
            predicted_box = bank.predict(timestamp, camera_transform, args.short_seconds, args.long_seconds)
            raw_prediction = raw_predictions[frame_index]
            if valid_box(raw_prediction):
                raw_box = xywh_to_xyxy(raw_prediction)
                motion = bank.score_candidate(raw_box, timestamp, camera_transform, camera_validity, args.short_seconds, args.long_seconds)
                if motion.score >= args.min_motion_score:
                    agreement = float(np.clip((motion.score - 0.5) * 2.0, 0.0, 1.0))
                    weight = args.blend_weight * agreement * (0.75 + 0.25 * camera_validity)
                    output_box = blend_xyxy(raw_box, predicted_box, weight)
                    bank.update(frame_index, timestamp, raw_box, 1.0, motion.score, camera_transform, args.long_seconds)
                else:
                    output_box = raw_box
            else:
                output_box = predicted_box
            output_predictions.append(xyxy_to_xywh(output_box))

        raw_metrics = sequence_metrics(raw_predictions, groundtruth)
        metrics = {
            "sequence": name,
            **sequence_metrics(output_predictions, groundtruth),
            "raw_samurai": raw_metrics,
            "camera_valid_fraction": valid_camera_frames / max(1, len(raw_predictions) - 1),
        }
        results.append(metrics)
        result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with (prediction_dir / f"{name}.txt").open("w", encoding="utf-8") as handle:
            for prediction in output_predictions:
                handle.write(",".join(f"{value:.4f}" for value in prediction) + "\n")
        payload = {"stage": "otb100_optflow_action_bank", "done": sequence_index, "total": len(available), "last_sequence": name, "last_result": metrics}
        if args.progress_json:
            args.progress_json.parent.mkdir(parents=True, exist_ok=True)
            args.progress_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload), flush=True)

    summary = summarize(results, args.target)
    summary["raw_samurai"] = summarize([
        {"success_auc": result["raw_samurai"]["success_auc"], "precision_20px": result["raw_samurai"]["precision_20px"], "frames": result["raw_samurai"]["frames"]}
        for result in results
    ], args.target)
    summary.update({
        "benchmark": "OTB100 available completed sequences",
        "benchmark_complete": len(results) == len(metadata),
        "target_met": bool(len(results) == len(metadata) and summary["target_met"]),
        "tracker": "existing SAMURAI boxes + LK/RANSAC CMC + causal 1s/3s Action Bank",
        "action_bank": {"short_seconds": args.short_seconds, "long_seconds": args.long_seconds, "fps": args.fps, "min_motion_score": args.min_motion_score, "blend_weight": args.blend_weight},
        "results": results,
    })
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"kind": "otb100_optflow_action_bank_done", **summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
