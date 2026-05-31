from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _safe_name(video_path: str) -> str:
    path = Path(video_path)
    stem = path.stem.lower()
    if stem in {"visible", "infrared", "ir"} and path.parent.name:
        stem = f"{path.parent.name}_{stem}".lower()
    for token in ("dji_fly_20260527_", "_hdrvideo"):
        stem = stem.replace(token, "")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem)


def _load_annotation_groups(path: Path) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            video = row.get("video_path") or row.get("source_video")
            if not video:
                continue
            groups.setdefault(video, []).append(row)
    return groups


def _write_gt(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _run_command(cmd: list[str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.write_text("", encoding="utf-8")
    with stdout_path.open("w", encoding="utf-8") as stdout:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            stdout.write(line)
            stdout.flush()
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"Command failed with exit code {code}: {' '.join(cmd)}")


def _infer_cmd(args: argparse.Namespace, profile: str, video: str, out: Path, max_frames: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "qstr_dronedet.cli",
        "infer",
        "--video",
        video,
        "--out",
        str(out),
        "--stream",
        "--frame-start",
        "0",
        "--frame-stride",
        str(args.frame_stride),
        "--max-frames",
        str(max_frames),
        "--yolo-weights",
        args.yolo_weights,
        "--yolo-conf",
        str(args.yolo_conf),
        "--yolo-tile-size",
        str(args.tile_size),
        "--yolo-tile-stride",
        str(args.tile_stride),
        "--yolo-device",
        args.device,
        "--max-yolo-candidates-per-frame",
        str(args.max_yolo_candidates_per_frame),
        "--max-candidates-per-frame",
        str(args.max_candidates_per_frame),
        "--crop-weights",
        args.crop_weights,
        "--temporal-weights",
        args.temporal_weights,
        "--tracklet-classifier-weights",
        args.tracklet_classifier_weights,
        "--tracklet-classifier-threshold",
        str(args.tracklet_classifier_threshold),
        "--tracklet-promotion-score-floor",
        str(args.tracklet_promotion_score_floor),
        "--tracklet-promotion-min-branch-drone",
        str(args.tracklet_promotion_min_branch_drone),
        "--tracklet-promotion-max-background",
        str(args.tracklet_promotion_max_background),
        "--tracklet-protect-temporal-only-hard-tiny",
        "--disable-motion-candidates",
        "--disable-frame-images",
    ]
    if args.disable_tracklet_promotion:
        cmd.append("--disable-tracklet-promotion")
    if args.tracklet_selective_promotion:
        cmd += [
            "--tracklet-selective-promotion",
            "--tracklet-selective-min-temporal-crop-delta",
            str(args.tracklet_selective_min_temporal_crop_delta),
            "--tracklet-selective-min-temporal-background-margin",
            str(args.tracklet_selective_min_temporal_background_margin),
            "--tracklet-selective-max-tracklet-background",
            str(args.tracklet_selective_max_tracklet_background),
            "--tracklet-selective-max-tracklet-objectness",
            str(args.tracklet_selective_max_tracklet_objectness),
            "--tracklet-selective-min-tracklet-rows",
            str(args.tracklet_selective_min_tracklet_rows),
            "--tracklet-selective-min-temporal-gain-rate",
            str(args.tracklet_selective_min_temporal_gain_rate),
            "--tracklet-selective-min-weak-detector-temporal-signal",
            str(args.tracklet_selective_min_weak_detector_temporal_signal),
            "--tracklet-selective-max-promoted-tracklets-per-sequence",
            str(args.tracklet_selective_max_promoted_tracklets_per_sequence),
        ]
        if args.tracklet_selective_allow_non_recovery_source:
            cmd.append("--tracklet-selective-allow-non-recovery-source")
    if profile != "yolo_only":
        cmd += [
            "--rtdetr-weights",
            args.rtdetr_weights,
            "--rtdetr-conf",
            str(args.rtdetr_conf),
            "--rtdetr-tile-size",
            str(args.tile_size),
            "--rtdetr-tile-stride",
            str(args.tile_stride),
            "--rtdetr-device",
            args.device,
            "--max-rtdetr-candidates-per-frame",
            str(args.max_rtdetr_candidates_per_frame),
        ]
    if profile == "yolo_rtdetr_no_gate":
        cmd.append("--disable-transformer-gate")
    if profile == "yolo_rtdetr_source_gate":
        cmd += ["--transformer-gate-min-final-drone-score", str(args.transformer_gate_min_final_drone_score)]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--rtdetr-weights", default="")
    parser.add_argument("--crop-weights", required=True)
    parser.add_argument("--temporal-weights", required=True)
    parser.add_argument("--tracklet-classifier-weights", required=True)
    parser.add_argument("--tracklet-classifier-threshold", type=float, default=0.50)
    parser.add_argument("--disable-tracklet-promotion", action="store_true")
    parser.add_argument("--tracklet-promotion-score-floor", type=float, default=0.22)
    parser.add_argument("--tracklet-promotion-min-branch-drone", type=float, default=0.40)
    parser.add_argument("--tracklet-promotion-max-background", type=float, default=0.68)
    parser.add_argument("--tracklet-selective-promotion", action="store_true")
    parser.add_argument("--tracklet-selective-min-temporal-crop-delta", type=float, default=0.05)
    parser.add_argument("--tracklet-selective-min-temporal-background-margin", type=float, default=-0.05)
    parser.add_argument("--tracklet-selective-max-tracklet-background", type=float, default=0.60)
    parser.add_argument("--tracklet-selective-max-tracklet-objectness", type=float, default=0.50)
    parser.add_argument("--tracklet-selective-min-tracklet-rows", type=int, default=2)
    parser.add_argument("--tracklet-selective-min-temporal-gain-rate", type=float, default=0.40)
    parser.add_argument("--tracklet-selective-min-weak-detector-temporal-signal", type=float, default=0.05)
    parser.add_argument("--tracklet-selective-allow-non-recovery-source", action="store_true")
    parser.add_argument("--tracklet-selective-max-promoted-tracklets-per-sequence", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--yolo-conf", type=float, default=0.05)
    parser.add_argument("--rtdetr-conf", type=float, default=0.05)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-stride", type=int, default=192)
    parser.add_argument("--max-yolo-candidates-per-frame", type=int, default=30)
    parser.add_argument("--max-rtdetr-candidates-per-frame", type=int, default=30)
    parser.add_argument("--max-candidates-per-frame", type=int, default=30)
    parser.add_argument("--transformer-gate-min-final-drone-score", type=float, default=0.20)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-video", type=int, default=0, help="0 means run through the max annotated frame for each video")
    parser.add_argument("--profiles", nargs="+", default=["yolo_only", "yolo_rtdetr_no_gate", "yolo_rtdetr_source_gate"])
    args = parser.parse_args()
    needs_rtdetr = any(profile != "yolo_only" for profile in args.profiles)
    if needs_rtdetr and not args.rtdetr_weights:
        raise ValueError("--rtdetr-weights is required when running RT-DETR profiles")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = _load_annotation_groups(Path(args.annotations))
    if not groups:
        raise ValueError(f"No video annotations found in {args.annotations}")
    sequences: list[str] = []
    for video, rows in groups.items():
        if not Path(video).exists():
            raise FileNotFoundError(f"Missing video path from annotations: {video}")
        sequence = _safe_name(video)
        sequences.append(sequence)
        max_gt_frame = max(int(float(row["frame_id"])) for row in rows)
        # qstr_dronedet.cli infer interprets --max-frames as the number of emitted
        # frames after --frame-stride, not as the maximum source frame id. Keep the
        # dense validation bounded to the annotated frame range.
        dense_frames = (max_gt_frame // max(1, int(args.frame_stride))) + 1
        if args.max_frames_per_video and args.max_frames_per_video > 0:
            dense_frames = min(dense_frames, args.max_frames_per_video)
        for profile in args.profiles:
            run_dir = out / profile / sequence
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_gt(run_dir / "frame_annotations.csv", rows)
            meta: dict[str, Any] = {
                "profile": profile,
                "sequence": sequence,
                "video": video,
                "max_gt_frame": max_gt_frame,
                "evaluated_frames": dense_frames,
                "frame_start": 0,
                "frame_stride": args.frame_stride,
                "stream": True,
                "disable_frame_images": True,
            }
            (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"START profile={profile} sequence={sequence} frames={dense_frames} out={run_dir}", flush=True)
            cmd = _infer_cmd(args, profile, video, run_dir, dense_frames)
            _run_command(cmd, run_dir / "stdout.log", run_dir / "stderr.log")
            print(f"DONE profile={profile} sequence={sequence}", flush=True)
    eval_cmd = [
        sys.executable,
        "tools/evaluate_dji_new2_profile_compare.py",
        "--root",
        str(out),
        "--sequences",
        *sequences,
        "--profiles",
        *args.profiles,
    ]
    _run_command(eval_cmd, out / "eval_stdout.log", out / "eval_stderr.log")
    (out / ".done").write_text("done\n", encoding="ascii")


if __name__ == "__main__":
    main()
