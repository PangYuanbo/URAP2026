from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image
import torch


def _write_synthetic_nps(root: Path, frames: int = 8) -> tuple[Path, Path]:
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True)
    gt_csv = root / "gt.csv"
    for frame_id in range(1, frames + 1):
        img = Image.new("RGB", (96, 64), color=(frame_id * 20 % 255, 20, 40))
        img.save(frames_dir / f"Clip_1_{frame_id:05d}.png")
    with gt_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq", "frame_id", "x1", "y1", "x2", "y2", "video_path"])
        writer.writeheader()
        for frame_id in range(1, frames + 1):
            writer.writerow(
                {
                    "seq": "Clip_1",
                    "frame_id": frame_id,
                    "x1": 10 + frame_id,
                    "y1": 12,
                    "x2": 24 + frame_id,
                    "y2": 26,
                    "video_path": f"Clip_1/Clip_1_{frame_id:05d}.png",
                }
            )
    return frames_dir, gt_csv


def _run_train(repo: Path, frames_dir: Path, gt_csv: Path, out_dir: Path, epochs: int = 1, *extra: str) -> str:
    cmd = [
        sys.executable,
        str(repo / "tools" / "train_native_video_detector.py"),
        "--frames-dir",
        str(frames_dir),
        "--gt-csv",
        str(gt_csv),
        "--out-dir",
        str(out_dir),
        "--max-samples",
        "4",
        "--epochs",
        str(epochs),
        "--batch-size",
        "2",
        "--image-size",
        "64",
        "--d-model",
        "32",
        "--encoder-layers",
        "1",
        "--decoder-layers",
        "1",
        "--num-queries",
        "4",
        "--future-len",
        "2",
        "--clip-len",
        "4",
        "--num-workers",
        "0",
        "--log-every",
        "1",
        "--save-every-steps",
        "2",
        "--ema",
        "--seed",
        "123",
        "--device",
        "cpu",
        *extra,
    ]
    proc = subprocess.run(cmd, cwd=repo, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout


def _start_event(stdout: str) -> dict[str, object]:
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        obj = json.loads(line)
        if obj.get("kind") == "native_video_train_start":
            return obj
    raise AssertionError(f"native_video_train_start not found in stdout:\n{stdout}")


def test_native_video_train_resume_roundtrip(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    frames_dir, gt_csv = _write_synthetic_nps(tmp_path / "data")
    out_dir = tmp_path / "run"

    first_stdout = _run_train(repo, frames_dir, gt_csv, out_dir)
    assert _start_event(first_stdout)["resume_mode"] == "fresh"
    latest = out_dir / "native_video_detector_latest.pt"
    assert latest.exists()
    first = torch.load(latest, map_location="cpu")
    assert first["global_step"] == 2
    assert first["batch"] == 2
    assert first["epoch"] == 1
    assert first["epoch_loss_so_far"] is not None
    assert "optimizer" in first
    assert "scaler" in first
    assert "ema_model" in first
    assert "scheduler" in first
    assert first["config"]["clip_len"] == 4
    assert first["config"]["future_len"] == 2
    assert first["config"]["num_queries"] == 4
    assert first["config"]["d_model"] == 32
    assert first["config"]["nhead"] == 4
    assert first["config"]["encoder_layers"] == 1
    assert first["config"]["decoder_layers"] == 1
    assert first["config"]["encoder_mode"] == "factorized"
    assert first["config"]["patch_stride"] == 8
    assert first["config"]["memory_attention"] == "none"
    assert first["config"]["memory_slots"] == 64
    assert first["config"]["motion_score_mode"] == "none"
    assert first["config"]["motion_score_weight"] == 1.0
    assert first["config"]["proposal_mode"] == "none"
    assert first["config"]["quality_score_mode"] == "none"
    assert first["config"]["dense_hard_negative_topk"] == 0
    assert first["config"]["dense_rank_weight"] == 0.0
    assert first["config"]["dense_rank_margin"] == 1.0
    assert first["config"]["dense_rank_negative_topk"] == 0
    assert first["config"]["dense_rank_positive_mode"] == "max"
    assert first["config"]["motion_obj_weight"] == 0.0
    assert first["config"]["dense_heatmap_weight"] == 0.0
    assert first["config"]["dense_heatmap_sigma"] == 0.02
    assert first["config"]["dense_heatmap_neg_weight"] == 0.02
    assert first["config"]["dense_heatmap_focal_gamma"] == 2.0
    assert first["config"]["quality_loss_weight"] == 0.0
    assert first["config"]["quality_warmup_steps"] == 0
    assert first["config"]["quality_ramp_steps"] == 0
    assert first["config"]["quality_positive_iou"] == 0.05
    assert first["config"]["quality_hard_negative_topk"] == 0
    assert first["config"]["quality_focal_gamma"] == 1.0
    assert first["config"]["seed"] == 123
    assert first["loss_contract"]["matching"] == "detr_hungarian_current_frame"
    assert set(first["loss_contract"]["bbox"]) == {"l1", "giou"}
    assert first["loss_contract"]["objectness"] == "focal_bce"
    assert first["loss_contract"]["dense_hard_negatives"] == "optional_topk_hard_negative_bce"
    assert first["loss_contract"]["dense_ranking"] == "optional_gt_positive_vs_topk_negative_margin"
    assert first["loss_contract"]["dense_ranking_positive_mode"] == "max_or_all_dense_positive_anchors"
    assert first["loss_contract"]["samurai_motion_score"] == "optional_dense_motion_objectness_branch"
    assert first["loss_contract"]["proposal_heatmap_head"] == "optional_separate_dense_center_proposal_head"
    assert first["loss_contract"]["dense_center_heatmap"] == "optional_gaussian_anchor_center_focal_bce"
    assert first["loss_contract"]["quality_score_head"] == "optional_dense_iou_quality_head"
    assert first["loss_contract"]["quality_score_loss"] == "optional_soft_iou_bce_with_hard_negatives"
    assert first["loss_contract"]["quality_loss_schedule"] == "optional_warmup_then_linear_ramp"
    assert first["loss_contract"]["future_chunk"] == "smooth_l1"
    start = _start_event(first_stdout)
    assert start["lr_scheduler"] == "cosine"
    assert start["encoder_layers"] == 1
    assert start["decoder_layers"] == 1
    assert start["seed"] == 123
    assert start["architecture"]["input"] == "4-frame clip"
    assert start["architecture"]["proposal_mode"] == "none"
    assert start["architecture"]["quality_score_mode"] == "none"
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["image_size"] == 64
    assert summary["clip_len"] == 4
    assert summary["future_len"] == 2
    assert summary["output_chunk_len"] == 3
    assert summary["num_queries"] == 4
    assert summary["d_model"] == 32
    assert summary["nhead"] == 4
    assert summary["encoder_layers"] == 1
    assert summary["decoder_layers"] == 1
    assert summary["encoder_mode"] == "factorized"
    assert summary["patch_stride"] == 8
    assert summary["memory_attention"] == "none"
    assert summary["memory_slots"] == 64
    assert summary["motion_score_mode"] == "none"
    assert summary["motion_score_weight"] == 1.0
    assert summary["proposal_mode"] == "none"
    assert summary["quality_score_mode"] == "none"
    assert summary["dense_hard_negative_topk"] == 0
    assert summary["dense_rank_weight"] == 0.0
    assert summary["dense_rank_margin"] == 1.0
    assert summary["dense_rank_negative_topk"] == 0
    assert summary["dense_rank_positive_mode"] == "max"
    assert summary["motion_obj_weight"] == 0.0
    assert summary["dense_heatmap_weight"] == 0.0
    assert summary["dense_heatmap_sigma"] == 0.02
    assert summary["dense_heatmap_neg_weight"] == 0.02
    assert summary["dense_heatmap_focal_gamma"] == 2.0
    assert summary["quality_loss_weight"] == 0.0
    assert summary["quality_warmup_steps"] == 0
    assert summary["quality_ramp_steps"] == 0
    assert summary["quality_positive_iou"] == 0.05
    assert summary["quality_hard_negative_topk"] == 0
    assert summary["quality_focal_gamma"] == 1.0
    assert summary["seed"] == 123
    assert summary["architecture"]["backbone"] == "small_conv_stem"
    assert summary["architecture"]["object_queries"] == 4
    assert summary["architecture"]["proposal_mode"] == "none"
    assert summary["architecture"]["quality_score_mode"] == "none"
    assert summary["architecture"]["output"] == "current_bbox_plus_2_future_bbox_chunk"
    assert summary["loss_contract"]["matching"] == "detr_hungarian_current_frame"
    assert set(summary["loss_contract"]["bbox"]) == {"l1", "giou"}
    assert summary["loss_contract"]["objectness"] == "focal_bce"
    assert summary["loss_contract"]["dense_ranking"] == "optional_gt_positive_vs_topk_negative_margin"
    assert summary["loss_contract"]["dense_ranking_positive_mode"] == "max_or_all_dense_positive_anchors"
    assert summary["loss_contract"]["samurai_motion_score"] == "optional_dense_motion_objectness_branch"
    assert summary["loss_contract"]["proposal_heatmap_head"] == "optional_separate_dense_center_proposal_head"
    assert summary["loss_contract"]["dense_center_heatmap"] == "optional_gaussian_anchor_center_focal_bce"
    assert summary["loss_contract"]["quality_score_head"] == "optional_dense_iou_quality_head"
    assert summary["loss_contract"]["quality_score_loss"] == "optional_soft_iou_bce_with_hard_negatives"
    assert summary["loss_contract"]["quality_loss_schedule"] == "optional_warmup_then_linear_ramp"
    assert summary["loss_contract"]["future_chunk"] == "smooth_l1"
    assert summary["parameter_count"]["trainable"] > 0
    assert summary["parameter_count"]["total"] >= summary["parameter_count"]["trainable"]
    assert not list(out_dir.glob("*.tmp.*"))

    resume_stdout = _run_train(repo, frames_dir, gt_csv, out_dir, 2, "--resume", str(latest))
    assert _start_event(resume_stdout)["resume_mode"] == "resume_next_epoch"
    resumed = torch.load(out_dir / "native_video_detector_latest.pt", map_location="cpu")
    assert resumed["global_step"] == 4
    assert resumed["epoch"] == 2
    assert resumed["batch"] == 2
    assert resumed["config"] == first["config"]
    assert "ema_model" in resumed
    assert "scheduler" in resumed
    assert (out_dir / "summary.json").exists()
    assert not list(out_dir.glob("*.tmp.*"))


def test_native_video_train_init_weights_allows_new_quality_head(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    frames_dir, gt_csv = _write_synthetic_nps(tmp_path / "data")
    source_dir = tmp_path / "source"
    init_dir = tmp_path / "init_quality"

    dense_args = ("--query-mode", "dense", "--dense-positive-topk", "4")
    _run_train(repo, frames_dir, gt_csv, source_dir, 1, *dense_args)
    source_latest = source_dir / "native_video_detector_latest.pt"
    assert source_latest.exists()

    init_stdout = _run_train(
        repo,
        frames_dir,
        gt_csv,
        init_dir,
        1,
        *dense_args,
        "--quality-score-mode",
        "iou",
        "--quality-loss-weight",
        "0.1",
        "--init-weights",
        str(source_latest),
    )
    start = _start_event(init_stdout)
    assert start["resume_mode"] == "init_weights"
    assert start["init_weights"] == str(source_latest.resolve())
    assert start["init_weights_stats"]["loaded"] > 0
    assert start["init_weights_stats"]["missing_from_checkpoint"] > 0

    initialized = torch.load(init_dir / "native_video_detector_latest.pt", map_location="cpu")
    assert initialized["global_step"] == 2
    assert initialized["config"]["quality_score_mode"] == "iou"
    assert initialized["config"]["quality_loss_weight"] == 0.1


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="native_video_resume_") as tmp:
        root = Path(tmp)
        test_native_video_train_resume_roundtrip(root / "resume")
        test_native_video_train_init_weights_allows_new_quality_head(root / "init")
