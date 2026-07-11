from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qstr_dronedet.native_video_detector import (
    NPSClipDataset,
    NativeVideoDetector,
    collate_nps_clips,
    native_video_detection_loss,
)


def clone_ema_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def update_ema_state(ema: dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    state = model.state_dict()
    with torch.no_grad():
        for key, value in state.items():
            if key not in ema:
                ema[key] = value.detach().clone()
            elif torch.is_floating_point(ema[key]):
                ema[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema[key].copy_(value.detach())


def build_lr_lambda(total_steps: int, warmup_steps: int, min_lr_ratio: float):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    min_lr_ratio = max(0.0, float(min_lr_ratio))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1.0 / warmup_steps, float(step + 1) / warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def scheduled_quality_loss_weight(base_weight: float, completed_steps: int, warmup_steps: int, ramp_steps: int) -> float:
    base_weight = float(base_weight)
    if base_weight <= 0.0:
        return 0.0
    next_step = int(completed_steps) + 1
    warmup_steps = max(0, int(warmup_steps))
    ramp_steps = max(0, int(ramp_steps))
    if next_step <= warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return base_weight
    progress = min(1.0, max(0.0, float(next_step - warmup_steps) / float(ramp_steps)))
    return base_weight * progress


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(device_arg)


def validate_attention_batch(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cuda" or args.encoder_mode != "factorized":
        return
    spatial_cells = math.ceil(float(args.image_size) / float(args.patch_stride)) ** 2
    effective_attention_batch = int(args.batch_size) * int(spatial_cells)
    max_effective_batch = 65535
    if effective_attention_batch > max_effective_batch:
        raise ValueError(
            "factorized CUDA attention effective batch is too large: "
            f"batch_size * spatial_cells = {args.batch_size} * {spatial_cells} = {effective_attention_batch}, "
            f"must be <= {max_effective_batch}. Reduce --batch-size or increase --patch-stride."
        )


def atomic_write_text(path: Path, text: str, marker: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{marker}")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def replace_with_retries(tmp_path: Path, path: Path, attempts: int = 10, delay_s: float = 0.25) -> None:
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            tmp_path.replace(path)
            return
        except OSError as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(delay_s * (attempt + 1))
    if last_error is not None:
        raise last_error


def prune_step_checkpoints(out_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = sorted(out_dir.glob("native_video_detector_step_*.pt"), key=lambda path: path.stat().st_mtime)
    stale = checkpoints[:-keep]
    for path in stale:
        try:
            path.unlink()
        except OSError as exc:
            print(
                json.dumps(
                    {
                        "kind": "native_video_checkpoint_prune_warning",
                        "path": str(path.resolve()),
                        "error": repr(exc),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )


def load_matching_model_state(model: torch.nn.Module, source_state: dict[str, torch.Tensor]) -> dict[str, object]:
    target_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[dict[str, object]] = []
    for key, value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            skipped.append({"key": key, "reason": "missing_in_current"})
            continue
        if tuple(target_value.shape) != tuple(value.shape):
            skipped.append(
                {
                    "key": key,
                    "reason": "shape_mismatch",
                    "checkpoint_shape": list(value.shape),
                    "current_shape": list(target_value.shape),
                }
            )
            continue
        compatible[key] = value.detach().to(device=target_value.device, dtype=target_value.dtype)
    merged_state = dict(target_state)
    merged_state.update(compatible)
    model.load_state_dict(merged_state)
    missing_from_checkpoint = [key for key in target_state if key not in compatible]
    return {
        "loaded": len(compatible),
        "current_total": len(target_state),
        "checkpoint_total": len(source_state),
        "missing_from_checkpoint": len(missing_from_checkpoint),
        "skipped": len(skipped),
        "skipped_examples": skipped[:12],
        "missing_examples": missing_from_checkpoint[:12],
    }


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return {"trainable": int(trainable), "total": int(total)}


def loss_contract() -> dict[str, object]:
    return {
        "matching": "detr_hungarian_current_frame",
        "bbox": ["l1", "giou"],
        "objectness": "focal_bce",
        "dense_objectness": "anchor_radius_or_topk_when_enabled",
        "dense_hard_negatives": "optional_topk_hard_negative_bce",
        "dense_ranking": "optional_gt_positive_vs_topk_negative_margin",
        "dense_ranking_positive_mode": "max_or_all_dense_positive_anchors",
        "action_chunk_consistency": "optional_dense_positive_future_chunk_smooth_l1_plus_future_objectness",
        "samurai_memory_quality": "optional_history_gt_quality_supervision_for_memory_weights",
        "samurai_memory_match": "optional_current_frame_dense_cells_match_motion_selected_history_memory_slots",
        "samurai_motion_score": "optional_dense_motion_objectness_branch",
        "proposal_heatmap_head": "optional_separate_dense_center_proposal_head",
        "dense_center_heatmap": "optional_gaussian_anchor_center_focal_bce",
        "quality_score_head": "optional_dense_iou_quality_head",
        "quality_score_loss": "optional_soft_iou_bce_with_hard_negatives",
        "quality_loss_schedule": "optional_warmup_then_linear_ramp",
        "future_chunk": "smooth_l1",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the native Octo-like video detector MVP.")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--gt-csv", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--future-len", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--encoder-mode", choices=["factorized", "global"], default="factorized")
    parser.add_argument("--patch-stride", type=int, choices=[4, 8, 16], default=8)
    parser.add_argument("--spatial-refine-layers", type=int, default=0)
    parser.add_argument("--spatial-refine-kernel", type=int, default=7)
    parser.add_argument("--spatial-refine-expansion", type=float, default=2.0)
    parser.add_argument("--motion-channels", action="store_true")
    parser.add_argument("--memory-mode", choices=["last", "samurai"], default="last")
    parser.add_argument("--box-size-scale", type=float, default=1.0)
    parser.add_argument("--query-mode", choices=["learned", "dense"], default="learned")
    parser.add_argument("--anchor-offset-cells", type=float, default=4.0)
    parser.add_argument("--dense-obj-source", choices=["token", "conv"], default="token")
    parser.add_argument("--memory-attention", choices=["none", "pooled_cross"], default="none")
    parser.add_argument("--memory-slots", type=int, default=64)
    parser.add_argument("--memory-match-mode", choices=["none", "slot_dot"], default="none")
    parser.add_argument("--memory-match-weight", type=float, default=0.0)
    parser.add_argument("--memory-match-temperature", type=float, default=5.0)
    parser.add_argument("--motion-score-mode", choices=["none", "samurai"], default="none")
    parser.add_argument("--motion-score-weight", type=float, default=1.0)
    parser.add_argument("--proposal-mode", choices=["none", "heatmap"], default="none")
    parser.add_argument("--quality-score-mode", choices=["none", "iou"], default="none")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--augment-hflip-prob", type=float, default=0.5)
    parser.add_argument("--augment-brightness", type=float, default=0.1)
    parser.add_argument("--augment-contrast", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--keep-step-checkpoints", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--sync-timing", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--box-weight", type=float, default=5.0)
    parser.add_argument("--giou-weight", type=float, default=2.0)
    parser.add_argument("--obj-weight", type=float, default=1.0)
    parser.add_argument("--future-weight", type=float, default=0.5)
    parser.add_argument("--noobj-weight", type=float, default=0.1)
    parser.add_argument("--obj-focal-gamma", type=float, default=2.0)
    parser.add_argument("--obj-focal-alpha", type=float, default=0.25)
    parser.add_argument("--dense-positive-radius", type=float, default=0.0)
    parser.add_argument("--dense-positive-topk", type=int, default=0)
    parser.add_argument("--dense-hard-negative-topk", type=int, default=0)
    parser.add_argument("--dense-rank-weight", type=float, default=0.0)
    parser.add_argument("--dense-rank-margin", type=float, default=1.0)
    parser.add_argument("--dense-rank-negative-topk", type=int, default=0)
    parser.add_argument("--dense-rank-positive-mode", choices=["max", "all"], default="max")
    parser.add_argument("--action-chunk-consistency-weight", type=float, default=0.0)
    parser.add_argument("--memory-quality-weight", type=float, default=0.0)
    parser.add_argument("--memory-quality-sigma", type=float, default=0.08)
    parser.add_argument("--memory-quality-recency-tau", type=float, default=0.0)
    parser.add_argument("--memory-quality-exclude-current", action="store_true")
    parser.add_argument("--motion-obj-weight", type=float, default=0.0)
    parser.add_argument("--dense-heatmap-weight", type=float, default=0.0)
    parser.add_argument("--dense-heatmap-sigma", type=float, default=0.02)
    parser.add_argument("--dense-heatmap-neg-weight", type=float, default=0.02)
    parser.add_argument("--dense-heatmap-focal-gamma", type=float, default=2.0)
    parser.add_argument("--memory-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--quality-loss-weight", type=float, default=0.0)
    parser.add_argument("--quality-warmup-steps", type=int, default=0)
    parser.add_argument("--quality-ramp-steps", type=int, default=0)
    parser.add_argument("--quality-positive-iou", type=float, default=0.05)
    parser.add_argument("--quality-hard-negative-topk", type=int, default=0)
    parser.add_argument("--quality-focal-gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-weights", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.resume is not None and args.init_weights is not None:
        raise ValueError("--resume and --init-weights are mutually exclusive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        seed_everything(args.seed)
    device = resolve_device(args.device)
    validate_attention_batch(args, device)
    if device.type == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda" and args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    dataset = NPSClipDataset(
        args.frames_dir,
        args.gt_csv,
        clip_len=args.clip_len,
        future_len=args.future_len,
        image_size=args.image_size,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        cache_dir=args.cache_dir,
        augment_hflip_prob=args.augment_hflip_prob,
        augment_brightness=args.augment_brightness,
        augment_contrast=args.augment_contrast,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "collate_fn": collate_nps_clips,
        "pin_memory": device.type == "cuda",
    }
    if args.seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed)
        loader_kwargs["generator"] = loader_generator
        loader_kwargs["worker_init_fn"] = seed_worker
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, args.prefetch_factor)
        loader_kwargs["persistent_workers"] = args.persistent_workers
    loader = DataLoader(dataset, **loader_kwargs)
    model = NativeVideoDetector(
        clip_len=args.clip_len,
        future_len=args.future_len,
        num_queries=args.num_queries,
        d_model=args.d_model,
        nhead=args.nhead,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        channels_last=args.channels_last,
        encoder_mode=args.encoder_mode,
        patch_stride=args.patch_stride,
        spatial_refine_layers=args.spatial_refine_layers,
        spatial_refine_kernel=args.spatial_refine_kernel,
        spatial_refine_expansion=args.spatial_refine_expansion,
        motion_channels=args.motion_channels,
        memory_mode=args.memory_mode,
        box_size_scale=args.box_size_scale,
        query_mode=args.query_mode,
        anchor_offset_cells=args.anchor_offset_cells,
        dense_obj_source=args.dense_obj_source,
        memory_attention=args.memory_attention,
        memory_slots=args.memory_slots,
        memory_match_mode=args.memory_match_mode,
        memory_match_weight=args.memory_match_weight,
        memory_match_temperature=args.memory_match_temperature,
        motion_score_mode=args.motion_score_mode,
        motion_score_weight=args.motion_score_weight,
        proposal_mode=args.proposal_mode,
        quality_score_mode=args.quality_score_mode,
    ).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    checkpoint_model = model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_train_steps = max(1, len(loader) * max(1, args.epochs))
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=build_lr_lambda(total_train_steps, args.warmup_steps, args.min_lr_ratio),
        )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema_state = clone_ema_state(checkpoint_model) if args.ema else None
    history: list[dict[str, float]] = []
    global_step = 0
    resume_epoch = 0
    resume_batch = 0
    start_epoch = 1
    resume_mode = "fresh"
    init_weights_stats: dict[str, object] | None = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        ckpt_config = dict(ckpt.get("config", {}))
        expected_config = {
            "clip_len": args.clip_len,
            "future_len": args.future_len,
            "num_queries": args.num_queries,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "encoder_layers": args.encoder_layers,
            "decoder_layers": args.decoder_layers,
            "encoder_mode": args.encoder_mode,
            "patch_stride": args.patch_stride,
            "spatial_refine_layers": args.spatial_refine_layers,
            "spatial_refine_kernel": args.spatial_refine_kernel,
            "spatial_refine_expansion": args.spatial_refine_expansion,
            "motion_channels": args.motion_channels,
            "memory_mode": args.memory_mode,
            "box_size_scale": args.box_size_scale,
            "query_mode": args.query_mode,
            "anchor_offset_cells": args.anchor_offset_cells,
            "dense_obj_source": args.dense_obj_source,
            "memory_attention": args.memory_attention,
            "memory_slots": args.memory_slots,
            "memory_match_mode": args.memory_match_mode,
            "memory_match_weight": args.memory_match_weight,
            "memory_match_temperature": args.memory_match_temperature,
            "motion_score_mode": args.motion_score_mode,
            "motion_score_weight": args.motion_score_weight,
            "proposal_mode": args.proposal_mode,
            "quality_score_mode": args.quality_score_mode,
            "dense_hard_negative_topk": args.dense_hard_negative_topk,
            "dense_rank_weight": args.dense_rank_weight,
            "dense_rank_margin": args.dense_rank_margin,
            "dense_rank_negative_topk": args.dense_rank_negative_topk,
            "dense_rank_positive_mode": args.dense_rank_positive_mode,
            "action_chunk_consistency_weight": args.action_chunk_consistency_weight,
            "memory_quality_weight": args.memory_quality_weight,
            "memory_quality_sigma": args.memory_quality_sigma,
            "memory_quality_recency_tau": args.memory_quality_recency_tau,
            "memory_quality_exclude_current": args.memory_quality_exclude_current,
            "motion_obj_weight": args.motion_obj_weight,
            "dense_heatmap_weight": args.dense_heatmap_weight,
            "dense_heatmap_sigma": args.dense_heatmap_sigma,
            "dense_heatmap_neg_weight": args.dense_heatmap_neg_weight,
            "dense_heatmap_focal_gamma": args.dense_heatmap_focal_gamma,
            "memory_match_loss_weight": args.memory_match_loss_weight,
            "quality_loss_weight": args.quality_loss_weight,
            "quality_warmup_steps": args.quality_warmup_steps,
            "quality_ramp_steps": args.quality_ramp_steps,
            "quality_positive_iou": args.quality_positive_iou,
            "quality_hard_negative_topk": args.quality_hard_negative_topk,
            "quality_focal_gamma": args.quality_focal_gamma,
            "seed": args.seed,
            "image_size": args.image_size,
        }
        if "encoder_mode" not in ckpt_config:
            ckpt_config["encoder_mode"] = "global"
        if "patch_stride" not in ckpt_config:
            ckpt_config["patch_stride"] = 16
        if "spatial_refine_layers" not in ckpt_config:
            ckpt_config["spatial_refine_layers"] = 0
        if "spatial_refine_kernel" not in ckpt_config:
            ckpt_config["spatial_refine_kernel"] = 7
        if "spatial_refine_expansion" not in ckpt_config:
            ckpt_config["spatial_refine_expansion"] = 2.0
        if "nhead" not in ckpt_config:
            ckpt_config["nhead"] = 4
        if "motion_channels" not in ckpt_config:
            ckpt_config["motion_channels"] = False
        if "memory_mode" not in ckpt_config:
            ckpt_config["memory_mode"] = "last"
        if "box_size_scale" not in ckpt_config:
            ckpt_config["box_size_scale"] = 1.0
        if "query_mode" not in ckpt_config:
            ckpt_config["query_mode"] = "learned"
        if "anchor_offset_cells" not in ckpt_config:
            ckpt_config["anchor_offset_cells"] = 4.0
        if "dense_obj_source" not in ckpt_config:
            ckpt_config["dense_obj_source"] = "token"
        if "memory_attention" not in ckpt_config:
            ckpt_config["memory_attention"] = "none"
        if "memory_slots" not in ckpt_config:
            ckpt_config["memory_slots"] = 64
        if "memory_match_mode" not in ckpt_config:
            ckpt_config["memory_match_mode"] = "none"
        if "memory_match_weight" not in ckpt_config:
            ckpt_config["memory_match_weight"] = 0.0
        if "memory_match_temperature" not in ckpt_config:
            ckpt_config["memory_match_temperature"] = 5.0
        if "motion_score_mode" not in ckpt_config:
            ckpt_config["motion_score_mode"] = "none"
        if "motion_score_weight" not in ckpt_config:
            ckpt_config["motion_score_weight"] = 1.0
        if "proposal_mode" not in ckpt_config:
            ckpt_config["proposal_mode"] = "none"
        if "quality_score_mode" not in ckpt_config:
            ckpt_config["quality_score_mode"] = "none"
        if "dense_hard_negative_topk" not in ckpt_config:
            ckpt_config["dense_hard_negative_topk"] = 0
        if "dense_rank_weight" not in ckpt_config:
            ckpt_config["dense_rank_weight"] = 0.0
        if "dense_rank_margin" not in ckpt_config:
            ckpt_config["dense_rank_margin"] = 1.0
        if "dense_rank_negative_topk" not in ckpt_config:
            ckpt_config["dense_rank_negative_topk"] = 0
        if "dense_rank_positive_mode" not in ckpt_config:
            ckpt_config["dense_rank_positive_mode"] = "max"
        if "action_chunk_consistency_weight" not in ckpt_config:
            ckpt_config["action_chunk_consistency_weight"] = 0.0
        if "memory_quality_weight" not in ckpt_config:
            ckpt_config["memory_quality_weight"] = 0.0
        if "memory_quality_sigma" not in ckpt_config:
            ckpt_config["memory_quality_sigma"] = 0.08
        if "memory_quality_recency_tau" not in ckpt_config:
            ckpt_config["memory_quality_recency_tau"] = 0.0
        if "memory_quality_exclude_current" not in ckpt_config:
            ckpt_config["memory_quality_exclude_current"] = False
        if "motion_obj_weight" not in ckpt_config:
            ckpt_config["motion_obj_weight"] = 0.0
        if "dense_heatmap_weight" not in ckpt_config:
            ckpt_config["dense_heatmap_weight"] = 0.0
        if "dense_heatmap_sigma" not in ckpt_config:
            ckpt_config["dense_heatmap_sigma"] = 0.02
        if "dense_heatmap_neg_weight" not in ckpt_config:
            ckpt_config["dense_heatmap_neg_weight"] = 0.02
        if "dense_heatmap_focal_gamma" not in ckpt_config:
            ckpt_config["dense_heatmap_focal_gamma"] = 2.0
        if "memory_match_loss_weight" not in ckpt_config:
            ckpt_config["memory_match_loss_weight"] = 0.0
        if "quality_loss_weight" not in ckpt_config:
            ckpt_config["quality_loss_weight"] = 0.0
        if "quality_warmup_steps" not in ckpt_config:
            ckpt_config["quality_warmup_steps"] = 0
        if "quality_ramp_steps" not in ckpt_config:
            ckpt_config["quality_ramp_steps"] = 0
        if "quality_positive_iou" not in ckpt_config:
            ckpt_config["quality_positive_iou"] = 0.05
        if "quality_hard_negative_topk" not in ckpt_config:
            ckpt_config["quality_hard_negative_topk"] = 0
        if "quality_focal_gamma" not in ckpt_config:
            ckpt_config["quality_focal_gamma"] = 1.0
        if "seed" not in ckpt_config:
            ckpt_config["seed"] = None
        mismatches = {
            key: {"checkpoint": ckpt_config.get(key), "current": value}
            for key, value in expected_config.items()
            if ckpt_config.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Resume checkpoint config mismatch: {mismatches}")
        checkpoint_model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and amp_enabled:
            scaler.load_state_dict(ckpt["scaler"])
        if args.ema:
            if "ema_model" in ckpt:
                ema_state = {key: value.detach().clone() for key, value in ckpt["ema_model"].items()}
            else:
                ema_state = clone_ema_state(checkpoint_model)
        history = list(ckpt.get("history", []))
        global_step = int(ckpt.get("global_step", 0))
        resume_epoch = int(ckpt.get("epoch", 0))
        resume_batch = int(ckpt.get("batch", 0))
        start_epoch = max(1, resume_epoch)
        resume_mode = "resume_mid_epoch_replays_epoch"
        if resume_batch >= len(loader):
            start_epoch = min(args.epochs + 1, resume_epoch + 1)
            resume_mode = "resume_next_epoch"
    elif args.init_weights is not None:
        init_ckpt = torch.load(args.init_weights, map_location=device)
        source_state_raw = init_ckpt.get("model", init_ckpt) if isinstance(init_ckpt, dict) else init_ckpt
        if not isinstance(source_state_raw, dict):
            raise ValueError(f"init weights do not contain a model state dict: {args.init_weights}")
        source_state = {
            str(key): value
            for key, value in source_state_raw.items()
            if isinstance(value, torch.Tensor)
        }
        init_weights_stats = load_matching_model_state(checkpoint_model, source_state)
        if args.ema:
            ema_state = clone_ema_state(checkpoint_model)
        resume_mode = "init_weights"

    if args.compile:
        model = torch.compile(model)

    param_counts = count_parameters(checkpoint_model)
    loss_spec = loss_contract()

    def save_checkpoint(name: str, epoch: int, batch_idx: int = 0, epoch_loss_so_far: float | None = None) -> Path:
        ckpt = {
            "model": checkpoint_model.state_dict(),
            "config": {
                "clip_len": args.clip_len,
                "future_len": args.future_len,
                "num_queries": args.num_queries,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "encoder_layers": args.encoder_layers,
                "decoder_layers": args.decoder_layers,
                "encoder_mode": args.encoder_mode,
                "patch_stride": args.patch_stride,
                "spatial_refine_layers": args.spatial_refine_layers,
                "spatial_refine_kernel": args.spatial_refine_kernel,
                "spatial_refine_expansion": args.spatial_refine_expansion,
                "motion_channels": args.motion_channels,
                "memory_mode": args.memory_mode,
                "box_size_scale": args.box_size_scale,
                "query_mode": args.query_mode,
                "anchor_offset_cells": args.anchor_offset_cells,
                "dense_obj_source": args.dense_obj_source,
                "memory_attention": args.memory_attention,
                "memory_slots": args.memory_slots,
                "memory_match_mode": args.memory_match_mode,
                "memory_match_weight": args.memory_match_weight,
                "memory_match_temperature": args.memory_match_temperature,
                "motion_score_mode": args.motion_score_mode,
                "motion_score_weight": args.motion_score_weight,
                "proposal_mode": args.proposal_mode,
                "quality_score_mode": args.quality_score_mode,
                "dense_hard_negative_topk": args.dense_hard_negative_topk,
                "dense_rank_weight": args.dense_rank_weight,
                "dense_rank_margin": args.dense_rank_margin,
                "dense_rank_negative_topk": args.dense_rank_negative_topk,
                "dense_rank_positive_mode": args.dense_rank_positive_mode,
                "action_chunk_consistency_weight": args.action_chunk_consistency_weight,
                "memory_quality_weight": args.memory_quality_weight,
                "memory_quality_sigma": args.memory_quality_sigma,
                "memory_quality_recency_tau": args.memory_quality_recency_tau,
                "memory_quality_exclude_current": args.memory_quality_exclude_current,
                "motion_obj_weight": args.motion_obj_weight,
                "dense_heatmap_weight": args.dense_heatmap_weight,
                "dense_heatmap_sigma": args.dense_heatmap_sigma,
                "dense_heatmap_neg_weight": args.dense_heatmap_neg_weight,
                "dense_heatmap_focal_gamma": args.dense_heatmap_focal_gamma,
                "memory_match_loss_weight": args.memory_match_loss_weight,
                "quality_loss_weight": args.quality_loss_weight,
                "quality_warmup_steps": args.quality_warmup_steps,
                "quality_ramp_steps": args.quality_ramp_steps,
                "quality_positive_iou": args.quality_positive_iou,
                "quality_hard_negative_topk": args.quality_hard_negative_topk,
                "quality_focal_gamma": args.quality_focal_gamma,
                "seed": args.seed,
                "image_size": args.image_size,
            },
            "loss_contract": loss_spec,
            "history": history,
            "epoch": epoch,
            "batch": batch_idx,
            "global_step": global_step,
            "epoch_loss_so_far": epoch_loss_so_far,
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }
        if ema_state is not None:
            ckpt["ema_model"] = ema_state
            ckpt["ema_decay"] = args.ema_decay
        path = args.out_dir / name
        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{global_step}")
        try:
            torch.save(ckpt, tmp_path)
            replace_with_retries(tmp_path, path)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "kind": "native_video_checkpoint_save_error",
                        "path": str(path.resolve()),
                        "tmp_path": str(tmp_path.resolve()),
                        "epoch": epoch,
                        "batch": batch_idx,
                        "global_step": global_step,
                        "error": repr(exc),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
            raise
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError as exc:
                    print(
                        json.dumps(
                            {
                                "kind": "native_video_checkpoint_tmp_cleanup_warning",
                                "tmp_path": str(tmp_path.resolve()),
                                "error": repr(exc),
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
        return path

    print(
        json.dumps(
            {
                "kind": "native_video_train_start",
                "device": str(device),
                "device_arg": args.device,
                "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "samples": len(dataset),
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "total_train_steps": total_train_steps,
                "lr": args.lr,
                "lr_scheduler": args.lr_scheduler,
                "warmup_steps": args.warmup_steps,
                "min_lr_ratio": args.min_lr_ratio,
                "augment_hflip_prob": args.augment_hflip_prob,
                "augment_brightness": args.augment_brightness,
                "augment_contrast": args.augment_contrast,
                "image_size": args.image_size,
                "clip_len": args.clip_len,
                "future_len": args.future_len,
                "output_chunk_len": args.future_len + 1,
                "num_queries": args.num_queries,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "encoder_layers": args.encoder_layers,
                "encoder_mode": args.encoder_mode,
                "patch_stride": args.patch_stride,
                "motion_channels": args.motion_channels,
                "memory_mode": args.memory_mode,
                "box_size_scale": args.box_size_scale,
                "query_mode": args.query_mode,
                "anchor_offset_cells": args.anchor_offset_cells,
                "dense_obj_source": args.dense_obj_source,
                "memory_attention": args.memory_attention,
                "memory_slots": args.memory_slots,
                "memory_match_mode": args.memory_match_mode,
                "memory_match_weight": args.memory_match_weight,
                "memory_match_temperature": args.memory_match_temperature,
                "motion_score_mode": args.motion_score_mode,
                "motion_score_weight": args.motion_score_weight,
                "proposal_mode": args.proposal_mode,
                "quality_score_mode": args.quality_score_mode,
                "decoder_layers": args.decoder_layers,
                "parameter_count": param_counts,
                "architecture": {
                    "input": f"{args.clip_len}-frame clip",
                    "backbone": "small_conv_stem",
                    "input_channels": 5 if args.motion_channels else 3,
                    "tokenization": f"conv_patch_stride_{args.patch_stride}",
                    "spatial_refine_layers": args.spatial_refine_layers,
                    "spatial_refine_kernel": args.spatial_refine_kernel,
                    "spatial_refine_expansion": args.spatial_refine_expansion,
                    "temporal_transformer": args.encoder_mode,
                    "motion_aware_memory": args.memory_mode,
                    "box_size_scale": args.box_size_scale,
                    "query_mode": args.query_mode,
                    "anchor_offset_cells": args.anchor_offset_cells,
                    "dense_obj_source": args.dense_obj_source,
                    "memory_attention": args.memory_attention,
                    "memory_slots": args.memory_slots,
                    "memory_match_mode": args.memory_match_mode,
                    "memory_match_weight": args.memory_match_weight,
                    "memory_match_temperature": args.memory_match_temperature,
                    "motion_score_mode": args.motion_score_mode,
                    "motion_score_weight": args.motion_score_weight,
                    "proposal_mode": args.proposal_mode,
                    "quality_score_mode": args.quality_score_mode,
                    "object_queries": args.num_queries,
                    "output": f"current_bbox_plus_{args.future_len}_future_bbox_chunk",
                },
                "loss_contract": loss_spec,
                "amp": amp_enabled,
                "channels_last": args.channels_last,
                "compile": args.compile,
                "tf32": bool(device.type == "cuda" and args.tf32),
                "cudnn_benchmark": bool(device.type == "cuda" and args.cudnn_benchmark),
                "seed": args.seed,
                "ema_enabled": bool(ema_state is not None),
                "ema_decay": args.ema_decay if ema_state is not None else None,
                "box_weight": args.box_weight,
                "giou_weight": args.giou_weight,
                "obj_weight": args.obj_weight,
                "future_weight": args.future_weight,
                "noobj_weight": args.noobj_weight,
                "obj_focal_gamma": args.obj_focal_gamma,
                "obj_focal_alpha": args.obj_focal_alpha,
                "dense_positive_radius": args.dense_positive_radius,
                "dense_positive_topk": args.dense_positive_topk,
                "dense_hard_negative_topk": args.dense_hard_negative_topk,
                "dense_rank_weight": args.dense_rank_weight,
                "dense_rank_margin": args.dense_rank_margin,
                "dense_rank_negative_topk": args.dense_rank_negative_topk,
                "dense_rank_positive_mode": args.dense_rank_positive_mode,
                "action_chunk_consistency_weight": args.action_chunk_consistency_weight,
                "memory_quality_weight": args.memory_quality_weight,
                "memory_quality_sigma": args.memory_quality_sigma,
                "memory_quality_recency_tau": args.memory_quality_recency_tau,
                "memory_quality_exclude_current": args.memory_quality_exclude_current,
                "motion_obj_weight": args.motion_obj_weight,
                "dense_heatmap_weight": args.dense_heatmap_weight,
                "dense_heatmap_sigma": args.dense_heatmap_sigma,
                "dense_heatmap_neg_weight": args.dense_heatmap_neg_weight,
                "dense_heatmap_focal_gamma": args.dense_heatmap_focal_gamma,
                "memory_match_loss_weight": args.memory_match_loss_weight,
                "quality_loss_weight": args.quality_loss_weight,
                "quality_warmup_steps": args.quality_warmup_steps,
                "quality_ramp_steps": args.quality_ramp_steps,
                "quality_positive_iou": args.quality_positive_iou,
                "quality_hard_negative_topk": args.quality_hard_negative_topk,
                "quality_focal_gamma": args.quality_focal_gamma,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
                "persistent_workers": bool(args.persistent_workers and args.num_workers > 0),
                "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir is not None else None,
                "resume": str(args.resume.resolve()) if args.resume is not None else None,
                "init_weights": str(args.init_weights.resolve()) if args.init_weights is not None else None,
                "init_weights_stats": init_weights_stats,
                "resume_mode": resume_mode,
                "resume_epoch": resume_epoch,
                "resume_batch": resume_batch,
                "start_epoch": start_epoch,
                "global_step": global_step,
            }
        ),
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses = []
        batch_wait_start = time.perf_counter()
        for batch_idx, batch in enumerate(loader, start=1):
            data_ms = (time.perf_counter() - batch_wait_start) * 1000.0
            compute_start = time.perf_counter()
            clips = batch["clip"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(clips)
                effective_quality_loss_weight = scheduled_quality_loss_weight(
                    args.quality_loss_weight,
                    global_step,
                    args.quality_warmup_steps,
                    args.quality_ramp_steps,
                )
                loss, metrics = native_video_detection_loss(
                    outputs,
                    batch["boxes"],
                    batch["future_boxes"],
                    batch["history_boxes"],
                    batch["history_frame_ids"],
                    batch["frame_id"],
                    box_weight=args.box_weight,
                    giou_weight=args.giou_weight,
                    obj_weight=args.obj_weight,
                    future_weight=args.future_weight,
                    noobj_weight=args.noobj_weight,
                    obj_focal_gamma=args.obj_focal_gamma,
                    obj_focal_alpha=args.obj_focal_alpha,
                    dense_positive_radius=args.dense_positive_radius,
                    dense_positive_topk=args.dense_positive_topk,
                    dense_hard_negative_topk=args.dense_hard_negative_topk,
                    dense_rank_weight=args.dense_rank_weight,
                    dense_rank_margin=args.dense_rank_margin,
                    dense_rank_negative_topk=args.dense_rank_negative_topk,
                    dense_rank_positive_mode=args.dense_rank_positive_mode,
                    action_chunk_consistency_weight=args.action_chunk_consistency_weight,
                    memory_quality_weight=args.memory_quality_weight,
                    memory_quality_sigma=args.memory_quality_sigma,
                    memory_quality_recency_tau=args.memory_quality_recency_tau,
                    memory_quality_exclude_current=args.memory_quality_exclude_current,
                    motion_obj_weight=args.motion_obj_weight,
                    dense_heatmap_weight=args.dense_heatmap_weight,
                    dense_heatmap_sigma=args.dense_heatmap_sigma,
                    dense_heatmap_neg_weight=args.dense_heatmap_neg_weight,
                    dense_heatmap_focal_gamma=args.dense_heatmap_focal_gamma,
                    memory_match_loss_weight=args.memory_match_loss_weight,
                    quality_loss_weight=effective_quality_loss_weight,
                    quality_positive_iou=args.quality_positive_iou,
                    quality_hard_negative_topk=args.quality_hard_negative_topk,
                    quality_focal_gamma=args.quality_focal_gamma,
                )
                metrics["quality_loss_weight_target"] = float(args.quality_loss_weight)
                metrics["quality_loss_weight_effective"] = float(effective_quality_loss_weight)
                metrics["quality_warmup_steps"] = int(args.quality_warmup_steps)
                metrics["quality_ramp_steps"] = int(args.quality_ramp_steps)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before_step = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            optimizer_step_ran = (not amp_enabled) or (scaler.get_scale() >= scale_before_step)
            if scheduler is not None and optimizer_step_ran:
                scheduler.step()
            if ema_state is not None:
                update_ema_state(ema_state, checkpoint_model, decay=args.ema_decay)
            if args.sync_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
            step_ms = (time.perf_counter() - compute_start) * 1000.0
            batch_wait_start = time.perf_counter()
            global_step += 1
            epoch_losses.append(metrics["loss"])
            if batch_idx == 1 or batch_idx % args.log_every == 0:
                clips_per_second = args.batch_size / max(step_ms / 1000.0, 1e-9)
                frames_per_second = (args.batch_size * args.clip_len) / max(step_ms / 1000.0, 1e-9)
                row = {
                    "kind": "native_video_train_progress",
                    "epoch": epoch,
                    "batch": batch_idx,
                    "batches_total": len(loader),
                    "global_step": global_step,
                    **metrics,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "data_ms": round(data_ms, 3),
                    "step_ms": round(step_ms, 3),
                    "clips_per_second": round(clips_per_second, 3),
                    "frames_per_second": round(frames_per_second, 3),
                    "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3)
                    if device.type == "cuda"
                    else 0.0,
                }
                print(json.dumps(row), flush=True)
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                epoch_loss_so_far = float(sum(epoch_losses) / max(1, len(epoch_losses)))
                step_path = save_checkpoint(f"native_video_detector_step_{global_step:07d}.pt", epoch, batch_idx, epoch_loss_so_far)
                prune_step_checkpoints(args.out_dir, args.keep_step_checkpoints)
                print(
                    json.dumps(
                        {
                            "kind": "native_video_step_checkpoint",
                            "epoch": epoch,
                            "batch": batch_idx,
                            "batches_total": len(loader),
                            "global_step": global_step,
                            "epoch_loss_so_far": epoch_loss_so_far,
                            "weights": str(step_path.resolve()),
                            "step_weights": str(step_path.resolve()),
                        }
                    ),
                    flush=True,
                )
        history.append({"epoch": float(epoch), "loss": float(sum(epoch_losses) / max(1, len(epoch_losses)))})
        epoch_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        final_path = save_checkpoint("native_video_detector.pt", epoch, len(loader), epoch_loss)
        latest_path = save_checkpoint("native_video_detector_latest.pt", epoch, len(loader), epoch_loss)
        epoch_path = save_checkpoint(f"native_video_detector_epoch_{epoch:03d}.pt", epoch, len(loader), epoch_loss)
        print(
            json.dumps(
                {
                    "kind": "native_video_epoch_checkpoint",
                    "epoch": epoch,
                    "epochs_total": args.epochs,
                    "weights": str(final_path.resolve()),
                    "latest_weights": str(latest_path.resolve()),
                    "epoch_weights": str(epoch_path.resolve()),
                }
            ),
            flush=True,
        )
    summary = {
        "weights": str((args.out_dir / "native_video_detector.pt").resolve()),
        "frames_dir": str(args.frames_dir.resolve()),
        "gt_csv": [str(path.resolve()) for path in args.gt_csv],
        "samples": len(dataset),
        "history": history,
        "device": str(device),
        "device_arg": args.device,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "augment_hflip_prob": args.augment_hflip_prob,
        "augment_brightness": args.augment_brightness,
        "augment_contrast": args.augment_contrast,
        "image_size": args.image_size,
        "clip_len": args.clip_len,
        "future_len": args.future_len,
        "output_chunk_len": args.future_len + 1,
        "num_queries": args.num_queries,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "encoder_mode": args.encoder_mode,
        "patch_stride": args.patch_stride,
        "spatial_refine_layers": args.spatial_refine_layers,
        "spatial_refine_kernel": args.spatial_refine_kernel,
        "spatial_refine_expansion": args.spatial_refine_expansion,
        "motion_channels": args.motion_channels,
        "memory_mode": args.memory_mode,
        "box_size_scale": args.box_size_scale,
        "query_mode": args.query_mode,
        "anchor_offset_cells": args.anchor_offset_cells,
        "dense_obj_source": args.dense_obj_source,
        "memory_attention": args.memory_attention,
        "memory_slots": args.memory_slots,
        "memory_match_mode": args.memory_match_mode,
        "memory_match_weight": args.memory_match_weight,
        "memory_match_temperature": args.memory_match_temperature,
        "motion_score_mode": args.motion_score_mode,
        "motion_score_weight": args.motion_score_weight,
        "proposal_mode": args.proposal_mode,
        "quality_score_mode": args.quality_score_mode,
        "parameter_count": param_counts,
        "architecture": {
            "input": f"{args.clip_len}-frame clip",
            "backbone": "small_conv_stem",
            "input_channels": 5 if args.motion_channels else 3,
            "tokenization": f"conv_patch_stride_{args.patch_stride}",
            "spatial_refine_layers": args.spatial_refine_layers,
            "spatial_refine_kernel": args.spatial_refine_kernel,
            "spatial_refine_expansion": args.spatial_refine_expansion,
            "temporal_transformer": args.encoder_mode,
            "motion_aware_memory": args.memory_mode,
            "box_size_scale": args.box_size_scale,
            "query_mode": args.query_mode,
            "anchor_offset_cells": args.anchor_offset_cells,
            "dense_obj_source": args.dense_obj_source,
            "memory_attention": args.memory_attention,
            "memory_slots": args.memory_slots,
            "memory_match_mode": args.memory_match_mode,
            "memory_match_weight": args.memory_match_weight,
            "memory_match_temperature": args.memory_match_temperature,
            "motion_score_mode": args.motion_score_mode,
            "motion_score_weight": args.motion_score_weight,
            "proposal_mode": args.proposal_mode,
            "quality_score_mode": args.quality_score_mode,
            "object_queries": args.num_queries,
            "output": f"current_bbox_plus_{args.future_len}_future_bbox_chunk",
        },
        "loss_contract": loss_spec,
        "ema_enabled": bool(ema_state is not None),
        "box_weight": args.box_weight,
        "giou_weight": args.giou_weight,
        "obj_weight": args.obj_weight,
        "future_weight": args.future_weight,
        "noobj_weight": args.noobj_weight,
        "obj_focal_gamma": args.obj_focal_gamma,
        "obj_focal_alpha": args.obj_focal_alpha,
        "dense_positive_radius": args.dense_positive_radius,
        "dense_positive_topk": args.dense_positive_topk,
        "dense_hard_negative_topk": args.dense_hard_negative_topk,
        "dense_rank_weight": args.dense_rank_weight,
        "dense_rank_margin": args.dense_rank_margin,
        "dense_rank_negative_topk": args.dense_rank_negative_topk,
        "dense_rank_positive_mode": args.dense_rank_positive_mode,
        "action_chunk_consistency_weight": args.action_chunk_consistency_weight,
        "memory_quality_weight": args.memory_quality_weight,
        "memory_quality_sigma": args.memory_quality_sigma,
        "memory_quality_recency_tau": args.memory_quality_recency_tau,
        "memory_quality_exclude_current": args.memory_quality_exclude_current,
        "motion_obj_weight": args.motion_obj_weight,
        "dense_heatmap_weight": args.dense_heatmap_weight,
        "dense_heatmap_sigma": args.dense_heatmap_sigma,
        "dense_heatmap_neg_weight": args.dense_heatmap_neg_weight,
        "dense_heatmap_focal_gamma": args.dense_heatmap_focal_gamma,
        "memory_match_loss_weight": args.memory_match_loss_weight,
        "quality_loss_weight": args.quality_loss_weight,
        "quality_warmup_steps": args.quality_warmup_steps,
        "quality_ramp_steps": args.quality_ramp_steps,
        "quality_positive_iou": args.quality_positive_iou,
        "quality_hard_negative_topk": args.quality_hard_negative_topk,
        "quality_focal_gamma": args.quality_focal_gamma,
        "seed": args.seed,
    }
    atomic_write_text(args.out_dir / "summary.json", json.dumps(summary, indent=2), marker=f"summary.{global_step}")
    print(json.dumps({"kind": "native_video_train_done", **summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
