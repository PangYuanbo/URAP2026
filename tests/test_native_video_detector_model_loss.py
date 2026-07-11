from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qstr_dronedet.native_video_detector import NativeVideoDetector, native_video_detection_loss


def test_native_video_detector_shapes_and_backward() -> None:
    torch.manual_seed(0)
    model = NativeVideoDetector(
        clip_len=8,
        future_len=4,
        num_queries=32,
        d_model=64,
        encoder_layers=1,
        decoder_layers=1,
    )
    clips = torch.rand(2, 8, 3, 64, 64)
    outputs = model(clips)

    assert outputs["boxes"].shape == (2, 32, 5, 4)
    assert outputs["logits"].shape == (2, 32, 5)
    assert torch.all(outputs["boxes"] >= 0.0)
    assert torch.all(outputs["boxes"] <= 1.0)

    targets = [
        torch.tensor([[0.45, 0.50, 0.12, 0.16]], dtype=torch.float32),
        torch.tensor([[0.25, 0.30, 0.08, 0.10], [0.65, 0.70, 0.10, 0.12]], dtype=torch.float32),
    ]
    future_targets = [
        [
            torch.tensor([[0.45, 0.50, 0.12, 0.16]], dtype=torch.float32),
            torch.tensor([[0.46, 0.50, 0.12, 0.16]], dtype=torch.float32),
            torch.tensor([[0.47, 0.50, 0.12, 0.16]], dtype=torch.float32),
            torch.tensor([[0.48, 0.50, 0.12, 0.16]], dtype=torch.float32),
            torch.tensor([[0.49, 0.50, 0.12, 0.16]], dtype=torch.float32),
        ],
        [
            torch.tensor([[0.25, 0.30, 0.08, 0.10], [0.65, 0.70, 0.10, 0.12]], dtype=torch.float32),
            torch.tensor([[0.26, 0.30, 0.08, 0.10], [0.66, 0.70, 0.10, 0.12]], dtype=torch.float32),
            torch.tensor([[0.27, 0.30, 0.08, 0.10], [0.67, 0.70, 0.10, 0.12]], dtype=torch.float32),
            torch.tensor([[0.28, 0.30, 0.08, 0.10], [0.68, 0.70, 0.10, 0.12]], dtype=torch.float32),
            torch.tensor([[0.29, 0.30, 0.08, 0.10], [0.69, 0.70, 0.10, 0.12]], dtype=torch.float32),
        ],
    ]
    loss, metrics = native_video_detection_loss(outputs, targets, future_targets)
    assert torch.isfinite(loss)
    assert metrics["matched"] == 3.0
    assert metrics["box_loss"] > 0.0
    assert metrics["giou_loss"] > 0.0
    assert metrics["future_loss"] > 0.0
    assert metrics["future_obj_pos"] == 12.0

    loss.backward()
    grads = [param.grad for param in model.parameters() if param.requires_grad]
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)


def test_native_video_detector_global_encoder_mode_shapes() -> None:
    model = NativeVideoDetector(
        clip_len=4,
        future_len=2,
        num_queries=8,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_mode="global",
    )
    outputs = model(torch.rand(2, 4, 3, 64, 64))
    assert outputs["boxes"].shape == (2, 8, 3, 4)
    assert outputs["logits"].shape == (2, 8, 3)


def test_native_video_detector_patch_stride_16_shapes() -> None:
    model = NativeVideoDetector(
        clip_len=4,
        future_len=2,
        num_queries=8,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        patch_stride=16,
    )
    outputs = model(torch.rand(2, 4, 3, 64, 64))
    assert outputs["boxes"].shape == (2, 8, 3, 4)
    assert outputs["logits"].shape == (2, 8, 3)


def test_native_video_detector_patch_stride_4_dense_shapes() -> None:
    model = NativeVideoDetector(
        clip_len=4,
        future_len=2,
        num_queries=8,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        patch_stride=4,
        query_mode="dense",
        dense_obj_source="conv",
        box_size_scale=0.02,
    )
    outputs = model(torch.rand(2, 4, 3, 64, 64))
    assert outputs["boxes"].shape == (2, 256, 3, 4)
    assert outputs["logits"].shape == (2, 256, 3)
    assert outputs["anchor_centers"].shape == (2, 256, 2)


def test_native_video_detector_spatial_refine_backward() -> None:
    torch.manual_seed(0)
    model = NativeVideoDetector(
        clip_len=4,
        future_len=1,
        num_queries=8,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        patch_stride=4,
        spatial_refine_layers=2,
        spatial_refine_kernel=3,
        spatial_refine_expansion=1.5,
        query_mode="dense",
        dense_obj_source="conv",
        box_size_scale=0.02,
    )
    outputs = model(torch.rand(2, 4, 3, 64, 64))
    assert outputs["boxes"].shape == (2, 256, 2, 4)
    targets = [
        torch.tensor([[0.20, 0.20, 0.01, 0.02]], dtype=torch.float32),
        torch.tensor([[0.60, 0.60, 0.01, 0.02]], dtype=torch.float32),
    ]
    loss, metrics = native_video_detection_loss(outputs, targets, dense_positive_topk=4)
    assert torch.isfinite(loss)
    assert metrics["dense_box_pos"] >= 8.0
    loss.backward()
    assert model.spatial_refine[0].block[0].weight.grad is not None
    assert model.dense_obj_conv[-1].weight.grad is not None


def test_native_video_detector_samurai_memory_motion_channels_backward() -> None:
    model = NativeVideoDetector(
        clip_len=5,
        future_len=2,
        num_queries=8,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        motion_channels=True,
        memory_mode="samurai",
    )
    outputs = model(torch.rand(2, 5, 3, 64, 64))
    assert outputs["boxes"].shape == (2, 8, 3, 4)
    assert outputs["logits"].shape == (2, 8, 3)
    assert outputs["memory_weights"].shape == (2, 5)
    assert torch.allclose(outputs["memory_weights"].sum(dim=1), torch.ones(2), atol=1e-5)

    targets = [
        torch.tensor([[0.20, 0.20, 0.10, 0.10]], dtype=torch.float32),
        torch.tensor([[0.60, 0.60, 0.10, 0.10]], dtype=torch.float32),
    ]
    future_targets = [
        [
            targets[0],
            torch.tensor([[0.22, 0.20, 0.10, 0.10]], dtype=torch.float32),
            torch.tensor([[0.24, 0.20, 0.10, 0.10]], dtype=torch.float32),
        ],
        [
            targets[1],
            torch.tensor([[0.62, 0.60, 0.10, 0.10]], dtype=torch.float32),
            torch.tensor([[0.64, 0.60, 0.10, 0.10]], dtype=torch.float32),
        ],
    ]
    loss, metrics = native_video_detection_loss(outputs, targets, future_targets)
    assert torch.isfinite(loss)
    assert metrics["matched"] == 2.0
    loss.backward()
    assert model.stem[0].weight.grad is not None
    assert model.memory_gate[-1].weight.grad is not None


def test_native_video_detector_tiny_box_size_prior() -> None:
    torch.manual_seed(0)
    model = NativeVideoDetector(
        clip_len=4,
        future_len=1,
        num_queries=8,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        box_size_scale=0.08,
    )
    outputs = model(torch.rand(2, 4, 3, 64, 64))
    assert outputs["boxes"].shape == (2, 8, 2, 4)
    assert torch.all(outputs["boxes"][..., :2] >= 0.0)
    assert torch.all(outputs["boxes"][..., :2] <= 1.0)
    assert torch.all(outputs["boxes"][..., 2:] >= 0.0)
    assert torch.all(outputs["boxes"][..., 2:] <= 0.08)


def test_native_video_detector_dense_query_mode_shapes_and_backward() -> None:
    torch.manual_seed(0)
    model = NativeVideoDetector(
        clip_len=4,
        future_len=1,
        num_queries=8,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        query_mode="dense",
        box_size_scale=0.02,
    )
    outputs = model(torch.rand(2, 4, 3, 64, 64))
    assert outputs["boxes"].shape == (2, 64, 2, 4)
    assert outputs["logits"].shape == (2, 64, 2)
    assert outputs["anchor_centers"].shape == (2, 64, 2)
    assert torch.all(outputs["boxes"][..., :2] >= 0.0)
    assert torch.all(outputs["boxes"][..., :2] <= 1.0)
    assert torch.all(outputs["boxes"][..., 2:] >= 0.0)
    assert torch.all(outputs["boxes"][..., 2:] <= 0.02)

    targets = [
        torch.tensor([[0.20, 0.20, 0.01, 0.02]], dtype=torch.float32),
        torch.tensor([[0.60, 0.60, 0.01, 0.02]], dtype=torch.float32),
    ]
    future_targets = [
        [targets[0], torch.tensor([[0.21, 0.20, 0.01, 0.02]], dtype=torch.float32)],
        [targets[1], torch.tensor([[0.61, 0.60, 0.01, 0.02]], dtype=torch.float32)],
    ]
    loss, metrics = native_video_detection_loss(
        outputs,
        targets,
        future_targets,
        dense_positive_radius=0.02,
        dense_positive_topk=4,
    )
    assert torch.isfinite(loss)
    assert metrics["matched"] == 2.0
    assert metrics["dense_obj_pos"] >= 8.0
    assert metrics["dense_box_pos"] >= 8.0
    loss.backward()
    assert model.dense_box_out.weight.grad is not None
    assert model.dense_obj_head[-1].weight.grad is not None


def test_native_video_detector_dense_conv_objectness_shapes_and_backward() -> None:
    torch.manual_seed(0)
    model = NativeVideoDetector(
        clip_len=4,
        future_len=1,
        num_queries=8,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        query_mode="dense",
        dense_obj_source="conv",
        box_size_scale=0.02,
    )
    outputs = model(torch.rand(2, 4, 3, 64, 64))
    assert outputs["boxes"].shape == (2, 64, 2, 4)
    assert outputs["logits"].shape == (2, 64, 2)
    assert torch.sigmoid(outputs["logits"]).mean() < 0.1

    targets = [
        torch.tensor([[0.20, 0.20, 0.01, 0.02]], dtype=torch.float32),
        torch.tensor([[0.60, 0.60, 0.01, 0.02]], dtype=torch.float32),
    ]
    future_targets = [
        [targets[0], torch.tensor([[0.21, 0.20, 0.01, 0.02]], dtype=torch.float32)],
        [targets[1], torch.tensor([[0.61, 0.60, 0.01, 0.02]], dtype=torch.float32)],
    ]
    loss, metrics = native_video_detection_loss(
        outputs,
        targets,
        future_targets,
        dense_positive_radius=0.02,
        dense_positive_topk=4,
    )
    assert torch.isfinite(loss)
    assert metrics["dense_box_pos"] >= 8.0
    loss.backward()
    assert model.dense_box_out.weight.grad is not None
    assert model.dense_obj_conv[-1].weight.grad is not None


def test_native_video_detector_dense_proposal_heatmap_head_backward() -> None:
    torch.manual_seed(0)
    model = NativeVideoDetector(
        clip_len=4,
        future_len=1,
        num_queries=8,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        query_mode="dense",
        dense_obj_source="conv",
        proposal_mode="heatmap",
        box_size_scale=0.02,
    )
    outputs = model(torch.rand(2, 4, 3, 64, 64))
    assert outputs["boxes"].shape == (2, 64, 2, 4)
    assert outputs["logits"].shape == (2, 64, 2)
    assert outputs["proposal_logits"].shape == (2, 64)

    targets = [
        torch.tensor([[0.20, 0.20, 0.01, 0.02]], dtype=torch.float32),
        torch.tensor([[0.60, 0.60, 0.01, 0.02]], dtype=torch.float32),
    ]
    future_targets = [
        [targets[0], torch.tensor([[0.21, 0.20, 0.01, 0.02]], dtype=torch.float32)],
        [targets[1], torch.tensor([[0.61, 0.60, 0.01, 0.02]], dtype=torch.float32)],
    ]
    loss, metrics = native_video_detection_loss(
        outputs,
        targets,
        future_targets,
        dense_positive_radius=0.02,
        dense_positive_topk=4,
        dense_heatmap_weight=1.0,
        dense_heatmap_sigma=0.02,
        dense_heatmap_neg_weight=0.01,
    )
    assert torch.isfinite(loss)
    assert metrics["dense_heatmap_source"] == "proposal_logits"
    assert metrics["dense_heatmap_loss"] > 0.0
    loss.backward()
    assert model.proposal_conv[-1].weight.grad is not None
    assert model.dense_obj_conv[-1].weight.grad is not None


def test_native_video_detector_dense_quality_head_backward() -> None:
    torch.manual_seed(0)
    model = NativeVideoDetector(
        clip_len=4,
        future_len=1,
        num_queries=8,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        query_mode="dense",
        dense_obj_source="conv",
        quality_score_mode="iou",
        box_size_scale=0.02,
    )
    outputs = model(torch.rand(2, 4, 3, 64, 64))
    assert outputs["boxes"].shape == (2, 64, 2, 4)
    assert outputs["logits"].shape == (2, 64, 2)
    assert outputs["quality_logits"].shape == (2, 64)

    targets = [
        outputs["boxes"][0, 0, 0].detach().reshape(1, 4),
        outputs["boxes"][1, 0, 0].detach().reshape(1, 4),
    ]
    loss, metrics = native_video_detection_loss(
        outputs,
        targets,
        dense_positive_topk=4,
        quality_loss_weight=1.0,
        quality_positive_iou=0.1,
        quality_hard_negative_topk=16,
        quality_focal_gamma=0.0,
    )
    assert torch.isfinite(loss)
    assert metrics["quality_loss"] > 0.0
    assert metrics["quality_target_pos"] >= 2.0
    assert metrics["quality_target_max"] > 0.999
    loss.backward()
    assert model.quality_conv[-1].weight.grad is not None
    assert model.dense_obj_conv[-1].weight.grad is not None


def test_native_video_detector_dense_samurai_pooled_memory_attention_backward() -> None:
    torch.manual_seed(0)
    model = NativeVideoDetector(
        clip_len=4,
        future_len=1,
        num_queries=8,
        d_model=32,
        nhead=4,
        encoder_layers=1,
        decoder_layers=1,
        query_mode="dense",
        dense_obj_source="conv",
        memory_mode="samurai",
        memory_attention="pooled_cross",
        memory_slots=4,
        box_size_scale=0.02,
    )
    outputs = model(torch.rand(2, 4, 3, 64, 64))
    assert outputs["boxes"].shape == (2, 64, 2, 4)
    assert outputs["logits"].shape == (2, 64, 2)
    assert outputs["memory_weights"].shape == (2, 4)
    assert torch.allclose(outputs["memory_weights"].sum(dim=1), torch.ones(2), atol=1e-5)

    targets = [
        torch.tensor([[0.20, 0.20, 0.01, 0.02]], dtype=torch.float32),
        torch.tensor([[0.60, 0.60, 0.01, 0.02]], dtype=torch.float32),
    ]
    future_targets = [
        [targets[0], torch.tensor([[0.21, 0.20, 0.01, 0.02]], dtype=torch.float32)],
        [targets[1], torch.tensor([[0.61, 0.60, 0.01, 0.02]], dtype=torch.float32)],
    ]
    loss, metrics = native_video_detection_loss(
        outputs,
        targets,
        future_targets,
        dense_positive_radius=0.02,
        dense_positive_topk=4,
    )
    assert torch.isfinite(loss)
    assert metrics["dense_obj_pos"] >= 8.0
    loss.backward()
    assert model.memory_cross_attn.in_proj_weight.grad is not None
    assert model.memory_attention_fusion[-1].weight.grad is not None
    assert model.dense_obj_conv[-1].weight.grad is not None


def test_native_video_detector_samurai_motion_score_branch_backward() -> None:
    torch.manual_seed(0)
    model = NativeVideoDetector(
        clip_len=4,
        future_len=1,
        num_queries=8,
        d_model=32,
        nhead=4,
        encoder_layers=1,
        decoder_layers=1,
        query_mode="dense",
        dense_obj_source="conv",
        memory_mode="samurai",
        memory_attention="pooled_cross",
        memory_slots=4,
        motion_score_mode="samurai",
        motion_score_weight=0.5,
        box_size_scale=0.02,
    )
    outputs = model(torch.rand(2, 4, 3, 64, 64))
    assert outputs["boxes"].shape == (2, 64, 2, 4)
    assert outputs["logits"].shape == (2, 64, 2)
    assert outputs["appearance_logits"].shape == (2, 64, 2)
    assert outputs["motion_logits"].shape == (2, 64, 2)
    assert float(outputs["motion_score_weight"].item()) == 0.5

    targets = [
        torch.tensor([[0.20, 0.20, 0.01, 0.02]], dtype=torch.float32),
        torch.tensor([[0.60, 0.60, 0.01, 0.02]], dtype=torch.float32),
    ]
    future_targets = [
        [targets[0], torch.tensor([[0.21, 0.20, 0.01, 0.02]], dtype=torch.float32)],
        [targets[1], torch.tensor([[0.61, 0.60, 0.01, 0.02]], dtype=torch.float32)],
    ]
    loss, metrics = native_video_detection_loss(
        outputs,
        targets,
        future_targets,
        dense_positive_radius=0.02,
        dense_positive_topk=4,
        motion_obj_weight=0.5,
    )
    assert torch.isfinite(loss)
    assert metrics["motion_obj_loss"] > 0.0
    assert metrics["motion_obj_weight"] == 0.5
    loss.backward()
    assert model.motion_score_head[-1].weight.grad is not None
    assert model.dense_obj_conv[-1].weight.grad is not None


def test_native_video_detector_samurai_memory_match_branch_backward() -> None:
    torch.manual_seed(0)
    model = NativeVideoDetector(
        clip_len=4,
        future_len=1,
        num_queries=8,
        d_model=32,
        nhead=4,
        encoder_layers=1,
        decoder_layers=1,
        query_mode="dense",
        dense_obj_source="conv",
        memory_mode="samurai",
        memory_attention="pooled_cross",
        memory_slots=4,
        memory_match_mode="slot_dot",
        memory_match_weight=0.25,
        memory_match_temperature=4.0,
        box_size_scale=0.02,
    )
    outputs = model(torch.rand(2, 4, 3, 64, 64))
    assert outputs["boxes"].shape == (2, 64, 2, 4)
    assert outputs["logits"].shape == (2, 64, 2)
    assert outputs["memory_match_logits"].shape == (2, 64)
    assert float(outputs["memory_match_weight"].item()) == 0.25

    targets = [
        torch.tensor([[0.20, 0.20, 0.01, 0.02]], dtype=torch.float32),
        torch.tensor([[0.60, 0.60, 0.01, 0.02]], dtype=torch.float32),
    ]
    future_targets = [
        [targets[0], torch.tensor([[0.21, 0.20, 0.01, 0.02]], dtype=torch.float32)],
        [targets[1], torch.tensor([[0.61, 0.60, 0.01, 0.02]], dtype=torch.float32)],
    ]
    loss, metrics = native_video_detection_loss(
        outputs,
        targets,
        future_targets,
        dense_positive_radius=0.02,
        dense_positive_topk=4,
        memory_match_loss_weight=1.0,
        dense_heatmap_sigma=0.02,
        dense_heatmap_neg_weight=0.01,
    )
    assert torch.isfinite(loss)
    assert metrics["memory_match_loss"] > 0.0
    assert metrics["memory_match_loss_weight"] == 1.0
    loss.backward()
    assert model.memory_gate[-1].weight.grad is not None
    assert model.memory_cross_attn.in_proj_weight.grad is not None
    assert model.dense_obj_conv[-1].weight.grad is not None


def test_native_video_loss_dense_hard_negative_mining_selects_negatives() -> None:
    torch.manual_seed(0)
    model = NativeVideoDetector(
        clip_len=4,
        future_len=1,
        num_queries=8,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        query_mode="dense",
        dense_obj_source="conv",
        box_size_scale=0.02,
    )
    outputs = model(torch.rand(1, 4, 3, 64, 64))
    targets = [torch.tensor([[0.20, 0.20, 0.01, 0.02]], dtype=torch.float32)]
    future_targets = [[targets[0], torch.tensor([[0.21, 0.20, 0.01, 0.02]], dtype=torch.float32)]]
    loss, metrics = native_video_detection_loss(
        outputs,
        targets,
        future_targets,
        dense_positive_radius=0.02,
        dense_positive_topk=4,
        dense_hard_negative_topk=16,
    )
    assert torch.isfinite(loss)
    assert metrics["dense_obj_pos"] >= 4.0
    assert metrics["dense_hard_neg_selected"] == 32.0
    loss.backward()
    assert model.dense_obj_conv[-1].weight.grad is not None


def test_native_video_loss_dense_ranking_penalizes_low_gt_scores() -> None:
    outputs = {
        "boxes": torch.tensor(
            [[[[0.20, 0.20, 0.02, 0.02]], [[0.50, 0.50, 0.02, 0.02]], [[0.70, 0.70, 0.02, 0.02]], [[0.30, 0.80, 0.02, 0.02]]]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "logits": torch.tensor([[[-1.0], [3.0], [2.0], [0.5]]], dtype=torch.float32, requires_grad=True),
        "anchor_centers": torch.tensor([[[0.20, 0.20], [0.50, 0.50], [0.70, 0.70], [0.30, 0.80]]], dtype=torch.float32),
    }
    targets = [torch.tensor([[0.20, 0.20, 0.02, 0.02]], dtype=torch.float32)]
    loss, metrics = native_video_detection_loss(
        outputs,
        targets,
        dense_positive_topk=1,
        dense_rank_weight=1.0,
        dense_rank_margin=1.0,
        dense_rank_negative_topk=2,
        obj_focal_gamma=0.0,
    )
    assert torch.isfinite(loss)
    assert metrics["dense_rank_loss"] > 0.0
    assert metrics["dense_rank_pairs"] == 2.0
    loss.backward()
    assert outputs["logits"].grad is not None
    assert outputs["logits"].grad[0, 0, 0] < 0.0
    assert outputs["logits"].grad[0, 1, 0] > 0.0


def test_native_video_loss_dense_ranking_all_positive_anchors() -> None:
    outputs = {
        "boxes": torch.tensor(
            [[[[0.20, 0.20, 0.02, 0.02]], [[0.21, 0.20, 0.02, 0.02]], [[0.50, 0.50, 0.02, 0.02]], [[0.70, 0.70, 0.02, 0.02]]]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "logits": torch.tensor([[[-1.0], [-0.5], [3.0], [2.0]]], dtype=torch.float32, requires_grad=True),
        "anchor_centers": torch.tensor([[[0.20, 0.20], [0.21, 0.20], [0.50, 0.50], [0.70, 0.70]]], dtype=torch.float32),
    }
    targets = [torch.tensor([[0.20, 0.20, 0.02, 0.02]], dtype=torch.float32)]
    loss, metrics = native_video_detection_loss(
        outputs,
        targets,
        dense_positive_radius=0.02,
        dense_rank_weight=1.0,
        dense_rank_margin=1.0,
        dense_rank_negative_topk=2,
        dense_rank_positive_mode="all",
        obj_focal_gamma=0.0,
    )
    assert torch.isfinite(loss)
    assert metrics["dense_rank_loss"] > 0.0
    assert metrics["dense_rank_pairs"] == 4.0
    assert metrics["dense_rank_positive_mode"] == "all"
    loss.backward()
    assert outputs["logits"].grad is not None
    assert outputs["logits"].grad[0, 0, 0] < 0.0
    assert outputs["logits"].grad[0, 1, 0] < 0.0
    assert outputs["logits"].grad[0, 2, 0] > 0.0


def test_native_video_loss_action_chunk_consistency_supervises_dense_future() -> None:
    outputs = {
        "boxes": torch.tensor(
            [
                [
                    [[0.20, 0.20, 0.02, 0.02], [0.20, 0.20, 0.02, 0.02], [0.20, 0.20, 0.02, 0.02]],
                    [[0.21, 0.20, 0.02, 0.02], [0.21, 0.20, 0.02, 0.02], [0.21, 0.20, 0.02, 0.02]],
                    [[0.60, 0.60, 0.02, 0.02], [0.60, 0.60, 0.02, 0.02], [0.60, 0.60, 0.02, 0.02]],
                    [[0.80, 0.80, 0.02, 0.02], [0.80, 0.80, 0.02, 0.02], [0.80, 0.80, 0.02, 0.02]],
                ]
            ],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "logits": torch.zeros(1, 4, 3, dtype=torch.float32, requires_grad=True),
        "anchor_centers": torch.tensor([[[0.20, 0.20], [0.21, 0.20], [0.60, 0.60], [0.80, 0.80]]], dtype=torch.float32),
    }
    targets = [torch.tensor([[0.20, 0.20, 0.02, 0.02]], dtype=torch.float32)]
    future_targets = [
        [
            targets[0],
            torch.tensor([[0.24, 0.20, 0.02, 0.02]], dtype=torch.float32),
            torch.tensor([[0.28, 0.20, 0.02, 0.02]], dtype=torch.float32),
        ]
    ]
    loss, metrics = native_video_detection_loss(
        outputs,
        targets,
        future_targets,
        dense_positive_topk=2,
        action_chunk_consistency_weight=1.0,
        obj_focal_gamma=0.0,
    )
    assert torch.isfinite(loss)
    assert metrics["action_chunk_consistency_weight"] == 1.0
    assert metrics["action_chunk_consistency_pos"] == 4.0
    assert metrics["action_chunk_consistency_loss"] > 0.0
    assert metrics["future_obj_pos"] == 4.0
    loss.backward()
    assert outputs["boxes"].grad is not None
    assert outputs["logits"].grad is not None
    assert outputs["boxes"].grad[0, 1, 1, 0] < 0.0
    assert outputs["logits"].grad[0, 1, 1] < 0.0


def test_native_video_loss_memory_quality_supervises_memory_weights() -> None:
    memory_weights = torch.tensor([[0.80, 0.10, 0.10]], dtype=torch.float32, requires_grad=True)
    outputs = {
        "boxes": torch.tensor(
            [[[[0.20, 0.20, 0.02, 0.02]], [[0.80, 0.80, 0.02, 0.02]]]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "logits": torch.zeros(1, 2, 1, dtype=torch.float32, requires_grad=True),
        "memory_weights": memory_weights,
    }
    targets = [torch.tensor([[0.20, 0.20, 0.02, 0.02]], dtype=torch.float32)]
    history_targets = [
        [
            torch.tensor([[0.80, 0.80, 0.02, 0.02]], dtype=torch.float32),
            torch.tensor([[0.20, 0.20, 0.02, 0.02]], dtype=torch.float32),
            torch.tensor([[0.21, 0.20, 0.02, 0.02]], dtype=torch.float32),
        ]
    ]
    loss, metrics = native_video_detection_loss(
        outputs,
        targets,
        history_targets=history_targets,
        memory_quality_weight=1.0,
        memory_quality_sigma=0.03,
        obj_focal_gamma=0.0,
    )
    assert torch.isfinite(loss)
    assert metrics["memory_quality_weight"] == 1.0
    assert metrics["memory_quality_samples"] == 1.0
    assert metrics["memory_quality_loss"] > 0.0
    loss.backward()
    assert memory_weights.grad is not None
    assert memory_weights.grad[0, 0] > -1e-6
    assert memory_weights.grad[0, 1] < 0.0


def test_native_video_loss_memory_quality_recency_prefers_recent_history() -> None:
    memory_weights = torch.tensor([[0.80, 0.10, 0.10]], dtype=torch.float32, requires_grad=True)
    outputs = {
        "boxes": torch.tensor(
            [[[[0.20, 0.20, 0.02, 0.02]], [[0.80, 0.80, 0.02, 0.02]]]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "logits": torch.zeros(1, 2, 1, dtype=torch.float32, requires_grad=True),
        "memory_weights": memory_weights,
    }
    targets = [torch.tensor([[0.20, 0.20, 0.02, 0.02]], dtype=torch.float32)]
    history_targets = [
        [
            torch.tensor([[0.20, 0.20, 0.02, 0.02]], dtype=torch.float32),
            torch.tensor([[0.21, 0.20, 0.02, 0.02]], dtype=torch.float32),
            torch.tensor([[0.20, 0.20, 0.02, 0.02]], dtype=torch.float32),
        ]
    ]
    loss, metrics = native_video_detection_loss(
        outputs,
        targets,
        history_targets=history_targets,
        memory_quality_weight=1.0,
        memory_quality_sigma=0.08,
        memory_quality_recency_tau=0.5,
        memory_quality_exclude_current=True,
        obj_focal_gamma=0.0,
    )
    assert torch.isfinite(loss)
    assert metrics["memory_quality_samples"] == 1.0
    assert metrics["memory_quality_exclude_current"] == 1.0
    assert metrics["memory_quality_target_entropy"] < 0.5
    loss.backward()
    assert memory_weights.grad is not None
    assert memory_weights.grad[0, 1] < memory_weights.grad[0, 0]
    assert abs(float(memory_weights.grad[0, 2])) < 1e-6


def test_native_video_loss_memory_quality_ignores_padded_current_frame_ids() -> None:
    memory_weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32, requires_grad=True)
    outputs = {
        "boxes": torch.tensor(
            [[[[0.20, 0.20, 0.02, 0.02]], [[0.80, 0.80, 0.02, 0.02]]]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "logits": torch.zeros(1, 2, 1, dtype=torch.float32, requires_grad=True),
        "memory_weights": memory_weights,
    }
    targets = [torch.tensor([[0.20, 0.20, 0.02, 0.02]], dtype=torch.float32)]
    history_targets = [
        [
            torch.tensor([[0.20, 0.20, 0.02, 0.02]], dtype=torch.float32),
            torch.tensor([[0.20, 0.20, 0.02, 0.02]], dtype=torch.float32),
            torch.tensor([[0.21, 0.20, 0.02, 0.02]], dtype=torch.float32),
            torch.tensor([[0.20, 0.20, 0.02, 0.02]], dtype=torch.float32),
        ]
    ]
    loss, metrics = native_video_detection_loss(
        outputs,
        targets,
        history_targets=history_targets,
        history_frame_ids=[[10, 10, 9, 10]],
        current_frame_ids=[10],
        memory_quality_weight=1.0,
        memory_quality_sigma=0.08,
        memory_quality_recency_tau=0.5,
        memory_quality_exclude_current=True,
        obj_focal_gamma=0.0,
    )
    assert torch.isfinite(loss)
    assert metrics["memory_quality_samples"] == 1.0
    assert metrics["memory_quality_target_entropy"] < 1e-4
    loss.backward()
    assert memory_weights.grad is not None
    assert memory_weights.grad[0, 2] < -3.9
    assert abs(float(memory_weights.grad[0, 0])) < 1e-6
    assert abs(float(memory_weights.grad[0, 1])) < 1e-6
    assert abs(float(memory_weights.grad[0, 3])) < 1e-6


def test_native_video_loss_dense_heatmap_guides_center_scores() -> None:
    proposal_logits = torch.tensor([[-2.0, 2.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    outputs = {
        "boxes": torch.tensor(
            [[[[0.20, 0.20, 0.02, 0.02]], [[0.50, 0.50, 0.02, 0.02]], [[0.70, 0.70, 0.02, 0.02]], [[0.30, 0.80, 0.02, 0.02]]]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "logits": torch.tensor([[[-2.0], [2.0], [0.0], [0.0]]], dtype=torch.float32, requires_grad=True),
        "proposal_logits": proposal_logits,
        "anchor_centers": torch.tensor([[[0.20, 0.20], [0.50, 0.50], [0.70, 0.70], [0.30, 0.80]]], dtype=torch.float32),
    }
    targets = [torch.tensor([[0.20, 0.20, 0.02, 0.02]], dtype=torch.float32)]
    loss, metrics = native_video_detection_loss(
        outputs,
        targets,
        box_weight=0.0,
        giou_weight=0.0,
        obj_weight=0.0,
        future_weight=0.0,
        dense_heatmap_weight=1.0,
        dense_heatmap_sigma=0.02,
        dense_heatmap_neg_weight=0.1,
        dense_heatmap_focal_gamma=0.0,
    )
    assert torch.isfinite(loss)
    assert metrics["dense_heatmap_loss"] > 0.0
    assert metrics["dense_heatmap_weight"] == 1.0
    assert metrics["dense_heatmap_source"] == "proposal_logits"
    assert metrics["dense_heatmap_pos_mass"] >= 1.0
    loss.backward()
    assert proposal_logits.grad is not None
    assert proposal_logits.grad[0, 0] < 0.0
    assert proposal_logits.grad[0, 1] > 0.0


def test_native_video_loss_handles_reordered_future_targets() -> None:
    outputs = {
        "boxes": torch.tensor(
            [[[[0.20, 0.20, 0.10, 0.10], [0.22, 0.20, 0.10, 0.10]], [[0.80, 0.80, 0.10, 0.10], [0.82, 0.80, 0.10, 0.10]]]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "logits": torch.zeros(1, 2, 2, dtype=torch.float32, requires_grad=True),
    }
    targets = [torch.tensor([[0.20, 0.20, 0.10, 0.10], [0.80, 0.80, 0.10, 0.10]], dtype=torch.float32)]
    future_targets = [
        [
            targets[0],
            torch.tensor([[0.82, 0.80, 0.10, 0.10], [0.22, 0.20, 0.10, 0.10]], dtype=torch.float32),
        ]
    ]
    loss, metrics = native_video_detection_loss(outputs, targets, future_targets)
    assert torch.isfinite(loss)
    assert metrics["matched"] == 2.0
    assert metrics["future_obj_pos"] == 2.0
    assert metrics["future_loss"] < 1e-6


def test_native_video_loss_focal_objectness_is_configurable() -> None:
    outputs = {
        "boxes": torch.tensor([[[[0.20, 0.20, 0.10, 0.10]], [[0.80, 0.80, 0.10, 0.10]]]], dtype=torch.float32, requires_grad=True),
        "logits": torch.tensor([[[0.0], [2.0]]], dtype=torch.float32, requires_grad=True),
    }
    targets = [torch.tensor([[0.20, 0.20, 0.10, 0.10]], dtype=torch.float32)]
    focal_loss, focal_metrics = native_video_detection_loss(outputs, targets, obj_focal_gamma=2.0, obj_focal_alpha=0.25)
    bce_loss, bce_metrics = native_video_detection_loss(outputs, targets, obj_focal_gamma=0.0)
    assert torch.isfinite(focal_loss)
    assert torch.isfinite(bce_loss)
    assert focal_metrics["obj_focal_gamma"] == 2.0
    assert bce_metrics["obj_focal_gamma"] == 0.0
    assert focal_metrics["obj_loss"] != bce_metrics["obj_loss"]


def test_native_video_loss_weights_are_configurable() -> None:
    outputs = {
        "boxes": torch.tensor([[[[0.20, 0.20, 0.10, 0.10]], [[0.80, 0.80, 0.10, 0.10]]]], dtype=torch.float32, requires_grad=True),
        "logits": torch.zeros(1, 2, 1, dtype=torch.float32, requires_grad=True),
    }
    targets = [torch.tensor([[0.30, 0.30, 0.10, 0.10]], dtype=torch.float32)]
    default_loss, default_metrics = native_video_detection_loss(outputs, targets, obj_focal_gamma=0.0)
    box_heavy_loss, box_heavy_metrics = native_video_detection_loss(outputs, targets, box_weight=10.0, obj_focal_gamma=0.0)
    no_obj_loss, no_obj_metrics = native_video_detection_loss(outputs, targets, obj_weight=0.0, obj_focal_gamma=0.0)
    assert torch.isfinite(default_loss)
    assert torch.isfinite(box_heavy_loss)
    assert torch.isfinite(no_obj_loss)
    assert box_heavy_loss > default_loss
    assert no_obj_loss < default_loss
    assert default_metrics["box_loss"] == box_heavy_metrics["box_loss"]
    assert no_obj_metrics["obj_loss"] == default_metrics["obj_loss"]


if __name__ == "__main__":
    test_native_video_detector_shapes_and_backward()
    test_native_video_detector_global_encoder_mode_shapes()
    test_native_video_detector_patch_stride_16_shapes()
    test_native_video_detector_patch_stride_4_dense_shapes()
    test_native_video_detector_spatial_refine_backward()
    test_native_video_detector_samurai_memory_motion_channels_backward()
    test_native_video_detector_tiny_box_size_prior()
    test_native_video_detector_dense_query_mode_shapes_and_backward()
    test_native_video_detector_dense_conv_objectness_shapes_and_backward()
    test_native_video_detector_dense_proposal_heatmap_head_backward()
    test_native_video_detector_dense_quality_head_backward()
    test_native_video_detector_dense_samurai_pooled_memory_attention_backward()
    test_native_video_detector_samurai_motion_score_branch_backward()
    test_native_video_detector_samurai_memory_match_branch_backward()
    test_native_video_loss_dense_hard_negative_mining_selects_negatives()
    test_native_video_loss_dense_ranking_penalizes_low_gt_scores()
    test_native_video_loss_dense_ranking_all_positive_anchors()
    test_native_video_loss_dense_heatmap_guides_center_scores()
    test_native_video_loss_action_chunk_consistency_supervises_dense_future()
    test_native_video_loss_memory_quality_supervises_memory_weights()
    test_native_video_loss_memory_quality_recency_prefers_recent_history()
    test_native_video_loss_memory_quality_ignores_padded_current_frame_ids()
    test_native_video_loss_handles_reordered_future_targets()
    test_native_video_loss_focal_objectness_is_configurable()
    test_native_video_loss_weights_are_configurable()
