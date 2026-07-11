from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=-1)


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=a.dtype, device=a.device)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))[None, :]
    return inter / (area_a + area_b - inter).clamp(min=1e-9)


def generalized_iou_loss(pred_cxcywh: torch.Tensor, target_cxcywh: torch.Tensor) -> torch.Tensor:
    pred = cxcywh_to_xyxy(pred_cxcywh)
    target = cxcywh_to_xyxy(target_cxcywh)
    lt = torch.maximum(pred[:, :2], target[:, :2])
    rb = torch.minimum(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = (area_p + area_t - inter).clamp(min=1e-9)
    iou = inter / union
    c_lt = torch.minimum(pred[:, :2], target[:, :2])
    c_rb = torch.maximum(pred[:, 2:], target[:, 2:])
    c_wh = (c_rb - c_lt).clamp(min=0)
    c_area = (c_wh[:, 0] * c_wh[:, 1]).clamp(min=1e-9)
    giou = iou - (c_area - union) / c_area
    return (1.0 - giou).mean()


def generalized_iou_matrix(pred_cxcywh: torch.Tensor, target_cxcywh: torch.Tensor) -> torch.Tensor:
    pred = cxcywh_to_xyxy(pred_cxcywh)
    target = cxcywh_to_xyxy(target_cxcywh)
    if pred.numel() == 0 or target.numel() == 0:
        return torch.zeros((pred.shape[0], target.shape[0]), dtype=pred.dtype, device=pred.device)
    lt = torch.maximum(pred[:, None, :2], target[None, :, :2])
    rb = torch.minimum(pred[:, None, 2:], target[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_p = ((pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0))[:, None]
    area_t = ((target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0))[None, :]
    union = (area_p + area_t - inter).clamp(min=1e-9)
    iou = inter / union
    c_lt = torch.minimum(pred[:, None, :2], target[None, :, :2])
    c_rb = torch.maximum(pred[:, None, 2:], target[None, :, 2:])
    c_wh = (c_rb - c_lt).clamp(min=0)
    c_area = (c_wh[..., 0] * c_wh[..., 1]).clamp(min=1e-9)
    return iou - (c_area - union) / c_area


def _detr_match(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, l1_weight: float = 5.0, giou_weight: float = 2.0) -> list[tuple[int, int]]:
    if target_boxes.numel() == 0:
        return []
    l1_cost = torch.cdist(pred_boxes, target_boxes, p=1)
    giou_cost = 1.0 - generalized_iou_matrix(pred_boxes, target_boxes)
    cost = (l1_weight * l1_cost + giou_weight * giou_cost).detach().cpu().numpy()
    query_idx, target_idx = linear_sum_assignment(cost)
    return [(int(q), int(t)) for q, t in zip(query_idx, target_idx)]


def _nearest_future_target(current_target: torch.Tensor, step_targets: torch.Tensor, target_idx: int) -> torch.Tensor:
    if step_targets.shape[0] == 1:
        return step_targets[0]
    if 0 <= target_idx < step_targets.shape[0]:
        fallback = step_targets[target_idx]
    else:
        fallback = step_targets[0]
    if current_target.numel() < 4 or step_targets.numel() == 0:
        return fallback
    distances = torch.norm(step_targets[:, :2] - current_target[:2], p=1, dim=1)
    return step_targets[int(torch.argmin(distances).item())]


def _memory_quality_distribution(
    current_targets: torch.Tensor,
    history_targets: list[torch.Tensor],
    history_frame_ids: list[int] | None,
    current_frame_id: int | None,
    sigma: float,
    recency_tau: float,
    exclude_current: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if current_targets.numel() == 0 or not history_targets:
        return None
    sigma = max(float(sigma), 1e-6)
    current_centers = current_targets[:, :2].to(device=device, dtype=dtype)
    scores: list[torch.Tensor] = []
    last_idx = len(history_targets) - 1
    for idx, step_targets in enumerate(history_targets):
        step_targets = step_targets.to(device=device, dtype=dtype)
        if step_targets.numel() == 0:
            scores.append(torch.zeros((), device=device, dtype=dtype))
            continue
        distances = torch.cdist(step_targets[:, :2], current_centers, p=2)
        min_distance = distances.min()
        quality = torch.exp(-0.5 * (min_distance / sigma).pow(2.0))
        age = max(0, last_idx - idx)
        is_current_frame = exclude_current and age == 0
        if (
            exclude_current
            and history_frame_ids is not None
            and current_frame_id is not None
            and idx < len(history_frame_ids)
            and int(history_frame_ids[idx]) == int(current_frame_id)
        ):
            is_current_frame = True
        if is_current_frame:
            quality = quality * 0.0
        elif recency_tau > 0.0:
            age_tensor = torch.tensor(float(age), device=device, dtype=dtype)
            quality = quality * torch.exp(-age_tensor / float(recency_tau))
        scores.append(quality)
    quality = torch.stack(scores)
    total = quality.sum()
    if float(total.detach().cpu()) <= 1e-8:
        return None
    return quality / total.clamp(min=1e-8)


def native_video_detection_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[torch.Tensor],
    future_targets: list[list[torch.Tensor]] | None = None,
    history_targets: list[list[torch.Tensor]] | None = None,
    history_frame_ids: list[list[int]] | None = None,
    current_frame_ids: list[int] | None = None,
    box_weight: float = 5.0,
    giou_weight: float = 2.0,
    obj_weight: float = 1.0,
    future_weight: float = 0.5,
    noobj_weight: float = 0.1,
    obj_focal_gamma: float = 2.0,
    obj_focal_alpha: float = 0.25,
    dense_positive_radius: float = 0.0,
    dense_positive_topk: int = 0,
    dense_hard_negative_topk: int = 0,
    dense_rank_weight: float = 0.0,
    dense_rank_margin: float = 1.0,
    dense_rank_negative_topk: int = 0,
    dense_rank_positive_mode: str = "max",
    action_chunk_consistency_weight: float = 0.0,
    memory_quality_weight: float = 0.0,
    memory_quality_sigma: float = 0.08,
    memory_quality_recency_tau: float = 0.0,
    memory_quality_exclude_current: bool = False,
    motion_obj_weight: float = 0.0,
    dense_heatmap_weight: float = 0.0,
    dense_heatmap_sigma: float = 0.02,
    dense_heatmap_neg_weight: float = 0.02,
    dense_heatmap_focal_gamma: float = 2.0,
    memory_match_loss_weight: float = 0.0,
    quality_loss_weight: float = 0.0,
    quality_positive_iou: float = 0.05,
    quality_hard_negative_topk: int = 0,
    quality_focal_gamma: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    boxes = outputs["boxes"]
    logits = outputs["logits"]
    if dense_rank_positive_mode not in {"max", "all"}:
        raise ValueError(f"unsupported dense_rank_positive_mode: {dense_rank_positive_mode}")
    bsz, num_queries, future_len, _ = boxes.shape
    device = boxes.device
    obj_target = torch.zeros((bsz, num_queries, future_len), dtype=torch.float32, device=device)
    anchor_centers = outputs.get("anchor_centers")
    box_losses: list[torch.Tensor] = []
    giou_losses: list[torch.Tensor] = []
    future_losses: list[torch.Tensor] = []
    action_chunk_consistency_losses: list[torch.Tensor] = []
    memory_quality_losses: list[torch.Tensor] = []
    dense_rank_groups: list[tuple[int, torch.Tensor]] = []
    matched = 0
    dense_obj_pos = 0
    dense_box_pos = 0
    action_chunk_consistency_pos = 0
    memory_quality_samples = 0
    memory_quality_target_entropy = 0.0
    heatmap_target = None
    if anchor_centers is not None and (dense_heatmap_weight > 0.0 or memory_match_loss_weight > 0.0):
        heatmap_target = torch.zeros((bsz, num_queries), dtype=torch.float32, device=device)
        dense_heatmap_sigma = max(float(dense_heatmap_sigma), 1e-6)
    quality_logits = outputs.get("quality_logits")
    if quality_logits is not None and quality_logits.ndim == 3:
        quality_logits = quality_logits[:, :, 0]
    quality_target = None
    if quality_logits is not None and quality_loss_weight > 0.0:
        quality_target = torch.zeros((bsz, num_queries), dtype=torch.float32, device=device)
    memory_weights = outputs.get("memory_weights")
    for batch_idx, target in enumerate(targets):
        target = target.to(device)
        if (
            memory_weights is not None
            and history_targets is not None
            and memory_quality_weight > 0.0
            and batch_idx < len(history_targets)
        ):
            target_distribution = _memory_quality_distribution(
                target,
                history_targets[batch_idx],
                history_frame_ids[batch_idx] if history_frame_ids is not None and batch_idx < len(history_frame_ids) else None,
                current_frame_ids[batch_idx] if current_frame_ids is not None and batch_idx < len(current_frame_ids) else None,
                memory_quality_sigma,
                memory_quality_recency_tau,
                memory_quality_exclude_current,
                device,
                memory_weights.dtype,
            )
            if target_distribution is not None and target_distribution.shape[0] == memory_weights.shape[1]:
                weights = memory_weights[batch_idx].clamp(min=1e-8)
                memory_quality_losses.append(-(target_distribution * weights.log()).sum())
                entropy = -(target_distribution * target_distribution.clamp(min=1e-8).log()).sum()
                memory_quality_target_entropy += float(entropy.detach().cpu())
                memory_quality_samples += 1
        if quality_target is not None and target.numel() > 0:
            pred_xyxy = cxcywh_to_xyxy(boxes[batch_idx, :, 0, :].detach())
            target_xyxy = cxcywh_to_xyxy(target)
            quality_target[batch_idx] = box_iou_xyxy(pred_xyxy, target_xyxy).max(dim=1).values.detach()
        matches = _detr_match(boxes[batch_idx, :, 0, :], target, l1_weight=box_weight, giou_weight=giou_weight)
        if heatmap_target is not None and target.numel() > 0:
            anchors = anchor_centers[batch_idx].to(device)
            for target_box in target:
                delta = (anchors - target_box[None, :2]) / dense_heatmap_sigma
                score = torch.exp(-0.5 * (delta * delta).sum(dim=1))
                heatmap_target[batch_idx] = torch.maximum(heatmap_target[batch_idx], score)
        if (
            anchor_centers is not None
            and target.numel() > 0
            and (dense_positive_radius > 0.0 or dense_positive_topk > 0)
        ):
            anchors = anchor_centers[batch_idx].to(device)
            for target_box in target:
                distances = torch.norm(anchors - target_box[None, :2], p=1, dim=1)
                positive = torch.zeros((num_queries,), dtype=torch.bool, device=device)
                if dense_positive_radius > 0.0:
                    positive |= (torch.abs(anchors[:, 0] - target_box[0]) <= dense_positive_radius) & (
                        torch.abs(anchors[:, 1] - target_box[1]) <= dense_positive_radius
                    )
                if dense_positive_topk > 0:
                    _, topk_idx = torch.topk(distances, k=min(int(dense_positive_topk), num_queries), largest=False)
                    positive[topk_idx] = True
                obj_target[batch_idx, positive, 0] = 1.0
                if positive.any():
                    dense_rank_groups.append((batch_idx, positive.detach().clone()))
                    dense_pred = boxes[batch_idx, positive, 0, :]
                    dense_target = target_box.reshape(1, 4).expand_as(dense_pred)
                    box_losses.append(F.smooth_l1_loss(dense_pred, dense_target))
                    giou_losses.append(generalized_iou_loss(dense_pred, dense_target))
                    dense_box_pos += int(positive.sum().detach().cpu())
                    if action_chunk_consistency_weight > 0.0 and future_targets is not None:
                        positive_count = int(positive.sum().detach().cpu())
                        for step in range(1, min(future_len, len(future_targets[batch_idx]))):
                            step_targets = future_targets[batch_idx][step].to(device)
                            if step_targets.numel() == 0:
                                continue
                            obj_target[batch_idx, positive, step] = 1.0
                            future_target = _nearest_future_target(target_box, step_targets, -1)
                            future_pred = boxes[batch_idx, positive, step, :]
                            dense_future_target = future_target.reshape(1, 4).expand_as(future_pred)
                            action_chunk_consistency_losses.append(F.smooth_l1_loss(future_pred, dense_future_target))
                            action_chunk_consistency_pos += positive_count
            dense_obj_pos += int(obj_target[batch_idx, :, 0].sum().detach().cpu())
        for query_idx, target_idx in matches:
            obj_target[batch_idx, query_idx, 0] = 1.0
            pred = boxes[batch_idx, query_idx : query_idx + 1, 0, :]
            tgt = target[target_idx : target_idx + 1]
            box_losses.append(F.smooth_l1_loss(pred, tgt))
            giou_losses.append(generalized_iou_loss(pred, tgt))
            matched += 1
            if future_targets is not None:
                for step in range(1, min(future_len, len(future_targets[batch_idx]))):
                    step_targets = future_targets[batch_idx][step].to(device)
                    if step_targets.numel() == 0:
                        continue
                    obj_target[batch_idx, query_idx, step] = 1.0
                    future_target = _nearest_future_target(target[target_idx], step_targets, target_idx)
                    future_losses.append(F.smooth_l1_loss(boxes[batch_idx, query_idx, step, :], future_target))
    def weighted_objectness_loss(raw_logits: torch.Tensor) -> torch.Tensor:
        loss_raw = F.binary_cross_entropy_with_logits(raw_logits, obj_target, reduction="none")
        if obj_focal_gamma <= 0:
            return loss_raw * obj_loss_weight
        obj_prob = torch.sigmoid(raw_logits)
        p_t = obj_prob * obj_target + (1.0 - obj_prob) * (1.0 - obj_target)
        focal_factor = (1.0 - p_t).clamp(min=0.0, max=1.0).pow(obj_focal_gamma)
        if obj_focal_alpha >= 0:
            alpha_t = obj_focal_alpha * obj_target + (1.0 - obj_focal_alpha) * (1.0 - obj_target)
            focal_factor = focal_factor * alpha_t
        return loss_raw * focal_factor * obj_loss_weight

    obj_loss_weight = torch.where(obj_target > 0.5, torch.ones_like(obj_target), torch.full_like(obj_target, noobj_weight))
    obj_loss_weighted = weighted_objectness_loss(logits)

    def reduce_objectness_loss(weighted_loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        selected_hard_negatives = 0
        if not (anchor_centers is not None and dense_obj_pos > 0 and dense_hard_negative_topk > 0):
            if anchor_centers is not None and dense_obj_pos > 0:
                normalizer = obj_target.sum().clamp(min=1.0)
                return weighted_loss.sum() / normalizer, normalizer, selected_hard_negatives
            normalizer = obj_loss_weight.sum().clamp(min=1.0)
            return weighted_loss.sum() / normalizer, normalizer, selected_hard_negatives
        selected_losses: list[torch.Tensor] = []
        for batch_idx in range(bsz):
            for step in range(future_len):
                positive = obj_target[batch_idx, :, step] > 0.5
                if positive.any():
                    selected_losses.append(weighted_loss[batch_idx, positive, step])
                negative = ~positive
                if negative.any():
                    neg_losses = weighted_loss[batch_idx, negative, step]
                    hard_count = min(int(dense_hard_negative_topk), int(neg_losses.numel()))
                    if hard_count > 0:
                        selected_losses.append(torch.topk(neg_losses, k=hard_count, largest=True).values)
                        selected_hard_negatives += hard_count
        if selected_losses:
            selected = torch.cat([loss.reshape(-1) for loss in selected_losses])
            normalizer = selected.new_tensor(float(max(dense_obj_pos + selected_hard_negatives * noobj_weight, 1.0)))
            return selected.sum() / normalizer, normalizer, selected_hard_negatives
        normalizer = obj_loss_weight.sum().clamp(min=1.0)
        return weighted_loss.sum() / normalizer, normalizer, selected_hard_negatives

    obj_loss, obj_loss_normalizer, dense_hard_neg_selected = reduce_objectness_loss(obj_loss_weighted)
    motion_logits = outputs.get("motion_logits")
    motion_obj_loss = logits.sum() * 0.0
    motion_hard_neg_selected = 0
    motion_obj_loss_normalizer = obj_loss_normalizer
    if motion_logits is not None and motion_obj_weight > 0.0:
        motion_obj_loss_weighted = weighted_objectness_loss(motion_logits)
        motion_obj_loss, motion_obj_loss_normalizer, motion_hard_neg_selected = reduce_objectness_loss(motion_obj_loss_weighted)
    proposal_logits = outputs.get("proposal_logits")
    heatmap_logits = proposal_logits
    dense_heatmap_source = "proposal_logits" if proposal_logits is not None else "final_logits"
    if heatmap_logits is not None and heatmap_logits.ndim == 3:
        heatmap_logits = heatmap_logits[:, :, 0]
    dense_heatmap_loss = logits.sum() * 0.0
    dense_heatmap_pos_mass = 0.0
    dense_heatmap_normalizer = logits.new_tensor(1.0)
    memory_match_logits = outputs.get("memory_match_logits")
    if memory_match_logits is not None and memory_match_logits.ndim == 3:
        memory_match_logits = memory_match_logits[:, :, 0]
    memory_match_loss = logits.sum() * 0.0
    memory_match_normalizer = logits.new_tensor(1.0)
    if heatmap_target is not None and dense_heatmap_weight > 0.0:
        current_logits = heatmap_logits if heatmap_logits is not None else logits[:, :, 0]
        heatmap_loss_raw = F.binary_cross_entropy_with_logits(current_logits, heatmap_target, reduction="none")
        if dense_heatmap_focal_gamma > 0.0:
            prob = torch.sigmoid(current_logits)
            p_t = prob * heatmap_target + (1.0 - prob) * (1.0 - heatmap_target)
            heatmap_loss_raw = heatmap_loss_raw * (1.0 - p_t).clamp(min=0.0, max=1.0).pow(float(dense_heatmap_focal_gamma))
        positive_weight = torch.ones_like(heatmap_target)
        negative_weight = torch.full_like(heatmap_target, float(max(dense_heatmap_neg_weight, 0.0)))
        heatmap_weight = torch.where(heatmap_target > 1e-3, positive_weight, negative_weight)
        dense_heatmap_pos_mass_tensor = heatmap_target.sum()
        dense_heatmap_pos_mass = float(dense_heatmap_pos_mass_tensor.detach().cpu())
        dense_heatmap_normalizer = (dense_heatmap_pos_mass_tensor + negative_weight[heatmap_target <= 1e-3].sum()).clamp(min=1.0)
        dense_heatmap_loss = (heatmap_loss_raw * heatmap_weight).sum() / dense_heatmap_normalizer
    elif heatmap_target is not None:
        dense_heatmap_pos_mass = float(heatmap_target.sum().detach().cpu())
    if heatmap_target is not None and memory_match_logits is not None and memory_match_loss_weight > 0.0:
        memory_match_loss_raw = F.binary_cross_entropy_with_logits(memory_match_logits, heatmap_target, reduction="none")
        if dense_heatmap_focal_gamma > 0.0:
            memory_match_prob = torch.sigmoid(memory_match_logits)
            memory_match_p_t = memory_match_prob * heatmap_target + (1.0 - memory_match_prob) * (1.0 - heatmap_target)
            memory_match_loss_raw = memory_match_loss_raw * (1.0 - memory_match_p_t).clamp(min=0.0, max=1.0).pow(
                float(dense_heatmap_focal_gamma)
            )
        positive_weight = torch.ones_like(heatmap_target)
        negative_weight = torch.full_like(heatmap_target, float(max(dense_heatmap_neg_weight, 0.0)))
        memory_match_weight = torch.where(heatmap_target > 1e-3, positive_weight, negative_weight)
        memory_match_pos_mass_tensor = heatmap_target.sum()
        memory_match_normalizer = (memory_match_pos_mass_tensor + negative_weight[heatmap_target <= 1e-3].sum()).clamp(min=1.0)
        memory_match_loss = (memory_match_loss_raw * memory_match_weight).sum() / memory_match_normalizer
    quality_loss = logits.sum() * 0.0
    quality_target_pos = 0
    quality_target_max = 0.0
    quality_hard_neg_selected = 0
    quality_loss_normalizer = logits.new_tensor(1.0)
    if quality_target is not None and quality_logits is not None and quality_loss_weight > 0.0:
        quality_positive_iou = max(0.0, min(float(quality_positive_iou), 1.0))
        quality_loss_raw = F.binary_cross_entropy_with_logits(quality_logits, quality_target, reduction="none")
        if quality_focal_gamma > 0.0:
            quality_prob = torch.sigmoid(quality_logits)
            quality_p_t = quality_prob * quality_target + (1.0 - quality_prob) * (1.0 - quality_target)
            quality_loss_raw = quality_loss_raw * (1.0 - quality_p_t).clamp(min=0.0, max=1.0).pow(float(quality_focal_gamma))
        quality_positive = quality_target >= quality_positive_iou
        quality_target_pos = int(quality_positive.sum().detach().cpu())
        quality_target_max = float(quality_target.max().detach().cpu()) if quality_target.numel() else 0.0
        quality_weight = torch.where(quality_positive, torch.ones_like(quality_target), torch.full_like(quality_target, noobj_weight))
        quality_weighted = quality_loss_raw * quality_weight
        selected_quality_losses: list[torch.Tensor] = []
        if quality_hard_negative_topk > 0:
            for batch_idx in range(bsz):
                positive = quality_positive[batch_idx]
                if positive.any():
                    selected_quality_losses.append(quality_weighted[batch_idx, positive])
                negative = ~positive
                if negative.any():
                    neg_losses = quality_weighted[batch_idx, negative]
                    hard_count = min(int(quality_hard_negative_topk), int(neg_losses.numel()))
                    if hard_count > 0:
                        selected_quality_losses.append(torch.topk(neg_losses, k=hard_count, largest=True).values)
                        quality_hard_neg_selected += hard_count
        if selected_quality_losses:
            selected_quality = torch.cat([loss.reshape(-1) for loss in selected_quality_losses])
            quality_loss_normalizer = selected_quality.new_tensor(float(max(quality_target_pos + quality_hard_neg_selected * noobj_weight, 1.0)))
            quality_loss = selected_quality.sum() / quality_loss_normalizer
        else:
            quality_loss_normalizer = quality_weight.sum().clamp(min=1.0)
            quality_loss = quality_weighted.sum() / quality_loss_normalizer
    box_loss = torch.stack(box_losses).mean() if box_losses else boxes.sum() * 0.0
    giou_loss = torch.stack(giou_losses).mean() if giou_losses else boxes.sum() * 0.0
    future_loss = torch.stack(future_losses).mean() if future_losses else boxes.sum() * 0.0
    action_chunk_consistency_loss = (
        torch.stack(action_chunk_consistency_losses).mean() if action_chunk_consistency_losses else boxes.sum() * 0.0
    )
    memory_quality_loss = torch.stack(memory_quality_losses).mean() if memory_quality_losses else logits.sum() * 0.0
    dense_rank_losses: list[torch.Tensor] = []
    dense_rank_pairs = 0
    if (
        anchor_centers is not None
        and dense_rank_weight > 0.0
        and dense_rank_margin > 0.0
        and dense_rank_negative_topk > 0
        and dense_rank_groups
    ):
        for batch_idx, positive in dense_rank_groups:
            pos_logits = logits[batch_idx, positive, 0]
            negative = obj_target[batch_idx, :, 0] <= 0.5
            neg_logits = logits[batch_idx, negative, 0]
            if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                continue
            hard_count = min(int(dense_rank_negative_topk), int(neg_logits.numel()))
            hard_neg = torch.topk(neg_logits, k=hard_count, largest=True).values
            if dense_rank_positive_mode == "all":
                rank_loss = F.relu(float(dense_rank_margin) - pos_logits[:, None] + hard_neg[None, :]).mean()
                dense_rank_pairs += hard_count * int(pos_logits.numel())
            else:
                pos_ref = pos_logits.max()
                rank_loss = F.relu(float(dense_rank_margin) - pos_ref + hard_neg).mean()
                dense_rank_pairs += hard_count
            dense_rank_losses.append(rank_loss)
    dense_rank_loss = torch.stack(dense_rank_losses).mean() if dense_rank_losses else logits.sum() * 0.0
    loss = (
        obj_weight * obj_loss
        + box_weight * box_loss
        + giou_weight * giou_loss
        + future_weight * future_loss
        + action_chunk_consistency_weight * action_chunk_consistency_loss
        + memory_quality_weight * memory_quality_loss
        + dense_rank_weight * dense_rank_loss
        + motion_obj_weight * motion_obj_loss
        + dense_heatmap_weight * dense_heatmap_loss
        + memory_match_loss_weight * memory_match_loss
        + quality_loss_weight * quality_loss
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "obj_loss": float(obj_loss.detach().cpu()),
        "box_loss": float(box_loss.detach().cpu()),
        "giou_loss": float(giou_loss.detach().cpu()),
        "future_loss": float(future_loss.detach().cpu()),
        "action_chunk_consistency_loss": float(action_chunk_consistency_loss.detach().cpu()),
        "action_chunk_consistency_weight": float(action_chunk_consistency_weight),
        "action_chunk_consistency_pos": float(action_chunk_consistency_pos),
        "memory_quality_loss": float(memory_quality_loss.detach().cpu()),
        "memory_quality_weight": float(memory_quality_weight),
        "memory_quality_sigma": float(memory_quality_sigma),
        "memory_quality_recency_tau": float(memory_quality_recency_tau),
        "memory_quality_exclude_current": float(bool(memory_quality_exclude_current)),
        "memory_quality_samples": float(memory_quality_samples),
        "memory_quality_target_entropy": float(memory_quality_target_entropy / max(memory_quality_samples, 1)),
        "dense_rank_loss": float(dense_rank_loss.detach().cpu()),
        "matched": float(matched),
        "future_obj_pos": float(obj_target[:, :, 1:].sum().detach().cpu()) if future_len > 1 else 0.0,
        "obj_focal_gamma": float(obj_focal_gamma),
        "dense_obj_pos": float(dense_obj_pos),
        "dense_box_pos": float(dense_box_pos),
        "dense_hard_neg_selected": float(dense_hard_neg_selected),
        "dense_rank_pairs": float(dense_rank_pairs),
        "dense_rank_weight": float(dense_rank_weight),
        "dense_rank_margin": float(dense_rank_margin),
        "dense_rank_negative_topk": float(dense_rank_negative_topk),
        "dense_rank_positive_mode": dense_rank_positive_mode,
        "motion_obj_loss": float(motion_obj_loss.detach().cpu()),
        "motion_obj_weight": float(motion_obj_weight),
        "motion_hard_neg_selected": float(motion_hard_neg_selected),
        "motion_obj_loss_normalizer": float(motion_obj_loss_normalizer.detach().cpu()),
        "obj_loss_normalizer": float(obj_loss_normalizer.detach().cpu()),
        "dense_heatmap_loss": float(dense_heatmap_loss.detach().cpu()),
        "dense_heatmap_weight": float(dense_heatmap_weight),
        "dense_heatmap_sigma": float(dense_heatmap_sigma),
        "dense_heatmap_neg_weight": float(dense_heatmap_neg_weight),
        "dense_heatmap_focal_gamma": float(dense_heatmap_focal_gamma),
        "dense_heatmap_source": dense_heatmap_source,
        "dense_heatmap_pos_mass": float(dense_heatmap_pos_mass),
        "dense_heatmap_loss_normalizer": float(dense_heatmap_normalizer.detach().cpu()),
        "memory_match_loss": float(memory_match_loss.detach().cpu()),
        "memory_match_loss_weight": float(memory_match_loss_weight),
        "memory_match_loss_normalizer": float(memory_match_normalizer.detach().cpu()),
        "quality_loss": float(quality_loss.detach().cpu()),
        "quality_loss_weight": float(quality_loss_weight),
        "quality_positive_iou": float(quality_positive_iou),
        "quality_hard_negative_topk": float(quality_hard_negative_topk),
        "quality_hard_neg_selected": float(quality_hard_neg_selected),
        "quality_focal_gamma": float(quality_focal_gamma),
        "quality_target_pos": float(quality_target_pos),
        "quality_target_max": float(quality_target_max),
        "quality_loss_normalizer": float(quality_loss_normalizer.detach().cpu()),
    }
