from __future__ import annotations

import torch
from torch import nn


class SpatialRefineBlock(nn.Module):
    """Lightweight high-resolution spatial refinement before temporal fusion."""

    def __init__(self, channels: int, kernel_size: int = 7, expansion: float = 2.0) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        hidden_channels = max(channels, int(round(float(channels) * float(expansion))))
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class NativeVideoDetector(nn.Module):
    """Small Octo-like video detector MVP.

    8-frame clip -> conv patch tokens -> temporal transformer encoder ->
    object query decoder -> current bbox plus future bbox/objectness chunk.

    ``future_len`` is the number of future frames after the current frame.
    The heads emit ``1 + future_len`` steps, with step 0 assigned to the
    current frame.
    """

    def __init__(
        self,
        clip_len: int = 8,
        future_len: int = 4,
        num_queries: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        channels_last: bool = False,
        encoder_mode: str = "factorized",
        patch_stride: int = 8,
        spatial_refine_layers: int = 0,
        spatial_refine_kernel: int = 7,
        spatial_refine_expansion: float = 2.0,
        motion_channels: bool = False,
        memory_mode: str = "last",
        box_size_scale: float = 1.0,
        query_mode: str = "learned",
        anchor_offset_cells: float = 4.0,
        dense_obj_source: str = "token",
        memory_attention: str = "none",
        memory_slots: int = 64,
        memory_match_mode: str = "none",
        memory_match_weight: float = 0.0,
        memory_match_temperature: float = 5.0,
        motion_score_mode: str = "none",
        motion_score_weight: float = 1.0,
        proposal_mode: str = "none",
        quality_score_mode: str = "none",
    ) -> None:
        super().__init__()
        self.clip_len = int(clip_len)
        self.future_len = int(future_len)
        self.chunk_len = self.future_len + 1
        self.num_queries = int(num_queries)
        self.d_model = int(d_model)
        self.channels_last = bool(channels_last)
        self.motion_channels = bool(motion_channels)
        if encoder_mode not in {"factorized", "global"}:
            raise ValueError(f"unsupported encoder_mode: {encoder_mode}")
        self.encoder_mode = encoder_mode
        if patch_stride not in {4, 8, 16}:
            raise ValueError(f"unsupported patch_stride: {patch_stride}")
        self.patch_stride = int(patch_stride)
        self.spatial_refine_layers = int(spatial_refine_layers)
        if self.spatial_refine_layers < 0:
            raise ValueError(f"spatial_refine_layers must be non-negative, got {spatial_refine_layers}")
        self.spatial_refine_kernel = int(spatial_refine_kernel)
        if self.spatial_refine_kernel <= 0 or self.spatial_refine_kernel % 2 == 0:
            raise ValueError(f"spatial_refine_kernel must be a positive odd integer, got {spatial_refine_kernel}")
        self.spatial_refine_expansion = float(spatial_refine_expansion)
        if self.spatial_refine_expansion <= 0.0:
            raise ValueError(f"spatial_refine_expansion must be positive, got {spatial_refine_expansion}")
        if memory_mode not in {"last", "samurai"}:
            raise ValueError(f"unsupported memory_mode: {memory_mode}")
        self.memory_mode = memory_mode
        self.box_size_scale = float(box_size_scale)
        if self.box_size_scale <= 0.0 or self.box_size_scale > 1.0:
            raise ValueError(f"box_size_scale must be in (0, 1], got {box_size_scale}")
        if query_mode not in {"learned", "dense"}:
            raise ValueError(f"unsupported query_mode: {query_mode}")
        self.query_mode = query_mode
        self.anchor_offset_cells = float(anchor_offset_cells)
        if self.anchor_offset_cells <= 0.0:
            raise ValueError(f"anchor_offset_cells must be positive, got {anchor_offset_cells}")
        if dense_obj_source not in {"token", "conv"}:
            raise ValueError(f"unsupported dense_obj_source: {dense_obj_source}")
        self.dense_obj_source = dense_obj_source
        if memory_attention not in {"none", "pooled_cross"}:
            raise ValueError(f"unsupported memory_attention: {memory_attention}")
        self.memory_attention = memory_attention
        self.memory_slots = int(memory_slots)
        if self.memory_slots <= 0:
            raise ValueError(f"memory_slots must be positive, got {memory_slots}")
        if memory_match_mode not in {"none", "slot_dot"}:
            raise ValueError(f"unsupported memory_match_mode: {memory_match_mode}")
        if memory_match_mode != "none" and self.query_mode != "dense":
            raise ValueError("memory_match_mode requires query_mode='dense'")
        self.memory_match_mode = memory_match_mode
        self.memory_match_weight = float(memory_match_weight)
        self.memory_match_temperature = max(1e-6, float(memory_match_temperature))
        if motion_score_mode not in {"none", "samurai"}:
            raise ValueError(f"unsupported motion_score_mode: {motion_score_mode}")
        if motion_score_mode != "none" and self.query_mode != "dense":
            raise ValueError("motion_score_mode requires query_mode='dense'")
        self.motion_score_mode = motion_score_mode
        self.motion_score_weight = float(motion_score_weight)
        if proposal_mode not in {"none", "heatmap"}:
            raise ValueError(f"unsupported proposal_mode: {proposal_mode}")
        if proposal_mode != "none" and self.query_mode != "dense":
            raise ValueError("proposal_mode requires query_mode='dense'")
        self.proposal_mode = proposal_mode
        if quality_score_mode not in {"none", "iou"}:
            raise ValueError(f"unsupported quality_score_mode: {quality_score_mode}")
        if quality_score_mode != "none" and self.query_mode != "dense":
            raise ValueError("quality_score_mode requires query_mode='dense'")
        self.quality_score_mode = quality_score_mode
        in_channels = 5 if self.motion_channels else 3
        stem_layers: list[nn.Module] = [
            nn.Conv2d(in_channels, d_model // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(d_model // 4, d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.SiLU(inplace=True),
        ]
        if self.patch_stride == 4:
            stem_layers.extend([
                nn.Conv2d(d_model // 2, d_model, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(d_model),
                nn.SiLU(inplace=True),
            ])
        else:
            stem_layers.extend([
                nn.Conv2d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(d_model),
                nn.SiLU(inplace=True),
            ])
            if self.patch_stride == 16:
                stem_layers.extend([
                    nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(d_model),
                    nn.SiLU(inplace=True),
                ])
        self.stem = nn.Sequential(*stem_layers)
        self.spatial_refine = nn.Sequential(
            *[
                SpatialRefineBlock(d_model, kernel_size=self.spatial_refine_kernel, expansion=self.spatial_refine_expansion)
                for _ in range(self.spatial_refine_layers)
            ]
        )
        self.frame_pos = nn.Parameter(torch.zeros(1, clip_len, 1, d_model))
        self.spatial_pos = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.query = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)
        self.memory_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(16, d_model // 2)),
            nn.GELU(),
            nn.Linear(max(16, d_model // 2), 1),
        )
        self.memory_fusion = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        if self.memory_attention == "pooled_cross":
            self.memory_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
            self.memory_attention_fusion = nn.Sequential(
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        self.box_out = nn.Linear(d_model, self.chunk_len * 4)
        self.box_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), self.box_out)
        self.obj_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, self.chunk_len))
        if self.query_mode == "dense":
            self.dense_box_out = nn.Linear(d_model, self.chunk_len * 4)
            self.dense_box_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), self.dense_box_out)
            if self.dense_obj_source == "conv":
                self.dense_obj_conv = nn.Sequential(
                    nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(d_model, self.chunk_len, kernel_size=1),
                )
            else:
                self.dense_obj_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, self.chunk_len))
            if self.motion_score_mode == "samurai":
                self.motion_score_head = nn.Sequential(
                    nn.LayerNorm(d_model * 4),
                    nn.Linear(d_model * 4, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, self.chunk_len),
                )
            if self.proposal_mode == "heatmap":
                self.proposal_conv = nn.Sequential(
                    nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(d_model, 1, kernel_size=1),
                )
            if self.quality_score_mode == "iou":
                self.quality_conv = nn.Sequential(
                    nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(d_model, 1, kernel_size=1),
                )
        self._init_tiny_box_prior()
        self._init_objectness_prior()
        self._init_motion_score_prior()

    def _init_tiny_box_prior(self) -> None:
        if self.box_size_scale >= 1.0:
            return
        initial_size = min(0.02, self.box_size_scale * 0.25)
        prior = min(0.95, max(1e-4, initial_size / self.box_size_scale))
        bias = torch.logit(torch.tensor(prior, dtype=self.box_out.bias.dtype))
        with torch.no_grad():
            for box_out in [self.box_out, getattr(self, "dense_box_out", None)]:
                if box_out is None:
                    continue
                for step in range(self.chunk_len):
                    box_out.bias[step * 4 + 2] = bias
                    box_out.bias[step * 4 + 3] = bias

    def _init_objectness_prior(self) -> None:
        prior = torch.tensor(0.01, dtype=self.box_out.bias.dtype)
        bias = torch.logit(prior)
        with torch.no_grad():
            for obj_head in [self.obj_head, getattr(self, "dense_obj_head", None)]:
                if obj_head is None:
                    continue
                final = obj_head[-1]
                if isinstance(final, nn.Linear):
                    final.bias.fill_(bias)
            dense_obj_conv = getattr(self, "dense_obj_conv", None)
            if dense_obj_conv is not None:
                final_conv = dense_obj_conv[-1]
                if isinstance(final_conv, nn.Conv2d):
                    final_conv.bias.fill_(bias)
            proposal_conv = getattr(self, "proposal_conv", None)
            if proposal_conv is not None:
                final_conv = proposal_conv[-1]
                if isinstance(final_conv, nn.Conv2d):
                    final_conv.bias.fill_(bias)
            quality_conv = getattr(self, "quality_conv", None)
            if quality_conv is not None:
                final_conv = quality_conv[-1]
                if isinstance(final_conv, nn.Conv2d):
                    final_conv.bias.fill_(bias)

    def _init_motion_score_prior(self) -> None:
        motion_score_head = getattr(self, "motion_score_head", None)
        if motion_score_head is None:
            return
        final = motion_score_head[-1]
        if isinstance(final, nn.Linear):
            with torch.no_grad():
                final.weight.zero_()
                final.bias.zero_()

    def _decode_boxes(self, raw_boxes: torch.Tensor) -> torch.Tensor:
        centers = torch.sigmoid(raw_boxes[..., :2])
        sizes = torch.sigmoid(raw_boxes[..., 2:]) * self.box_size_scale
        return torch.cat([centers, sizes], dim=-1)

    def _decode_dense_boxes(self, raw_boxes: torch.Tensor, feat_h: int, feat_w: int) -> tuple[torch.Tensor, torch.Tensor]:
        ys = (torch.arange(feat_h, device=raw_boxes.device, dtype=raw_boxes.dtype) + 0.5) / max(1, feat_h)
        xs = (torch.arange(feat_w, device=raw_boxes.device, dtype=raw_boxes.dtype) + 0.5) / max(1, feat_w)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        anchors = torch.stack((xx, yy), dim=-1).reshape(1, feat_h * feat_w, 2)
        cell = torch.tensor(
            [1.0 / max(1, feat_w), 1.0 / max(1, feat_h)],
            device=raw_boxes.device,
            dtype=raw_boxes.dtype,
        ).reshape(1, 1, 1, 2)
        offsets = (torch.sigmoid(raw_boxes[..., :2]) - 0.5) * self.anchor_offset_cells * cell
        centers = (anchors[:, :, None, :] + offsets).clamp(0.0, 1.0)
        sizes = torch.sigmoid(raw_boxes[..., 2:]) * self.box_size_scale
        return torch.cat([centers, sizes], dim=-1), anchors

    def _augment_motion_channels(self, clips: torch.Tensor) -> torch.Tensor:
        if not self.motion_channels:
            return clips
        prev = torch.cat([clips[:, :1], clips[:, :-1]], dim=1)
        prev_diff = (clips - prev).abs().mean(dim=2, keepdim=True)
        anchor_diff = (clips - clips[:, :1]).abs().mean(dim=2, keepdim=True)
        return torch.cat([clips, prev_diff, anchor_diff], dim=2)

    def _select_memory(self, frame_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # frame_tokens: B,T,S,C. The learned gate is the detector-free analog of
        # SAMURAI's motion-aware memory selection: choose reliable historical
        # video evidence before decoding object queries.
        current_memory = frame_tokens[:, -1]
        if self.memory_mode == "last":
            weights = torch.zeros(frame_tokens.shape[0], frame_tokens.shape[1], device=frame_tokens.device, dtype=frame_tokens.dtype)
            weights[:, -1] = 1.0
            return current_memory, weights, current_memory
        pooled = frame_tokens.mean(dim=2)
        gate_logits = self.memory_gate(pooled).squeeze(-1)
        weights = torch.softmax(gate_logits, dim=1)
        selected = (frame_tokens * weights[:, :, None, None]).sum(dim=1)
        fused = self.memory_fusion(torch.cat([current_memory, selected], dim=-1))
        return current_memory + fused, weights, selected

    def _pool_memory_slots(self, memory: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
        # Compress dense spatial memory before cross-attention. Full SxS
        # attention is impractical at stride 4, while fixed slots preserve a
        # global memory bank that every current-frame cell can query.
        bsz, _, channels = memory.shape
        slots_h = max(1, int(self.memory_slots**0.5))
        slots_w = max(1, (self.memory_slots + slots_h - 1) // slots_h)
        memory_map = memory.reshape(bsz, feat_h, feat_w, channels).permute(0, 3, 1, 2)
        pooled = torch.nn.functional.adaptive_avg_pool2d(memory_map, (slots_h, slots_w))
        slots = pooled.flatten(2).transpose(1, 2)
        return slots[:, : self.memory_slots, :]

    def _apply_memory_attention(
        self,
        memory: torch.Tensor,
        selected_memory: torch.Tensor,
        feat_h: int,
        feat_w: int,
    ) -> torch.Tensor:
        if self.memory_attention == "none":
            return memory
        memory_slots = self._pool_memory_slots(selected_memory, feat_h, feat_w)
        attended, _ = self.memory_cross_attn(memory, memory_slots, memory_slots, need_weights=False)
        fused = self.memory_attention_fusion(torch.cat([memory, attended], dim=-1))
        return memory + fused

    def _dense_motion_logits(
        self,
        memory: torch.Tensor,
        selected_memory: torch.Tensor,
        frame_tokens: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.motion_score_mode == "none":
            return None
        current = frame_tokens[:, -1]
        prev = frame_tokens[:, -2] if frame_tokens.shape[1] >= 2 else current
        prev2 = frame_tokens[:, -3] if frame_tokens.shape[1] >= 3 else prev
        velocity = current - prev
        acceleration = current - (prev * 2.0) + prev2
        motion_features = torch.cat([memory, selected_memory, velocity, acceleration], dim=-1)
        return self.motion_score_head(motion_features)

    def _memory_match_logits(
        self,
        memory: torch.Tensor,
        selected_memory: torch.Tensor,
        feat_h: int,
        feat_w: int,
    ) -> torch.Tensor | None:
        if self.memory_match_mode == "none":
            return None
        memory_slots = self._pool_memory_slots(selected_memory, feat_h, feat_w)
        current_norm = torch.nn.functional.normalize(memory, dim=-1)
        slot_norm = torch.nn.functional.normalize(memory_slots, dim=-1)
        affinity = torch.matmul(current_norm, slot_norm.transpose(1, 2)).amax(dim=-1)
        return affinity * self.memory_match_temperature

    def forward(self, clips: torch.Tensor) -> dict[str, torch.Tensor]:
        # clips: B,T,3,H,W
        clips = self._augment_motion_channels(clips)
        bsz, timesteps, channels, height, width = clips.shape
        x = clips.reshape(bsz * timesteps, channels, height, width)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        feat = self.stem(x)
        feat = self.spatial_refine(feat)
        _, channels_out, feat_h, feat_w = feat.shape
        feat_maps = feat.reshape(bsz, timesteps, channels_out, feat_h, feat_w)
        tokens = feat_maps.flatten(3).transpose(2, 3)
        ys = torch.linspace(-1.0, 1.0, feat_h, device=feat.device, dtype=feat.dtype)
        xs = torch.linspace(-1.0, 1.0, feat_w, device=feat.device, dtype=feat.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        spatial = torch.stack((xx, yy), dim=-1).reshape(1, 1, feat_h * feat_w, 2)
        spatial = self.spatial_pos(spatial).to(dtype=tokens.dtype)
        tokens = tokens + self.frame_pos[:, :timesteps] + spatial
        if self.encoder_mode == "factorized":
            temporal_tokens = tokens.permute(0, 2, 1, 3).reshape(bsz * feat_h * feat_w, timesteps, channels_out)
            temporal_memory = self.encoder(temporal_tokens)
            frame_tokens = temporal_memory.reshape(bsz, feat_h * feat_w, timesteps, channels_out).permute(0, 2, 1, 3)
        else:
            encoded = self.encoder(tokens.reshape(bsz, timesteps * feat_h * feat_w, channels_out))
            frame_tokens = encoded.reshape(bsz, timesteps, feat_h * feat_w, channels_out)
        memory, memory_weights, selected_memory = self._select_memory(frame_tokens)
        memory = self._apply_memory_attention(memory, selected_memory, feat_h, feat_w)
        if self.query_mode == "dense":
            raw_boxes = self.dense_box_head(memory).reshape(bsz, feat_h * feat_w, self.chunk_len, 4)
            boxes, anchors = self._decode_dense_boxes(raw_boxes, feat_h, feat_w)
            memory_map = None
            if self.dense_obj_source == "conv":
                memory_map = memory.reshape(bsz, feat_h, feat_w, channels_out).permute(0, 3, 1, 2)
                logits = self.dense_obj_conv(memory_map).flatten(2).transpose(1, 2)
            else:
                logits = self.dense_obj_head(memory).reshape(bsz, feat_h * feat_w, self.chunk_len)
            memory_match_logits = self._memory_match_logits(memory, selected_memory, feat_h, feat_w)
            if memory_match_logits is not None:
                logits = logits + (self.memory_match_weight * memory_match_logits[:, :, None])
            proposal_logits = None
            if self.proposal_mode == "heatmap":
                if memory_map is None:
                    memory_map = memory.reshape(bsz, feat_h, feat_w, channels_out).permute(0, 3, 1, 2)
                proposal_logits = self.proposal_conv(memory_map).flatten(2).squeeze(1)
            quality_logits = None
            if self.quality_score_mode == "iou":
                if memory_map is None:
                    memory_map = memory.reshape(bsz, feat_h, feat_w, channels_out).permute(0, 3, 1, 2)
                quality_logits = self.quality_conv(memory_map).flatten(2).squeeze(1)
            motion_logits = self._dense_motion_logits(memory, selected_memory, frame_tokens)
            if motion_logits is not None:
                appearance_logits = logits
                logits = appearance_logits + (self.motion_score_weight * motion_logits)
                result = {
                    "boxes": boxes,
                    "logits": logits,
                    "appearance_logits": appearance_logits,
                    "motion_logits": motion_logits,
                    "motion_score_weight": torch.tensor(self.motion_score_weight, device=logits.device, dtype=logits.dtype),
                    "memory_weights": memory_weights,
                    "anchor_centers": anchors.expand(bsz, -1, -1),
                }
                if memory_match_logits is not None:
                    result["memory_match_logits"] = memory_match_logits
                    result["memory_match_weight"] = torch.tensor(self.memory_match_weight, device=logits.device, dtype=logits.dtype)
                    result["memory_match_temperature"] = torch.tensor(
                        self.memory_match_temperature,
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                if proposal_logits is not None:
                    result["proposal_logits"] = proposal_logits
                if quality_logits is not None:
                    result["quality_logits"] = quality_logits
                return result
            result = {
                "boxes": boxes,
                "logits": logits,
                "memory_weights": memory_weights,
                "anchor_centers": anchors.expand(bsz, -1, -1),
            }
            if memory_match_logits is not None:
                result["memory_match_logits"] = memory_match_logits
                result["memory_match_weight"] = torch.tensor(self.memory_match_weight, device=logits.device, dtype=logits.dtype)
                result["memory_match_temperature"] = torch.tensor(
                    self.memory_match_temperature,
                    device=logits.device,
                    dtype=logits.dtype,
                )
            if proposal_logits is not None:
                result["proposal_logits"] = proposal_logits
            if quality_logits is not None:
                result["quality_logits"] = quality_logits
            return result
        queries = self.query.expand(bsz, -1, -1)
        decoded = self.decoder(queries, memory)
        raw_boxes = self.box_head(decoded).reshape(bsz, self.num_queries, self.chunk_len, 4)
        boxes = self._decode_boxes(raw_boxes)
        logits = self.obj_head(decoded).reshape(bsz, self.num_queries, self.chunk_len)
        return {"boxes": boxes, "logits": logits, "memory_weights": memory_weights}
