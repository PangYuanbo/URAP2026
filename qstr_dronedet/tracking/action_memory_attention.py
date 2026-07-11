from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActionMemoryAttentionOutput:
    logits: torch.Tensor
    short_context: torch.Tensor
    long_context: torch.Tensor
    short_attention: torch.Tensor
    long_attention: torch.Tensor
    write_probability: torch.Tensor


class ActionMemoryCrossAttentionRanker(torch.nn.Module):
    """Query candidate state against causal short/long Action Memory banks."""

    def __init__(
        self,
        in_dim: int,
        token_start: int,
        token_dim: int,
        short_tokens: int,
        long_tokens: int,
        hidden: int = 192,
        memory_dim: int | None = None,
        heads: int = 4,
        memory_layers: int = 1,
        dropout: float = 0.05,
        query_indices: list[int] | None = None,
        use_static_head: bool = True,
    ) -> None:
        super().__init__()
        self.token_start = int(token_start)
        self.token_dim = int(token_dim)
        self.short_tokens = int(short_tokens)
        self.long_tokens = int(long_tokens)
        self.short_width = self.token_dim * self.short_tokens
        self.long_width = self.token_dim * self.long_tokens
        token_stop = self.token_start + self.short_width + self.long_width
        static_indices = [index for index in range(in_dim) if not self.token_start <= index < token_stop]
        if not static_indices:
            raise ValueError("Action Memory ranker requires candidate/static query features")
        self.register_buffer("static_indices", torch.tensor(static_indices, dtype=torch.long), persistent=False)
        self.query_from_static = query_indices is None
        resolved_query_indices = static_indices if query_indices is None else [int(index) for index in query_indices]
        self.register_buffer("query_indices", torch.tensor(resolved_query_indices, dtype=torch.long), persistent=False)
        self.use_static_head = bool(use_static_head)
        self.memory_dim = int(memory_dim or max(48, hidden // 2))
        if self.memory_dim % heads:
            raise ValueError("memory_dim must be divisible by heads")

        self.static_net = torch.nn.Sequential(
            torch.nn.Linear(len(static_indices), hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        if self.query_from_static:
            self.query_projection = torch.nn.Sequential(
                torch.nn.Linear(hidden, self.memory_dim), torch.nn.LayerNorm(self.memory_dim)
            )
        else:
            self.action_query_projection = torch.nn.Sequential(
                torch.nn.Linear(len(resolved_query_indices), hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, self.memory_dim),
                torch.nn.LayerNorm(self.memory_dim),
            )
        self.token_projection = torch.nn.Sequential(
            torch.nn.Linear(self.token_dim + 2, self.memory_dim),
            torch.nn.LayerNorm(self.memory_dim),
            torch.nn.SiLU(),
        )
        self.short_encoder = torch.nn.TransformerEncoder(
            self._encoder_layer(heads, dropout), num_layers=memory_layers
        )
        self.long_encoder = torch.nn.TransformerEncoder(
            self._encoder_layer(heads, dropout), num_layers=memory_layers
        )
        self.short_cross_attention = torch.nn.MultiheadAttention(
            self.memory_dim, heads, dropout=dropout, batch_first=True
        )
        self.long_cross_attention = torch.nn.MultiheadAttention(
            self.memory_dim, heads, dropout=dropout, batch_first=True
        )
        self.context_norm = torch.nn.LayerNorm(self.memory_dim)
        self.fusion_gate = torch.nn.Sequential(
            torch.nn.Linear(3 * self.memory_dim, 2), torch.nn.Sigmoid()
        )
        head_input = 3 * self.memory_dim + (hidden if self.use_static_head else 0)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(head_input, hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )
        write_hidden = max(32, hidden // 2)
        write_input = hidden + 2 * self.memory_dim if self.use_static_head else 3 * self.memory_dim
        self.write_head = torch.nn.Sequential(
            torch.nn.Linear(write_input, write_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(write_hidden, 1),
        )

    def _encoder_layer(self, heads: int, dropout: float) -> torch.nn.TransformerEncoderLayer:
        return torch.nn.TransformerEncoderLayer(
            d_model=self.memory_dim,
            nhead=heads,
            dim_feedforward=4 * self.memory_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def _memory_tokens(self, tokens: torch.Tensor, bank_id: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = tokens[..., 0] > 0.5
        count = tokens.shape[1]
        age = torch.linspace(0.0, 1.0, count, device=tokens.device, dtype=tokens.dtype)
        age = age.view(1, count, 1).expand(tokens.shape[0], -1, -1)
        bank = torch.full_like(age, bank_id)
        embedded = self.token_projection(torch.cat((tokens, age, bank), dim=-1))
        any_valid = valid.any(dim=1, keepdim=True)
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~any_valid.squeeze(1)
        embedded = embedded.masked_fill((~safe_valid).unsqueeze(-1), 0.0)
        return embedded, ~safe_valid, any_valid

    def _query_bank(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        encoder: torch.nn.TransformerEncoder,
        attention: torch.nn.MultiheadAttention,
        bank_id: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory, padding_mask, any_valid = self._memory_tokens(tokens, bank_id)
        memory = encoder(memory, src_key_padding_mask=padding_mask)
        context, weights = attention(
            query,
            memory,
            memory,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        context = self.context_norm(context) * any_valid.to(context.dtype).unsqueeze(-1)
        weights = weights * any_valid.to(weights.dtype).view(-1, 1, 1, 1)
        return context.squeeze(1), weights.squeeze(2)

    def forward_with_details(self, features: torch.Tensor) -> ActionMemoryAttentionOutput:
        static = self.static_net(features.index_select(1, self.static_indices))
        query = (
            self.query_projection(static)
            if self.query_from_static
            else self.action_query_projection(features.index_select(1, self.query_indices))
        ).unsqueeze(1)
        short_start = self.token_start
        short_stop = short_start + self.short_width
        long_stop = short_stop + self.long_width
        short = features[:, short_start:short_stop].reshape(-1, self.short_tokens, self.token_dim)
        long = features[:, short_stop:long_stop].reshape(-1, self.long_tokens, self.token_dim)
        short_context, short_attention = self._query_bank(
            query, short, self.short_encoder, self.short_cross_attention, 0.0
        )
        long_context, long_attention = self._query_bank(
            query, long, self.long_encoder, self.long_cross_attention, 1.0
        )
        query_vector = query.squeeze(1)
        gates = self.fusion_gate(torch.cat((query_vector, short_context, long_context), dim=1))
        short_context = short_context * gates[:, :1]
        long_context = long_context * gates[:, 1:]
        head_parts = [query_vector, short_context, long_context]
        if self.use_static_head:
            head_parts.insert(0, static)
        logits = self.head(torch.cat(head_parts, dim=1)).squeeze(-1)
        write_parts = [short_context, long_context]
        write_parts.insert(0, static if self.use_static_head else query_vector)
        write_probability = torch.sigmoid(self.write_head(torch.cat(write_parts, dim=1)).squeeze(-1))
        return ActionMemoryAttentionOutput(
            logits=logits,
            short_context=short_context,
            long_context=long_context,
            short_attention=short_attention,
            long_attention=long_attention,
            write_probability=write_probability,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_with_details(features).logits
