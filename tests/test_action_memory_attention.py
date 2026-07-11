from __future__ import annotations

import torch

from qstr_dronedet.tracking.action_memory_attention import ActionMemoryCrossAttentionRanker


def _model() -> ActionMemoryCrossAttentionRanker:
    return ActionMemoryCrossAttentionRanker(
        in_dim=18,
        token_start=4,
        token_dim=2,
        short_tokens=3,
        long_tokens=4,
        hidden=32,
        memory_dim=16,
        heads=4,
        dropout=0.0,
    ).eval()


def test_cross_attention_handles_empty_banks_without_nan() -> None:
    output = _model().forward_with_details(torch.zeros(2, 18))
    assert torch.isfinite(output.logits).all()
    assert torch.count_nonzero(output.short_context) == 0
    assert torch.count_nonzero(output.long_context) == 0
    assert torch.count_nonzero(output.short_attention) == 0
    assert torch.count_nonzero(output.long_attention) == 0


def test_current_candidate_query_changes_memory_readout() -> None:
    torch.manual_seed(7)
    features = torch.zeros(2, 18)
    features[:, 4:10] = torch.tensor([1.0, 0.2, 1.0, 0.8, 1.0, -0.3])
    features[:, 10:18] = torch.tensor([1.0, 0.1, 1.0, 0.4, 1.0, 0.7, 1.0, -0.2])
    features[1, :4] = torch.tensor([2.0, -1.0, 0.5, 1.5])
    output = _model().forward_with_details(features)
    assert not torch.allclose(output.short_context[0], output.short_context[1])
    assert not torch.allclose(output.long_context[0], output.long_context[1])
    assert output.write_probability.shape == (2,)


def test_invalid_memory_tokens_receive_zero_attention() -> None:
    features = torch.zeros(1, 18)
    features[0, 4:10] = torch.tensor([1.0, 0.2, 0.0, 50.0, 1.0, 0.7])
    output = _model().forward_with_details(features)
    assert torch.count_nonzero(output.short_attention[..., 1]) == 0
