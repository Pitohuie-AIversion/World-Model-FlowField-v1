"""Unit tests for Physical Condition Embedding and AdaLN."""

import torch
import pytest
from src.models.conditioning import PhysicalConditionEmbedding, AdaLN, AdaLNZeroBlock


def test_physical_condition_embedding():
    embed = PhysicalConditionEmbedding(embed_dim=128)
    re = torch.tensor([1e4, 5e4, 1e5, 5e5])
    sc = torch.tensor([0.1, 1.0, 5.0, 10.0])

    e_c = embed(re, sc)
    assert e_c.shape == (4, 128)
    assert not torch.isnan(e_c).any()


def test_adaln_zero_init():
    """Verify AdaLN with zero initialization initially acts as standard LayerNorm."""
    dim = 64
    cond_dim = 128
    adaln = AdaLN(dim=dim, cond_dim=cond_dim, zero_init=True)

    x = torch.randn(2, 10, dim)
    c = torch.randn(2, cond_dim)

    out_cond = adaln(x, c)
    out_uncond = adaln(x, None)

    # With zero init, projection produces gamma=0, beta=0, so out = 1 * norm(x) + 0
    assert torch.allclose(out_cond, out_uncond, atol=1e-5)


def test_adaln_zero_block():
    dim = 64
    cond_dim = 128
    block = AdaLNZeroBlock(dim=dim, cond_dim=cond_dim)

    c = torch.randn(4, cond_dim)
    params = block(c)
    assert len(params) == 6
    for p in params:
        assert p.shape == (4, dim)
        assert torch.allclose(p, torch.zeros_like(p), atol=1e-6)
