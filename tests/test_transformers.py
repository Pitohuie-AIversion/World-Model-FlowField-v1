"""Unit tests for Latent and Direct Spatio-Temporal Transformers."""

import torch
import pytest
from src.models.latent_transformer import LatentSTTransformer
from src.models.direct_transformer import DirectSTTransformer


def test_latent_st_transformer_shapes():
    b, l, c_z, h_z, w_z = 2, 4, 64, 8, 16
    model = LatentSTTransformer(
        latent_channels=c_z,
        embed_dim=128,
        cond_dim=64,
        depth=2,
        num_heads=4,
        history_length=l,
        prediction_mode="direct",
    )

    z_hist = torch.randn(b, l, c_z, h_z, w_z)
    re = torch.tensor([1e4, 5e4])
    sc = torch.tensor([0.1, 1.0])

    # 1. Conditioned forward
    out = model(z_hist, re=re, sc=sc)
    assert out.shape == (b, 1, c_z, h_z, w_z), f"Unexpected output shape: {out.shape}"

    # 2. Unconditioned forward (protocol A)
    out_uncond = model(z_hist, re=None, sc=None)
    assert out_uncond.shape == (b, 1, c_z, h_z, w_z)

    # 3. Residual prediction mode
    model_res = LatentSTTransformer(
        latent_channels=c_z,
        embed_dim=128,
        cond_dim=64,
        depth=2,
        num_heads=4,
        history_length=l,
        prediction_mode="residual",
    )
    out_res = model_res(z_hist, re=re, sc=sc)
    assert out_res.shape == (b, 1, c_z, h_z, w_z)


def test_direct_st_transformer_shapes():
    b, l, c, ny, nx = 2, 4, 4, 32, 64
    model = DirectSTTransformer(
        in_channels=c,
        patch_size=(8, 8),
        embed_dim=128,
        cond_dim=64,
        depth=2,
        num_heads=4,
        history_length=l,
        prediction_mode="direct",
    )

    q_hist = torch.randn(b, l, c, ny, nx)
    re = torch.tensor([1e4, 5e4])
    sc = torch.tensor([0.1, 1.0])

    # Conditioned forward
    out = model(q_hist, re=re, sc=sc)
    assert out.shape == (b, 1, c, ny, nx), f"Unexpected output shape: {out.shape}"

    # Residual mode
    model_res = DirectSTTransformer(
        in_channels=c,
        patch_size=(8, 8),
        embed_dim=128,
        cond_dim=64,
        depth=2,
        num_heads=4,
        history_length=l,
        prediction_mode="residual",
    )
    out_res = model_res(q_hist, re=re, sc=sc)
    assert out_res.shape == (b, 1, c, ny, nx)
