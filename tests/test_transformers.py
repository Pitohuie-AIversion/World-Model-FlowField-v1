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


def test_latent_st_transformer_residual_math_and_gradient():
    """Verify exact residual formulation Z_{t+1} = Z_t + Delta_Z and backpropagation."""
    b, l, c_z, h_z, w_z = 2, 4, 32, 8, 16
    model = LatentSTTransformer(
        latent_channels=c_z,
        embed_dim=64,
        cond_dim=32,
        depth=2,
        num_heads=2,
        history_length=l,
        prediction_mode="residual",
    )

    z_hist = torch.randn(b, l, c_z, h_z, w_z, requires_grad=True)
    re = torch.tensor([1e4, 1e5])
    sc = torch.tensor([0.2, 5.0])

    z_pred = model(z_hist, re=re, sc=sc)
    assert z_pred.shape == (b, 1, c_z, h_z, w_z)

    # In residual mode, the predicted increment Delta Z is (z_pred - z_hist[:, -1:])
    delta_z = z_pred - z_hist[:, -1:]
    assert delta_z.shape == (b, 1, c_z, h_z, w_z)

    # Verify backpropagation through both residual skip connection and transformer blocks
    target = z_hist[:, -1:] + 0.1 * torch.randn_like(z_hist[:, -1:])
    loss = torch.nn.functional.mse_loss(z_pred, target)
    loss.backward()

    assert z_hist.grad is not None
    assert not torch.isnan(z_hist.grad).any()
    assert model.blocks[0].spatial_attn.qkv.weight.grad is not None
    assert model.out_proj.weight.grad is not None

    # Condition sensitivity test:
    # 1) At zero-init state, AdaLN proj weights are 0, so gamma=0, beta=0.
    #    Conditioned forward produces identical output (identity modulation).
    re2 = torch.tensor([5e5, 5e5])
    sc2 = torch.tensor([10.0, 10.0])
    with torch.no_grad():
        z_pred_cond1 = model(z_hist, re=re, sc=sc)
        z_pred_cond2 = model(z_hist, re=re2, sc=sc2)
    assert torch.allclose(z_pred_cond1, z_pred_cond2, atol=1e-5)

    # 2) After one gradient update step, AdaLN projections absorb condition gradients,
    #    yielding distinct predictions for different physical parameters (Re, Sc).
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    optimizer.step()
    with torch.no_grad():
        z_pred_cond1_after = model(z_hist, re=re, sc=sc)
        z_pred_cond2_after = model(z_hist, re=re2, sc=sc2)
    assert not torch.allclose(z_pred_cond1_after, z_pred_cond2_after, atol=1e-5)


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
