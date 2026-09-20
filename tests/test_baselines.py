"""Unit tests for Persistence and FNO baselines."""

import torch
import pytest
from src.baselines.persistence import PersistenceBaseline
from src.baselines.fno import FNO2D


def test_persistence_baseline():
    b, l, c, ny, nx = 2, 4, 4, 32, 64
    hist = torch.randn(b, l, c, ny, nx)

    model = PersistenceBaseline()
    # Single-step prediction
    pred_1 = model(hist, horizon=1)
    assert pred_1.shape == (b, 1, c, ny, nx)
    assert torch.allclose(pred_1, hist[:, -1:])

    # Multi-step prediction
    pred_5 = model(hist, horizon=5)
    assert pred_5.shape == (b, 5, c, ny, nx)
    for h in range(5):
        assert torch.allclose(pred_5[:, h], hist[:, -1, :, :])


def test_fno2d_baseline():
    b, l, c, ny, nx = 2, 4, 4, 32, 64
    hist = torch.randn(b, l, c, ny, nx)

    # In channels: L * C = 4 * 4 = 16
    fno = FNO2D(
        in_channels=16,
        out_channels=4,
        modes1=8,
        modes2=8,
        width=32,
        num_layers=2,
    )

    out = fno(hist)
    assert out.shape == (b, 1, c, ny, nx), f"Unexpected FNO output shape: {out.shape}"
    assert not torch.isnan(out).any()


def test_pde_transformer_baseline():
    """Verify PDETransformer baseline accepts history and predicts next state directly on mesh."""
    from src.baselines.pde_transformer import PDETransformer

    b, l, c, ny, nx = 2, 4, 4, 32, 64
    hist = torch.randn(b, l, c, ny, nx, requires_grad=True)
    re = torch.tensor([1e4, 5e4])
    sc = torch.tensor([0.2, 2.0])

    model = PDETransformer(
        in_channels=c,
        patch_size=(8, 8),
        embed_dim=64,
        cond_dim=32,
        depth=2,
        num_heads=2,
        history_length=l,
        prediction_mode="residual",
    )

    out = model(hist, re=re, sc=sc)
    assert out.shape == (b, 1, c, ny, nx)

    loss = out.sum()
    loss.backward()
    assert hist.grad is not None
    assert model.out_proj.weight.grad is not None


def test_latent_forecaster_unified_pipeline():
    """Verify end-to-end LatentForecaster pipeline: q_hist -> Z_hist -> ST-Transformer -> Z_next -> Decoder."""
    from src.models.encoder import Encoder2D
    from src.models.decoder import Decoder2D
    from src.models.latent_transformer import LatentSTTransformer
    from src.models.latent_forecaster import LatentForecaster

    b, l, c, ny, nx = 2, 4, 4, 32, 64
    encoder = Encoder2D(in_channels=c, latent_channels=32, base_channels=16)
    decoder = Decoder2D(latent_channels=32, out_channels=c, base_channels=16, project_pressure=True)
    transformer = LatentSTTransformer(
        latent_channels=32,
        embed_dim=64,
        cond_dim=32,
        depth=2,
        num_heads=2,
        history_length=l,
        prediction_mode="residual",
    )

    forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder, freeze_representation=False)

    q_hist = torch.randn(b, l, c, ny, nx, requires_grad=True)
    re = torch.tensor([1e4, 1e5])
    sc = torch.tensor([0.5, 1.0])

    # 1. Single-step via unified forward(horizon=1)
    q_pred_1 = forecaster(q_hist, re=re, sc=sc, horizon=1)
    assert q_pred_1.shape == (b, 1, c, ny, nx)

    # 2. Multi-step rollout via unified forward(horizon=3)
    q_pred_3 = forecaster(q_hist, re=re, sc=sc, horizon=3)
    assert q_pred_3.shape == (b, 3, c, ny, nx)

    # 3. Check pressure zero-mean projection on output
    p_pred = q_pred_1[:, :, 2]
    mean_p = p_pred.mean(dim=(-2, -1))
    assert torch.allclose(mean_p, torch.zeros_like(mean_p), atol=1e-5)


def test_fair_baseline_unified_contract():
    """Verify all models (Persistence, FNO, PDE-Transformer, LatentForecaster) adhere to fair contract."""
    from src.baselines.persistence import PersistenceBaseline
    from src.baselines.fno import FNO2D
    from src.baselines.pde_transformer import PDETransformer
    from src.models.encoder import Encoder2D
    from src.models.decoder import Decoder2D
    from src.models.latent_transformer import LatentSTTransformer
    from src.models.latent_forecaster import LatentForecaster

    b, l, c, ny, nx = 2, 4, 4, 32, 64
    q_hist = torch.randn(b, l, c, ny, nx)

    m_persistence = PersistenceBaseline()
    m_fno = FNO2D(in_channels=l * c, out_channels=c, modes1=8, modes2=8, width=32, num_layers=2)
    m_pde_trans = PDETransformer(in_channels=c, patch_size=(8, 8), embed_dim=64, cond_dim=32, depth=2, num_heads=2)

    enc = Encoder2D(in_channels=c, latent_channels=32, base_channels=16)
    dec = Decoder2D(latent_channels=32, out_channels=c, base_channels=16)
    trans = LatentSTTransformer(latent_channels=32, embed_dim=64, cond_dim=32, depth=2, num_heads=2)
    m_latent = LatentForecaster(encoder=enc, transformer=trans, decoder=dec)

    models = [m_persistence, m_fno, m_pde_trans, m_latent]

    for model in models:
        # Each model must accept the identical physical history tensor and return (B, 1, 4, Ny, Nx)
        if isinstance(model, PersistenceBaseline):
            out = model(q_hist, horizon=1)
        elif isinstance(model, FNO2D):
            out = model(q_hist)
        else:
            out = model(q_hist)

        assert out.shape == (b, 1, c, ny, nx), f"Model {type(model).__name__} output shape mismatch: {out.shape}"
        assert not torch.isnan(out).any(), f"Model {type(model).__name__} produced NaNs"

