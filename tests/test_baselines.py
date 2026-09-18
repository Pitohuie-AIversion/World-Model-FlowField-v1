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
