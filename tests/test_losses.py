"""Unit tests for Field, Divergence, Vorticity and Rollout loss functions."""

import torch
import pytest
from src.losses.field import FieldLoss
from src.losses.divergence import DivergenceLoss
from src.losses.vorticity import VorticityLoss
from src.losses.rollout import RolloutLoss


def test_field_losses():
    pred = torch.randn(2, 4, 32, 64)
    target = pred.clone()

    loss_fn = FieldLoss(loss_type="mse")
    assert loss_fn(pred, target).item() == pytest.approx(0.0, abs=1e-6)

    loss_fn_rel = FieldLoss(loss_type="relative_l2")
    assert loss_fn_rel(pred, target).item() == pytest.approx(0.0, abs=1e-6)


def test_divergence_loss():
    # Construct zero divergence field
    ny, nx = 64, 128
    y = torch.linspace(-1.0, 1.0 - 2.0 / ny, ny)
    x = torch.linspace(0.0, 1.0 - 1.0 / nx, nx)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    u = -torch.sin(2.0 * torch.pi * xx) * torch.sin(torch.pi * yy)
    v = -2.0 * torch.cos(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)
    p = torch.zeros_like(u)
    s = torch.zeros_like(u)

    q = torch.stack([u, v, p, s], dim=0).unsqueeze(0)  # (1, 4, ny, nx)

    div_loss_fn = DivergenceLoss(domain_size=(2.0, 1.0))
    loss_val = div_loss_fn(q)
    assert loss_val.item() < 1e-6


def test_vorticity_loss():
    q1 = torch.randn(2, 4, 32, 64)
    q2 = q1.clone()

    vort_loss_fn = VorticityLoss()
    assert vort_loss_fn(q1, q2).item() == pytest.approx(0.0, abs=1e-6)


def test_rollout_loss():
    b, h, c, ny, nx = 2, 4, 4, 16, 32
    pred = torch.randn(b, h, c, ny, nx)
    target = pred.clone()

    rollout_fn = RolloutLoss(weight_mode="discount")
    assert rollout_fn(pred, target).item() == pytest.approx(0.0, abs=1e-6)
