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
    assert FieldLoss(loss_type="mse")(pred, target).item() == pytest.approx(0.0, abs=1e-6)
    assert FieldLoss(loss_type="relative_l2")(pred, target).item() == pytest.approx(0.0, abs=1e-6)


def test_divergence_loss():
    nx, ny = 64, 128
    x = torch.arange(nx, dtype=torch.float32) / nx
    y = -1.0 + torch.arange(ny, dtype=torch.float32) * (2.0 / ny)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    u = -torch.sin(2.0 * torch.pi * xx) * torch.sin(torch.pi * yy)
    v = -2.0 * torch.cos(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)
    p = torch.zeros_like(u)
    s = torch.zeros_like(u)
    q = torch.stack([u, v, p, s], dim=0).unsqueeze(0)
    assert DivergenceLoss(domain_size=(1.0, 2.0))(q).item() < 1e-6


def test_vorticity_loss():
    q1 = torch.randn(2, 4, 32, 64)
    q2 = q1.clone()
    assert VorticityLoss()(q1, q2).item() == pytest.approx(0.0, abs=1e-6)


def test_rollout_loss():
    pred = torch.randn(2, 4, 4, 16, 32)
    target = pred.clone()
    assert RolloutLoss(weight_mode="discount")(pred, target).item() == pytest.approx(0.0, abs=1e-6)
