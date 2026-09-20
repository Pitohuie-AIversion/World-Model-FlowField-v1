"""Unit tests for HistoryBuffer autoregressive rollout."""

import torch
import pytest
from src.models.history_buffer import HistoryBuffer


def test_history_buffer_fifo():
    buf = HistoryBuffer(history_length=4)
    init = torch.tensor([[[0.0]], [[1.0]], [[2.0]], [[3.0]]]).transpose(0, 1)  # shape (1, 4, 1)
    buf.reset(init)

    assert torch.allclose(buf.current, init)

    next_val = torch.tensor([[[4.0]]])  # shape (1, 1, 1)
    buf.push(next_val)

    expected = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]]).transpose(0, 1)
    assert torch.allclose(buf.current, expected)


def test_history_buffer_rollout():
    buf = HistoryBuffer(history_length=3)
    # Start with [1, 2, 3]
    init = torch.tensor([[[1.0], [2.0], [3.0]]])  # (1, 3, 1)
    buf.reset(init)

    # Simple step_fn: next = last + 1
    def step_fn(hist, cond=None):
        return hist[:, -1:] + 1.0

    traj = buf.rollout(step_fn, steps=5)
    # Expected rollout: [4, 5, 6, 7, 8]
    expected_traj = torch.tensor([[[4.0], [5.0], [6.0], [7.0], [8.0]]])
    assert torch.allclose(traj, expected_traj)
    # Buffer should now hold [6, 7, 8]
    expected_buf = torch.tensor([[[6.0], [7.0], [8.0]]])
    assert torch.allclose(buf.current, expected_buf)


def test_latent_space_rollout_with_condition():
    """Verify HistoryBuffer with 4D spatial latent tensors (B, L, C_z, H_z, W_z) and condition."""
    b, l, c_z, h_z, w_z = 2, 4, 16, 8, 16
    initial_z = torch.randn(b, l, c_z, h_z, w_z)
    cond = torch.randn(b, 32)

    buf = HistoryBuffer(history_length=l)
    buf.reset(initial_z)
    assert buf.current.shape == (b, l, c_z, h_z, w_z)

    # Step function simulating a latent transition: next_z = mean(hist) + linear(cond)
    def latent_step_fn(hist: torch.Tensor, c: torch.Tensor):
        # Ensure hist has length L
        assert hist.shape[1] == l
        next_step = hist[:, -1:] * 0.9 + c[:, :c_z].view(b, 1, c_z, 1, 1) * 0.1
        return next_step

    # Rollout for 30 steps in latent space
    horizon = 30
    trajectory = buf.rollout(latent_step_fn, steps=horizon, condition=cond)
    assert trajectory.shape == (b, horizon, c_z, h_z, w_z)
    assert not torch.isnan(trajectory).any()

    # Final buffer state must have length L
    assert buf.current.shape == (b, l, c_z, h_z, w_z)

