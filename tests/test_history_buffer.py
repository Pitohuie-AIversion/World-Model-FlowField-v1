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
