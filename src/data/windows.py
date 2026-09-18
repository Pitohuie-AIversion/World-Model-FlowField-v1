"""Sliding window indexing and temporal trajectory slicing utilities."""

from typing import List, Tuple
import torch


def generate_window_indices(
    total_timesteps: int,
    history_length: int = 4,
    horizon: int = 1,
    stride: int = 1,
) -> List[Tuple[int, int]]:
    """Generate (start_idx, end_idx) pairs for sliding windows.

    Each window spans [start_idx : start_idx + history_length + horizon].

    Args:
        total_timesteps: Length of trajectory T (e.g. 200).
        history_length: Length of history L (default: 4).
        horizon: Length of rollout future H (default: 1).
        stride: Stride between window start times.

    Returns:
        List of (start_t, split_t, end_t) tuples where:
            history is in [start_t, split_t)
            future is in [split_t, end_t)
    """
    window_len = history_length + horizon
    if total_timesteps < window_len:
        raise ValueError(
            f"Trajectory length {total_timesteps} is shorter than window length {window_len}"
        )

    indices = []
    for start in range(0, total_timesteps - window_len + 1, stride):
        split = start + history_length
        end = split + horizon
        indices.append((start, split, end))

    return indices


def slice_trajectory_window(
    trajectory: torch.Tensor,
    start: int,
    split: int,
    end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice a single window from trajectory tensor of shape (T, C, Ny, Nx).

    Returns:
        history: (L, C, Ny, Nx)
        future: (H, C, Ny, Nx)
    """
    history = trajectory[start:split]
    future = trajectory[split:end]
    return history, future
