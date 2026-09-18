"""Baseline 0: Persistence (Identity prediction) model.

Predicts that future states remain identical to the latest observed history frame:
    q_hat_(t+h) = q_t for all h >= 1
Serves as the zero-intelligence lower bound.
"""

from typing import Optional
import torch
import torch.nn as nn


class PersistenceBaseline(nn.Module):
    """Persistence baseline: copies the last frame of history forward."""

    def __init__(self):
        super().__init__()
        # Dummy parameter to make it a valid PyTorch module
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        history: torch.Tensor,
        horizon: int = 1,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            history: History tensor of shape (B, L, C, Ny, Nx).
            horizon: Number of future steps H to predict.
            condition: Ignored condition argument for API compatibility.

        Returns:
            Future states of shape (B, H, C, Ny, Nx).
        """
        last_frame = history[:, -1:]  # (B, 1, C, Ny, Nx)
        if horizon == 1:
            return last_frame
        return last_frame.expand(-1, horizon, -1, -1, -1).clone()
