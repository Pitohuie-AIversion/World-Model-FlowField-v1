"""Divergence physical loss for enforcing incompressibility: div(u) = 0."""

from typing import Tuple
import torch
import torch.nn as nn
from src.utils.fft_derivatives import compute_divergence


class DivergenceLoss(nn.Module):
    """Calculates mean squared divergence of predicted velocity fields (u, v).

    Args:
        domain_size: (Ly, Lx) spatial domain sizes. Defaults to (2.0, 1.0).
    """

    def __init__(self, domain_size: Tuple[float, float] = (2.0, 1.0)):
        super().__init__()
        self.domain_size = domain_size

    def forward(self, pred_q: torch.Tensor) -> torch.Tensor:
        """Compute divergence loss.

        Args:
            pred_q: Physical fields of shape (..., C, Ny, Nx), where channel 0 is u, channel 1 is v.

        Returns:
            loss: Scalar mean squared divergence.
        """
        # Extract u and v
        u = pred_q[..., 0, :, :]
        v = pred_q[..., 1, :, :]

        div = compute_divergence(u, v, domain_size=self.domain_size)
        return torch.mean(div**2)
