"""Vorticity loss for preserving fine vortex structures: ||omega_hat - omega||^2."""

from typing import Tuple
import torch
import torch.nn as nn
from src.utils.fft_derivatives import compute_vorticity


class VorticityLoss(nn.Module):
    """Calculates mean squared or relative error between predicted and target vorticity.

    Args:
        domain_size: (Ly, Lx) spatial domain sizes. Defaults to (2.0, 1.0).
        relative: If True, computes relative L2 error rather than absolute MSE.
        eps: Epsilon for relative division.
    """

    def __init__(
        self,
        domain_size: Tuple[float, float] = (2.0, 1.0),
        relative: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.domain_size = domain_size
        self.relative = relative
        self.eps = eps

    def forward(self, pred_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        """Compute vorticity loss.

        Args:
            pred_q: Predicted physical fields (..., C, Ny, Nx), channels 0, 1 are u, v.
            target_q: Target physical fields (..., C, Ny, Nx), channels 0, 1 are u, v.

        Returns:
            loss: Scalar vorticity error loss.
        """
        u_pred = pred_q[..., 0, :, :]
        v_pred = pred_q[..., 1, :, :]
        omega_pred = compute_vorticity(u_pred, v_pred, domain_size=self.domain_size)

        u_target = target_q[..., 0, :, :]
        v_target = target_q[..., 1, :, :]
        omega_target = compute_vorticity(u_target, v_target, domain_size=self.domain_size)

        if self.relative:
            diff = torch.norm(omega_pred - omega_target, p=2, dim=(-2, -1))
            norm = torch.norm(omega_target, p=2, dim=(-2, -1))
            return torch.mean(diff / (norm + self.eps))
        else:
            return torch.mean((omega_pred - omega_target) ** 2)
