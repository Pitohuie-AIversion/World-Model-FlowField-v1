"""Divergence loss for the Closure-R4 shear_flow spatial contract."""

from typing import Tuple
import torch
import torch.nn as nn

from src.utils.fft_derivatives import compute_divergence
from src.utils.physics_contract import SHEAR_FLOW_DOMAIN_SIZE_XY


class DivergenceLoss(nn.Module):
    """Mean squared incompressibility residual on (..., C, Nx, Ny)."""

    def __init__(self, domain_size: Tuple[float, float] = SHEAR_FLOW_DOMAIN_SIZE_XY):
        super().__init__()
        self.domain_size = domain_size

    def forward(self, pred_q: torch.Tensor) -> torch.Tensor:
        u = pred_q[..., 0, :, :]
        v = pred_q[..., 1, :, :]
        div = compute_divergence(u, v, domain_size=self.domain_size)
        return torch.mean(div**2)
