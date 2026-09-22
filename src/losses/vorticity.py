"""Vorticity consistency loss for the Closure-R4 spatial contract."""

from typing import Tuple
import torch
import torch.nn as nn

from src.utils.fft_derivatives import compute_vorticity
from src.utils.physics_contract import SHEAR_FLOW_DOMAIN_SIZE_XY


class VorticityLoss(nn.Module):
    """Vorticity error for fields stored as (..., C, Nx, Ny)."""

    def __init__(
        self,
        domain_size: Tuple[float, float] = SHEAR_FLOW_DOMAIN_SIZE_XY,
        relative: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.domain_size = domain_size
        self.relative = relative
        self.eps = eps

    def forward(self, pred_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
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
        return torch.mean((omega_pred - omega_target) ** 2)
