"""Multi-scale kinetic energy spectrum loss for preserving turbulent cascades and preventing spectral dissipation."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from src.utils.fft_derivatives import compute_batched_radial_energy_spectrum


class EnergySpectrumLoss(nn.Module):
    """Calculates shell-integrated kinetic energy spectrum loss E(k) via 2D RFFT.

    Penalizes deviations in the turbulent energy cascade:
        E(k) = \\sum_{|k'| \\in shell_k} 0.5 * (|u_hat(k')|^2 + |v_hat(k')|^2)

    Mitigates two primary failure modes of autoregressive flow prediction:
    1. Spectral dissipation / over-smoothing: loss of high-wavenumber turbulent eddies.
    2. Spectral aliasing / shock: artificial energy buildup at high frequencies.

    Args:
        domain_size: (Lx, Ly) physical extent of domain. Defaults to (1.0, 2.0).
        loss_type: Loss metric formulation:
            - 'log_l1': Mean absolute error in log10 space (recommended for multi-decade cascades).
            - 'log_l2': Mean squared error in log10 space.
            - 'relative': Relative error |E_pred - E_target| / (E_target + eps).
        high_freq_weight: High-wavenumber emphasis coefficient alpha >= 0.0.
            Scales wavenumber weights as w_k = 1.0 + alpha * (k / k_max), normalized to mean 1.0.
        eps: Small constant to stabilize log10 and division operations (default: 1e-8).
    """

    def __init__(
        self,
        domain_size: Tuple[float, float] = (1.0, 2.0),
        loss_type: str = "log_l1",
        high_freq_weight: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        valid_loss_types = ("log_l1", "log_l2", "relative")
        if loss_type not in valid_loss_types:
            raise ValueError(f"Unknown loss_type: '{loss_type}'. Must be one of {valid_loss_types}")
        if high_freq_weight < 0.0:
            raise ValueError(f"high_freq_weight must be >= 0.0, got {high_freq_weight}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0.0, got {eps}")

        self.domain_size = domain_size
        self.loss_type = loss_type
        self.high_freq_weight = high_freq_weight
        self.eps = eps

    def _get_wavenumber_weights(self, num_bins: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Construct normalized wavenumber weights w_k emphasizing high frequencies if alpha > 0."""
        if self.high_freq_weight == 0.0:
            return torch.ones(num_bins, device=device, dtype=dtype)
        k_normalized = torch.linspace(0.0, 1.0, num_bins, device=device, dtype=dtype)
        raw_weights = 1.0 + self.high_freq_weight * k_normalized
        # Normalize weights so that mean(w) == 1.0, preserving loss scale across alpha values
        return raw_weights / torch.mean(raw_weights)

    def _compute_spectrum_error(
        self,
        pred_q: torch.Tensor,
        target_q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute 1D energy spectra and raw point-wise spectral differences."""
        assert pred_q.shape[-3] >= 2 and target_q.shape[-3] >= 2, (
            f"Tensors must have >= 2 channels for [u, v], got pred {pred_q.shape} vs target {target_q.shape}"
        )
        u_pred, v_pred = pred_q[..., 0, :, :], pred_q[..., 1, :, :]
        u_target, v_target = target_q[..., 0, :, :], target_q[..., 1, :, :]

        _, e_pred = compute_batched_radial_energy_spectrum(u_pred, v_pred, domain_size=self.domain_size)
        _, e_target = compute_batched_radial_energy_spectrum(u_target, v_target, domain_size=self.domain_size)

        num_bins = e_target.shape[-1]
        weights = self._get_wavenumber_weights(num_bins, e_target.device, e_target.dtype)

        if self.loss_type == "log_l1":
            log_pred = torch.log10(e_pred + self.eps)
            log_target = torch.log10(e_target + self.eps)
            diff = torch.abs(log_pred - log_target)
        elif self.loss_type == "log_l2":
            log_pred = torch.log10(e_pred + self.eps)
            log_target = torch.log10(e_target + self.eps)
            diff = (log_pred - log_target) ** 2
        elif self.loss_type == "relative":
            diff = torch.abs(e_pred - e_target) / (e_target + self.eps)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        weighted_diff = diff * weights
        return weighted_diff, e_pred, e_target

    def forward(self, pred_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        """Compute scalar energy spectrum loss.

        Args:
            pred_q: Predicted physical flow fields of shape (..., C, Nx, Ny).
            target_q: Target physical flow fields of shape (..., C, Nx, Ny).

        Returns:
            Scalar tensor representing spectrum loss.
        """
        weighted_diff, _, _ = self._compute_spectrum_error(pred_q, target_q)
        return torch.mean(weighted_diff)

    def compute_loss_and_bands(
        self,
        pred_q: torch.Tensor,
        target_q: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute scalar loss alongside low, mid, and high wavenumber band errors.

        Returns:
            total_loss: Differentiable scalar loss tensor.
            metrics: Dictionary containing float diagnostics for logging.
        """
        weighted_diff, _, _ = self._compute_spectrum_error(pred_q, target_q)
        total_loss = torch.mean(weighted_diff)

        # Compute detached band-wise diagnostics
        with torch.no_grad():
            num_bins = weighted_diff.shape[-1]
            k_low = num_bins // 3
            k_mid = 2 * (num_bins // 3)

            low_err = torch.mean(weighted_diff[..., :k_low]).item() if k_low > 0 else 0.0
            mid_err = torch.mean(weighted_diff[..., k_low:k_mid]).item() if k_mid > k_low else 0.0
            high_err = torch.mean(weighted_diff[..., k_mid:]).item() if num_bins > k_mid else 0.0

            metrics = {
                "spec_loss_total": float(total_loss.item()),
                "spec_loss_low": float(low_err),
                "spec_loss_mid": float(mid_err),
                "spec_loss_high": float(high_err),
            }
        return total_loss, metrics
