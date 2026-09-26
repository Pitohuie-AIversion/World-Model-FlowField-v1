"""Probabilistic Latent Dynamics Module for ProbLatent-R1.

Implements:
1. VarianceHead2D: Conditioned spatial variance projection head with strict positivity.
2. initialize_from_g0: Numerically stable initialization from empirical residual second moment.
3. sample_next_latent: Reparameterized sampling from conditional Gaussian N(mu, variance).
4. gaussian_nll_latent_loss: Numerically stable Gaussian NLL loss in latent space.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.checkpoint import compute_g1_bias_init_from_g0


class VarianceHead2D(nn.Module):
    """2D Spatial Latent Variance Head for ProbLatent-R1.

    Maps latent transformer feature representations to per-token conditional variance
    with guaranteed strict positivity (softplus activation + variance floor).

    Args:
        embed_dim: Feature embedding dimension of the transformer (e.g. 256).
        latent_channels: Number of latent channels C_z (e.g. 64).
        variance_floor: Numerical minimum variance floor epsilon (default: 1e-4).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        latent_channels: int = 64,
        variance_floor: float = 1e-4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_channels = latent_channels
        self.variance_floor = float(variance_floor)

        # Linear projection: embed_dim -> latent_channels
        self.linear = nn.Linear(embed_dim, latent_channels)
        self.reset_parameters()

    def reset_parameters(self):
        """Default initialization: zero weights, zero bias."""
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def initialize_from_g0(
        self,
        v_g0: torch.Tensor,
        min_margin: float = 1e-5,
    ) -> torch.Tensor:
        """Initialize variance head to match empirical G0 baseline variance.

        Sets linear weights to zero and inverts softplus on bias so that:
            softplus(b_init) + variance_floor == effective_g0.

        Args:
            v_g0: Empirical per-channel second moment tensor of shape (latent_channels,).
            min_margin: Safety margin applied when v_g0 <= variance_floor.

        Returns:
            effective_g0: The clamped target variance tensor.
        """
        b_init, effective_g0 = compute_g1_bias_init_from_g0(
            v_g0=v_g0,
            variance_floor=self.variance_floor,
            min_margin=min_margin,
        )
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.copy_(
                b_init.to(dtype=self.linear.bias.dtype, device=self.linear.bias.device)
            )
        return effective_g0

    def forward(
        self,
        features: torch.Tensor,
        h_z: int,
        w_z: int,
    ) -> torch.Tensor:
        """Forward pass predicting strictly positive conditional variance.

        Args:
            features: Feature tensor of shape (B, 1, N, embed_dim) or (B, N, embed_dim).
            h_z: Latent height H_z.
            w_z: Latent width W_z.

        Returns:
            variance: Predicted conditional variance of shape (B, 1, C_z, H_z, W_z).
        """
        has_time_dim = (features.dim() == 4)
        if not has_time_dim:
            features = features.unsqueeze(1)

        b, t, n, d = features.shape
        if n != h_z * w_z:
            raise ValueError(
                f"Spatial token count mismatch: got N={n}, expected {h_z}x{w_z}={h_z * w_z}"
            )

        raw_var = self.linear(features)  # (B, T, N, C_z)
        raw_var = raw_var.view(b, t, h_z, w_z, self.latent_channels).permute(0, 1, 4, 2, 3)

        # Guaranteed strict positivity: softplus(x) + variance_floor >= variance_floor
        variance = F.softplus(raw_var) + self.variance_floor
        return variance


def sample_next_latent(
    mu: torch.Tensor,
    variance: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample next latent state from conditional Gaussian N(mu, variance).

    Args:
        mu: Mean prediction tensor of shape (B, 1, C_z, H_z, W_z).
        variance: Variance prediction tensor of shape (B, 1, C_z, H_z, W_z).
        generator: Optional PyTorch Generator for reproducible sampling.
        seed: Optional integer seed (creates temporary seeded generator if generator is None).

    Returns:
        z_sample: Sampled next latent state of shape (B, 1, C_z, H_z, W_z).
    """
    std = torch.sqrt(variance)
    if seed is not None and generator is None:
        gen = torch.Generator(device=mu.device if mu.device.type != "mps" else "cpu")
        gen.manual_seed(seed)
    else:
        gen = generator

    noise = torch.randn(mu.shape, generator=gen, dtype=mu.dtype, device=mu.device)
    return mu + std * noise


def gaussian_nll_latent_loss(
    mu: torch.Tensor,
    target: torch.Tensor,
    variance: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute Gaussian Negative Log-Likelihood loss in latent space.

    Args:
        mu: Predicted mean tensor of shape (B, ..., C_z, H_z, W_z).
        target: Target latent tensor of same shape.
        variance: Predicted variance tensor of same shape.
        eps: Minimum variance epsilon for numerical log stability.

    Returns:
        loss: Scalar mean NLL loss.
    """
    return F.gaussian_nll_loss(input=mu, target=target, var=variance, eps=eps, reduction="mean")
