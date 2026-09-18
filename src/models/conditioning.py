"""Physical condition embedding and AdaLN (Adaptive Layer Norm) conditioning module.

Converts (Re, Sc) parameters via logarithmic transformation and MLP projection,
and injects into Transformer blocks via AdaLN modulation.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class PhysicalConditionEmbedding(nn.Module):
    """Encodes physical parameters (Re, Sc) into continuous condition vectors.

    Args:
        embed_dim: Output dimension of condition embedding.
        hidden_dim: Hidden dimension of MLP. Defaults to 2 * embed_dim.
        use_log: Whether to apply log10 transformation to Re and Sc. Default True.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: Optional[int] = None,
        use_log: bool = True,
    ):
        super().__init__()
        self.use_log = use_log
        hidden_dim = hidden_dim or (2 * embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.SiLU(),
        )

    def forward(self, re: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            re: Reynolds number tensor of shape (B,) or (B, 1).
            sc: Schmidt number tensor of shape (B,) or (B, 1).

        Returns:
            Condition embedding tensor of shape (B, embed_dim).
        """
        if re.ndim == 1:
            re = re.unsqueeze(-1)
        if sc.ndim == 1:
            sc = sc.unsqueeze(-1)

        if self.use_log:
            re = torch.log10(torch.clamp(re.float(), min=1e-6))
            sc = torch.log10(torch.clamp(sc.float(), min=1e-6))

        c = torch.cat([re, sc], dim=-1)  # (B, 2)
        return self.mlp(c)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization (AdaLN) modulation module.

    Computes scale (gamma) and shift (beta) parameters from condition embedding
    to modulate normalized hidden states:
        out = (1 + gamma) * LayerNorm(x) + beta

    Args:
        dim: Dimension of features to normalize.
        cond_dim: Dimension of condition embedding vector.
        eps: Epsilon for LayerNorm.
        zero_init: If True, initialize linear projection to zero weights/biases
                   so that modulation starts as identity transformation.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        eps: float = 1e-6,
        zero_init: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)

        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, ..., dim).
            cond: Optional condition tensor of shape (B, cond_dim).
                  If None, standard unmodulated LayerNorm is performed.

        Returns:
            Modulated tensor of shape (B, ..., dim).
        """
        norm_x = self.norm(x)
        if cond is None:
            return norm_x

        scale_shift = self.proj(cond)  # (B, 2 * dim)
        # Reshape to match x dimensions: (B, 1, ..., 1, 2 * dim)
        while scale_shift.ndim < x.ndim:
            scale_shift = scale_shift.unsqueeze(1)

        gamma, beta = torch.chunk(scale_shift, 2, dim=-1)
        return (1.0 + gamma) * norm_x + beta


class AdaLNZeroBlock(nn.Module):
    """DiT-style AdaLN modulation generating gamma, beta, and gating alpha.

    Produces:
        gamma1, beta1, alpha1 (for attention / spatial mixing)
        gamma2, beta2, alpha2 (for MLP / feedforward)
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim),
        )
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Returns (gamma1, beta1, alpha1, gamma2, beta2, alpha2) matching shape."""
        params = self.mod(cond)  # (B, 6 * dim)
        chunks = torch.chunk(params, 6, dim=-1)
        return chunks
