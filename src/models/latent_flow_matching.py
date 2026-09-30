"""Residual Latent Conditional Flow Matching Module for Fluid World Models.

Implements Residual Latent Conditional Flow Matching (Latent CFM) with an
Optimal-Transport-inspired straight probability path directly on normalized
latent residuals (aligned with the ArchesWeatherGen paradigm: deterministic prediction
+ normalized residual + flow matching + autoregressive ensemble generation):

1. SinusoidalTimeEmbedding: Continuous flow time tau in [0, 1] projection.
2. LatentPeriodicResBlock2D: Residual convolutional block with circular padding
   and AdaLN condition modulation respecting 2D periodic boundary conditions.
3. LatentSpatialAttention2D: Lightweight spatial self-attention on latent tokens.
4. LatentVelocityNet2D: Lightweight spatial velocity field network v_theta(x_tau, tau, context_mu, cond).
5. ODESolver: Numerical ODE integrators (Euler, Midpoint, Heun, RK4) for continuous-time sampling.
6. LatentFlowMatcher: Residual flow matcher supporting per-channel residual scale normalization,
   exact deterministic fallback, straight-path training loss, and multi-sample ensemble rollout.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.conditioning import PhysicalConditionEmbedding


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous flow time tau in [0, 1].

    Args:
        embed_dim: Dimension of time embedding output.
        max_period: Maximum frequency period (default: 10000.0).
    """

    def __init__(self, embed_dim: int = 128, max_period: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        half = embed_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """Map continuous flow time tau to embedding vector.

        Args:
            tau: Tensor of flow times in [0, 1], shape (B,) or (B, 1).

        Returns:
            Embedding tensor of shape (B, embed_dim).
        """
        if tau.ndim == 0:
            tau = tau.unsqueeze(0)
        if tau.ndim == 1:
            tau = tau.unsqueeze(-1)  # (B, 1)

        # Compute sin/cos components: (B, 1) * (1, half) -> (B, half)
        args = tau.float() * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embed_dim)
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class LatentPeriodicResBlock2D(nn.Module):
    """Residual convolutional block with circular padding and AdaLN modulation.

    Preserves 2D toroidal periodic boundaries of shear flow in the latent space.

    Args:
        channels: Number of input and output feature channels.
        cond_dim: Dimension of conditioning embedding vector.
        groups: Number of groups for GroupNorm (default: 8).
    """

    def __init__(self, channels: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.channels = channels
        self.cond_dim = cond_dim
        num_groups = min(groups, channels)

        self.norm1 = nn.GroupNorm(num_groups, channels, affine=False)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )

        self.norm2 = nn.GroupNorm(num_groups, channels, affine=False)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )

        # AdaLN projection: outputs scale (gamma1, gamma2) and shift (beta1, beta2)
        # shape: (B, 4 * channels)
        self.cond_proj = nn.Linear(cond_dim, 4 * channels)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Args:

        x: Feature tensor of shape (B, C, H, W).
        cond: Condition tensor of shape (B, cond_dim).
        """
        # Project conditioning to modulation parameters
        mod = self.cond_proj(cond)  # (B, 4 * C)
        gamma1, beta1, gamma2, beta2 = torch.chunk(mod, 4, dim=-1)

        # Reshape to broadcast over spatial dimensions: (B, C, 1, 1)
        gamma1 = gamma1.unsqueeze(-1).unsqueeze(-1)
        beta1 = beta1.unsqueeze(-1).unsqueeze(-1)
        gamma2 = gamma2.unsqueeze(-1).unsqueeze(-1)
        beta2 = beta2.unsqueeze(-1).unsqueeze(-1)

        # First modulated conv
        h = self.norm1(x)
        h = (1.0 + gamma1) * h + beta1
        h = self.act(h)
        h = self.conv1(h)

        # Second modulated conv
        h = self.norm2(h)
        h = (1.0 + gamma2) * h + beta2
        h = self.act(h)
        h = self.conv2(h)

        return x + h


class LatentSpatialAttention2D(nn.Module):
    """Lightweight 2D spatial self-attention on latent tokens.

    Captures non-local spatial coherence across Kelvin-Helmholtz vortices.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        h_norm = self.norm(x)
        qkv = self.qkv(h_norm)  # (B, 3*C, H, W)
        qkv = qkv.view(b, 3, self.num_heads, self.head_dim, n).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        out = self.proj(out)
        return x + out


class LatentVelocityNet2D(nn.Module):
    """Latent Velocity Field Network v_theta(x_tau, tau, context_mu, cond).

    Predicts the time-dependent velocity field dx_tau / dtau in the latent space.

    Args:
        latent_channels: Number of latent state channels C_z (default: 64).
        hidden_channels: Internal convolutional feature channels (default: 128).
        cond_dim: Physical condition dimension (default: 128).
        time_dim: Flow time embedding dimension (default: 128).
        num_blocks: Number of periodic residual blocks (default: 4).
        use_spatial_attn: Whether to include spatial self-attention in middle blocks.
        zero_init: If True, initializes output convolution to zero weights and biases.
    """

    def __init__(
        self,
        latent_channels: int = 64,
        hidden_channels: int = 128,
        cond_dim: int = 128,
        time_dim: int = 128,
        num_blocks: int = 4,
        use_spatial_attn: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.cond_dim = cond_dim
        self.time_dim = time_dim

        # Continuous flow time encoder
        self.time_embed = SinusoidalTimeEmbedding(embed_dim=time_dim)

        # Joint conditioning projection: time_dim + cond_dim -> hidden_channels
        self.joint_cond_mlp = nn.Sequential(
            nn.Linear(time_dim + cond_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Input projection: concatenates x_tau (C_z) and context mean mu (C_z) -> 2 * C_z
        self.in_conv = nn.Conv2d(
            latent_channels * 2,
            hidden_channels,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )

        # Sequential periodic residual blocks with periodic circular padding
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                LatentPeriodicResBlock2D(
                    channels=hidden_channels,
                    cond_dim=hidden_channels,
                    groups=8,
                )
            )

        # Optional mid-level spatial attention
        self.use_spatial_attn = use_spatial_attn
        if use_spatial_attn:
            self.attn = LatentSpatialAttention2D(hidden_channels, num_heads=4)
        else:
            self.attn = None

        # Output projection back to latent_channels: predicts velocity field v_theta
        self.out_norm = nn.GroupNorm(min(8, hidden_channels), hidden_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(
            hidden_channels,
            latent_channels,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )

        if zero_init:
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        context_mu: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate predicted velocity field v_theta(x_tau, tau, context_mu, cond).

        Args:
            x_tau: State tensor along flow path at time tau, shape (B, C_z, H_z, W_z).
            tau: Flow time tensor, shape (B,) or (B, 1) or scalar float.
            context_mu: Context prior mean tensor, shape (B, C_z, H_z, W_z).
            cond: Physical condition embedding tensor, shape (B, cond_dim).

        Returns:
            v_pred: Predicted velocity field dx/dtau of shape (B, C_z, H_z, W_z).
        """
        # Ensure tau is tensor
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor([tau], dtype=x_tau.dtype, device=x_tau.device).repeat(x_tau.shape[0])
        elif tau.numel() == 1 and x_tau.shape[0] > 1:
            tau = tau.repeat(x_tau.shape[0])

        # Flow time embedding
        t_emb = self.time_embed(tau)  # (B, time_dim)

        # Joint conditioning vector
        joint_c = torch.cat([t_emb, cond], dim=-1)  # (B, time_dim + cond_dim)
        c_emb = self.joint_cond_mlp(joint_c)  # (B, hidden_channels)

        # Concatenate flow state and context mean along channel dimension
        h = torch.cat([x_tau, context_mu], dim=1)  # (B, 2 * C_z, H_z, W_z)
        h = self.in_conv(h)

        mid_idx = len(self.blocks) // 2
        for i, block in enumerate(self.blocks):
            h = block(h, cond=c_emb)
            if self.attn is not None and i == mid_idx:
                h = self.attn(h)

        h = self.out_act(self.out_norm(h))
        v_pred = self.out_conv(h)
        return v_pred


class ODESolver:
    """Numerical ODE Integrator for Latent Continuous Normalizing Flows."""

    SUPPORTED_SOLVERS = ("euler", "midpoint", "heun", "rk4")

    @classmethod
    def integrate(
        cls,
        func: nn.Module,
        x0: torch.Tensor,
        context_mu: torch.Tensor,
        cond: torch.Tensor,
        num_steps: int = 10,
        solver: str = "midpoint",
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Integrate dx/dtau = func(x, tau, context_mu, cond) from tau=0 to tau=1.

        Args:
            func: Velocity field callable (e.g. LatentVelocityNet2D).
            x0: Initial state at tau=0 of shape (B, C_z, H_z, W_z).
            context_mu: Conditioning mean prior of shape (B, C_z, H_z, W_z).
            cond: Physical condition vector of shape (B, cond_dim).
            num_steps: Number of integration steps (default: 10).
            solver: One of ('euler', 'midpoint', 'heun', 'rk4').
            return_trajectory: If True, also returns list of states at each step.

        Returns:
            x1: Integrated terminal state at tau=1 of shape (B, C_z, H_z, W_z).
            trajectory: (Optional) List of tensors at each integration step if return_trajectory=True.
        """
        solver = solver.lower()
        if solver not in cls.SUPPORTED_SOLVERS:
            raise ValueError(f"Unsupported ODE solver '{solver}'. Choose from {cls.SUPPORTED_SOLVERS}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive integer, got {num_steps}")

        dt = 1.0 / float(num_steps)
        x = x0
        device = x0.device
        dtype = x0.dtype
        b = x0.shape[0]

        trajectory = [x] if return_trajectory else None

        for step in range(num_steps):
            tau_val = float(step * dt)
            tau = torch.full((b, 1), tau_val, device=device, dtype=dtype)

            if solver == "euler":
                v = func(x, tau, context_mu, cond)
                x = x + dt * v

            elif solver == "midpoint":
                v = func(x, tau, context_mu, cond)
                tau_mid = torch.full((b, 1), tau_val + 0.5 * dt, device=device, dtype=dtype)
                x_mid = x + 0.5 * dt * v
                v_mid = func(x_mid, tau_mid, context_mu, cond)
                x = x + dt * v_mid

            elif solver == "heun":
                v1 = func(x, tau, context_mu, cond)
                tau_next = torch.full((b, 1), min(1.0, tau_val + dt), device=device, dtype=dtype)
                x_pred = x + dt * v1
                v2 = func(x_pred, tau_next, context_mu, cond)
                x = x + 0.5 * dt * (v1 + v2)

            elif solver == "rk4":
                v1 = func(x, tau, context_mu, cond)

                tau_half = torch.full((b, 1), tau_val + 0.5 * dt, device=device, dtype=dtype)
                v2 = func(x + 0.5 * dt * v1, tau_half, context_mu, cond)
                v3 = func(x + 0.5 * dt * v2, tau_half, context_mu, cond)

                tau_next = torch.full((b, 1), min(1.0, tau_val + dt), device=device, dtype=dtype)
                v4 = func(x + dt * v3, tau_next, context_mu, cond)

                x = x + (dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4)

            if return_trajectory:
                trajectory.append(x)

        if return_trajectory:
            return x, trajectory
        return x


class LatentFlowMatcher(nn.Module):
    """Residual Latent Conditional Flow Matcher with OT-inspired probability path.

    Adopts the ArchesWeatherGen paradigm:
    1. Deterministic forecaster D0 provides the predictable component mu_t;
    2. Residual r_t = Z_{t+1} - mu_t is normalized by channel RMS scale s_c = sqrt(E[r_c^2] + eps);
    3. Conditional Flow Matching models the normalized residual distribution x_1 = r / s;
    4. Base distribution x_0 ~ N(0, I) has naturally matched unit scale;
    5. After ODE sampling x_1, denormalization r = x_1 * s restores true physical variance;
    6. Prediction is Z_{t+1} = mu_t + r.

    Args:
        latent_channels: Latent channel count C_z (default: 64).
        cond_dim: Physical condition dimension (default: 128).
        hidden_channels: Velocity network channels (default: 128).
        num_blocks: Number of periodic residual blocks (default: 4).
        use_spatial_attn: Whether to include spatial attention in velocity network.
        target_mode: 'residual' (learns z_{t+1} - mu_t) or 'direct' (learns z_{t+1}).
                     Default: 'residual'.
        sigma_min: Numerical boundary smoothness for probability path (default: 1e-4).
        zero_init: Zero-initialize velocity head for baseline parity at initialization.
        residual_scale: Optional per-channel scaling tensor (C_z,) or (1, C_z, 1, 1).
    """

    def __init__(
        self,
        latent_channels: int = 64,
        cond_dim: int = 128,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        use_spatial_attn: bool = True,
        target_mode: str = "residual",
        sigma_min: float = 1e-4,
        zero_init: bool = True,
        residual_scale: Optional[Union[torch.Tensor, List[float]]] = None,
    ):
        super().__init__()
        assert target_mode in ("residual", "direct"), f"Unknown target_mode: {target_mode}"
        self.latent_channels = latent_channels
        self.cond_dim = cond_dim
        self.hidden_channels = hidden_channels
        self.target_mode = target_mode
        self.sigma_min = float(sigma_min)

        # Condition embedder for (Re, Sc)
        self.cond_embed = PhysicalConditionEmbedding(embed_dim=cond_dim)

        # Velocity field neural network
        self.velocity_net = LatentVelocityNet2D(
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            cond_dim=cond_dim,
            time_dim=cond_dim,
            num_blocks=num_blocks,
            use_spatial_attn=use_spatial_attn,
            zero_init=zero_init,
        )

        # Residual scale buffer: s_c = sqrt(E[r_c^2] + eps)
        self.register_buffer("residual_scale", None)
        if residual_scale is not None:
            self.set_residual_scale(residual_scale)

    def set_residual_scale(self, scale: Union[torch.Tensor, List[float]]) -> None:
        """Set per-channel residual scale tensor s_c = sqrt(E[r_c^2] + eps).

        Normalizes raw latent residual to unit second moment prior to CFM training,
        ensuring MSE loss treats all latent channels equally (ArchesWeatherGen paradigm).

        Args:
            scale: Tensor or sequence of shape (C_z,) or (1, C_z, 1, 1).
        """
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32)
        valid_1d = (scale_tensor.ndim == 1 and scale_tensor.shape[0] == self.latent_channels)
        valid_4d = (scale_tensor.ndim == 4 and scale_tensor.shape == (1, self.latent_channels, 1, 1))
        if not (valid_1d or valid_4d):
            raise ValueError(
                f"residual_scale must have shape ({self.latent_channels},) or (1, {self.latent_channels}, 1, 1), "
                f"got shape {tuple(scale_tensor.shape)}"
            )
        if not torch.isfinite(scale_tensor).all():
            raise ValueError("residual_scale contains non-finite values (NaN or Inf)")
        if (scale_tensor <= 0).any():
            raise ValueError("residual_scale values must be strictly positive (> 0) for all channels")

        if scale_tensor.ndim == 1:
            scale_tensor = scale_tensor.view(1, -1, 1, 1)
        self.register_buffer("residual_scale", scale_tensor)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Strict governance hook for loading checkpoints containing residual_scale.

        Enforces:
        1. When key is present in state_dict:
           - Must be a valid torch.Tensor.
           - Must strictly validate shape, finiteness (no NaN/Inf), and positivity (> 0).
           - Any violation records to error_msgs for strict fail-closed rejection.
        2. Seamlessly registers validated scale tensor so strict=True checkpoint loading succeeds.
        """
        key = prefix + "residual_scale"
        if key in state_dict:
            val = state_dict[key]
            if val is None:
                error_msgs.append(f"Buffer '{key}' in state_dict cannot be None")
            elif not isinstance(val, torch.Tensor):
                error_msgs.append(f"Buffer '{key}' must be a torch.Tensor, got {type(val)}")
            else:
                try:
                    self.set_residual_scale(val)
                except ValueError as e:
                    error_msgs.append(f"Invalid '{key}' in state_dict: {str(e)}")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @classmethod
    def from_residual_stats(
        cls,
        stats_data: Dict[str, Any],
        latent_channels: int = 64,
        eps: float = 1e-6,
        **kwargs,
    ) -> "LatentFlowMatcher":
        """Instantiate LatentFlowMatcher with residual scale from Phase 0 statistics."""
        if not isinstance(stats_data, dict) or "statistics" not in stats_data:
            raise ValueError("stats_data missing required 'statistics' dictionary")
        if "channel_residual_second_moment_g0" not in stats_data["statistics"]:
            raise ValueError("stats_data missing required 'statistics.channel_residual_second_moment_g0'")

        second_moments = stats_data["statistics"]["channel_residual_second_moment_g0"]
        second_moments_tensor = torch.as_tensor(second_moments, dtype=torch.float32)
        if not torch.isfinite(second_moments_tensor).all():
            raise ValueError("channel_residual_second_moment_g0 contains non-finite values")
        if (second_moments_tensor < 0).any():
            raise ValueError("channel_residual_second_moment_g0 contains negative values")

        scale = torch.sqrt(second_moments_tensor + eps)
        return cls(latent_channels=latent_channels, residual_scale=scale, **kwargs)

    def _resolve_condition(
        self,
        b: int,
        device: torch.device,
        dtype: torch.dtype,
        re: Optional[torch.Tensor] = None,
        sc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Helper to compute or default condition embedding vector."""
        if re is not None and sc is not None:
            c = self.cond_embed(re, sc)
            if c.shape[0] == 1 and b > 1:
                c = c.expand(b, -1)
            return c
        # Default condition vector (zeros)
        return torch.zeros(b, self.cond_dim, device=device, dtype=dtype)

    def compute_loss(
        self,
        z_next: torch.Tensor,
        mu: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        sc: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        custom_tau: Optional[torch.Tensor] = None,
        custom_x0: Optional[torch.Tensor] = None,
        context: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute CFM regression loss with residual scale normalization.

        Args:
            z_next: True next latent state, shape (B, 1, C_z, H_z, W_z) or (B, C_z, H_z, W_z).
            mu: Deterministic mean prediction, shape matching z_next.
            re: Reynolds number tensor (B,) or (B, 1).
            sc: Schmidt number tensor (B,) or (B, 1).
            generator: Optional PyTorch Generator for reproducible random sampling.
            custom_tau: Optional pre-sampled flow time tensor for deterministic testing.
            custom_x0: Optional pre-sampled base noise tensor for deterministic testing.
            context: Optional Context contract containing physical parameters (re, sc).

        Returns:
            Dict containing 'loss' (scalar mean squared error), 'v_pred', 'u_target', 'x_tau'.
        """
        from src.contracts.context import resolve_context
        ctx = resolve_context(context=context, re=re, sc=sc)
        re, sc = ctx.to_re_sc() if ctx is not None else (None, None)

        if z_next.ndim == 5:
            z_next = z_next.squeeze(1)
        if mu.ndim == 5:
            mu = mu.squeeze(1)

        b, c_z, hz, wz = z_next.shape
        device = z_next.device
        dtype = z_next.dtype

        # 1. Determine target x_1
        if self.target_mode == "residual":
            raw_residual = z_next - mu  # Ground-truth residual
            if self.residual_scale is not None:
                scale = self.residual_scale.to(device=device, dtype=dtype)
                x1 = raw_residual / scale
            else:
                x1 = raw_residual
        else:
            x1 = z_next  # Absolute latent target

        # 2. Sample base noise x_0 ~ N(0, I)
        if custom_x0 is not None:
            x0 = custom_x0.to(device=device, dtype=dtype)
            if x0.ndim == 5:
                x0 = x0.squeeze(1)
        else:
            x0 = torch.randn(b, c_z, hz, wz, generator=generator, device=device, dtype=dtype)

        # 3. Sample flow time tau ~ U(0, 1)
        if custom_tau is not None:
            tau = custom_tau.to(device=device, dtype=dtype)
            if tau.ndim == 1:
                tau = tau.unsqueeze(-1)
        else:
            tau = torch.rand(b, 1, generator=generator, device=device, dtype=dtype)

        # 4. Construct straight conditional probability path
        # x_tau = (1 - (1 - sigma_min) * tau) * x0 + tau * x1
        tau_expanded = tau.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        x_tau = (1.0 - (1.0 - self.sigma_min) * tau_expanded) * x0 + tau_expanded * x1

        # 5. Target conditional vector field: u_tau = x1 - (1 - sigma_min) * x0
        u_target = x1 - (1.0 - self.sigma_min) * x0

        # 6. Physical condition embedding
        cond = self._resolve_condition(b=b, device=device, dtype=dtype, re=re, sc=sc)

        # 7. Predict velocity field
        v_pred = self.velocity_net(
            x_tau=x_tau,
            tau=tau,
            context_mu=mu,
            cond=cond,
        )

        # 8. Regression loss: mean squared error across all elements
        loss = F.mse_loss(v_pred, u_target)

        return {
            "loss": loss,
            "v_pred": v_pred,
            "u_target": u_target,
            "x_tau": x_tau,
        }

    def sample_next_latent(
        self,
        mu: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        sc: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        solver: str = "midpoint",
        noise_scale: float = 1.0,
        deterministic_fallback: bool = False,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        custom_x0: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        context: Optional[Any] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Sample next latent state by integrating the learned neural ODE from tau=0 to tau=1.

        Args:
            mu: Deterministic prior mean, shape (B, 1, C_z, H_z, W_z) or (B, C_z, H_z, W_z).
            re: Optional Reynolds number tensor (B,).
            sc: Optional Schmidt number tensor (B,).
            num_steps: Number of numerical integration steps (default: 10).
            solver: ODE solver name ('euler', 'midpoint', 'heun', 'rk4').
            noise_scale: Standard deviation scaling for base noise (default: 1.0).
                         Acts as a sampling temperature / dispersion knob.
                         NOTE: setting noise_scale=0.0 starts the ODE from x0=0. In a trained
                         network with learned non-zero velocity, this yields a mean-drift trajectory
                         and does NOT guarantee exact D0 parity.
            deterministic_fallback: If True, bypasses ODE sampling completely and returns mu directly.
                                     Guarantees strict zero-error D0 structural parity.
            generator: Optional PyTorch Generator.
            seed: Optional integer seed for reproducibility.
            custom_x0: Optional initial base noise tensor for testing.
            return_trajectory: If True, also returns list of intermediate latent tensors.
            context: Optional Context contract containing physical parameters (re, sc).

        Returns:
            z_sample: Sampled next latent state of shape (B, 1, C_z, H_z, W_z).
            trajectory: (Optional) Intermediate trajectory states if return_trajectory=True.
        """
        from src.contracts.context import resolve_context
        ctx = resolve_context(context=context, re=re, sc=sc)
        re, sc = ctx.to_re_sc() if ctx is not None else (None, None)

        orig_ndim = mu.ndim
        if orig_ndim == 5:
            mu_2d = mu.squeeze(1)
        else:
            mu_2d = mu

        # 1. Exact deterministic fallback
        if deterministic_fallback:
            if return_trajectory:
                return (mu.clone() if orig_ndim == 5 else mu_2d.clone()), [mu.clone() if orig_ndim == 5 else mu_2d.clone()]
            return mu.clone() if orig_ndim == 5 else mu_2d.clone()

        b, c_z, hz, wz = mu_2d.shape
        device = mu_2d.device
        dtype = mu_2d.dtype

        if seed is not None and generator is None:
            gen = torch.Generator(device=device if device.type != "mps" else "cpu")
            gen.manual_seed(seed)
        else:
            gen = generator

        # Initial noise state at tau = 0
        if custom_x0 is not None:
            x0 = custom_x0.to(device=device, dtype=dtype)
            if x0.ndim == 5:
                x0 = x0.squeeze(1)
            x0 = x0 * float(noise_scale)
        else:
            if float(noise_scale) == 0.0:
                x0 = torch.zeros(b, c_z, hz, wz, device=device, dtype=dtype)
            else:
                x0 = torch.randn(b, c_z, hz, wz, generator=gen, device=device, dtype=dtype) * float(noise_scale)

        # Condition embedding
        cond = self._resolve_condition(b=b, device=device, dtype=dtype, re=re, sc=sc)

        # Integrate ODE from tau=0 to tau=1
        if return_trajectory:
            x1, traj = ODESolver.integrate(
                func=self.velocity_net,
                x0=x0,
                context_mu=mu_2d,
                cond=cond,
                num_steps=num_steps,
                solver=solver,
                return_trajectory=True,
            )
        else:
            x1 = ODESolver.integrate(
                func=self.velocity_net,
                x0=x0,
                context_mu=mu_2d,
                cond=cond,
                num_steps=num_steps,
                solver=solver,
                return_trajectory=False,
            )

        # Rescale normalized residual back to true physical scale
        if self.target_mode == "residual":
            if self.residual_scale is not None:
                scale = self.residual_scale.to(device=device, dtype=dtype)
                x1_scaled = x1 * scale
            else:
                x1_scaled = x1
            z_sample = mu_2d + x1_scaled
        else:
            z_sample = x1

        if orig_ndim == 5:
            z_sample = z_sample.unsqueeze(1)

        if return_trajectory:
            traj_out = []
            scale = self.residual_scale.to(device=device, dtype=dtype) if self.residual_scale is not None else None
            for t_item in traj:
                if self.target_mode == "residual":
                    t_scaled = t_item * scale if scale is not None else t_item
                    s = mu_2d + t_scaled
                else:
                    s = t_item
                if orig_ndim == 5:
                    s = s.unsqueeze(1)
                traj_out.append(s)
            return z_sample, traj_out

        return z_sample

    def sample_ensemble(
        self,
        mu: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        sc: Optional[torch.Tensor] = None,
        num_samples: int = 8,
        num_steps: int = 10,
        solver: str = "midpoint",
        noise_scale: float = 1.0,
        deterministic_fallback: bool = False,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        custom_x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample an ensemble of K independent trajectories in parallel.

        Args:
            mu: Prior mean tensor of shape (B, 1, C_z, H_z, W_z).
            re: Reynolds number tensor (B,).
            sc: Schmidt number tensor (B,).
            num_samples: Number of sample realizations K (default: 8).
            num_steps: ODE integration steps per sample.
            solver: ODE solver name.
            noise_scale: Sampling temperature / dispersion knob.
            deterministic_fallback: If True, strictly returns mu without ODE sampling.
            generator: Optional PyTorch Generator.
            seed: Optional integer seed.
            custom_x0: Optional standard normal base noise tensor for Common Random Numbers.
                       Can be (B, K, 1, C_z, Hz, Wz) or (B * K, 1, C_z, Hz, Wz) or (B * K, C_z, Hz, Wz).

        Returns:
            z_ensemble: Sampled tensor of shape (B, K, 1, C_z, H_z, W_z).
        """
        b = mu.shape[0]
        k = num_samples

        if deterministic_fallback:
            return mu.unsqueeze(1).repeat(1, k, 1, 1, 1, 1)

        # Expand batch by K for parallel execution
        mu_exp = mu.repeat_interleave(k, dim=0)  # (B * K, 1, C_z, H_z, W_z)
        re_exp = re.repeat_interleave(k, dim=0) if re is not None else None
        sc_exp = sc.repeat_interleave(k, dim=0) if sc is not None else None

        flat_custom_x0 = None
        if custom_x0 is not None:
            if custom_x0.ndim == 6:  # (B, K, 1, C_z, Hz, Wz)
                c_z, hz, wz = custom_x0.shape[3], custom_x0.shape[4], custom_x0.shape[5]
                flat_custom_x0 = custom_x0.view(b * k, c_z, hz, wz)
            elif custom_x0.ndim == 5:  # (B * K, 1, C_z, Hz, Wz)
                flat_custom_x0 = custom_x0.squeeze(1)
            elif custom_x0.ndim == 4:  # (B * K, C_z, Hz, Wz)
                flat_custom_x0 = custom_x0

        z_samples_flat = self.sample_next_latent(
            mu=mu_exp,
            re=re_exp,
            sc=sc_exp,
            num_steps=num_steps,
            solver=solver,
            noise_scale=noise_scale,
            deterministic_fallback=False,
            generator=generator,
            seed=seed,
            custom_x0=flat_custom_x0,
            return_trajectory=False,
        )  # (B * K, 1, C_z, H_z, W_z)

        # Reshape to (B, K, 1, C_z, H_z, W_z)
        c_z = z_samples_flat.shape[2]
        hz = z_samples_flat.shape[3]
        wz = z_samples_flat.shape[4]
        return z_samples_flat.view(b, k, 1, c_z, hz, wz)
