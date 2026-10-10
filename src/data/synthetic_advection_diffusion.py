"""Synthetic Periodic Scalar Advection-Diffusion Analytical Dynamics Module.

Provides benchmark trajectories for 2D scalar fields governed by the periodic advection-diffusion equation:
    d omega / dt + u0 * d omega / dx + v0 * d omega / dy = nu * laplacian(omega)

Domain: [0, Lx) x [0, Ly) with periodic boundary conditions.
Physical specification:
- Velocity (u0, v0) is constant and uniform across space and time.
- Viscosity nu >= 0 is constant.
- For bandlimited periodic initial conditions, exact analytical solutions are obtained
  via closed-form Fourier mode propagation:
    hat{omega}(k, t) = hat{omega}(k, 0) * exp( -i * (u0 * kx + v0 * ky) * t - nu * |k|^2 * t )

Key engineering contracts:
1. Closed-form analytical propagation with machine-precision accuracy in float64 / float32.
2. Trajectory-level identity isolation: train, validation, and test splits are strictly
   partitioned by initial condition / trajectory seeds (no cross-time frame leakage).
3. Explicit enstrophy balance: instantaneous enstrophy Z(t) = 0.5 * integral(omega^2) and
   dissipation rate dZ/dt = -nu * integral(|grad(omega)|^2) are tracked analytically.
"""

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.fft

from src.utils.fft_derivatives import get_wavenumbers, spectral_grad_2d


@dataclass
class AdvectionDiffusionConfig:
    """Configuration for periodic scalar advection-diffusion analytical dynamics."""
    nx: int = 64
    ny: int = 64
    lx: float = 1.0
    ly: float = 1.0
    u0: float = 0.5
    v0: float = 0.5
    nu: float = 0.001
    base_wavenumber: int = 1
    perturbation_modes: Tuple[Tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (2, 1))
    perturbation_amplitude: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "lx": self.lx,
            "ly": self.ly,
            "u0": self.u0,
            "v0": self.v0,
            "nu": self.nu,
            "base_wavenumber": self.base_wavenumber,
            "perturbation_modes": [list(m) for m in self.perturbation_modes],
            "perturbation_amplitude": self.perturbation_amplitude,
        }


def compute_tensor_digest(tensor: torch.Tensor) -> Dict[str, Any]:
    """Compute deterministic SHA-256 fingerprint and summary statistics of a PyTorch tensor."""
    raw_bytes = tensor.contiguous().cpu().numpy().tobytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    return {
        "sha256": sha256,
        "shape": list(tensor.shape),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


class PeriodicScalarAdvectionDiffusion:
    """Analytical solver and generator for 2D periodic scalar advection-diffusion fields."""

    def __init__(self, cfg: Optional[AdvectionDiffusionConfig] = None):
        self.cfg = cfg or AdvectionDiffusionConfig()

    def generate_initial_condition(
        self,
        seed: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, List[float]]:
        """Generate a smooth periodic initial scalar field with reproducible random phases.

        Base field:
            omega_0(x, y) = 2.0 * cos(2*pi*k0*x/Lx) * cos(2*pi*k0*y/Ly)
                          = cos(2*pi*k0*(x/Lx + y/Ly)) + cos(2*pi*k0*(x/Lx - y/Ly))
        Perturbations:
            sum_{m, n} eps * sin(2*pi*(m*x/Lx + n*y/Ly) + phi_{m,n})

        Returns:
            field: Tensor of shape (1, nx, ny)
            phases: List of float phases corresponding to each perturbation mode
        """
        gen = torch.Generator(device="cpu").manual_seed(seed)

        nx, ny = self.cfg.nx, self.cfg.ny
        lx, ly = self.cfg.lx, self.cfg.ly
        k0 = self.cfg.base_wavenumber

        x = torch.arange(nx, dtype=dtype, device=device) * (lx / nx)
        y = torch.arange(ny, dtype=dtype, device=device) * (ly / ny)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # Base quadrupole field
        base_field = 2.0 * torch.cos(2.0 * math.pi * k0 * X / lx) * torch.cos(2.0 * math.pi * k0 * Y / ly)

        pert_field = torch.zeros_like(base_field)
        phases: List[float] = []

        for m, n in self.cfg.perturbation_modes:
            phi = float(torch.rand(1, generator=gen).item() * 2.0 * math.pi)
            phases.append(phi)
            arg = 2.0 * math.pi * (m * X / lx + n * Y / ly) + phi
            pert_field = pert_field + self.cfg.perturbation_amplitude * torch.sin(arg)

        field = (base_field + pert_field).unsqueeze(0)  # (1, nx, ny)
        return field, phases

    def step_algebraic(
        self,
        t: float,
        phases: List[float],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Exact algebraic analytical solution at continuous time t >= 0.

        Exploits exact modal decomposition under constant advection and linear diffusion:
        For mode (kx, ky) with amplitude A and phase phi:
            omega(x, y, t) = A * exp( -nu * (kx^2 + ky^2) * t ) *
                             func( kx * (x - u0 * t) + ky * (y - v0 * t) + phi )

        Returns:
            field: Tensor of shape (1, nx, ny)
        """
        nx, ny = self.cfg.nx, self.cfg.ny
        lx, ly = self.cfg.lx, self.cfg.ly
        k0 = self.cfg.base_wavenumber
        u0, v0, nu = self.cfg.u0, self.cfg.v0, self.cfg.nu

        x = torch.arange(nx, dtype=dtype, device=device) * (lx / nx)
        y = torch.arange(ny, dtype=dtype, device=device) * (ly / ny)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # Advected coordinates
        X_adv = X - u0 * t
        Y_adv = Y - v0 * t

        # Base modes:
        # 2*cos(kx*X)*cos(ky*Y) = cos(kx*X + ky*Y) + cos(kx*X - ky*Y)
        # Both modes (k0, k0) and (k0, -k0) have |k|^2 = (2*pi*k0/lx)^2 + (2*pi*k0/ly)^2
        kx0 = 2.0 * math.pi * k0 / lx
        ky0 = 2.0 * math.pi * k0 / ly
        k_base_sq = kx0**2 + ky0**2
        decay_base = math.exp(-nu * k_base_sq * t)

        base_t = decay_base * 2.0 * torch.cos(kx0 * X_adv) * torch.cos(ky0 * Y_adv)

        pert_t = torch.zeros_like(base_t)
        for (m, n), phi in zip(self.cfg.perturbation_modes, phases):
            km = 2.0 * math.pi * m / lx
            kn = 2.0 * math.pi * n / ly
            k_mode_sq = km**2 + kn**2
            decay_mode = math.exp(-nu * k_mode_sq * t)
            arg = km * X_adv + kn * Y_adv + phi
            pert_t = pert_t + (self.cfg.perturbation_amplitude * decay_mode) * torch.sin(arg)

        field_t = (base_t + pert_t).unsqueeze(0)  # (1, nx, ny)
        return field_t

    def step_spectral(
        self,
        field_0: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """Propagate arbitrary 2D periodic field analytically in Fourier space.

        hat{omega}(k, t) = hat{omega}(k, 0) * exp( -i * (u0 * kx + v0 * ky) * t - nu * |k|^2 * t )

        Args:
            field_0: Tensor of shape (..., nx, ny), real-valued.
            t: Continuous propagation time in seconds.

        Returns:
            field_t: Propagated field of same shape and dtype.
        """
        if t == 0.0:
            return field_0.clone()

        nx, ny = field_0.shape[-2], field_0.shape[-1]
        lx, ly = self.cfg.lx, self.cfg.ly
        u0, v0, nu = self.cfg.u0, self.cfg.v0, self.cfg.nu

        device = field_0.device
        orig_dtype = field_0.dtype
        calc_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
        f_calc = field_0.to(dtype=calc_dtype)

        kx_base, ky_base = get_wavenumbers(nx, ny, lx, ly, device, calc_dtype)
        if field_0.ndim > 2:
            kx = kx_base.view(*([1] * (field_0.ndim - 2)), nx, 1)
            ky = ky_base.view(*([1] * (field_0.ndim - 2)), 1, ny // 2 + 1)
        else:
            kx, ky = kx_base, ky_base

        # Forward 2D Real FFT
        f_hat = torch.fft.rfft2(f_calc, dim=(-2, -1))

        # Phase velocity and diffusion decay factor
        phase_shift = -(u0 * kx + v0 * ky) * t
        diff_decay = -nu * (kx**2 + ky**2) * t
        prop_factor = torch.exp(torch.complex(diff_decay, phase_shift))

        f_hat_t = f_hat * prop_factor

        # Zero Nyquist derivative mode if even dimension
        if nx % 2 == 0:
            f_hat_t[..., nx // 2, :] = 0.0
        if ny % 2 == 0:
            f_hat_t[..., :, ny // 2] = 0.0

        f_t = torch.fft.irfft2(f_hat_t, s=(nx, ny), dim=(-2, -1))
        return f_t.to(dtype=orig_dtype)

    def generate_trajectory(
        self,
        seed: int,
        num_steps: int,
        dt: float,
        use_algebraic: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate a single trajectory of length (num_steps + 1).

        Returns:
            Tensor of shape (num_steps + 1, 1, nx, ny) representing frames at t = 0, dt, 2*dt, ..., T.
        """
        field_0, phases = self.generate_initial_condition(seed=seed, dtype=dtype, device=device)
        frames = [field_0]

        for step in range(1, num_steps + 1):
            t = float(step * dt)
            if use_algebraic:
                field_t = self.step_algebraic(t=t, phases=phases, dtype=dtype, device=device)
            else:
                field_t = self.step_spectral(field_0=field_0, t=t)
            frames.append(field_t)

        return torch.stack(frames, dim=0)  # (T+1, 1, nx, ny)

    def compute_enstrophy(self, field: torch.Tensor) -> torch.Tensor:
        """Compute spatial mean enstrophy Z = 0.5 * mean(omega^2).

        Args:
            field: Tensor of shape (..., nx, ny).

        Returns:
            Tensor of shape (...) with mean enstrophy values.
        """
        return 0.5 * torch.mean(field**2, dim=(-2, -1))

    def compute_enstrophy_dissipation_rate(self, field: torch.Tensor) -> torch.Tensor:
        """Compute theoretical enstrophy dissipation rate dZ/dt = -nu * integral(|grad(omega)|^2) / Area.

        In mean spatial units (area-normalized):
            dZ/dt = -nu * mean( (d omega / dx)^2 + (d omega / dy)^2 )

        Args:
            field: Tensor of shape (..., nx, ny).

        Returns:
            Tensor of shape (...) with dissipation rate (always <= 0 when nu >= 0).
        """
        if self.cfg.nu == 0.0:
            return torch.zeros(field.shape[:-2], dtype=field.dtype, device=field.device)

        domain_size = (self.cfg.lx, self.cfg.ly)
        df_dx, df_dy = spectral_grad_2d(field, domain_size=domain_size)
        grad_sq_mean = torch.mean(df_dx**2 + df_dy**2, dim=(-2, -1))
        return -self.cfg.nu * grad_sq_mean


def generate_trajectory_dataset(
    num_trajectories: int,
    num_steps: int,
    dt: float,
    seed_base: int = 1000,
    cfg: Optional[AdvectionDiffusionConfig] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, List[int]]:
    """Generate a batch of independent trajectories partitioned strictly by initial condition seeds.

    Args:
        num_trajectories: Number of independent trajectories to synthesize.
        num_steps: Number of forward time steps per trajectory.
        dt: Time increment per step.
        seed_base: Base integer for trajectory seeds: seed_i = seed_base + i.
        cfg: Configuration parameters.
        dtype: PyTorch floating point type.
        device: PyTorch device.

    Returns:
        trajectories: Tensor of shape (num_trajectories, num_steps + 1, 1, nx, ny)
        seeds: List of integer seeds used for trajectory identities
    """
    solver = PeriodicScalarAdvectionDiffusion(cfg=cfg)
    trajs = []
    seeds = []

    for i in range(num_trajectories):
        s = seed_base + i
        seeds.append(s)
        traj = solver.generate_trajectory(seed=s, num_steps=num_steps, dt=dt, dtype=dtype, device=device)
        trajs.append(traj)

    dataset_tensor = torch.stack(trajs, dim=0)  # (N, T+1, 1, nx, ny)
    return dataset_tensor, seeds


def build_trajectory_windows(
    trajectories: torch.Tensor,
    history_len: int,
    future_len: int,
    stride: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice trajectory tensors into history and future forecasting windows.

    Args:
        trajectories: Tensor of shape (N, T+1, C, H, W)
        history_len: Window length L for past history
        future_len: Horizon length H for future rollout
        stride: Stride between window starting indices

    Returns:
        history: Tensor of shape (B, L, C, H, W)
        future: Tensor of shape (B, H, C, H, W)
    """
    if trajectories.ndim != 5:
        raise ValueError(f"Expected 5D trajectory tensor [N, T+1, C, H, W], got shape {trajectories.shape}")

    n_trajs, total_steps, c, h, w = trajectories.shape
    total_window_len = history_len + future_len

    if total_steps < total_window_len:
        raise ValueError(
            f"Trajectory length {total_steps} is insufficient for window length {total_window_len} "
            f"(history={history_len} + future={future_len})."
        )

    hist_list = []
    fut_list = []

    for t_idx in range(n_trajs):
        traj = trajectories[t_idx]  # (T+1, C, H, W)
        for start in range(0, total_steps - total_window_len + 1, stride):
            mid = start + history_len
            end = mid + future_len
            hist_list.append(traj[start:mid])
            fut_list.append(traj[mid:end])

    history = torch.stack(hist_list, dim=0)  # (B, L, C, H, W)
    future = torch.stack(fut_list, dim=0)    # (B, H, C, H, W)
    return history, future


def verify_trajectory_split_isolation(
    train_trajectories: torch.Tensor,
    val_trajectories: torch.Tensor,
    test_trajectories: Optional[torch.Tensor] = None,
    min_dist_threshold: float = 1e-3,
) -> Dict[str, float]:
    """Verify strictly zero data leakage across trajectory splits by initial states.

    Computes minimum pairwise L2 distance between t=0 initial fields across partitions.

    Args:
        train_trajectories: Tensor of shape (N_train, T+1, C, H, W)
        val_trajectories: Tensor of shape (N_val, T+1, C, H, W)
        test_trajectories: Optional tensor of shape (N_test, T+1, C, H, W)
        min_dist_threshold: Threshold below which initial states are flagged as leaked.

    Returns:
        Dict with minimum distances between split pairs.
    """
    train_init = train_trajectories[:, 0].flatten(start_dim=1)  # (N_train, D)
    val_init = val_trajectories[:, 0].flatten(start_dim=1)      # (N_val, D)

    # Val vs Train
    dist_val_train = torch.cdist(val_init, train_init, p=2)
    min_vt = float(torch.min(dist_val_train).item())
    if min_vt < min_dist_threshold:
        raise ValueError(
            f"Data leakage detected between Validation and Train initial states: "
            f"min L2 distance {min_vt:.6e} < {min_dist_threshold}."
        )

    results = {"val_vs_train_min_l2": min_vt}

    if test_trajectories is not None:
        test_init = test_trajectories[:, 0].flatten(start_dim=1)
        dist_test_train = torch.cdist(test_init, train_init, p=2)
        min_tt = float(torch.min(dist_test_train).item())
        if min_tt < min_dist_threshold:
            raise ValueError(
                f"Data leakage detected between Test and Train initial states: "
                f"min L2 distance {min_tt:.6e} < {min_dist_threshold}."
            )

        dist_test_val = torch.cdist(test_init, val_init, p=2)
        min_tv = float(torch.min(dist_test_val).item())
        if min_tv < min_dist_threshold:
            raise ValueError(
                f"Data leakage detected between Test and Validation initial states: "
                f"min L2 distance {min_tv:.6e} < {min_dist_threshold}."
            )

        results["test_vs_train_min_l2"] = min_tt
        results["test_vs_val_min_l2"] = min_tv

    return results
