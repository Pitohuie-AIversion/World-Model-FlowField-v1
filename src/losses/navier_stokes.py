"""Navier-Stokes and Passive Tracer PDE residual loss modules for 2D incompressible shear flow.

Domain specifications for The Well shear_flow:
    Physical domain: [0, Lx] x [0, Ly] with Lx = 1.0, Ly = 2.0.
    Tensor spatial layout: (..., Nx, Ny) where dim -2 is x (Lx=1.0) and dim -1 is y (Ly=2.0).
    Governing Equations:
        Momentum:
            partial_t u + (u partial_x u + v partial_y u) + partial_x p - nu * Laplacian(u) = 0
            partial_t v + (u partial_x v + v partial_y v) + partial_y p - nu * Laplacian(v) = 0
            where nu = 1 / Re.
        Tracer Transport (passive scalar):
            partial_t s + (u partial_x s + v partial_y s) - kappa * Laplacian(s) = 0
            where kappa = nu / Sc = 1 / (Re * Sc).
        Incompressibility:
            div(u) = partial_x u + partial_y v = 0.

Time discretization:
    Trapezoidal / Crank-Nicolson interval residual across adjacent states [q_n, q_{n+1}]:
        A_u(q) = u partial_x u + v partial_y u + partial_x p - nu * Laplacian(u)
        r_u = (u_{n+1} - u_n) / dt + 0.5 * (A_u(q_{n+1}) + A_u(q_n))
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.utils.fft_derivatives import (
    compute_laplacian_2d,
    dealias_field_2d,
    project_zero_mean_pressure,
    spectral_grad_xy,
)


def _ensure_trajectory_dim(
    pred: torch.Tensor,
    q0: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    """Ensure tensor has sequence dimension (B, T, C, Nx, Ny).

    If q0 is provided, prepends q0 to form a sequence of length H + 1,
    yielding H transition intervals.
    """
    if pred.ndim == 4:
        # (B, C, Nx, Ny) -> (B, 1, C, Nx, Ny)
        pred = pred.unsqueeze(1)

    if pred.ndim != 5:
        raise ValueError(f"Expected pred tensor with 4 or 5 dims, got shape {pred.shape}")

    if q0 is not None:
        if q0.ndim == 4:
            q0 = q0.unsqueeze(1)
        if q0.ndim != 5 or q0.shape[1] != 1:
            raise ValueError(f"Expected q0 tensor with shape (B, C, Nx, Ny) or (B, 1, C, Nx, Ny), got {q0.shape}")
        full_traj = torch.cat([q0, pred], dim=1)
        num_intervals = pred.shape[1]
    else:
        full_traj = pred
        num_intervals = pred.shape[1] - 1
        if num_intervals < 1:
            raise ValueError(
                "Cannot compute temporal PDE residual on single-frame prediction without q0. "
                "Provide historical state q0 or a multi-step rollout trajectory."
            )

    return full_traj, num_intervals


def _broadcast_physics_param(param: Union[float, torch.Tensor], target_shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Broadcast scalar or 1D batch physics parameter (Re, Sc, dt) to field dimensions (B, 1, 1, 1)."""
    if isinstance(param, (int, float)):
        return torch.tensor(float(param), device=device, dtype=torch.float32).view(*([1] * len(target_shape)))
    elif isinstance(param, torch.Tensor):
        param = param.to(device=device, dtype=torch.float32)
        if param.ndim == 0:
            return param.view(*([1] * len(target_shape)))
        elif param.ndim == 1:
            # (B,) -> (B, 1, 1, 1) or (B, 1, 1, 1, 1)
            b = param.shape[0]
            if b != target_shape[0]:
                raise ValueError(f"Batch dimension mismatch: param has {b}, target has {target_shape[0]}")
            view_shape = [b] + [1] * (len(target_shape) - 1)
            return param.view(*view_shape)
        else:
            return param
    else:
        raise TypeError(f"Expected float or torch.Tensor for physics parameter, got {type(param)}")


class NavierStokesMomentumResidualLoss(nn.Module):
    """Computes mean-squared Navier-Stokes momentum PDE residual loss.

    r_u = partial_t u + (u partial_x u + v partial_y u) + partial_x p - nu * Laplacian(u)
    r_v = partial_t v + (u partial_x v + v partial_y v) + partial_y p - nu * Laplacian(v)

    Discretized via trapezoidal rule on each time interval:
        r_{u, n+1/2} = (u_{n+1} - u_n) / dt + 0.5 * (A_u(q_{n+1}) + A_u(q_n))
        r_{v, n+1/2} = (v_{n+1} - v_n) / dt + 0.5 * (A_v(q_{n+1}) + A_v(q_n))

    Args:
        domain_size: (Lx, Ly) physical domain size. Defaults to (1.0, 2.0).
        scale_u: Normalization scale a_u for u residual.
        scale_v: Normalization scale a_v for v residual.
        dealias: If True, applies Orszag 2/3 dealiasing to non-linear advection terms.
    """

    def __init__(
        self,
        domain_size: Tuple[float, float] = (1.0, 2.0),
        scale_u: float = 1.0,
        scale_v: float = 1.0,
        dealias: bool = True,
    ):
        super().__init__()
        self.domain_size = domain_size
        self.scale_u = float(scale_u)
        self.scale_v = float(scale_v)
        self.dealias = bool(dealias)

        if self.scale_u <= 0 or self.scale_v <= 0:
            raise ValueError(f"Residual scales must be positive, got scale_u={scale_u}, scale_v={scale_v}")

    def compute_spatial_operator(
        self,
        q: torch.Tensor,
        nu: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial momentum operator A(q) = (u . grad) u + grad(p) - nu * Laplacian(u).

        Args:
            q: Physical fields tensor of shape (..., C, Nx, Ny) with C >= 3 (u, v, p).
            nu: Kinematic viscosity tensor (1 / Re), broadcastable to velocity field.

        Returns:
            A_u: Spatial operator for u momentum, shape (..., Nx, Ny).
            A_v: Spatial operator for v momentum, shape (..., Nx, Ny).
        """
        if q.shape[-3] < 3:
            raise ValueError(f"Momentum residual requires at least 3 channels (u, v, p), got {q.shape[-3]}")

        u = q[..., 0, :, :]
        v = q[..., 1, :, :]
        p = q[..., 2, :, :]

        p_gauge = project_zero_mean_pressure(p)

        # Gradients
        du_dx, du_dy = spectral_grad_xy(u, domain_size=self.domain_size)
        dv_dx, dv_dy = spectral_grad_xy(v, domain_size=self.domain_size)
        dp_dx, dp_dy = spectral_grad_xy(p_gauge, domain_size=self.domain_size)

        # Laplacians
        lap_u = compute_laplacian_2d(u, domain_size=self.domain_size)
        lap_v = compute_laplacian_2d(v, domain_size=self.domain_size)

        # Advective non-linear acceleration: (u * du/dx + v * du/dy)
        advect_u = u * du_dx + v * du_dy
        advect_v = u * dv_dx + v * dv_dy

        if self.dealias:
            advect_u = dealias_field_2d(advect_u)
            advect_v = dealias_field_2d(advect_v)

        A_u = advect_u + dp_dx - nu * lap_u
        A_v = advect_v + dp_dy - nu * lap_v

        return A_u, A_v

    def forward(
        self,
        pred_phys: torch.Tensor,
        re: Union[float, torch.Tensor],
        dt: Union[float, torch.Tensor] = 0.1,
        q0_phys: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute momentum residual loss and diagnostics.

        Args:
            pred_phys: Decoded physical fields, shape (B, H, C, Nx, Ny) or (B, C, Nx, Ny).
                       Channels 0, 1, 2 must correspond to u, v, p.
            re: Reynolds number, scalar or shape (B,).
            dt: Time step between consecutive frames, scalar or shape (B,).
            q0_phys: Optional initial historical physical state (B, C, Nx, Ny).

        Returns:
            loss: Scalar normalized momentum residual loss.
            stats: Dictionary of residual statistics (RMSE of r_u, r_v, total loss).
        """
        full_traj, num_intervals = _ensure_trajectory_dim(pred_phys, q0=q0_phys)

        device = full_traj.device
        b = full_traj.shape[0]

        # Broadcast Re and dt
        # full_traj shape: (B, T, C, Nx, Ny) -> broadcast to (B, 1, Nx, Ny)
        nu_tensor = _broadcast_physics_param(1.0 / re if isinstance(re, (int, float)) else 1.0 / re, (b, 1, 1, 1), device)
        dt_tensor = _broadcast_physics_param(dt, (b, 1, 1, 1), device)

        if torch.any(nu_tensor <= 0):
            raise ValueError("Reynolds number must be strictly positive (nu > 0).")
        if torch.any(dt_tensor <= 0):
            raise ValueError("Time step dt must be strictly positive.")

        # Compute spatial operator A(q) for all frames along trajectory
        # Merge B and T for batched spectral operations
        t_steps = full_traj.shape[1]
        c, nx, ny = full_traj.shape[2], full_traj.shape[3], full_traj.shape[4]
        q_flattened = full_traj.view(b * t_steps, c, nx, ny)

        nu_flattened = nu_tensor.unsqueeze(1).expand(b, t_steps, 1, 1, 1).reshape(b * t_steps, 1, 1)

        A_u_flat, A_v_flat = self.compute_spatial_operator(q_flattened, nu_flattened)
        A_u_all = A_u_flat.view(b, t_steps, nx, ny)
        A_v_all = A_v_flat.view(b, t_steps, nx, ny)

        u_all = full_traj[:, :, 0, :, :]
        v_all = full_traj[:, :, 1, :, :]

        # Interval residuals across [n, n+1]
        u_n = u_all[:, :-1]
        u_next = u_all[:, 1:]
        v_n = v_all[:, :-1]
        v_next = v_all[:, 1:]

        A_u_n = A_u_all[:, :-1]
        A_u_next = A_u_all[:, 1:]
        A_v_n = A_v_all[:, :-1]
        A_v_next = A_v_all[:, 1:]

        # Trapezoidal rule: r = (q_{n+1} - q_n) / dt + 0.5 * (A(q_{n+1}) + A(q_n))
        # Note dt_tensor has shape (B, 1, 1, 1)
        r_u = (u_next - u_n) / dt_tensor + 0.5 * (A_u_next + A_u_n)
        r_v = (v_next - v_n) / dt_tensor + 0.5 * (A_v_next + A_v_n)

        loss_u = torch.mean((r_u / self.scale_u) ** 2)
        loss_v = torch.mean((r_v / self.scale_v) ** 2)
        total_loss = loss_u + loss_v

        ru_rmse = torch.sqrt(torch.mean(r_u**2)).detach().item()
        rv_rmse = torch.sqrt(torch.mean(r_v**2)).detach().item()

        stats = {
            "loss_momentum": total_loss.detach().item(),
            "loss_momentum_u": loss_u.detach().item(),
            "loss_momentum_v": loss_v.detach().item(),
            "res_momentum_u_rmse": ru_rmse,
            "res_momentum_v_rmse": rv_rmse,
        }

        return total_loss, stats


class TracerAdvectionDiffusionResidualLoss(nn.Module):
    """Computes mean-squared passive tracer advection-diffusion PDE residual loss.

    r_s = partial_t s + (u partial_x s + v partial_y s) - kappa * Laplacian(s)
    where kappa = 1 / (Re * Sc).

    Discretized via trapezoidal rule on each time interval:
        r_{s, n+1/2} = (s_{n+1} - s_n) / dt + 0.5 * (A_s(q_{n+1}) + A_s(q_n))
        where A_s(q) = u partial_x s + v partial_y s - kappa * Laplacian(s).

    Args:
        domain_size: (Lx, Ly) physical domain size. Defaults to (1.0, 2.0).
        scale_s: Normalization scale a_s for tracer residual.
        dealias: If True, applies Orszag 2/3 dealiasing to non-linear tracer advection.
    """

    def __init__(
        self,
        domain_size: Tuple[float, float] = (1.0, 2.0),
        scale_s: float = 1.0,
        dealias: bool = True,
    ):
        super().__init__()
        self.domain_size = domain_size
        self.scale_s = float(scale_s)
        self.dealias = bool(dealias)

        if self.scale_s <= 0:
            raise ValueError(f"Residual scale_s must be positive, got {scale_s}")

    def compute_spatial_operator(
        self,
        q: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spatial tracer operator A_s(q) = (u . grad) s - kappa * Laplacian(s).

        Args:
            q: Physical fields tensor of shape (..., C, Nx, Ny) with C >= 4 (u, v, p, s).
            kappa: Tracer diffusivity tensor (1 / (Re * Sc)).

        Returns:
            A_s: Spatial operator for tracer transport, shape (..., Nx, Ny).
        """
        if q.shape[-3] < 4:
            raise ValueError(f"Tracer residual requires at least 4 channels (u, v, p, s), got {q.shape[-3]}")

        u = q[..., 0, :, :]
        v = q[..., 1, :, :]
        s = q[..., 3, :, :]

        ds_dx, ds_dy = spectral_grad_xy(s, domain_size=self.domain_size)
        lap_s = compute_laplacian_2d(s, domain_size=self.domain_size)

        advect_s = u * ds_dx + v * ds_dy
        if self.dealias:
            advect_s = dealias_field_2d(advect_s)

        A_s = advect_s - kappa * lap_s
        return A_s

    def forward(
        self,
        pred_phys: torch.Tensor,
        re: Union[float, torch.Tensor],
        sc: Union[float, torch.Tensor],
        dt: Union[float, torch.Tensor] = 0.1,
        q0_phys: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute tracer residual loss and diagnostics.

        Args:
            pred_phys: Decoded physical fields, shape (B, H, C, Nx, Ny) or (B, C, Nx, Ny).
                       Channel 3 must correspond to tracer s.
            re: Reynolds number, scalar or shape (B,).
            sc: Schmidt number, scalar or shape (B,).
            dt: Time step between consecutive frames, scalar or shape (B,).
            q0_phys: Optional initial historical physical state (B, C, Nx, Ny).

        Returns:
            loss: Scalar normalized tracer residual loss.
            stats: Dictionary of residual statistics (RMSE of r_s, total loss).
        """
        full_traj, num_intervals = _ensure_trajectory_dim(pred_phys, q0=q0_phys)

        device = full_traj.device
        b = full_traj.shape[0]

        # kappa = 1 / (Re * Sc)
        re_tensor = _broadcast_physics_param(re, (b, 1, 1, 1), device)
        sc_tensor = _broadcast_physics_param(sc, (b, 1, 1, 1), device)
        dt_tensor = _broadcast_physics_param(dt, (b, 1, 1, 1), device)

        if torch.any(re_tensor <= 0) or torch.any(sc_tensor <= 0):
            raise ValueError("Reynolds and Schmidt numbers must be strictly positive.")
        if torch.any(dt_tensor <= 0):
            raise ValueError("Time step dt must be strictly positive.")

        kappa_tensor = 1.0 / (re_tensor * sc_tensor)

        t_steps = full_traj.shape[1]
        c, nx, ny = full_traj.shape[2], full_traj.shape[3], full_traj.shape[4]
        q_flattened = full_traj.view(b * t_steps, c, nx, ny)

        kappa_flattened = kappa_tensor.unsqueeze(1).expand(b, t_steps, 1, 1, 1).reshape(b * t_steps, 1, 1)

        A_s_flat = self.compute_spatial_operator(q_flattened, kappa_flattened)
        A_s_all = A_s_flat.view(b, t_steps, nx, ny)

        s_all = full_traj[:, :, 3, :, :]

        s_n = s_all[:, :-1]
        s_next = s_all[:, 1:]
        A_s_n = A_s_all[:, :-1]
        A_s_next = A_s_all[:, 1:]

        # Trapezoidal rule
        r_s = (s_next - s_n) / dt_tensor + 0.5 * (A_s_next + A_s_n)

        loss_s = torch.mean((r_s / self.scale_s) ** 2)
        rs_rmse = torch.sqrt(torch.mean(r_s**2)).detach().item()

        stats = {
            "loss_tracer": loss_s.detach().item(),
            "res_tracer_s_rmse": rs_rmse,
        }

        return loss_s, stats


class NavierStokesPDELoss(nn.Module):
    """Composite Navier-Stokes momentum and passive tracer PDE residual loss module.

    L_PDE = lambda_mom * L_momentum + lambda_tr * L_tracer

    Args:
        domain_size: (Lx, Ly) physical domain size.
        lambda_mom: Weight for Navier-Stokes momentum residual.
        lambda_tr: Weight for tracer advection-diffusion residual.
        scale_u: Normalization scale for u momentum residual.
        scale_v: Normalization scale for v momentum residual.
        scale_s: Normalization scale for tracer transport residual.
        dealias: If True, applies 2/3 Orszag dealiasing to non-linear advection.
    """

    def __init__(
        self,
        domain_size: Tuple[float, float] = (1.0, 2.0),
        lambda_mom: float = 1.0,
        lambda_tr: float = 1.0,
        scale_u: float = 1.0,
        scale_v: float = 1.0,
        scale_s: float = 1.0,
        dealias: bool = True,
    ):
        super().__init__()
        self.domain_size = domain_size
        self.lambda_mom = float(lambda_mom)
        self.lambda_tr = float(lambda_tr)

        self.momentum_loss = NavierStokesMomentumResidualLoss(
            domain_size=domain_size,
            scale_u=scale_u,
            scale_v=scale_v,
            dealias=dealias,
        )
        self.tracer_loss = TracerAdvectionDiffusionResidualLoss(
            domain_size=domain_size,
            scale_s=scale_s,
            dealias=dealias,
        )

    def forward(
        self,
        pred_phys: torch.Tensor,
        re: Union[float, torch.Tensor],
        sc: Union[float, torch.Tensor],
        dt: Union[float, torch.Tensor] = 0.1,
        q0_phys: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined PDE residual loss.

        Args:
            pred_phys: Decoded physical fields (B, H, C, Nx, Ny) or (B, C, Nx, Ny).
            re: Reynolds number.
            sc: Schmidt number.
            dt: Time step.
            q0_phys: Optional preceding physical state.

        Returns:
            loss: Combined scalar loss.
            stats: Dictionary containing diagnostic statistics for all residual components.
        """
        device = pred_phys.device
        total_loss = torch.tensor(0.0, device=device, dtype=pred_phys.dtype)
        all_stats: Dict[str, float] = {}

        if self.lambda_mom > 0:
            loss_mom, stats_mom = self.momentum_loss(pred_phys, re=re, dt=dt, q0_phys=q0_phys)
            total_loss = total_loss + self.lambda_mom * loss_mom
            all_stats.update(stats_mom)

        if self.lambda_tr > 0:
            loss_tr, stats_tr = self.tracer_loss(pred_phys, re=re, sc=sc, dt=dt, q0_phys=q0_phys)
            total_loss = total_loss + self.lambda_tr * loss_tr
            all_stats.update(stats_tr)

        all_stats["loss_pde_total"] = total_loss.detach().item()
        return total_loss, all_stats
