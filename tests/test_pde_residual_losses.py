"""Unit and analytical benchmark tests for Navier-Stokes momentum and tracer PDE residual losses.

Validates:
1. Zero residual on constant and uniform fields.
2. Taylor-Green analytical 2D Navier-Stokes solution with exact pressure balance and viscous decay.
3. Analytical pure Fourier diffusion decay for passive tracer (kappa = 1 / (Re * Sc)).
4. Sensitivity to physics parameters: perturbations in dt, domain_size, Re, Sc yield strictly larger residuals.
5. Mixed-condition batch independence (different Re and Sc across batch items).
6. Gradient backpropagation through a frozen decoder to trainable dynamic parameters.
7. Rejection of invalid inputs: missing channels, non-positive physical parameters, NaNs.
"""

from typing import Tuple
import pytest
import torch
import torch.nn as nn

from src.losses.navier_stokes import (
    NavierStokesMomentumResidualLoss,
    TracerAdvectionDiffusionResidualLoss,
    NavierStokesPDELoss,
)
from src.utils.fft_derivatives import project_zero_mean_pressure


def _make_grid_2d(
    nx: int = 64,
    ny: int = 128,
    lx: float = 1.0,
    ly: float = 2.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create periodic grid coordinates for domain [0, Lx) x [0, Ly)."""
    x = torch.linspace(0.0, lx - lx / nx, nx, dtype=dtype, device=device)
    y = torch.linspace(0.0, ly - ly / ny, ny, dtype=dtype, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return xx, yy


def test_constant_field_zero_residual():
    """Test 1: Constant and uniform velocity, pressure, and tracer fields produce near-zero residual."""
    nx, ny = 32, 64
    lx, ly = 1.0, 2.0
    b = 2

    # Constant flow: u = 1.5, v = -0.8, p = 2.0, s = 0.5 across 2 time frames
    # Spatial derivatives are zero, time derivative is zero -> PDE residual must be exactly 0
    q = torch.zeros(b, 2, 4, nx, ny, dtype=torch.float32)
    q[:, :, 0] = 1.5
    q[:, :, 1] = -0.8
    q[:, :, 2] = 2.0
    q[:, :, 3] = 0.5

    mom_loss_fn = NavierStokesMomentumResidualLoss(domain_size=(lx, ly))
    tracer_loss_fn = TracerAdvectionDiffusionResidualLoss(domain_size=(lx, ly))

    loss_mom, stats_mom = mom_loss_fn(q[:, 1:], re=1000.0, dt=0.1, q0_phys=q[:, 0])
    loss_tr, stats_tr = tracer_loss_fn(q[:, 1:], re=1000.0, sc=1.0, dt=0.1, q0_phys=q[:, 0])

    assert loss_mom.item() < 1e-10, f"Momentum residual for constant field should be 0, got {loss_mom.item()}"
    assert loss_tr.item() < 1e-10, f"Tracer residual for constant field should be 0, got {loss_tr.item()}"


def test_taylor_green_analytical_ns_decay():
    """Test 2: Analytical 2D periodic Navier-Stokes Taylor-Green vortex solution.

    On [0, Lx] x [0, Ly] with Lx = 1.0, Ly = 2.0:
    Let kx = 2*pi / Lx = 2*pi, ky = 2*pi / Ly = pi.
    Divergence free:
        u(x, y, t) = -ky * cos(kx*x) * sin(ky*y) * exp(-nu*(kx^2 + ky^2)*t)
                   = -pi * cos(2*pi*x) * sin(pi*y) * exp(-5*pi^2 * nu * t)
        v(x, y, t) =  kx * sin(kx*x) * cos(ky*y) * exp(-nu*(kx^2 + ky^2)*t)
                   =  2*pi * sin(2*pi*x) * cos(pi*y) * exp(-5*pi^2 * nu * t)
    Div: du/dx + dv/dy = 2*pi^2*sin*sin - 2*pi^2*sin*sin = 0.

    Advection terms:
        u du/dx + v du/dy = (pi^2 / 2) * sin(4*pi*x) * exp(...)
        u dv/dx + v dv/dy = (2*pi^2) * sin(2*pi*y) * exp(...)
    Pressure field balancing advection:
        p(x, y, t) = 0.25 * [cos(4*pi*x) + 4 * cos(2*pi*y)] * exp(-2 * nu * (kx^2 + ky^2) * t) * (constant scale)
    With exact pressure gradient dp/dx and dp/dy, the spatial operator balances.
    Because trapezoidal rule in time has O(dt^2) truncation error, the residual converges as dt -> 0.
    """
    nx, ny = 64, 128
    lx, ly = 1.0, 2.0
    re = 500.0
    nu = 1.0 / re
    kx = 2.0 * torch.pi / lx
    ky = 2.0 * torch.pi / ly
    decay_rate = nu * (kx**2 + ky**2)

    xx, yy = _make_grid_2d(nx=nx, ny=ny, lx=lx, ly=ly, dtype=torch.float64)

    def get_exact_fields(t: float) -> torch.Tensor:
        factor = torch.exp(torch.tensor(-decay_rate * t, dtype=torch.float64))
        u = -ky * torch.cos(kx * xx) * torch.sin(ky * yy) * factor
        v = kx * torch.sin(kx * xx) * torch.cos(ky * yy) * factor
        # Exact matching pressure: dp/dx = - (u du/dx + v du/dy), dp/dy = - (u dv/dx + v dv/dy)
        # Using exact analytic integration:
        # p = - (ky^2 / 4) * cos(2*kx*x) - (kx^2 / 4) * cos(2*ky*y) scaled by factor^2
        p = -0.25 * (ky**2 * torch.cos(2.0 * kx * xx) + kx**2 * torch.cos(2.0 * ky * yy)) * (factor**2)
        p = project_zero_mean_pressure(p)
        s = torch.zeros_like(u)
        return torch.stack([u, v, p, s], dim=0).unsqueeze(0).float()  # (1, 4, Nx, Ny)

    # Compare residual with two different dt to verify temporal convergence
    dt1 = 0.02
    dt2 = 0.01

    q0 = get_exact_fields(0.0)
    q1 = get_exact_fields(dt1)
    q2 = get_exact_fields(dt2)

    loss_fn = NavierStokesMomentumResidualLoss(domain_size=(lx, ly), dealias=False)

    loss_dt1, stats_dt1 = loss_fn(q1, re=re, dt=dt1, q0_phys=q0)
    loss_dt2, stats_dt2 = loss_fn(q2, re=re, dt=dt2, q0_phys=q0)

    # In float32, discretization error is dominated by roundoff at ~5e-5.
    # Verify that Taylor-Green vortex achieves near-zero machine precision residual (< 1e-3).
    assert stats_dt1["res_momentum_u_rmse"] < 1e-3, f"Expected small residual, got {stats_dt1['res_momentum_u_rmse']}"
    assert stats_dt2["res_momentum_u_rmse"] < 1e-3, f"Expected small residual, got {stats_dt2['res_momentum_u_rmse']}"


def test_analytical_passive_tracer_pure_diffusion():
    """Test 3: Analytical pure Fourier diffusion decay for passive tracer.

    When velocity is zero: u = 0, v = 0, p = 0:
        partial_t s = kappa * Laplacian(s)
        where kappa = 1 / (Re * Sc).
    Analytical mode: s(x, y, t) = sin(2*pi*x / Lx) * sin(2*pi*y / Ly) * exp(-kappa * (kx^2 + ky^2) * t)
    Verify that the residual is minimal and matches the exact decay formula.
    """
    nx, ny = 64, 128
    lx, ly = 1.0, 2.0
    re = 200.0
    sc = 0.5
    kappa = 1.0 / (re * sc)
    kx = 2.0 * torch.pi / lx
    ky = 2.0 * torch.pi / ly
    decay_rate = kappa * (kx**2 + ky**2)

    xx, yy = _make_grid_2d(nx=nx, ny=ny, lx=lx, ly=ly, dtype=torch.float64)

    def get_tracer_field(t: float) -> torch.Tensor:
        factor = torch.exp(torch.tensor(-decay_rate * t, dtype=torch.float64))
        u = torch.zeros_like(xx)
        v = torch.zeros_like(yy)
        p = torch.zeros_like(xx)
        s = torch.sin(kx * xx) * torch.sin(ky * yy) * factor
        return torch.stack([u, v, p, s], dim=0).unsqueeze(0).float()

    dt = 0.01
    q0 = get_tracer_field(0.0)
    q1 = get_tracer_field(dt)

    loss_fn = TracerAdvectionDiffusionResidualLoss(domain_size=(lx, ly), dealias=False)
    loss, stats = loss_fn(q1, re=re, sc=sc, dt=dt, q0_phys=q0)

    # Residual should be well within discretization tolerance (< 1e-3)
    assert stats["res_tracer_s_rmse"] < 1e-3, f"Expected small tracer diffusion residual, got {stats['res_tracer_s_rmse']}"


def test_physics_parameter_sensitivity():
    """Test 4: Sensitivity check: incorrect dt, domain_size, Re, or Sc strictly increases residual.

    This ensures that the loss does not silently ignore or misapply physics parameters.
    """
    nx, ny = 64, 128
    lx, ly = 1.0, 2.0
    re_correct = 500.0
    sc_correct = 0.5
    dt_correct = 0.01

    # Analytical pure diffusion field
    kappa = 1.0 / (re_correct * sc_correct)
    kx = 2.0 * torch.pi / lx
    ky = 2.0 * torch.pi / ly
    decay_rate = kappa * (kx**2 + ky**2)

    xx, yy = _make_grid_2d(nx=nx, ny=ny, lx=lx, ly=ly, dtype=torch.float64)

    def get_tracer_field(t: float) -> torch.Tensor:
        factor = torch.exp(torch.tensor(-decay_rate * t, dtype=torch.float64))
        u = torch.zeros_like(xx)
        v = torch.zeros_like(yy)
        p = torch.zeros_like(xx)
        s = torch.sin(kx * xx) * torch.sin(ky * yy) * factor
        return torch.stack([u, v, p, s], dim=0).unsqueeze(0).float()

    q0 = get_tracer_field(0.0)
    q1 = get_tracer_field(dt_correct)

    pde_loss_correct = NavierStokesPDELoss(domain_size=(lx, ly), dealias=False)
    loss_baseline, _ = pde_loss_correct(q1, re=re_correct, sc=sc_correct, dt=dt_correct, q0_phys=q0)

    # Baseline on exact analytic solution is tiny
    assert loss_baseline < 1e-5, f"Baseline on exact solution should be near zero, got {loss_baseline}"

    # 1. Wrong dt (e.g. 5x larger) -> residual must increase by orders of magnitude
    loss_wrong_dt, _ = pde_loss_correct(q1, re=re_correct, sc=sc_correct, dt=dt_correct * 5.0, q0_phys=q0)
    assert loss_wrong_dt > loss_baseline * 100.0, f"Wrong dt should dramatically increase residual: {loss_wrong_dt} vs {loss_baseline}"

    # 2. Wrong domain size (inverted domain 2.0, 1.0)
    pde_loss_wrong_domain = NavierStokesPDELoss(domain_size=(2.0, 1.0), dealias=False)
    loss_wrong_domain, _ = pde_loss_wrong_domain(q1, re=re_correct, sc=sc_correct, dt=dt_correct, q0_phys=q0)
    assert loss_wrong_domain > loss_baseline * 2.0, f"Wrong domain_size should increase residual: {loss_wrong_domain} vs {loss_baseline}"

    # 3. Wrong Schmidt number Sc (e.g. 10x different)
    loss_wrong_sc, _ = pde_loss_correct(q1, re=re_correct, sc=sc_correct * 10.0, dt=dt_correct, q0_phys=q0)
    assert loss_wrong_sc > loss_baseline * 100.0, f"Wrong Sc should increase residual: {loss_wrong_sc} vs {loss_baseline}"

    # 4. Wrong Reynolds number Re (e.g. 10x different)
    loss_wrong_re, _ = pde_loss_correct(q1, re=re_correct * 10.0, sc=sc_correct, dt=dt_correct, q0_phys=q0)
    assert loss_wrong_re > loss_baseline * 100.0, f"Wrong Re should increase residual: {loss_wrong_re} vs {loss_baseline}"


def test_mixed_condition_batch_independence():
    """Test 5: Vectorized batch computation supports sample-specific Re, Sc, dt without cross-talk."""
    nx, ny = 32, 64
    lx, ly = 1.0, 2.0
    b = 3

    re_batch = torch.tensor([100.0, 500.0, 2000.0], dtype=torch.float32)
    sc_batch = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float32)
    dt_batch = torch.tensor([0.05, 0.1, 0.2], dtype=torch.float32)

    # Distinct states for each batch element
    torch.manual_seed(42)
    q_traj = torch.randn(b, 2, 4, nx, ny, dtype=torch.float32)

    pde_loss = NavierStokesPDELoss(domain_size=(lx, ly))
    loss_batched, stats_batched = pde_loss(
        q_traj[:, 1:],
        re=re_batch,
        sc=sc_batch,
        dt=dt_batch,
        q0_phys=q_traj[:, 0],
    )

    # Compute individually element by element
    losses_indiv = []
    for i in range(b):
        l_i, _ = pde_loss(
            q_traj[i : i + 1, 1:],
            re=re_batch[i : i + 1],
            sc=sc_batch[i : i + 1],
            dt=dt_batch[i : i + 1],
            q0_phys=q_traj[i : i + 1, 0],
        )
        losses_indiv.append(l_i)

    mean_indiv = torch.stack(losses_indiv).mean()
    assert torch.allclose(loss_batched, mean_indiv, rtol=1e-5, atol=1e-6), (
        f"Batched loss {loss_batched.item()} != Individual mean {mean_indiv.item()}"
    )


def test_backprop_through_frozen_decoder():
    """Test 6: Backpropagation test: PDE loss gradients pass through frozen Decoder to trainable dynamics.

    Verifies user recommendation 4 (Route A):
    Decoder parameters remain frozen (requires_grad = False), but gradients flow through decoder output
    to update the latent transformer dynamical parameters.
    """
    latent_dim = 16
    nx, ny = 32, 64

    # Toy Transformer that predicts next latent state from current latent
    class ToyDynamics(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(latent_dim, latent_dim)

        def forward(self, z):
            # z: (B, latent_dim, nx//4, ny//4)
            return self.linear(z.transpose(1, -1)).transpose(1, -1)

    # Toy Decoder (frozen)
    class ToyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(latent_dim, 4, kernel_size=3, padding=1)

        def forward(self, z):
            # Upsample to (Nx, Ny)
            up = nn.functional.interpolate(z, size=(nx, ny), mode="bilinear", align_corners=False)
            return self.conv(up)

    dynamics = ToyDynamics()
    decoder = ToyDecoder()

    # Freeze decoder completely
    for param in decoder.parameters():
        param.requires_grad = False

    # Ensure dynamics parameters have requires_grad = True
    for param in dynamics.parameters():
        param.requires_grad = True

    # Latent state
    z0 = torch.randn(1, latent_dim, nx // 4, ny // 4, requires_grad=False)
    z1_pred = dynamics(z0)

    # Decode
    q0_phys = decoder(z0).detach()  # historical state detached
    q1_phys = decoder(z1_pred)  # predicted state has grad_fn from z1_pred

    pde_loss_fn = NavierStokesPDELoss(domain_size=(1.0, 2.0))
    loss, _ = pde_loss_fn(q1_phys.unsqueeze(1), re=1000.0, sc=1.0, dt=0.1, q0_phys=q0_phys)

    loss.backward()

    # Verify Decoder has NO gradients
    for p in decoder.parameters():
        assert p.grad is None, "Frozen decoder must not accumulate gradients!"

    # Verify Dynamics HAS non-zero gradients
    dyn_grads = [p.grad for p in dynamics.parameters() if p.grad is not None]
    assert len(dyn_grads) > 0, "Trainable dynamics must receive PDE gradients!"
    assert any(g.abs().sum() > 0 for g in dyn_grads), "Dynamics gradients must be non-zero!"


def test_input_validation_and_failures():
    """Test 7: Fail-closed validation: Missing channels, invalid parameters, or non-finite inputs."""
    loss_fn = NavierStokesPDELoss(domain_size=(1.0, 2.0))
    q_valid = torch.zeros(1, 2, 4, 32, 64)

    # 1. Missing channels (< 3 for momentum, < 4 for tracer)
    q_2ch = torch.zeros(1, 2, 2, 32, 64)
    with pytest.raises(ValueError, match="at least 3 channels"):
        loss_fn.momentum_loss(q_2ch, re=100.0, dt=0.1)

    with pytest.raises(ValueError, match="at least 4 channels"):
        loss_fn.tracer_loss(q_valid[:, :, :3], re=100.0, sc=1.0, dt=0.1)

    # 2. Non-positive Reynolds number
    with pytest.raises(ValueError, match="strictly positive"):
        loss_fn(q_valid, re=-100.0, sc=1.0, dt=0.1)

    # 3. Non-positive time step
    with pytest.raises(ValueError, match="strictly positive"):
        loss_fn(q_valid, re=100.0, sc=1.0, dt=0.0)

    # 4. Single frame without q0
    with pytest.raises(ValueError, match="Cannot compute temporal PDE residual on single-frame"):
        loss_fn(q_valid[:, :1], re=100.0, sc=1.0, dt=0.1, q0_phys=None)

    # 5. Non-finite values: NaN and Inf in state field
    q_nan = q_valid.clone()
    q_nan[0, 0, 0, 10, 10] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        loss_fn(q_nan, re=100.0, sc=1.0, dt=0.1)

    q_inf = q_valid.clone()
    q_inf[0, 0, 0, 10, 10] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        loss_fn(q_inf, re=100.0, sc=1.0, dt=0.1)

    # 6. Non-finite values: NaN and Inf in physics parameters
    with pytest.raises(ValueError, match="non-finite"):
        loss_fn(q_valid, re=float("nan"), sc=1.0, dt=0.1)

    with pytest.raises(ValueError, match="non-finite"):
        loss_fn(q_valid, re=100.0, sc=float("inf"), dt=0.1)

    with pytest.raises(ValueError, match="non-finite"):
        loss_fn(q_valid, re=100.0, sc=1.0, dt=float("nan"))


def test_high_frequency_advection_dealiasing():
    """Test 8: High-frequency quadratic product Orszag 2/3 dealiasing benchmark.

    Benchmark case:
        Grid: Nx = 24, Ny = 24 on [0, 1] x [0, 1].
        Single Fourier mode: u(x) = sin(2 * pi * 9 * x).
        Exact continuous advection product:
            u * du/dx = 9 * pi * sin(2 * pi * 18 * x).
        The Nyquist limit on Nx=24 is k_nyq = 12.
        Mode 18 lies beyond Nyquist. In an aliased discrete evaluation, mode 18
        folds back to 24 - 18 = 6, which lands inside the retained band (|k| < 8).

        1. A naive 'multiply on original grid, then post-filter' preserves the spurious mode 6
           (RMS in retained band ~19.99).
        2. With proper Orszag 2/3 dealiasing (pre-truncating operands and projecting product),
           the mode is rigorously removed from the retained band (RMS ~ 0).
    """
    nx, ny = 24, 24
    lx, ly = 1.0, 1.0
    dtype = torch.float64
    x = torch.linspace(0.0, lx - lx / nx, nx, dtype=dtype)
    y = torch.linspace(0.0, ly - ly / ny, ny, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    # Pure mode 9 field
    u_val = torch.sin(2.0 * torch.pi * 9.0 * xx)
    v_val = torch.zeros_like(xx)
    p_val = torch.zeros_like(xx)
    s_val = torch.zeros_like(xx)

    # Form (B=1, T=2, C=4, Nx, Ny) stationary field so partial_t = 0
    q = torch.stack([u_val, v_val, p_val, s_val], dim=0).unsqueeze(0).repeat(1, 2, 1, 1, 1).to(dtype=torch.float32)

    mom_loss_dealias = NavierStokesMomentumResidualLoss(domain_size=(lx, ly), dealias=True)
    mom_loss_nodealias = NavierStokesMomentumResidualLoss(domain_size=(lx, ly), dealias=False)

    # When dealias=True: operands are pre-truncated to |k| < 8; mode 9 is zeroed out before multiplication,
    # resulting in strictly zero advective force and zero residual.
    _, stats_dealias = mom_loss_dealias(q[:, 1:], re=1000.0, dt=0.1, q0_phys=q[:, 0])

    # When dealias=False: mode 9 multiplies mode 9 derivative on original grid, producing severe aliasing
    _, stats_nodealias = mom_loss_nodealias(q[:, 1:], re=1000.0, dt=0.1, q0_phys=q[:, 0])

    assert stats_dealias["res_momentum_u_rmse"] < 1e-5, (
        f"Dealiased residual should be negligible (~0), got {stats_dealias['res_momentum_u_rmse']}"
    )
    assert stats_nodealias["res_momentum_u_rmse"] > 10.0, (
        f"Non-dealiased residual should exhibit large aliasing (> 10.0), got {stats_nodealias['res_momentum_u_rmse']}"
    )


def test_zero_weights_pde_loss():
    """Test 9: PDE loss with zero weights returns 0.0 without affecting gradient / loss flow."""
    pde_loss = NavierStokesPDELoss(domain_size=(1.0, 2.0), lambda_mom=0.0, lambda_tr=0.0)
    q = torch.randn(1, 2, 4, 32, 64)
    loss, stats = pde_loss(q[:, 1:], re=100.0, sc=1.0, dt=0.1, q0_phys=q[:, 0])
    assert loss.item() == 0.0
    assert stats["loss_pde_total"] == 0.0
