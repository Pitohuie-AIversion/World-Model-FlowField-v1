"""Unit and regression tests for synthetic periodic scalar advection-diffusion dynamics."""

import math
import pytest
import torch

from src.data.synthetic_advection_diffusion import (
    AdvectionDiffusionConfig,
    PeriodicScalarAdvectionDiffusion,
    build_trajectory_windows,
    generate_trajectory_dataset,
    verify_trajectory_split_isolation,
)


def test_t0_identity_float32_and_float64():
    """Verify that at t=0, analytical propagation yields the exact initial state."""
    # float32 check
    cfg = AdvectionDiffusionConfig(nx=32, ny=32, u0=0.5, v0=0.5, nu=0.01)
    solver = PeriodicScalarAdvectionDiffusion(cfg=cfg)
    field_0_f32, phases_f32 = solver.generate_initial_condition(seed=42, dtype=torch.float32)

    step_alg_f32 = solver.step_algebraic(t=0.0, phases=phases_f32, dtype=torch.float32)
    step_spec_f32 = solver.step_spectral(field_0=field_0_f32, t=0.0)

    assert torch.allclose(step_alg_f32, field_0_f32, atol=1e-6, rtol=1e-6)
    assert torch.allclose(step_spec_f32, field_0_f32, atol=1e-6, rtol=1e-6)

    # float64 check
    field_0_f64, phases_f64 = solver.generate_initial_condition(seed=42, dtype=torch.float64)
    step_alg_f64 = solver.step_algebraic(t=0.0, phases=phases_f64, dtype=torch.float64)
    step_spec_f64 = solver.step_spectral(field_0=field_0_f64, t=0.0)

    assert torch.allclose(step_alg_f64, field_0_f64, atol=1e-12, rtol=1e-12)
    assert torch.allclose(step_spec_f64, field_0_f64, atol=1e-12, rtol=1e-12)


def test_inviscid_advection_conservation():
    """Verify that with nu=0 (inviscid transport), spatial enstrophy is strictly conserved over time."""
    cfg = AdvectionDiffusionConfig(nx=32, ny=32, u0=1.0, v0=-0.5, nu=0.0)
    solver = PeriodicScalarAdvectionDiffusion(cfg=cfg)

    traj = solver.generate_trajectory(seed=77, num_steps=10, dt=0.1, dtype=torch.float64)
    assert traj.shape == (11, 1, 32, 32)

    z_series = [solver.compute_enstrophy(traj[k]).item() for k in range(11)]
    z_0 = z_series[0]

    for k, z_k in enumerate(z_series):
        rel_change = abs(z_k - z_0) / z_0
        assert rel_change < 1e-10, f"Step {k}: Enstrophy changed by {rel_change:.3e} in inviscid transport!"


def test_zero_velocity_pure_diffusion():
    """Verify that with u0=0, v0=0, nu>0, enstrophy strictly decays monotonically with zero spatial drift."""
    # 1. Monotonic decay on perturbed field
    cfg = AdvectionDiffusionConfig(nx=32, ny=32, u0=0.0, v0=0.0, nu=0.05)
    solver = PeriodicScalarAdvectionDiffusion(cfg=cfg)

    traj = solver.generate_trajectory(seed=123, num_steps=8, dt=0.05, dtype=torch.float64)
    z_series = [solver.compute_enstrophy(traj[k]).item() for k in range(9)]

    for k in range(len(z_series) - 1):
        assert z_series[k + 1] < z_series[k], f"Step {k}: Enstrophy failed to decay monotonically under diffusion!"

    # 2. Location of global maximum strictly preserved when perturbation is zero
    cfg_unperturbed = AdvectionDiffusionConfig(nx=32, ny=32, u0=0.0, v0=0.0, nu=0.05, perturbation_amplitude=0.0)
    solver_unpert = PeriodicScalarAdvectionDiffusion(cfg=cfg_unperturbed)
    traj_unpert = solver_unpert.generate_trajectory(seed=42, num_steps=6, dt=0.05, dtype=torch.float64)

    argmax_0 = torch.argmax(traj_unpert[0, 0]).item()
    for k in range(1, 7):
        argmax_k = torch.argmax(traj_unpert[k, 0]).item()
        assert argmax_k == argmax_0, f"Step {k}: Peak drifted without advection velocity!"


def test_single_fourier_mode_decay_rate():
    """Verify that a single Fourier mode decays at exact theoretical rate exp(-nu * |k|^2 * t)."""
    lx, ly = 1.0, 1.0
    nx, ny = 32, 32
    nu = 0.02
    t = 0.5

    cfg = AdvectionDiffusionConfig(
        nx=nx, ny=ny, lx=lx, ly=ly, u0=0.0, v0=0.0, nu=nu,
        base_wavenumber=1, perturbation_amplitude=0.0,
    )
    solver = PeriodicScalarAdvectionDiffusion(cfg=cfg)

    # Initial condition has base quadrupole: 2*cos(2*pi*x)*cos(2*pi*y)
    # kx = 2*pi, ky = 2*pi -> |k|^2 = (2*pi)^2 + (2*pi)^2 = 8*pi^2
    k_sq = 8.0 * (math.pi**2)
    expected_decay = math.exp(-nu * k_sq * t)

    f0, phases = solver.generate_initial_condition(seed=1, dtype=torch.float64)
    ft_alg = solver.step_algebraic(t=t, phases=phases, dtype=torch.float64)
    ft_spec = solver.step_spectral(field_0=f0, t=t)

    # In L2 norm, decay factor is exactly expected_decay
    decay_alg = float((ft_alg.norm() / f0.norm()).item())
    decay_spec = float((ft_spec.norm() / f0.norm()).item())

    assert abs(decay_alg - expected_decay) < 1e-12
    assert abs(decay_spec - expected_decay) < 1e-12

    # Algebraic and spectral solutions agree to machine precision
    diff_alg_spec = float((ft_alg - ft_spec).abs().max().item())
    assert diff_alg_spec < 1e-12


def test_semigroup_consistency():
    """Verify that time stepping satisfies the semigroup identity: S(t1 + t2) = S(t2) o S(t1)."""
    cfg = AdvectionDiffusionConfig(nx=32, ny=32, u0=0.7, v0=0.3, nu=0.005)
    solver = PeriodicScalarAdvectionDiffusion(cfg=cfg)

    f0, _ = solver.generate_initial_condition(seed=999, dtype=torch.float64)
    t1, t2 = 0.15, 0.25

    # Direct propagation S(t1 + t2)
    f_direct = solver.step_spectral(f0, t=t1 + t2)

    # Two-step propagation S(t2)(S(t1))
    f_step1 = solver.step_spectral(f0, t=t1)
    f_step2 = solver.step_spectral(f_step1, t=t2)

    assert torch.allclose(f_direct, f_step2, atol=1e-9, rtol=1e-9)


def test_trajectory_isolation_and_no_leakage():
    """Verify trajectory dataset synthesis enforces strict initial condition identity partitioning."""
    cfg = AdvectionDiffusionConfig(nx=32, ny=32)

    train_trajs, train_seeds = generate_trajectory_dataset(
        num_trajectories=4, num_steps=2, dt=0.05, seed_base=1000, cfg=cfg,
    )
    val_trajs, val_seeds = generate_trajectory_dataset(
        num_trajectories=2, num_steps=2, dt=0.05, seed_base=2000, cfg=cfg,
    )
    test_trajs, test_seeds = generate_trajectory_dataset(
        num_trajectories=2, num_steps=2, dt=0.05, seed_base=3000, cfg=cfg,
    )

    # Seeds must be completely disjoint
    assert set(train_seeds).isdisjoint(set(val_seeds))
    assert set(train_seeds).isdisjoint(set(test_seeds))
    assert set(val_seeds).isdisjoint(set(test_seeds))

    # Clean verification passes
    iso = verify_trajectory_split_isolation(train_trajs, val_trajs, test_trajs)
    assert iso["val_vs_train_min_l2"] > 1e-3
    assert iso["test_vs_train_min_l2"] > 1e-3
    assert iso["test_vs_val_min_l2"] > 1e-3

    # Deliberate contamination must be detected and rejected
    leaked_val = val_trajs.clone()
    leaked_val[0] = train_trajs[0]
    with pytest.raises(ValueError, match="Data leakage detected"):
        verify_trajectory_split_isolation(train_trajs, leaked_val)


def test_trajectory_windowing_contract():
    """Verify build_trajectory_windows slices trajectories into [B, L, C, H, W] and [B, H, C, H, W]."""
    cfg = AdvectionDiffusionConfig(nx=32, ny=32)
    trajs, _ = generate_trajectory_dataset(
        num_trajectories=3, num_steps=10, dt=0.05, seed_base=100, cfg=cfg,
    )
    # trajs shape: (3, 11, 1, 32, 32)
    hist, fut = build_trajectory_windows(trajs, history_len=3, future_len=5, stride=2)

    assert hist.ndim == 5 and hist.shape[1] == 3 and hist.shape[2:] == (1, 32, 32)
    assert fut.ndim == 5 and fut.shape[1] == 5 and fut.shape[2:] == (1, 32, 32)
    assert hist.shape[0] == fut.shape[0]

    # Verify no NaN or Inf
    assert torch.isfinite(hist).all()
    assert torch.isfinite(fut).all()


def test_enstrophy_dissipation_rate_matches_derivative():
    """Verify that spectral dissipation rate matches the discrete enstrophy difference."""
    nu = 0.005
    cfg = AdvectionDiffusionConfig(nx=64, ny=64, u0=0.5, v0=0.5, nu=nu)
    solver = PeriodicScalarAdvectionDiffusion(cfg=cfg)

    f0, phases = solver.generate_initial_condition(seed=55, dtype=torch.float64)
    dt = 1e-3
    f1 = solver.step_algebraic(t=dt, phases=phases, dtype=torch.float64)

    z0 = solver.compute_enstrophy(f0).item()
    z1 = solver.compute_enstrophy(f1).item()
    discrete_rate = (z1 - z0) / dt

    diss_0 = solver.compute_enstrophy_dissipation_rate(f0).item()
    diss_1 = solver.compute_enstrophy_dissipation_rate(f1).item()
    avg_diss = 0.5 * (diss_0 + diss_1)

    rel_error = abs(discrete_rate - avg_diss) / abs(avg_diss)
    assert rel_error < 0.01, f"Dissipation rate mismatch: discrete {discrete_rate:.6f} vs analytical {avg_diss:.6f}"
