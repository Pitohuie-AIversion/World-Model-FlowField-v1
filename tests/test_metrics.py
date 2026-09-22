"""Unit tests for Field, Spectral, Tracer, Compute, and Rollout metrics."""

import torch
import pytest
from src.metrics.field import evaluate_field_metrics, compute_vrmse, compute_nmse
from src.metrics.spectral import compute_radial_energy_spectrum, compute_spectral_error
from src.metrics.tracer import compute_tracer_metrics
from src.metrics.compute import count_parameters
from src.metrics.rollout import evaluate_rollout_trajectory
from src.baselines.persistence import PersistenceBaseline


def test_field_metrics():
    target = torch.randn(2, 4, 32, 64)
    pred = target + 0.05 * torch.randn_like(target)

    metrics = evaluate_field_metrics(pred, target)
    for c in ["u", "v", "p", "s"]:
        assert f"vrmse_{c}" in metrics
        assert f"nmse_{c}" in metrics
        assert f"max_err_{c}" in metrics
        assert metrics[f"vrmse_{c}"] > 0.0

    assert "vrmse_mean" in metrics


def test_spectral_metrics():
    u = torch.randn(32, 64)
    v = torch.randn(32, 64)
    k_bins, e_k = compute_radial_energy_spectrum(u, v, domain_size=(1.0, 2.0))
    assert len(k_bins) == len(e_k)
    assert (e_k >= 0.0).all()

    spec_err = compute_spectral_error(u, v, u, v, domain_size=(1.0, 2.0))
    assert spec_err["spec_err_total"] == pytest.approx(0.0, abs=1e-5)


def test_tracer_metrics():
    target_s = torch.rand(2, 32, 64)
    pred_s = target_s.clone()

    res = compute_tracer_metrics(pred_s, target_s)
    assert res["tracer_var_retention"] == pytest.approx(1.0, abs=1e-4)
    assert res["tracer_out_of_bounds_rate"] == 0.0
    assert res["tracer_mass_error"] == pytest.approx(0.0, abs=1e-4)


def test_compute_metrics():
    model = PersistenceBaseline()
    res = count_parameters(model)
    assert "params_total_m" in res
    assert "params_trainable_m" in res


def test_rollout_evaluation():
    b, h, c, nx, ny = 2, 5, 4, 32, 64
    target = torch.randn(b, h, c, nx, ny)
    pred = target.clone()

    res = evaluate_rollout_trajectory(pred, target, evaluation_steps=[1, 5])
    assert "step_1" in res
    assert "step_5" in res
    assert res["step_1"]["vrmse_mean"] == pytest.approx(0.0, abs=1e-5)
    assert res["step_1"]["rmse_mean"] == pytest.approx(0.0, abs=1e-5)
    assert "div_rmse" in res["step_1"]
    assert "div_max" in res["step_1"]
    assert "vort_rmse" in res["step_1"]
    assert "ke_rel_err" in res["step_1"]
    assert "enstrophy_rel_err" in res["step_1"]
    assert "tracer_mean_err" in res["step_1"]
    assert "energy_spectrum_mae" in res["step_1"]


def test_spectral_single_fourier_mode():
    """Verify single Fourier mode falls into exactly the correct radial wavenumber bin."""
    nx, ny = 64, 128
    lx, ly = 1.0, 2.0
    # delta_k = 2*pi / max(lx, ly) = 2*pi / 2.0 = pi
    # Mode nx=2, ny=0: kx = 2*(2*pi/1.0) = 4*pi, ky = 0 => k_mag = 4*pi => bin 4
    x = (torch.arange(nx, dtype=torch.float32) * (lx / nx)).unsqueeze(1)
    y = (torch.arange(ny, dtype=torch.float32) * (ly / ny)).unsqueeze(0)
    u = torch.cos(4.0 * torch.pi * x).repeat(1, ny)
    v = torch.zeros(nx, ny)

    k_bins, e_k = compute_radial_energy_spectrum(u, v, domain_size=(lx, ly))
    assert torch.argmax(e_k).item() == 4
    assert (e_k[4] / e_k.sum()).item() > 0.99

    # Mode nx=0, ny=3: kx = 0, ky = 3*(2*pi/2.0) = 3*pi => k_mag = 3*pi => bin 3
    u_y = torch.zeros(nx, ny)
    v_y = torch.cos(3.0 * torch.pi * y).repeat(nx, 1)
    _, e_k_y = compute_radial_energy_spectrum(u_y, v_y, domain_size=(lx, ly))
    assert torch.argmax(e_k_y).item() == 3
    assert (e_k_y[3] / e_k_y.sum()).item() > 0.99


def test_spectral_two_fourier_modes():
    """Verify two Fourier modes produce spectrum peaks at exactly the right bin positions."""
    nx, ny = 64, 128
    lx, ly = 1.0, 2.0
    x = (torch.arange(nx, dtype=torch.float32) * (lx / nx)).unsqueeze(1)
    # Mode 1: k1 = 4*pi (bin 4)
    # Mode 2: k2 = 16*pi (bin 16)
    u = (torch.cos(4.0 * torch.pi * x) + torch.cos(16.0 * torch.pi * x)).repeat(1, ny)
    v = torch.zeros(nx, ny)

    k_bins, e_k = compute_radial_energy_spectrum(u, v, domain_size=(lx, ly))
    top_bins = torch.topk(e_k, k=2).indices.tolist()
    assert 4 in top_bins
    assert 16 in top_bins


def test_spectral_energy_conservation_under_rfft():
    """Verify Parseval's energy conservation under radial RFFT binning for band-limited fields."""
    nx, ny = 64, 128
    lx, ly = 1.0, 2.0
    x = (torch.arange(nx, dtype=torch.float32) * (lx / nx)).unsqueeze(1)
    y = (torch.arange(ny, dtype=torch.float32) * (ly / ny)).unsqueeze(0)
    u = torch.cos(4.0 * torch.pi * x).repeat(1, ny) + 0.5 * torch.sin(3.0 * torch.pi * y).repeat(nx, 1)
    v = 0.3 * torch.cos(2.0 * torch.pi * x).repeat(1, ny) * torch.cos(2.0 * torch.pi * y).repeat(nx, 1)

    ke_phys = 0.5 * (u**2 + v**2).mean()
    k_bins, e_k = compute_radial_energy_spectrum(u, v, domain_size=(lx, ly))
    ke_spec = e_k.sum()

    assert ke_spec.item() == pytest.approx(ke_phys.item(), rel=1e-4)


def test_tracer_near_zero_mean_robustness():
    """Verify mass error metric remains finite, well-scaled and non-amplified when tracer spatial mean is ~0."""
    target_s = torch.randn(2, 32, 64)
    target_s = target_s - target_s.mean(dim=(-2, -1), keepdim=True)
    assert torch.abs(torch.mean(target_s)).item() < 1e-6

    pred_s = target_s + 0.02 * torch.randn_like(target_s)
    res = compute_tracer_metrics(pred_s, target_s)

    assert not torch.isnan(torch.tensor(res["tracer_mass_error"]))
    assert not torch.isinf(torch.tensor(res["tracer_mass_error"]))
    assert res["tracer_mass_error"] < 0.5
    assert res["tracer_mean_err"] < 0.1


def test_tracer_maximum_principle_bounds():
    """Verify maximum principle bounds evaluation against initial state."""
    initial_s = torch.zeros(2, 32, 64)
    initial_s[:, :16, :] = 1.0
    initial_s[:, 16:, :] = -1.0  # bounds are [-1.0, 1.0]

    future_s = initial_s * 0.5  # decayed bounds [-0.5, 0.5]
    pred_s = initial_s * 0.7    # bounds [-0.7, 0.7]

    # Without initial_s, evaluated against future_s => falsely flagged as out of bounds
    res_no_init = compute_tracer_metrics(pred_s, future_s)
    assert res_no_init["tracer_out_of_bounds_rate"] > 0.0

    # With initial_s, correctly recognized as obeying maximum principle
    res_with_init = compute_tracer_metrics(pred_s, future_s, initial_s=initial_s)
    assert res_with_init["tracer_out_of_bounds_rate"] == 0.0

    # Prediction that genuinely violates physical maximum principle
    pred_violating = initial_s * 1.5  # bounds [-1.5, 1.5]
    res_violating = compute_tracer_metrics(pred_violating, future_s, initial_s=initial_s)
    assert res_violating["tracer_out_of_bounds_rate"] > 0.0


