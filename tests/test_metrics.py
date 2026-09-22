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
    b, h, c, ny, nx = 2, 5, 4, 32, 64
    target = torch.randn(b, h, c, ny, nx)
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

