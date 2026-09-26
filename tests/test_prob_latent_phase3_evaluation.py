"""Unit test suite for ProbLatent-R1 Phase 3 Evaluation Pipeline."""

import json
import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scripts.evaluate_prob_latent_phase3 import (
    compute_gaussian_crps,
    compute_prediction_intervals,
    compute_ensemble_spread_skill,
    compute_physical_quantiles_coverage,
    apply_pressure_gauge,
    verify_phase3_preflight_gate,
    evaluate_single_step_probability,
    validate_phase3_metrics_finite,
)


class TestCRPSAndPredictionIntervalsMath:
    """Mathematical and numerical validation of scoring rules and interval calibration metrics."""

    def test_gaussian_crps_exact_analytical_value(self):
        """When y == mu and var == 1, CRPS = (sqrt(2) - 1) / sqrt(pi)."""
        mu = torch.tensor([0.0])
        target = torch.tensor([0.0])
        var = torch.tensor([1.0])
        expected = (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)
        crps = compute_gaussian_crps(mu, target, var).item()
        assert crps == pytest.approx(expected, rel=1e-6)

    def test_gaussian_crps_symmetry(self):
        """CRPS(y, mu, var) must be strictly symmetric with respect to (y - mu)."""
        mu = torch.tensor([1.5])
        target_pos = torch.tensor([2.5])
        target_neg = torch.tensor([0.5])
        var = torch.tensor([0.4])

        crps_pos = compute_gaussian_crps(mu, target_pos, var).item()
        crps_neg = compute_gaussian_crps(mu, target_neg, var).item()
        assert crps_pos == pytest.approx(crps_neg, rel=1e-7)

    def test_gaussian_crps_converges_to_mae_as_variance_vanishes(self):
        """When sigma -> 0, Gaussian CRPS converges smoothly to deterministic MAE |y - mu|."""
        mu = torch.tensor([2.0])
        target = torch.tensor([3.5])
        expected_mae = 1.5
        small_var = torch.tensor([1e-8])
        crps = compute_gaussian_crps(mu, target, small_var).item()
        assert crps == pytest.approx(expected_mae, rel=1e-3)

    def test_prediction_intervals_nominal_coverage_on_synthetic_gaussian(self):
        """Empirical coverage on synthetic Gaussian draws aligns with nominal levels."""
        torch.manual_seed(42)
        n_samples = 20000
        mu = torch.zeros(n_samples)
        var = torch.ones(n_samples)
        target = torch.randn(n_samples)

        intervals = compute_prediction_intervals(
            mu=mu,
            target=target,
            var=var,
            nominal_levels=(0.50, 0.80, 0.90, 0.95),
        )

        assert intervals["50"]["picp"] == pytest.approx(0.50, abs=0.015)
        assert intervals["80"]["picp"] == pytest.approx(0.80, abs=0.015)
        assert intervals["90"]["picp"] == pytest.approx(0.90, abs=0.015)
        assert intervals["95"]["picp"] == pytest.approx(0.95, abs=0.015)

        # Monotonicity of interval width
        assert intervals["50"]["mpiw"] < intervals["80"]["mpiw"]
        assert intervals["80"]["mpiw"] < intervals["90"]["mpiw"]
        assert intervals["90"]["mpiw"] < intervals["95"]["mpiw"]

    def test_interval_calibration_error_directional_accounting(self):
        """Verify calibration error tracking when G1 over-covers (e.g. 94.94% at nominal 90%)."""
        # Target with narrower spread than predicted var=1.0 -> will over-cover
        torch.manual_seed(42)
        n_samples = 10000
        mu = torch.zeros(n_samples)
        var = torch.ones(n_samples)
        # Scale target to 0.7 to artificially produce over-coverage ~94.9%
        target = torch.randn(n_samples) * 0.7

        intervals = compute_prediction_intervals(
            mu=mu, target=target, var=var, nominal_levels=(0.90,)
        )
        picp = intervals["90"]["picp"]
        assert picp > 0.90  # Over-coverage
        signed_err = intervals["90"]["signed_calibration_error"]
        abs_err = intervals["90"]["absolute_calibration_error"]
        assert signed_err == pytest.approx(picp - 0.90, abs=1e-5)
        assert abs_err == pytest.approx(abs(picp - 0.90), abs=1e-5)


class TestSpreadSkillAndPhysicalMetrics:
    """Validation of Spread-Skill RMS formulation, physical quantiles, and pressure gauge."""

    def test_rms_spread_vs_mean_std_and_rmse_vs_mae(self):
        """Verify that RMS spread = sqrt(mean(var)) is distinct from mean(std) on heteroscedastic spread."""
        K = 10
        # Spatial grid 2x2 with differing variances across grid points
        # Grid point (0,0): var = 1.0, Grid point (0,1): var = 9.0
        # mean(var) = 5.0 -> RMS spread = sqrt(5.0) approx 2.236
        # std: [1.0, 3.0] -> mean(std) = 2.0
        # RMS spread > mean(std) by Jensen's inequality
        torch.manual_seed(42)
        samps = torch.zeros(K, 4, 2, 2)
        samps[:, 0, 0, 0] = torch.randn(K) * 1.0
        samps[:, 0, 0, 1] = torch.randn(K) * 3.0

        target = torch.zeros(4, 2, 2)
        target[0, 0, 0] = 0.5
        target[0, 0, 1] = 1.5

        res = compute_ensemble_spread_skill(samps, target)
        u_metrics = res["per_variable"]["u"]

        # Check Bessel's correction factor across all spatial points
        var_sample = samps[:, 0].var(dim=0, unbiased=True)
        expected_mean_var = var_sample.mean().item()
        expected_rms_spread = math.sqrt(expected_mean_var)
        assert u_metrics["rms_spread"] == pytest.approx(expected_rms_spread, rel=1e-5)

        # Finite-K inflation factor check: sqrt((K+1)/K)
        expected_finite_k = math.sqrt((K + 1) / K)
        assert res["finite_k_inflation_factor"] == pytest.approx(expected_finite_k, rel=1e-5)
        assert u_metrics["finite_k_adjusted_spread"] == pytest.approx(expected_rms_spread * expected_finite_k, rel=1e-5)

        # Denominator is strictly RMSE, not MAE
        expected_mse = ((samps.mean(dim=0)[0] - target[0]) ** 2).mean().item()
        expected_rmse = math.sqrt(expected_mse)
        assert u_metrics["rmse"] == pytest.approx(expected_rmse, rel=1e-5)
        assert u_metrics["spread_skill_ratio"] == pytest.approx(expected_rms_spread / expected_rmse, rel=1e-5)

    def test_pressure_gauge_zero_mean(self):
        """Verify that apply_pressure_gauge enforces zero mean on pressure channel without touching others."""
        field = torch.randn(4, 16, 16)
        field[2] += 10.0  # Add large pressure offset
        gauged = apply_pressure_gauge(field, pressure_channel=2)

        # Pressure channel spatial mean must be zero
        assert gauged[2].mean().item() == pytest.approx(0.0, abs=1e-6)
        # Velocity and tracer channels must be untouched
        assert torch.allclose(gauged[0], field[0])
        assert torch.allclose(gauged[1], field[1])
        assert torch.allclose(gauged[3], field[3])

    def test_physical_space_empirical_quantiles_computation(self):
        """Verify empirical quantile calculation across sample dimension."""
        K = 20
        torch.manual_seed(42)
        samps = torch.randn(K, 4, 8, 8)
        target = torch.zeros(4, 8, 8)

        cov = compute_physical_quantiles_coverage(samps, target)
        assert "overall" in cov and "velocity" in cov
        assert 0.0 <= cov["overall"]["picp_80"] <= 1.0
        assert 0.0 <= cov["overall"]["picp_90"] <= 1.0
        assert cov["overall"]["mpiw_80"] < cov["overall"]["mpiw_90"]


class MockIdentifiedDataset(Dataset):
    """Synthetic dataset producing full cryptographic identity fields conforming to ShearFlowDataset."""
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        return {
            "history": torch.zeros(4, 4, 16, 16),
            "future": torch.zeros(1, 4, 16, 16),
            "re": torch.tensor(10000.0),
            "sc": torch.tensor(0.1),
            "source_file": item["source_file"],
            "traj_idx": torch.tensor(item["traj_idx"], dtype=torch.long),
            "start_t": torch.tensor(item["start_t"], dtype=torch.long),
            "cluster_id": torch.tensor(item["cluster_id"], dtype=torch.long),
        }


class TestIdentityAndGroupingFailClosed:
    """Verify trajectory identity grouping, collision prevention, and missing field rejection."""

    def test_two_files_with_same_traj_idx_not_merged(self):
        """When two files have the same traj_idx=1, they must not be merged into a single trajectory."""
        entries = [
            {"source_file": "data/test/shear_flow_A.hdf5", "traj_idx": 1, "start_t": 0, "cluster_id": 1},
            {"source_file": "data/test/shear_flow_A.hdf5", "traj_idx": 1, "start_t": 8, "cluster_id": 1},
            {"source_file": "data/train/shear_flow_B.hdf5", "traj_idx": 1, "start_t": 0, "cluster_id": 2},
            {"source_file": "data/train/shear_flow_B.hdf5", "traj_idx": 1, "start_t": 8, "cluster_id": 2},
        ]
        loader = DataLoader(MockIdentifiedDataset(entries), batch_size=2)

        class MockG0Head(nn.Module):
            def forward(self, x):
                return torch.ones(x.shape[0], 1, 64, 4, 4)

        class MockForecaster(nn.Module):
            def eval(self):
                pass
            def predict_distribution_single_step(self, q, re=None, sc=None, variance_head=None):
                b = q.shape[0]
                mu = torch.zeros(b, 1, 64, 4, 4)
                var = torch.ones(b, 1, 64, 4, 4) * (0.5 if variance_head is not None else 0.25)
                return mu, var
            def encoder(self, q):
                b = q.shape[0]
                return torch.zeros(b, 1, 64, 4, 4)

        res = evaluate_single_step_probability(
            forecaster=MockForecaster(),
            g0_head=MockG0Head(),
            test_loader=loader,
            device=torch.device("cpu"),
        )

        trajs = res["trajectory_diagnostics"]
        assert len(trajs) == 2, f"Expected 2 distinct trajectories, got {len(trajs)}"
        traj_ids = {t["trajectory_id"] for t in trajs}
        assert "shear_flow_A.hdf5::sim_01" in traj_ids
        assert "shear_flow_B.hdf5::sim_01" in traj_ids
        for t in trajs:
            assert t["windows"] == 2

    def test_missing_identity_fields_fails_closed(self):
        """If any identity field is missing from the batch, evaluate_single_step_probability fails closed."""
        class DefectiveDataset(Dataset):
            def __len__(self):
                return 2
            def __getitem__(self, idx):
                return {
                    "history": torch.zeros(4, 4, 16, 16),
                    "future": torch.zeros(1, 4, 16, 16),
                    # Missing source_file, traj_idx, start_t, cluster_id
                }

        loader = DataLoader(DefectiveDataset(), batch_size=2)
        with pytest.raises(ValueError, match="missing required identity field"):
            evaluate_single_step_probability(
                forecaster=nn.Identity(),
                g0_head=nn.Identity(),
                test_loader=loader,
                device=torch.device("cpu"),
            )


class TestPreflightGateFailClosed:
    """Verify pre-flight gate fails closed on hash tampering, non-finite values, or NLL drift."""

    def test_preflight_gate_fails_on_hash_mismatch(self, tmp_path):
        record_file = tmp_path / "phase2_record.json"
        with open(record_file, "w") as f:
            json.dump({
                "cryptographic_bindings": {
                    "d0_checkpoint": {"sha256": "expected_sha_123"},
                    "stats_file": {"sha256": "dummy"},
                    "data_protocol": {"split_hash": "dummy", "normalizer_hash": "dummy"},
                },
                "artifacts": {
                    "best_g1_variance_head": {"sha256": "dummy"},
                    "g0_baseline_initialization": {"sha256": "dummy_g0"},
                },
                "training_summary": {"best_val_nll": -0.206643},
            }, f)

        d0_file = tmp_path / "d0.pt"
        d0_file.write_text("actual_content")

        with pytest.raises(ValueError, match="D0 SHA mismatch"):
            verify_phase3_preflight_gate(
                d0_checkpoint_path=str(d0_file),
                g0_checkpoint_path="dummy",
                stats_path="dummy",
                normalizer_path="dummy",
                split_file="dummy",
                phase2_record_path=str(record_file),
                g1_variance_head_path="dummy",
                forecaster=nn.Identity(),
                val_loader=[],
                device=torch.device("cpu"),
            )

    def test_preflight_gate_fails_on_g0_checkpoint_mismatch(self, tmp_path):
        """Gate must reject when g0_checkpoint_path hash does not match phase2 record."""
        from src.utils.provenance import compute_file_sha256

        d0_file = tmp_path / "d0.pt"
        d0_file.write_text("d0")
        stats_file = tmp_path / "stats.json"
        stats_file.write_text("stats")
        g0_file = tmp_path / "g0.pt"
        g0_file.write_text("g0_tampered")

        record_file = tmp_path / "phase2_record.json"
        with open(record_file, "w") as f:
            json.dump({
                "cryptographic_bindings": {
                    "d0_checkpoint": {"sha256": compute_file_sha256(str(d0_file))},
                    "stats_file": {"sha256": compute_file_sha256(str(stats_file))},
                    "data_protocol": {"split_hash": "dummy", "normalizer_hash": "dummy"},
                },
                "artifacts": {
                    "best_g1_variance_head": {"sha256": "dummy"},
                    "g0_baseline_initialization": {"sha256": "expected_g0_sha_xyz"},
                },
                "training_summary": {"best_val_nll": -0.206643},
            }, f)

        with pytest.raises(ValueError, match="G0 baseline checkpoint SHA mismatch"):
            verify_phase3_preflight_gate(
                d0_checkpoint_path=str(d0_file),
                g0_checkpoint_path=str(g0_file),
                stats_path=str(stats_file),
                normalizer_path="dummy",
                split_file="dummy",
                phase2_record_path=str(record_file),
                g1_variance_head_path="dummy",
                forecaster=nn.Identity(),
                val_loader=[],
                device=torch.device("cpu"),
            )

    def test_preflight_gate_rejects_nan_and_inf(self, tmp_path):
        """Gate must reject when reproduced val NLL or batches contain NaN/Inf."""
        from src.utils.provenance import compute_file_sha256, compute_split_hash_from_file, compute_normalizer_hash
        from src.data.normalization import FieldNormalizer

        split_file = tmp_path / "split.json"
        with open(split_file, "w") as f:
            json.dump({"train": []}, f)
        split_h = compute_split_hash_from_file(str(split_file))

        norm_file = tmp_path / "norm.pt"
        torch.save({"mean": torch.zeros(4), "std": torch.ones(4)}, norm_file)
        norm_obj = FieldNormalizer()
        norm_obj.load_state_dict(torch.load(norm_file, weights_only=True))
        norm_h = compute_normalizer_hash(norm_obj)

        d0_file = tmp_path / "d0.pt"
        d0_file.write_text("d0")
        stats_file = tmp_path / "stats.json"
        stats_file.write_text("stats")
        g0_file = tmp_path / "g0.pt"
        g0_file.write_text("g0")
        g1_file = tmp_path / "g1.pt"
        g1_file.write_text("g1")

        record_file = tmp_path / "phase2_record.json"
        with open(record_file, "w") as f:
            json.dump({
                "cryptographic_bindings": {
                    "d0_checkpoint": {"sha256": compute_file_sha256(str(d0_file))},
                    "stats_file": {"sha256": compute_file_sha256(str(stats_file))},
                    "data_protocol": {"split_hash": split_h, "normalizer_hash": norm_h},
                },
                "artifacts": {
                    "best_g1_variance_head": {"sha256": compute_file_sha256(str(g1_file))},
                    "g0_baseline_initialization": {"sha256": compute_file_sha256(str(g0_file))},
                },
                "training_summary": {"best_val_nll": -0.206643},
            }, f)

        class NanForecaster(nn.Module):
            def eval(self):
                pass
            def predict_distribution_single_step(self, q, re=None, sc=None):
                b = q.shape[0]
                mu = torch.zeros(b, 1, 64, 4, 4)
                var = torch.full((b, 1, 64, 4, 4), float("nan"))
                return mu, var
            def encoder(self, q):
                b = q.shape[0]
                return torch.zeros(b, 1, 64, 4, 4)

        loader = DataLoader(MockIdentifiedDataset([{
            "source_file": "f.hdf5", "traj_idx": 0, "start_t": 0, "cluster_id": 0
        }]), batch_size=1)

        with pytest.raises(ValueError, match="non-finite distribution"):
            verify_phase3_preflight_gate(
                d0_checkpoint_path=str(d0_file),
                g0_checkpoint_path=str(g0_file),
                stats_path=str(stats_file),
                normalizer_path=str(norm_file),
                split_file=str(split_file),
                phase2_record_path=str(record_file),
                g1_variance_head_path=str(g1_file),
                forecaster=NanForecaster(),
                val_loader=loader,
                device=torch.device("cpu"),
            )


class TestMetricsFinitenessValidation:
    """Verify recursive finite checks for final evaluation report."""

    def test_validate_phase3_metrics_finite_catches_nan(self):
        valid_dict = {
            "metrics": {
                "nll": -0.20,
                "nested": [{"val": 1.5}, {"val": 2.5}],
            }
        }
        validate_phase3_metrics_finite(valid_dict)

        invalid_dict = {
            "metrics": {
                "nll": -0.20,
                "nested": [{"val": 1.5}, {"val": float("nan")}],
            }
        }
        with pytest.raises(ValueError, match="non-finite float at root.metrics.nested\\[1\\].val = nan"):
            validate_phase3_metrics_finite(invalid_dict)
