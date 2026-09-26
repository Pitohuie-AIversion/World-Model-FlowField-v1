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
    verify_phase3_preflight_gate,
    evaluate_single_step_probability,
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
        # Target draws from true N(0, 1)
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


class DummyDataset(Dataset):
    def __init__(self, n=4):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return {
            "history": torch.zeros(4, 4, 16, 16),
            "future": torch.zeros(1, 4, 16, 16),
            "re": torch.tensor(10000.0),
            "sc": torch.tensor(0.1),
            "traj_idx": torch.tensor(idx),
        }


class TestPreflightGateFailClosed:
    """Verify pre-flight gate fails closed on hash tampering or NLL drift."""

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
                },
                "training_summary": {"best_val_nll": -0.206643},
            }, f)

        d0_file = tmp_path / "d0.pt"
        d0_file.write_text("actual_content")

        with pytest.raises(ValueError, match="D0 SHA mismatch"):
            verify_phase3_preflight_gate(
                d0_checkpoint_path=str(d0_file),
                stats_path="dummy",
                normalizer_path="dummy",
                split_file="dummy",
                phase2_record_path=str(record_file),
                g1_variance_head_path="dummy",
                forecaster=nn.Identity(),
                val_loader=[],
                device=torch.device("cpu"),
            )

    def test_preflight_gate_fails_when_recomputed_val_nll_drifts(self, tmp_path, monkeypatch):
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
        d0_sha = compute_file_sha256(str(d0_file))

        stats_file = tmp_path / "stats.json"
        stats_file.write_text("stats")
        stats_sha = compute_file_sha256(str(stats_file))

        g1_file = tmp_path / "g1.pt"
        g1_file.write_text("g1")
        g1_sha = compute_file_sha256(str(g1_file))

        record_file = tmp_path / "phase2_record.json"
        with open(record_file, "w") as f:
            json.dump({
                "cryptographic_bindings": {
                    "d0_checkpoint": {"sha256": d0_sha},
                    "stats_file": {"sha256": stats_sha},
                    "data_protocol": {"split_hash": split_h, "normalizer_hash": norm_h},
                },
                "artifacts": {
                    "best_g1_variance_head": {"sha256": g1_sha},
                },
                "training_summary": {"best_val_nll": -0.206643},
            }, f)

        class MockForecaster(nn.Module):
            def eval(self):
                pass
            def predict_distribution_single_step(self, q, re=None, sc=None):
                b = q.shape[0]
                return torch.zeros(b, 1, 64, 4, 4), torch.ones(b, 1, 64, 4, 4)
            def encoder(self, q):
                b = q.shape[0]
                # Offset target so NLL is not -0.206643
                return torch.ones(b, 1, 64, 4, 4) * 2.0

        loader = DataLoader(DummyDataset(n=2), batch_size=2)
        forecaster = MockForecaster()

        # Recomputed NLL on target=2.0, mu=0.0, var=1.0 will be 0.5 * (0 + 4) = 2.0,
        # which heavily deviates from expected -0.206643 -> must fail closed
        with pytest.raises(RuntimeError, match="Pre-flight gate failed: recomputed val NLL"):
            verify_phase3_preflight_gate(
                d0_checkpoint_path=str(d0_file),
                stats_path=str(stats_file),
                normalizer_path=str(norm_file),
                split_file=str(split_file),
                phase2_record_path=str(record_file),
                g1_variance_head_path=str(g1_file),
                forecaster=forecaster,
                val_loader=loader,
                device=torch.device("cpu"),
                tolerance=1e-4,
            )


class TestSingleStepProbabilisticEvaluationParity:
    """Verify mean parity assertion and metrics formulation in single-step evaluation."""

    def test_single_step_evaluation_passes_and_checks_mean_parity(self):
        class MockG0Head(nn.Module):
            def forward(self, x):
                return torch.ones(x.shape[0], 1, 64, 4, 4)

        class MockForecaster(nn.Module):
            def eval(self):
                pass
            def predict_distribution_single_step(self, q, re=None, sc=None, variance_head=None):
                b = q.shape[0]
                # mu is identical across G0 and G1
                mu = torch.zeros(b, 1, 64, 4, 4)
                if variance_head is not None:
                    var = torch.ones(b, 1, 64, 4, 4) * 0.5
                else:
                    var = torch.ones(b, 1, 64, 4, 4) * 0.25
                return mu, var
            def encoder(self, q):
                b = q.shape[0]
                return torch.zeros(b, 1, 64, 4, 4)

        forecaster = MockForecaster()
        g0_head = MockG0Head()
        loader = DataLoader(DummyDataset(n=4), batch_size=2)

        res = evaluate_single_step_probability(
            forecaster=forecaster,
            g0_head=g0_head,
            test_loader=loader,
            device=torch.device("cpu"),
            nominal_levels=(0.50, 0.80, 0.90, 0.95),
        )

        assert res["mean_parity_check"]["status"] == "PASS"
        assert res["dataset_summary"]["windows_evaluated"] == 4
        # Since target=0, mu=0:
        # G1 has var=0.25 < G0 var=0.5 -> G1 NLL must be lower than G0 NLL
        assert res["G1_heteroscedastic_model"]["nll"] < res["G0_homoscedastic_baseline"]["nll"]
        assert res["comparison_g1_vs_g0"]["nll_improved"] is True
        assert res["comparison_g1_vs_g0"]["delta_nll"] < 0.0
        assert len(res["channel_diagnostics"]) == 64

    def test_single_step_evaluation_fails_when_mean_parity_violated(self):
        class MockG0Head(nn.Module):
            pass

        class CorruptedMeanForecaster(nn.Module):
            def eval(self):
                pass
            def predict_distribution_single_step(self, q, re=None, sc=None, variance_head=None):
                b = q.shape[0]
                if variance_head is not None:
                    mu = torch.zeros(b, 1, 64, 4, 4)
                else:
                    mu = torch.ones(b, 1, 64, 4, 4) * 0.1  # Corrupted mean discrepancy
                return mu, torch.ones(b, 1, 64, 4, 4)
            def encoder(self, q):
                b = q.shape[0]
                return torch.zeros(b, 1, 64, 4, 4)

        forecaster = CorruptedMeanForecaster()
        g0_head = MockG0Head()
        loader = DataLoader(DummyDataset(n=2), batch_size=2)

        with pytest.raises(AssertionError, match="Mean parity check failed"):
            evaluate_single_step_probability(
                forecaster=forecaster,
                g0_head=g0_head,
                test_loader=loader,
                device=torch.device("cpu"),
            )
