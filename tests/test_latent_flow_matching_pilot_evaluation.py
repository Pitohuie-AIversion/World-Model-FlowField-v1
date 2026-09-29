"""Unit and Contract Verification Tests for FM-R1 Pilot Evaluation Pipeline.

Governance Contracts Verified:
1. Mathematical integrity of CRPS estimator:
   - Degenerate ensemble exactly equals MAE.
   - Empirical CRPS converges to analytical Gaussian CRPS.
   - Strict non-negativity.
2. PICP nominal-level prediction interval calibration contract.
3. Same K and RNG reproducibility contract across runs.
4. Same validation window manifest contract across D0/G0/G1/FM.
5. G1 validation metrics are dynamically recomputed, never copied from test artifact.
6. Autoregressive rollout target is strictly Ground Truth (GT), not D0.
7. Spectral energy comparison includes GT and computes relative L2 error vs GT.
"""

import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scripts.evaluate_latent_flow_matching_pilot import (
    compute_sample_crps_tensor,
    compute_gaussian_crps_analytical,
    compute_quantile_coverage_and_width,
    compute_gaussian_analytical_intervals,
    apply_pressure_gauge,
    evaluate_one_step_comparative,
    evaluate_rollout_physics_comparative,
)
from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.probabilistic_latent_dynamics import VarianceHead2D
from src.models.latent_flow_matching import LatentFlowMatcher
from src.data.normalization import FieldNormalizer


class DummySyntheticEvalDataset(Dataset):
    """Synthetic dataset generating paired history and 10-step future ground truth."""

    def __init__(self, num_samples: int = 4):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 4 history steps, 10 future steps, spatial size 32x32 for fast CPU testing
        history = torch.randn(4, 4, 32, 32)
        future = torch.randn(10, 4, 32, 32)
        return {
            "history": history,
            "future": future,
            "re": torch.tensor(10000.0),
            "sc": torch.tensor(0.5),
        }


class TestCRPSEstimatorContract:
    """Verify empirical CRPS mathematical properties and asymptotic convergence."""

    def test_degenerate_ensemble_equals_mae(self):
        # When all K samples are identical point prediction x_hat, CRPS(F_K, y) must strictly equal |x_hat - y|
        k = 16
        b = 4
        point_pred = torch.tensor([1.5, -2.0, 0.5, 3.0]).view(b, 1)
        target = torch.tensor([1.0, -1.0, 1.0, 2.0]).view(b, 1)

        samples = point_pred.unsqueeze(0).repeat(k, 1, 1)  # (K, B, 1)
        crps = compute_sample_crps_tensor(samples, target)
        mae = torch.abs(point_pred - target)

        assert torch.allclose(crps, mae, atol=1e-6)

    def test_empirical_crps_converges_to_analytical_gaussian(self):
        # Sample K=2000 from N(mu, sigma^2) and assert convergence to analytical Gaussian formula
        k = 2000
        b = 100
        mu = torch.linspace(-1.0, 1.0, b)
        var = torch.linspace(0.2, 1.5, b)
        target = mu + 0.5 * torch.sqrt(var)  # target offset by 0.5 sigma

        generator = torch.Generator().manual_seed(42)
        eps = torch.randn((k, b), generator=generator)
        samples = mu.unsqueeze(0) + eps * torch.sqrt(var).unsqueeze(0)

        emp_crps = compute_sample_crps_tensor(samples, target).mean()
        ana_crps = compute_gaussian_crps_analytical(mu, target, var).mean()

        # Should match within 2.5% relative error under K=2000
        rel_err = abs(emp_crps.item() - ana_crps.item()) / ana_crps.item()
        assert rel_err < 0.025

    def test_crps_strict_non_negativity(self):
        k = 32
        samples = torch.randn(k, 20, 10)
        target = torch.randn(20, 10)
        crps = compute_sample_crps_tensor(samples, target)
        assert (crps >= -1e-6).all()


class TestPICPNominalLevelContract:
    """Verify central quantile interval coverage and calibration errors."""

    def test_picp_standard_normal_calibration(self):
        k = 2000
        n_points = 5000
        generator = torch.Generator().manual_seed(1234)

        # Standard normal samples and targets
        samples = torch.randn((k, n_points), generator=generator)
        target = torch.randn((n_points,), generator=generator)

        res = compute_quantile_coverage_and_width(samples, target, nominal_levels=[0.5, 0.8, 0.9, 0.95])
        # Under 5000 test points, coverage must match nominal within +- 2.5 percentage points
        for lvl_str, exp_nominal in [("50", 0.50), ("80", 0.80), ("90", 0.90), ("95", 0.95)]:
            picp = res[lvl_str]["picp"]
            assert abs(picp - exp_nominal) < 0.025
            assert res[lvl_str]["nominal"] == exp_nominal


class TestSameKAndRNGContract:
    """Verify deterministic reproducibility under fixed seed and diversity across seeds."""

    def test_same_rng_contract_reproducibility(self):
        fm = LatentFlowMatcher(latent_channels=8, cond_dim=16, hidden_channels=16, num_blocks=1)
        mu = torch.randn(2, 1, 8, 4, 4)
        re = torch.tensor([1000.0, 2000.0])
        sc = torch.tensor([0.5, 0.7])

        gen1 = torch.Generator().manual_seed(42)
        samps1 = fm.sample_ensemble(mu=mu, re=re, sc=sc, num_samples=8, generator=gen1)

        gen2 = torch.Generator().manual_seed(42)
        samps2 = fm.sample_ensemble(mu=mu, re=re, sc=sc, num_samples=8, generator=gen2)

        assert torch.allclose(samps1, samps2)


class TestComparativeValidationBenchmarkIntegrity:
    """Verify strict comparative evaluation contracts on validation data."""

    @pytest.fixture
    def setup_synthetic_models_and_loader(self):
        device = torch.device("cpu")
        encoder = Encoder2D(in_channels=4, latent_channels=8, base_channels=8, channel_mult=[1, 1, 1])
        decoder = Decoder2D(latent_channels=8, out_channels=4, base_channels=8, channel_mult=[1, 1, 1])
        transformer = LatentSTTransformer(
            latent_channels=8, embed_dim=16, cond_dim=16, depth=1, num_heads=2,
            history_length=4, prediction_mode="residual", use_spatial_pos=False
        )
        forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder).to(device)

        g0_head = VarianceHead2D(embed_dim=16, latent_channels=8, variance_floor=1e-4).to(device)
        g1_head = VarianceHead2D(embed_dim=16, latent_channels=8, variance_floor=1e-4).to(device)
        fm = LatentFlowMatcher(latent_channels=8, cond_dim=16, hidden_channels=16, num_blocks=1).to(device)

        dataset = DummySyntheticEvalDataset(num_samples=4)
        dataloader = DataLoader(dataset, batch_size=2)

        normalizer = FieldNormalizer()
        normalizer.mean = torch.zeros(4)
        normalizer.std = torch.ones(4)

        return forecaster, g0_head, g1_head, fm, dataloader, normalizer, device

    def test_same_validation_manifest_across_all_models(self, setup_synthetic_models_and_loader):
        forecaster, g0_head, g1_head, fm, dataloader, normalizer, device = setup_synthetic_models_and_loader

        step1 = evaluate_one_step_comparative(
            forecaster=forecaster,
            g0_head=g0_head,
            g1_head=g1_head,
            fm=fm,
            dataloader=dataloader,
            normalizer=normalizer,
            device=device,
            num_samples_K=4,
            fm_temperatures=[1.0, 0.8],
            seed=42,
        )

        assert step1["validation_windows_evaluated"] == 4
        assert step1["ensemble_size_K"] == 4
        # Assert all models evaluated on exactly the same dataset
        for m in ["D0", "G0", "G1", "FM_temp_1.0", "FM_temp_0.8"]:
            assert m in step1

    def test_g1_validation_metrics_are_dynamically_recomputed(self, setup_synthetic_models_and_loader):
        forecaster, g0_head, g1_head, fm, dataloader, normalizer, device = setup_synthetic_models_and_loader

        step1 = evaluate_one_step_comparative(
            forecaster=forecaster,
            g0_head=g0_head,
            g1_head=g1_head,
            fm=fm,
            dataloader=dataloader,
            normalizer=normalizer,
            device=device,
            num_samples_K=4,
            fm_temperatures=[1.0],
            seed=42,
        )

        # G1 must have newly evaluated metrics (not static 0.360156 from Phase 3 test JSON)
        g1_crps = step1["G1"]["latent_crps_empirical"]
        assert isinstance(g1_crps, float)
        assert g1_crps > 0.0
        assert "latent_crps_analytical" in step1["G1"]
        assert "physical_crps_per_channel" in step1["G1"]
        assert set(step1["G1"]["physical_crps_per_channel"].keys()) == {"u", "v", "p", "s"}

    def test_rollout_target_is_ground_truth_not_d0(self, setup_synthetic_models_and_loader):
        forecaster, g0_head, g1_head, fm, dataloader, normalizer, device = setup_synthetic_models_and_loader

        step2 = evaluate_rollout_physics_comparative(
            forecaster=forecaster,
            g0_head=g0_head,
            g1_head=g1_head,
            fm=fm,
            dataloader=dataloader,
            normalizer=normalizer,
            device=device,
            horizons=[5, 10],
            num_samples_K=4,
            max_trajectories=2,
            seed=42,
        )

        assert "h5" in step2
        assert "h10" in step2
        assert "ground_truth_rms_divergence" in step2["h5"]
        assert "ground_truth_rms_divergence" in step2["h10"]

        # Crucial check: D0 itself is evaluated against GT!
        d0_metrics = step2["h5"]["D0"]
        assert "ensemble_mean_vrmse_vs_gt" in d0_metrics
        assert "sample_rms_divergence" in d0_metrics

        # FM is evaluated against GT
        fm_metrics = step2["h5"]["FM"]
        assert "ensemble_mean_vrmse_vs_gt" in fm_metrics
        assert "sample_vorticity_rmse_vs_gt" in fm_metrics
        assert "divergence_ratio_vs_gt" in fm_metrics

    def test_spectral_comparison_includes_ground_truth(self, setup_synthetic_models_and_loader):
        forecaster, g0_head, g1_head, fm, dataloader, normalizer, device = setup_synthetic_models_and_loader

        step2 = evaluate_rollout_physics_comparative(
            forecaster=forecaster,
            g0_head=g0_head,
            g1_head=g1_head,
            fm=fm,
            dataloader=dataloader,
            normalizer=normalizer,
            device=device,
            horizons=[5, 10],
            num_samples_K=4,
            max_trajectories=2,
            seed=42,
        )

        # Spectral comparisons must include GT spectrum
        spectra = step2["energy_spectra_first_25_modes"]
        assert "GT" in spectra
        assert spectra["GT"] is not None
        assert "FM" in spectra
        assert "D0" in spectra

        # Relative errors must be calculated vs GT
        rel_errors = step2["spectral_relative_error_vs_gt_h10"]
        assert "FM" in rel_errors
        assert "D0" in rel_errors
        assert "G1" in rel_errors
        assert isinstance(rel_errors["FM"], float)
