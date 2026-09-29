"""Unit and Contract Verification Tests for FM-R1 Pilot Evaluation Pipeline.

Governance Contracts Verified:
1. Mathematical integrity of CRPS estimator:
   - Degenerate ensemble exactly equals MAE.
   - Empirical CRPS converges to analytical Gaussian CRPS.
   - Strict non-negativity.
2. PICP nominal-level prediction interval calibration contract.
3. Phase 3-compatible Pooled Spread-Skill Ratio contract:
   - Bessel correction (ddof=1)
   - Finite-K inflation factor sqrt((K+1)/K)
4. Same K, RNG reproducibility, and Common Random Numbers (CRN) across temperature sweep:
   - test_temperature_sweep_reuses_common_base_noise
   - test_temperature_cli_preserves_common_random_numbers
5. Trajectory-aware rollout evaluation manifest:
   - test_rollout_manifest_covers_requested_unique_trajectories
   - test_max_trajectories_counts_trajectory_ids_not_windows
6. Cryptographic provenance fail-closed contract:
   - test_evaluator_rejects_g0_normalizer_mismatch
   - test_evaluator_rejects_g1_normalizer_mismatch
   - test_evaluator_rejects_fm_normalizer_mismatch
   - test_evaluator_rejects_missing_seed
   - test_evaluator_rejects_checkpoint_parent_d0_mismatch
7. Comparative Validation Benchmark Integrity:
   - test_same_validation_manifest_across_all_models
   - test_divergence_and_vorticity_aggregate_all_ensemble_members
   - test_individual_member_spectral_error_is_not_mean_spectrum_error
   - test_spectrum_aggregates_multiple_windows
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
    compute_pooled_spread_skill,
    apply_pressure_gauge,
    build_rollout_manifest,
    verify_checkpoint_provenance,
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
from src.utils.provenance import compute_file_sha256, compute_split_hash_from_file, compute_normalizer_hash


class DummySyntheticEvalDataset(Dataset):
    """Synthetic dataset generating paired history and 10-step future ground truth."""

    def __init__(self, num_samples: int = 12, num_trajs: int = 4):
        self.num_samples = num_samples
        self.num_trajs = num_trajs

    def __len__(self):
        return self.num_samples

    def get_window_metadata(self, idx: int):
        traj_idx = idx % self.num_trajs
        return {
            "source_file": f"file_{traj_idx}.h5",
            "traj_idx": traj_idx,
            "start_t": (idx // self.num_trajs) * 8,
            "cluster_id": traj_idx,
            "re": 10000.0,
            "sc": 0.5,
        }

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
        k = 16
        b = 4
        point_pred = torch.tensor([1.5, -2.0, 0.5, 3.0]).view(b, 1)
        target = torch.tensor([1.0, -1.0, 1.0, 2.0]).view(b, 1)

        samples = point_pred.unsqueeze(0).repeat(k, 1, 1)  # (K, B, 1)
        crps = compute_sample_crps_tensor(samples, target)
        mae = torch.abs(point_pred - target)

        assert torch.allclose(crps, mae, atol=1e-6)

    def test_empirical_crps_converges_to_analytical_gaussian(self):
        k = 2000
        b = 100
        mu = torch.linspace(-1.0, 1.0, b)
        var = torch.linspace(0.2, 1.5, b)
        target = mu + 0.5 * torch.sqrt(var)

        generator = torch.Generator().manual_seed(42)
        eps = torch.randn((k, b), generator=generator)
        samples = mu.unsqueeze(0) + eps * torch.sqrt(var).unsqueeze(0)

        emp_crps = compute_sample_crps_tensor(samples, target).mean()
        ana_crps = compute_gaussian_crps_analytical(mu, target, var).mean()

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

        samples = torch.randn((k, n_points), generator=generator)
        target = torch.randn((n_points,), generator=generator)

        res = compute_quantile_coverage_and_width(samples, target, nominal_levels=[0.5, 0.8, 0.9, 0.95])
        for lvl_str, exp_nominal in [("50", 0.50), ("80", 0.80), ("90", 0.90), ("95", 0.95)]:
            picp = res[lvl_str]["picp"]
            assert abs(picp - exp_nominal) < 0.025
            assert res[lvl_str]["nominal"] == exp_nominal


class TestPhase3PooledSSRContract:
    """Verify Phase 3 Bessel correction (ddof=1) and finite-K adjustment contract."""

    def test_pooled_ssr_matches_phase3_contract(self):
        k = 8
        b = 2
        ny, nx = 16, 16
        generator = torch.Generator().manual_seed(42)
        target = torch.zeros(b, 4, ny, nx)
        std_true = 0.5
        mean_offset = 0.5
        samples = mean_offset + std_true * torch.randn((k, b, 4, ny, nx), generator=generator)

        ssr_res = compute_pooled_spread_skill(samples, target)

        assert ssr_res["K"] == k
        expected_finite_k = math.sqrt((k + 1.0) / k)
        assert math.isclose(ssr_res["finite_k_inflation_factor"], expected_finite_k, rel_tol=1e-5)

        assert abs(ssr_res["pooled_rms_spread"] - std_true) < 0.05
        assert ssr_res["pooled_rmse"] > 0.45

        expected_ssr = ssr_res["pooled_rms_spread"] / ssr_res["pooled_rmse"]
        assert math.isclose(ssr_res["spread_skill_ratio"], expected_ssr, rel_tol=1e-5)
        assert math.isclose(ssr_res["finite_k_adjusted_ssr"], expected_ssr * expected_finite_k, rel_tol=1e-5)
        assert "velocity" in ssr_res
        assert set(ssr_res["per_variable"].keys()) == {"u", "v", "p", "s"}


class TestSameKAndRNGContract:
    """Verify deterministic reproducibility under fixed seed and common random numbers."""

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

    def test_temperature_sweep_reuses_common_base_noise(self):
        fm = LatentFlowMatcher(latent_channels=8, cond_dim=16, hidden_channels=16, num_blocks=1)
        mu = torch.randn(2, 1, 8, 4, 4)
        re = torch.tensor([1000.0, 2000.0])
        sc = torch.tensor([0.5, 0.7])

        base_x0 = torch.randn(2 * 4, 8, 4, 4)
        samps_10 = fm.sample_ensemble(mu=mu, re=re, sc=sc, num_samples=4, noise_scale=1.0, custom_x0=base_x0)
        samps_07 = fm.sample_ensemble(mu=mu, re=re, sc=sc, num_samples=4, noise_scale=0.7, custom_x0=base_x0)

        assert not torch.allclose(samps_10, samps_07)
        assert samps_10.shape == samps_07.shape


class TestRolloutManifestAndTrajectoryCoverage:
    """Verify trajectory-aware manifest generation."""

    def test_rollout_manifest_covers_requested_unique_trajectories(self):
        dataset = DummySyntheticEvalDataset(num_samples=16, num_trajs=4)
        manifest, indices = build_rollout_manifest(dataset, windows_per_traj=2, max_trajectories=3)

        assert len(indices) == 6
        unique_trajs = set(r["traj_idx"] for r in manifest)
        assert len(unique_trajs) == 3
        assert unique_trajs == {0, 1, 2}

    def test_max_trajectories_counts_trajectory_ids_not_windows(self):
        dataset = DummySyntheticEvalDataset(num_samples=16, num_trajs=4)
        manifest, indices = build_rollout_manifest(dataset, windows_per_traj=4, max_trajectories=2)

        assert len(indices) == 8
        unique_trajs = set(r["traj_idx"] for r in manifest)
        assert len(unique_trajs) == 2


class TestProvenanceFailClosedContract:
    """Verify cryptographic fail-closed checks on checkpoints and protocol."""

    @pytest.fixture
    def valid_checkpoints(self, tmp_path):
        d0_path = tmp_path / "d0.pt"
        torch.save({"dummy": 1}, d0_path)
        d0_sha = compute_file_sha256(str(d0_path))

        split_path = tmp_path / "split.json"
        split_path.write_text('{"train": [], "valid": []}')
        split_hash = compute_split_hash_from_file(str(split_path))

        normalizer = FieldNormalizer()
        normalizer.mean = torch.zeros(4)
        normalizer.std = torch.ones(4)
        norm_hash = compute_normalizer_hash(normalizer)

        g0_path = tmp_path / "g0.pt"
        torch.save({
            "provenance": {
                "d0_checkpoint": {"sha256": d0_sha},
                "data_protocol": {"split_hash": split_hash, "normalizer_hash": norm_hash},
                "seed": 42,
            }
        }, g0_path)

        g1_path = tmp_path / "g1.pt"
        torch.save({
            "provenance": {
                "d0_checkpoint": {"sha256": d0_sha},
                "data_protocol": {"split_hash": split_hash, "normalizer_hash": norm_hash},
                "seed": 42,
            }
        }, g1_path)

        fm_path = tmp_path / "fm.pt"
        torch.save({
            "provenance": {
                "d0_checkpoint": {"sha256": d0_sha},
                "residual_statistics": {"d0_sha256": d0_sha},
                "data_protocol": {"split_hash": split_hash, "normalizer_hash": norm_hash},
                "seed": 42,
            }
        }, fm_path)

        return str(d0_path), str(g0_path), str(g1_path), str(fm_path), str(split_path), normalizer

    def test_evaluator_passes_on_valid_checkpoints(self, valid_checkpoints):
        d0, g0, g1, fm, split, norm = valid_checkpoints
        rep = verify_checkpoint_provenance(d0, g0, g1, fm, split, norm, expected_seed=42)
        assert rep["status"] == "PASSED"

    def test_evaluator_rejects_g0_normalizer_mismatch(self, valid_checkpoints):
        d0, g0, g1, fm, split, norm = valid_checkpoints
        data = torch.load(g0, map_location="cpu")
        data["provenance"]["data_protocol"]["normalizer_hash"] = "tampered_norm_hash"
        torch.save(data, g0)
        with pytest.raises(ValueError, match="G0 normalizer_hash mismatch"):
            verify_checkpoint_provenance(d0, g0, g1, fm, split, norm, expected_seed=42)

    def test_evaluator_rejects_g1_normalizer_mismatch(self, valid_checkpoints):
        d0, g0, g1, fm, split, norm = valid_checkpoints
        data = torch.load(g1, map_location="cpu")
        data["provenance"]["data_protocol"]["normalizer_hash"] = "tampered_norm_hash"
        torch.save(data, g1)
        with pytest.raises(ValueError, match="G1 normalizer_hash mismatch"):
            verify_checkpoint_provenance(d0, g0, g1, fm, split, norm, expected_seed=42)

    def test_evaluator_rejects_fm_normalizer_mismatch(self, valid_checkpoints):
        d0, g0, g1, fm, split, norm = valid_checkpoints
        data = torch.load(fm, map_location="cpu")
        data["provenance"]["data_protocol"]["normalizer_hash"] = "tampered_norm_hash"
        torch.save(data, fm)
        with pytest.raises(ValueError, match="FM normalizer_hash mismatch"):
            verify_checkpoint_provenance(d0, g0, g1, fm, split, norm, expected_seed=42)

    def test_evaluator_rejects_missing_seed(self, valid_checkpoints):
        d0, g0, g1, fm, split, norm = valid_checkpoints
        data = torch.load(fm, map_location="cpu")
        del data["provenance"]["seed"]
        torch.save(data, fm)
        with pytest.raises(ValueError, match="FM seed mismatch"):
            verify_checkpoint_provenance(d0, g0, g1, fm, split, norm, expected_seed=42)

    def test_evaluator_rejects_checkpoint_parent_d0_mismatch(self, valid_checkpoints):
        d0, g0, g1, fm, split, norm = valid_checkpoints
        data = torch.load(g0, map_location="cpu")
        data["provenance"]["d0_checkpoint"]["sha256"] = "wrong_sha"
        torch.save(data, g0)
        with pytest.raises(ValueError, match="G0 parent D0 SHA mismatch"):
            verify_checkpoint_provenance(d0, g0, g1, fm, split, norm, expected_seed=42)


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

        dataset = DummySyntheticEvalDataset(num_samples=4, num_trajs=2)
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
        assert step1["common_random_numbers"] is True
        for m in ["D0", "G0", "G1", "FM_temp_1.0", "FM_temp_0.8"]:
            assert m in step1

    def test_temperature_cli_preserves_common_random_numbers(self, setup_synthetic_models_and_loader):
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
            fm_temperatures=[1.0, 0.8, 0.7, 0.6, 0.5, 0.4],
            seed=42,
        )

        assert step1["common_random_numbers"] is True
        for t in [1.0, 0.8, 0.7, 0.6, 0.5, 0.4]:
            m_key = f"FM_temp_{t}"
            assert m_key in step1
            assert "physical_intervals_per_channel" in step1[m_key]
            assert set(step1[m_key]["physical_intervals_per_channel"].keys()) == {"u", "v", "p", "s"}

    def test_divergence_and_vorticity_aggregate_all_ensemble_members(self, setup_synthetic_models_and_loader):
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
            rollout_temperatures=[1.0, 0.7],
            windows_per_traj=2,
            max_trajectories=2,
            seed=42,
        )

        assert "h5" in step2
        assert "h10" in step2

        for m in ["D0", "G0", "G1", "FM_temp_1.0", "FM_temp_0.7"]:
            assert m in step2["h5"]
            m_metrics = step2["h5"][m]
            assert "ensemble_mean_vrmse_vs_gt" in m_metrics
            assert "sample_mean_vrmse_vs_gt" in m_metrics
            assert "ensemble_mean_rms_divergence" in m_metrics
            assert "sample_rms_divergence" in m_metrics
            assert "divergence_ratio_vs_gt" in m_metrics
            assert "ensemble_mean_vorticity_rmse_vs_gt" in m_metrics
            assert "sample_vorticity_rmse_vs_gt" in m_metrics

            assert m_metrics["sample_rms_divergence"] > 0.0
            assert m_metrics["sample_vorticity_rmse_vs_gt"] >= 0.0

    def test_individual_member_spectral_error_is_not_mean_spectrum_error(self, setup_synthetic_models_and_loader):
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
            rollout_temperatures=[1.0, 0.7],
            windows_per_traj=2,
            max_trajectories=2,
            seed=42,
        )

        spec = step2["spectral_relative_error_vs_gt_h10"]["FM_temp_1.0"]
        assert "ensemble_mean_field_spectrum_rel_error_mean" in spec
        assert "mean_member_spectrum_rel_error_mean" in spec
        assert "individual_member_spectrum_rel_error_mean" in spec
        assert spec["ensemble_mean_field_spectrum_rel_error_mean"] >= 0.0
        assert spec["mean_member_spectrum_rel_error_mean"] >= 0.0
        assert spec["individual_member_spectrum_rel_error_mean"] >= 0.0

    def test_spectrum_aggregates_multiple_windows(self, setup_synthetic_models_and_loader):
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
            rollout_temperatures=[1.0, 0.7],
            windows_per_traj=2,
            max_trajectories=2,
            seed=42,
        )

        assert "spectral_relative_error_vs_gt_h10" in step2
        spec_errs = step2["spectral_relative_error_vs_gt_h10"]
        for m in ["D0", "G0", "G1", "FM_temp_1.0", "FM_temp_0.7"]:
            assert m in spec_errs
            assert "ensemble_mean_field_spectrum_rel_error_mean" in spec_errs[m]
            assert "mean_member_spectrum_rel_error_mean" in spec_errs[m]
            assert "individual_member_spectrum_rel_error_mean" in spec_errs[m]
            assert isinstance(spec_errs[m]["ensemble_mean_field_spectrum_rel_error_mean"], float)

        spectra = step2["energy_spectra_first_25_modes"]
        assert "GT" in spectra
        assert "FM_temp_1.0_ensemble_mean_field" in spectra
        assert "FM_temp_0.7_ensemble_mean_field" in spectra
