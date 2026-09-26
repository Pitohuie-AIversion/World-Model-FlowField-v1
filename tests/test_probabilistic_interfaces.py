"""Comprehensive Unit and Contract Tests for ProbLatent-R1 Phase 1 Interfaces.

Validates:
1. VarianceHead2D:
   - Output shape (B, 1, C_z, H_z, W_z).
   - Strict positivity and variance floor enforcement (variance >= epsilon).
   - Alignment with empirical G0 baseline via compute_g1_bias_init_from_g0.
   - Gradient flow to variance head parameters.
2. LatentSTTransformer:
   - Deterministic forward bit-wise invariance whether variance head is attached or not.
   - predict_distribution produces exact mu matching forward, plus strictly positive variance.
   - Fail-closed behavior when predict_distribution is called without an attached variance head.
3. LatentForecaster Probabilistic Rollout:
   - sample_rollout returns complete dictionary with correct dimensions for (B, K, H, ...).
   - Reproducibility: identical seed produces bit-wise identical sample trajectories.
   - Diversity: different seeds produce diverse sample trajectories (sample variance > 0).
   - Independent history buffers: no cross-sample contamination across the K rollouts.
4. Latent Space Gaussian NLL Loss:
   - Finite loss values and correct gradient propagation.
"""

from typing import Dict, Optional, Tuple
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.latent_transformer import LatentSTTransformer
from src.models.latent_forecaster import LatentForecaster
from src.models.probabilistic_latent_dynamics import (
    VarianceHead2D,
    sample_next_latent,
    gaussian_nll_latent_loss,
)


class TestVarianceHead2DContract:
    """Verify VarianceHead2D architecture, numerical boundaries, and G0 initialization."""

    def test_variance_head_output_shape(self):
        b, n, embed_dim = 2, 64, 32
        hz, wz = 8, 8
        c_z = 16
        head = VarianceHead2D(embed_dim=embed_dim, latent_channels=c_z, variance_floor=1e-4)

        features = torch.randn(b, 1, n, embed_dim)
        var = head(features, h_z=hz, w_z=wz)

        assert var.shape == (b, 1, c_z, hz, wz)
        assert (var >= 1e-4).all()

    def test_variance_floor_strict_positivity_under_extreme_inputs(self):
        head = VarianceHead2D(embed_dim=32, latent_channels=8, variance_floor=1e-4)
        # Drive raw linear output to large negative values via negative bias
        with torch.no_grad():
            head.linear.bias.fill_(-100.0)
        extreme_features = torch.full((2, 1, 16, 32), 0.0)
        var = head(extreme_features, h_z=4, w_z=4)

        # Must never drop below variance_floor
        assert not torch.isnan(var).any()
        assert not torch.isinf(var).any()
        assert (var >= 1e-4).all()
        # softplus(-100) -> 0, so var -> variance_floor
        assert torch.allclose(var, torch.full_like(var, 1e-4), atol=1e-6)

    def test_variance_head_initialization_from_g0_matches_empirically(self):
        embed_dim = 32
        c_z = 8
        hz, wz = 4, 4
        head = VarianceHead2D(embed_dim=embed_dim, latent_channels=c_z, variance_floor=1e-4)

        v_g0 = torch.tensor([0.05, 0.1, 0.2, 0.5, 0.001, 1e-5, 0.0, 1.2], dtype=torch.float32)
        effective_g0 = head.initialize_from_g0(v_g0=v_g0, min_margin=1e-5)

        # Before any training, features of arbitrary magnitude should produce effective_g0
        features = torch.randn(4, 1, hz * wz, embed_dim)
        pred_var = head(features, h_z=hz, w_z=wz)

        # Across all batch items, spatial tokens, and channels:
        for c in range(c_z):
            channel_var = pred_var[:, :, c, :, :]
            target_val = effective_g0[c].item()
            assert torch.allclose(channel_var, torch.full_like(channel_var, target_val), atol=1e-5)

    def test_variance_head_gradient_flow(self):
        head = VarianceHead2D(embed_dim=16, latent_channels=4, variance_floor=1e-4)
        features = torch.randn(2, 1, 16, 16, requires_grad=True)
        var = head(features, h_z=4, w_z=4)
        loss = var.sum()
        loss.backward()

        assert head.linear.weight.grad is not None
        assert head.linear.bias.grad is not None
        assert not torch.isnan(head.linear.weight.grad).any()
        assert not torch.isnan(head.linear.bias.grad).any()


class TestLatentSTTransformerProbabilisticInterface:
    """Verify LatentSTTransformer deterministic invariance and distribution prediction."""

    @pytest.fixture
    def transformer(self):
        return LatentSTTransformer(
            latent_channels=8,
            embed_dim=16,
            cond_dim=8,
            depth=1,
            num_heads=2,
            history_length=2,
            prediction_mode="direct",
            use_spatial_pos=False,
        )

    def test_deterministic_forward_bitwise_invariance_with_or_without_variance_head(self, transformer):
        z_hist = torch.randn(2, 2, 8, 4, 4)
        re = torch.tensor([1000.0, 2000.0])
        sc = torch.tensor([0.5, 1.0])

        out_before = transformer(z_hist, re=re, sc=sc)

        # Attach variance head
        v_head = VarianceHead2D(embed_dim=16, latent_channels=8, variance_floor=1e-4)
        transformer.attach_variance_head(v_head)
        assert transformer.has_variance_head is True

        out_after = transformer(z_hist, re=re, sc=sc)

        # Deterministic forward must remain 100% bit-wise identical
        torch.testing.assert_close(out_before, out_after)

    def test_predict_distribution_matches_forward_mu(self, transformer):
        v_head = VarianceHead2D(embed_dim=16, latent_channels=8, variance_floor=1e-4)
        transformer.attach_variance_head(v_head)

        z_hist = torch.randn(2, 2, 8, 4, 4)
        re = torch.tensor([1000.0, 2000.0])
        sc = torch.tensor([0.5, 1.0])

        mu_det = transformer(z_hist, re=re, sc=sc)
        mu_prob, var_prob = transformer.predict_distribution(z_hist, re=re, sc=sc)

        torch.testing.assert_close(mu_det, mu_prob)
        assert var_prob.shape == mu_prob.shape
        assert (var_prob >= 1e-4).all()

    def test_predict_distribution_fails_closed_when_variance_head_missing(self, transformer):
        assert transformer.has_variance_head is False
        z_hist = torch.randn(2, 2, 8, 4, 4)

        with pytest.raises(RuntimeError, match="No variance head attached"):
            transformer.predict_distribution(z_hist)


class TestLatentForecasterSampleRollout:
    """Verify single-source multi-trajectory probabilistic rollout in LatentForecaster."""

    @pytest.fixture
    def forecaster(self):
        enc = Encoder2D(in_channels=4, latent_channels=8, base_channels=8)
        dec = Decoder2D(latent_channels=8, out_channels=4, base_channels=8)
        trans = LatentSTTransformer(
            latent_channels=8,
            embed_dim=16,
            cond_dim=8,
            depth=1,
            num_heads=2,
            history_length=2,
            prediction_mode="direct",
            use_spatial_pos=False,
        )
        head = VarianceHead2D(embed_dim=16, latent_channels=8, variance_floor=1e-4)
        trans.attach_variance_head(head)
        return LatentForecaster(encoder=enc, transformer=trans, decoder=dec)

    def test_sample_rollout_shapes_and_keys(self, forecaster):
        b, l, c_in, ny, nx = 2, 2, 4, 16, 16
        k = 4
        horizon = 3
        q_hist = torch.randn(b, l, c_in, ny, nx)

        results = forecaster.sample_rollout(
            q_hist=q_hist,
            horizon=horizon,
            num_samples=k,
            seed=42,
            decode_samples=True,
        )

        assert "deterministic_rollout" in results
        assert "sample_trajectories" in results
        assert "ensemble_mean" in results
        assert "latent_samples" in results
        assert "latent_variances" in results

        assert results["deterministic_rollout"].shape == (b, horizon, c_in, ny, nx)
        assert results["sample_trajectories"].shape == (b, k, horizon, c_in, ny, nx)
        assert results["ensemble_mean"].shape == (b, horizon, c_in, ny, nx)
        assert results["latent_samples"].shape == (b, k, horizon, 8, 2, 2)
        assert results["latent_variances"].shape == (b, k, horizon, 8, 2, 2)

    def test_sample_rollout_reproducibility_with_identical_seed(self, forecaster):
        q_hist = torch.randn(1, 2, 4, 16, 16)

        res1 = forecaster.sample_rollout(q_hist=q_hist, horizon=3, num_samples=3, seed=123)
        res2 = forecaster.sample_rollout(q_hist=q_hist, horizon=3, num_samples=3, seed=123)

        torch.testing.assert_close(res1["latent_samples"], res2["latent_samples"])
        torch.testing.assert_close(res1["sample_trajectories"], res2["sample_trajectories"])

    def test_sample_rollout_diversity_with_different_seeds(self, forecaster):
        q_hist = torch.randn(1, 2, 4, 16, 16)

        res1 = forecaster.sample_rollout(q_hist=q_hist, horizon=3, num_samples=3, seed=100)
        res2 = forecaster.sample_rollout(q_hist=q_hist, horizon=3, num_samples=3, seed=200)

        # Different seeds must produce different sample trajectories
        assert not torch.allclose(res1["latent_samples"], res2["latent_samples"])

    def test_controlled_noise_trajectory_isolation(self, forecaster):
        """Verify rigorously that altering path 0 noise does not alter path 1 at any horizon step."""
        q_hist = torch.randn(1, 2, 4, 16, 16)
        horizon = 3
        k = 2

        # 1. Baseline noise sequence for 3 steps, shape (B, K, 1, C_z, Hz, Wz) = (1, 2, 1, 8, 2, 2)
        torch.manual_seed(42)
        base_noise = [torch.randn(1, k, 1, 8, 2, 2) for _ in range(horizon)]

        res_base = forecaster.sample_rollout(
            q_hist=q_hist,
            horizon=horizon,
            num_samples=k,
            custom_noise_sequence=base_noise,
            decode_samples=True,
        )

        # 2. Perturbed noise sequence: ONLY alter Path 0 at Step 0, leave Path 1 identical at all steps
        pert_noise = [n.clone() for n in base_noise]
        pert_noise[0][0, 0] += 5.0  # Large perturbation to path 0 at step 0

        res_pert = forecaster.sample_rollout(
            q_hist=q_hist,
            horizon=horizon,
            num_samples=k,
            custom_noise_sequence=pert_noise,
            decode_samples=True,
        )

        # Path 0 must change because its noise changed
        assert not torch.allclose(res_pert["latent_samples"][:, 0], res_base["latent_samples"][:, 0])
        assert not torch.allclose(res_pert["sample_trajectories"][:, 0], res_base["sample_trajectories"][:, 0])

        # Path 1 must remain 100% bit-wise identical across all horizon steps (latent and physical)
        torch.testing.assert_close(res_pert["latent_samples"][:, 1], res_base["latent_samples"][:, 1], atol=0.0, rtol=0.0)
        torch.testing.assert_close(res_pert["sample_trajectories"][:, 1], res_base["sample_trajectories"][:, 1], atol=0.0, rtol=0.0)
        torch.testing.assert_close(res_pert["latent_variances"][:, 1], res_base["latent_variances"][:, 1], atol=0.0, rtol=0.0)

    def test_batch_and_conditioning_isolation_multibatch(self, forecaster):
        """Verify B>1 multi-batch and distinct Re/Sc conditioning maintains strict isolation."""
        b, l, c_in, ny, nx = 2, 2, 4, 16, 16
        k = 2
        horizon = 2

        q_hist = torch.randn(b, l, c_in, ny, nx)
        re = torch.tensor([1000.0, 5000.0])
        sc = torch.tensor([0.1, 1.0])

        torch.manual_seed(99)
        custom_noise = [torch.randn(b, k, 1, 8, 2, 2) for _ in range(horizon)]

        # Joint batch rollout (B=2)
        res_joint = forecaster.sample_rollout(
            q_hist=q_hist,
            re=re,
            sc=sc,
            horizon=horizon,
            num_samples=k,
            custom_noise_sequence=custom_noise,
            decode_samples=True,
        )

        # Isolated single-batch rollouts (B=1 each)
        noise_b0 = [n[0:1] for n in custom_noise]
        noise_b1 = [n[1:2] for n in custom_noise]

        res_b0 = forecaster.sample_rollout(
            q_hist=q_hist[0:1],
            re=re[0:1],
            sc=sc[0:1],
            horizon=horizon,
            num_samples=k,
            custom_noise_sequence=noise_b0,
            decode_samples=True,
        )

        res_b1 = forecaster.sample_rollout(
            q_hist=q_hist[1:2],
            re=re[1:2],
            sc=sc[1:2],
            horizon=horizon,
            num_samples=k,
            custom_noise_sequence=noise_b1,
            decode_samples=True,
        )

        # Batch 0 and Batch 1 results in joint run must exactly equal isolated runs
        torch.testing.assert_close(res_joint["latent_samples"][0:1], res_b0["latent_samples"], atol=0.0, rtol=0.0)
        torch.testing.assert_close(res_joint["latent_samples"][1:2], res_b1["latent_samples"], atol=0.0, rtol=0.0)
        torch.testing.assert_close(res_joint["sample_trajectories"][0:1], res_b0["sample_trajectories"], atol=0.0, rtol=0.0)
        torch.testing.assert_close(res_joint["sample_trajectories"][1:2], res_b1["sample_trajectories"], atol=0.0, rtol=0.0)


class TestStructuralParityAndGovernance:
    """Verify refactoring parity, freeze governance, and optimizer parameter updates."""

    def test_refactor_structural_parity_direct_and_residual_modes(self):
        """Verify forward_features refactoring preserves exact mathematical parity in all modes."""
        for pred_mode in ["direct", "residual"]:
            for use_pos in [True, False]:
                model = LatentSTTransformer(
                    latent_channels=8,
                    embed_dim=16,
                    cond_dim=8,
                    depth=1,
                    num_heads=2,
                    history_length=2,
                    prediction_mode=pred_mode,
                    use_spatial_pos=use_pos,
                )
                z_hist = torch.randn(2, 2, 8, 4, 4)
                re = torch.tensor([1000.0, 2000.0])
                sc = torch.tensor([0.2, 0.8])

                # 1. Standard forward
                mu_forward = model(z_hist, re=re, sc=sc)

                # 2. Manual evaluation of pipeline via forward_features
                features, (b, l, c_z, hz, wz) = model.forward_features(z_hist, re=re, sc=sc)
                delta = model.out_proj(features).view(b, 1, hz, wz, c_z).permute(0, 1, 4, 2, 3)
                expected_mu = z_hist[:, -1:] + delta if pred_mode == "residual" else delta

                # Must be 100% bit-wise identical
                assert torch.equal(mu_forward, expected_mu)

                # 3. Predict distribution mu
                v_head = VarianceHead2D(embed_dim=16, latent_channels=8)
                model.attach_variance_head(v_head)
                mu_dist, _ = model.predict_distribution(z_hist, re=re, sc=sc)
                assert torch.equal(mu_forward, mu_dist)

    def test_freeze_for_variance_training_and_optimizer_step(self):
        """Verify Phase 2 training step updates ONLY variance head while freezing all D0 parameters."""
        enc = Encoder2D(in_channels=4, latent_channels=8, base_channels=8)
        dec = Decoder2D(latent_channels=8, out_channels=4, base_channels=8)
        trans = LatentSTTransformer(
            latent_channels=8,
            embed_dim=16,
            cond_dim=8,
            depth=1,
            num_heads=2,
            history_length=2,
            prediction_mode="direct",
            use_spatial_pos=False,
        )
        head = VarianceHead2D(embed_dim=16, latent_channels=8, variance_floor=1e-4)
        trans.attach_variance_head(head)
        forecaster = LatentForecaster(encoder=enc, transformer=trans, decoder=dec)

        # Apply strict Phase 2 freezing contract
        forecaster.freeze_for_variance_training()

        # Check requires_grad flags
        assert all(not p.requires_grad for p in forecaster.encoder.parameters())
        assert all(not p.requires_grad for p in forecaster.decoder.parameters())
        assert not forecaster.transformer.in_proj.weight.requires_grad
        assert not forecaster.transformer.out_proj.weight.requires_grad
        assert all(p.requires_grad for p in forecaster.transformer.variance_head.parameters())

        # Save initial copies of weights
        init_enc_weight = forecaster.encoder.in_conv.weight.clone()
        init_dec_weight = forecaster.decoder.out_conv[2].weight.clone()
        init_trans_in_proj = forecaster.transformer.in_proj.weight.clone()
        init_trans_out_proj = forecaster.transformer.out_proj.weight.clone()
        init_vhead_weight = forecaster.transformer.variance_head.linear.weight.clone()

        # Train step with Gaussian NLL loss
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, forecaster.parameters()),
            lr=0.01,
        )
        optimizer.zero_grad()

        q_hist = torch.randn(2, 2, 4, 16, 16)
        target_z = torch.randn(2, 1, 8, 2, 2)
        mu, var = forecaster.predict_distribution_single_step(q_hist)
        loss = gaussian_nll_latent_loss(mu=mu, target=target_z, variance=var)
        loss.backward()

        # Verify gradients: variance head MUST have grad; frozen parts MUST have None
        assert forecaster.transformer.variance_head.linear.weight.grad is not None
        assert forecaster.transformer.variance_head.linear.bias.grad is not None
        assert forecaster.transformer.out_proj.weight.grad is None
        assert forecaster.transformer.in_proj.weight.grad is None
        assert forecaster.encoder.in_conv.weight.grad is None
        assert forecaster.decoder.out_conv[2].weight.grad is None

        optimizer.step()

        # Verify weight update: variance head updated; frozen parts remain bit-wise identical
        assert not torch.equal(forecaster.transformer.variance_head.linear.weight, init_vhead_weight)
        assert torch.equal(forecaster.encoder.in_conv.weight, init_enc_weight)
        assert torch.equal(forecaster.decoder.out_conv[2].weight, init_dec_weight)
        assert torch.equal(forecaster.transformer.in_proj.weight, init_trans_in_proj)
        assert torch.equal(forecaster.transformer.out_proj.weight, init_trans_out_proj)


class TestGaussianNLLLatentLoss:
    """Verify latent space Gaussian NLL loss implementation."""

    def test_gaussian_nll_latent_loss_decreases_as_mu_approaches_target(self):
        target = torch.randn(2, 1, 8, 4, 4)
        var = torch.full_like(target, 0.5)

        mu_far = target + 2.0
        mu_close = target + 0.1

        loss_far = gaussian_nll_latent_loss(mu=mu_far, target=target, variance=var)
        loss_close = gaussian_nll_latent_loss(mu=mu_close, target=target, variance=var)

        assert loss_close.item() < loss_far.item()
        assert not torch.isnan(loss_close)
        assert not torch.isinf(loss_close)
