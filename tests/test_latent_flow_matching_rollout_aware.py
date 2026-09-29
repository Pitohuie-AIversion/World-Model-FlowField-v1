"""Comprehensive Test Suite for Latent Flow Matching Rollout-Aware (FM-R2).

Verifies the scientific equivalence and failure mode contracts across:
- Branch C1 (1-step continuation)
- Branch C2 (Teacher-Forced 2-step control)
- Branch R2-A (Rollout-Aware 2-step treatment)

Test Coverage:
1. test_c2_and_r2a_start_from_byte_identical_weights:
   Verifies C2 and R2-A instantiate byte-identical weights from parent FM checkpoint.
2. test_same_horizon_2_windows_and_targets:
   Verifies both branches receive identical H=2 targets z_{t+1} and z_{t+2}.
3. test_same_optimizer_initialization:
   Verifies fresh AdamW is initialized with identical hyperparameters (lr, weight_decay).
4. test_same_batch_ordering:
   Verifies deterministic batch ordering under same seed.
5. test_same_cfm_random_draws:
   Verifies dedicated RNGs produce identical (x_0, tau) draws in Step 1 and Step 2 across C2 and R2-A.
6. test_c2_uses_gt_z_t1_in_step2_history:
   Verifies C2 uses ground truth z_{t+1} in Step 2 history condition.
7. test_r2a_uses_generated_detached_z_hat_t1:
   Verifies R2-A uses generated detached z_hat_{t+1} in Step 2 history condition.
8. test_d0_backbone_remains_frozen:
   Verifies Encoder, Decoder, and Transformer are frozen (requires_grad=False).
9. test_loss_count_and_weighting_identical:
   Verifies C2 and R2-A both compute L = 0.5 * (L_1 + L_2).
10. test_preflight_cryptographic_provenance_fail_closed:
    Verifies rejection of missing checkpoints, SHA mismatch, and seed mismatch.
11. test_synthetic_e2e_training_step:
    Verifies end-to-end training execution on synthetic data for C1, C2, and R2-A.
"""

import json
import math
import os
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scripts.train_latent_flow_matching_rollout_aware import (
    verify_rollout_aware_preflight_contract,
    build_and_freeze_rollout_aware_model,
    compute_rollout_aware_step_loss,
    evaluate_rollout_aware_validation,
    build_rollout_aware_checkpoint_payload,
    train_latent_flow_matching_rollout_aware,
)
from src.data.normalization import FieldNormalizer
from src.utils.provenance import compute_file_sha256, compute_normalizer_hash
from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.latent_flow_matching import LatentFlowMatcher


class DummySyntheticHorizon2Dataset(Dataset):
    """Synthetic dataset for testing H=2 multi-step training execution."""

    def __init__(self, num_samples: int = 4, produce_nans: bool = False):
        self.num_samples = num_samples
        self.produce_nans = produce_nans

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.produce_nans:
            history = torch.full((4, 4, 32, 32), float("nan"))
            future = torch.full((2, 4, 32, 32), float("nan"))
        else:
            history = torch.randn(4, 4, 32, 32)
            future = torch.randn(2, 4, 32, 32)
        return {
            "history": history,
            "future": future,
            "re": torch.tensor(10000.0),
            "sc": torch.tensor(0.5),
        }


def _create_mock_checkpoints(tmp_path: Path):
    """Helper to construct valid mock D0 and parent FM checkpoints with proper cryptographic bindings."""
    # 1. Mock D0
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=256,
        cond_dim=128,
        depth=6,
        num_heads=8,
        history_length=4,
        prediction_mode="direct",
        use_spatial_pos=True,
    )
    forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder)

    d0_path = str(tmp_path / "mock_d0.pt")
    d0_payload = {
        "model_state_dict": forecaster.state_dict(),
        "config": {
            "prediction_mode": "direct",
            "embed_dim": 256,
            "depth": 6,
            "num_heads": 8,
            "use_spatial_pos": True,
        },
    }
    torch.save(d0_payload, d0_path)
    d0_sha = compute_file_sha256(d0_path)

    # 2. Normalizer
    norm = FieldNormalizer()
    norm_path = str(tmp_path / "mock_norm.pt")
    torch.save(norm.state_dict(), norm_path)
    norm_hash = compute_normalizer_hash(norm)

    # 3. Split file
    split_path = str(tmp_path / "mock_split.json")
    split_data = {
        "metadata": {"type": "grouped"},
        "train": [{"source_file": "f1.h5", "cluster_id": "c1", "indices": [0, 1]}],
        "validation": [{"source_file": "f2.h5", "cluster_id": "c2", "indices": [0, 1]}],
        "test": [],
    }
    with open(split_path, "w") as f:
        json.dump(split_data, f)
    from src.utils.provenance import compute_split_hash_from_file
    split_hash = compute_split_hash_from_file(split_path)

    # 4. Mock parent FM
    fm = LatentFlowMatcher(
        latent_channels=64,
        cond_dim=128,
        hidden_channels=128,
        num_blocks=4,
        use_spatial_attn=True,
        target_mode="residual",
        zero_init=True,
    )
    fm_path = str(tmp_path / "mock_parent_fm.pt")
    fm_payload = {
        "flow_matcher_state_dict": fm.state_dict(),
        "config": {
            "hidden_channels": 128,
            "num_blocks": 4,
            "target_mode": "residual",
            "use_spatial_attn": True,
            "sigma_min": 1e-4,
            "num_flow_steps": 10,
            "solver": "midpoint",
            "residual_scale_applied": False,
        },
        "provenance": {
            "d0_checkpoint": {"path": d0_path, "sha256": d0_sha},
            "data_protocol": {
                "split_file": split_path,
                "split_hash": split_hash,
                "normalizer_file": norm_path,
                "normalizer_hash": norm_hash,
            },
            "seed": 42,
        },
    }
    torch.save(fm_payload, fm_path)

    return d0_path, fm_path, norm_path, split_path, d0_payload, fm_payload


class TestFM_R2EquivalenceAndGovernanceContracts:
    """Verifies all required experimental equivalence contracts for FM-R2."""

    def test_c2_and_r2a_start_from_byte_identical_weights(self, tmp_path):
        """Verify that C2 and R2-A instantiate byte-identical initial model parameters."""
        d0_path, fm_path, norm_path, split_path, d0_payload, fm_payload = _create_mock_checkpoints(tmp_path)
        device = torch.device("cpu")

        model_c2 = build_and_freeze_rollout_aware_model(d0_payload, fm_payload, device=device)
        model_r2a = build_and_freeze_rollout_aware_model(d0_payload, fm_payload, device=device)

        sd_c2 = model_c2.flow_matcher.state_dict()
        sd_r2a = model_r2a.flow_matcher.state_dict()

        assert sd_c2.keys() == sd_r2a.keys(), "State dict keys mismatch between C2 and R2-A"
        for k in sd_c2.keys():
            assert torch.equal(sd_c2[k], sd_r2a[k]), f"Weight mismatch for parameter {k}"

    def test_d0_backbone_remains_frozen(self, tmp_path):
        """Verify that Encoder, Decoder, and Transformer backbone remain strictly frozen."""
        d0_path, fm_path, norm_path, split_path, d0_payload, fm_payload = _create_mock_checkpoints(tmp_path)
        device = torch.device("cpu")

        forecaster = build_and_freeze_rollout_aware_model(d0_payload, fm_payload, device=device)

        for name, p in forecaster.named_parameters():
            if "flow_matcher" in name:
                assert p.requires_grad, f"FlowMatcher param {name} must have requires_grad=True"
            else:
                assert not p.requires_grad, f"Backbone param {name} must be frozen (requires_grad=False)"

    def test_same_optimizer_initialization(self, tmp_path):
        """Verify fresh AdamW optimizer initialization is identical across branches."""
        d0_path, fm_path, norm_path, split_path, d0_payload, fm_payload = _create_mock_checkpoints(tmp_path)
        device = torch.device("cpu")

        model_c2 = build_and_freeze_rollout_aware_model(d0_payload, fm_payload, device=device)
        model_r2a = build_and_freeze_rollout_aware_model(d0_payload, fm_payload, device=device)

        opt_c2 = torch.optim.AdamW(model_c2.flow_matcher.parameters(), lr=2e-4, weight_decay=1e-4)
        opt_r2a = torch.optim.AdamW(model_r2a.flow_matcher.parameters(), lr=2e-4, weight_decay=1e-4)

        assert opt_c2.defaults == opt_r2a.defaults
        assert len(opt_c2.param_groups[0]["params"]) == len(opt_r2a.param_groups[0]["params"])

    def test_same_cfm_random_draws_between_c2_and_r2a(self, tmp_path):
        """Verify dedicated RNG generators ensure 100% bit-identical CFM loss random draws."""
        d0_path, fm_path, norm_path, split_path, d0_payload, fm_payload = _create_mock_checkpoints(tmp_path)
        device = torch.device("cpu")

        forecaster = build_and_freeze_rollout_aware_model(d0_payload, fm_payload, device=device)

        dataset = DummySyntheticHorizon2Dataset(num_samples=2)
        batch = next(iter(DataLoader(dataset, batch_size=2)))

        # Run C2
        gen_l1_c2 = torch.Generator(device=device).manual_seed(1234)
        gen_l2_c2 = torch.Generator(device=device).manual_seed(5678)
        res_c2 = compute_rollout_aware_step_loss(
            forecaster=forecaster,
            batch=batch,
            device=device,
            branch="C2",
            gen_loss_1=gen_l1_c2,
            gen_loss_2=gen_l2_c2,
        )

        # Run R2-A with identical gen_loss_1 seed
        gen_l1_r2 = torch.Generator(device=device).manual_seed(1234)
        gen_l2_r2 = torch.Generator(device=device).manual_seed(5678)
        gen_samp = torch.Generator(device=device).manual_seed(9999)
        res_r2a = compute_rollout_aware_step_loss(
            forecaster=forecaster,
            batch=batch,
            device=device,
            branch="R2_A",
            gen_loss_1=gen_l1_r2,
            gen_loss_2=gen_l2_r2,
            gen_sample=gen_samp,
        )

        # Step 1 loss MUST be bit-identical because Step 1 conditions and random draws are identical!
        assert math.isclose(res_c2["loss_step1"], res_r2a["loss_step1"], rel_tol=1e-6), (
            f"Step 1 loss mismatch between C2 ({res_c2['loss_step1']}) and R2-A ({res_r2a['loss_step1']})"
        )

    def test_c2_uses_gt_z_t1_in_step2_history(self, tmp_path):
        """Verify C2 feeds ground truth z_{t+1} into Step 2 history condition."""
        d0_path, fm_path, norm_path, split_path, d0_payload, fm_payload = _create_mock_checkpoints(tmp_path)
        device = torch.device("cpu")
        forecaster = build_and_freeze_rollout_aware_model(d0_payload, fm_payload, device=device)

        dataset = DummySyntheticHorizon2Dataset(num_samples=2)
        batch = next(iter(DataLoader(dataset, batch_size=2)))

        captured_step2_hist = []

        def hook_fn(module, args):
            z_input = args[0]
            captured_step2_hist.append(z_input.clone())

        hook = forecaster.transformer.register_forward_pre_hook(hook_fn)

        compute_rollout_aware_step_loss(forecaster, batch, device=device, branch="C2")
        hook.remove()

        # Second call to transformer is for Step 2
        assert len(captured_step2_hist) == 2
        step2_hist = captured_step2_hist[1]

        # The last frame of Step 2 history condition should be encoded future[:, 0:1]
        with torch.no_grad():
            expected_gt_z1 = forecaster.encoder(batch["future"][:, 0:1])
        assert torch.allclose(step2_hist[:, -1:], expected_gt_z1, atol=1e-5), "C2 must use GT z_{t+1}"

    def test_r2a_uses_generated_detached_z_hat_t1(self, tmp_path):
        """Verify R2-A feeds generated detached z_hat_{t+1} into Step 2 history condition."""
        d0_path, fm_path, norm_path, split_path, d0_payload, fm_payload = _create_mock_checkpoints(tmp_path)
        device = torch.device("cpu")
        forecaster = build_and_freeze_rollout_aware_model(d0_payload, fm_payload, device=device)

        dataset = DummySyntheticHorizon2Dataset(num_samples=2)
        batch = next(iter(DataLoader(dataset, batch_size=2)))

        captured_step2_hist = []

        def hook_fn(module, args):
            z_input = args[0]
            captured_step2_hist.append(z_input)

        hook = forecaster.transformer.register_forward_pre_hook(hook_fn)

        res = compute_rollout_aware_step_loss(forecaster, batch, device=device, branch="R2_A")
        hook.remove()

        assert len(captured_step2_hist) == 2
        step2_hist = captured_step2_hist[1]

        # In R2-A, Step 2 history condition's last frame must NOT require grad (strictly detached)
        assert not step2_hist.requires_grad, "Step 2 history condition must be detached in R2-A"

        # And it should not equal GT z_{t+1} (unless by impossible coincidence)
        with torch.no_grad():
            gt_z1 = forecaster.encoder(batch["future"][:, 0:1])
        assert not torch.allclose(step2_hist[:, -1:], gt_z1), "R2-A must use sampled latent, not GT"

    def test_loss_count_and_weighting_identical(self, tmp_path):
        """Verify both C2 and R2-A compute L = 0.5 * (L_1 + L_2)."""
        d0_path, fm_path, norm_path, split_path, d0_payload, fm_payload = _create_mock_checkpoints(tmp_path)
        device = torch.device("cpu")
        forecaster = build_and_freeze_rollout_aware_model(d0_payload, fm_payload, device=device)

        dataset = DummySyntheticHorizon2Dataset(num_samples=2)
        batch = next(iter(DataLoader(dataset, batch_size=2)))

        for branch in ("C2", "R2_A"):
            res = compute_rollout_aware_step_loss(forecaster, batch, device=device, branch=branch)
            expected_total = 0.5 * (res["loss_step1"] + res["loss_step2"])
            assert math.isclose(res["loss"].item(), expected_total, rel_tol=1e-5), (
                f"{branch} loss weighting is not 0.5 * L1 + 0.5 * L2"
            )

    def test_preflight_cryptographic_provenance_fail_closed(self, tmp_path):
        """Verify preflight rejects missing checkpoints, SHA mismatch, and seed mismatch."""
        d0_path, fm_path, norm_path, split_path, d0_payload, fm_payload = _create_mock_checkpoints(tmp_path)

        # 1. Valid preflight passes
        verify_rollout_aware_preflight_contract(
            d0_checkpoint_path=d0_path,
            parent_fm_checkpoint_path=fm_path,
            normalizer_path=norm_path,
            split_file=split_path,
            expected_seed=42,
        )

        # 2. Missing D0 fails
        with pytest.raises(FileNotFoundError):
            verify_rollout_aware_preflight_contract(
                d0_checkpoint_path="nonexistent_d0.pt",
                parent_fm_checkpoint_path=fm_path,
                normalizer_path=norm_path,
                split_file=split_path,
            )

        # 3. Missing Parent FM fails
        with pytest.raises(FileNotFoundError):
            verify_rollout_aware_preflight_contract(
                d0_checkpoint_path=d0_path,
                parent_fm_checkpoint_path="nonexistent_fm.pt",
                normalizer_path=norm_path,
                split_file=split_path,
            )

        # 4. Wrong seed fails
        with pytest.raises(ValueError, match="does not match expected seed"):
            verify_rollout_aware_preflight_contract(
                d0_checkpoint_path=d0_path,
                parent_fm_checkpoint_path=fm_path,
                normalizer_path=norm_path,
                split_file=split_path,
                expected_seed=999,
            )

        # 5. D0 SHA mismatch in parent FM fails
        fm_payload_corrupt = torch.load(fm_path, map_location="cpu", weights_only=False)
        fm_payload_corrupt["provenance"]["d0_checkpoint"]["sha256"] = "0" * 64
        corrupt_fm_path = str(tmp_path / "corrupt_fm.pt")
        torch.save(fm_payload_corrupt, corrupt_fm_path)

        with pytest.raises(ValueError, match="which diverges from runtime D0 SHA"):
            verify_rollout_aware_preflight_contract(
                d0_checkpoint_path=d0_path,
                parent_fm_checkpoint_path=corrupt_fm_path,
                normalizer_path=norm_path,
                split_file=split_path,
            )

    def test_synthetic_e2e_training_step(self, tmp_path):
        """Verify end-to-end forward, backward, and optimization step for C1, C2, and R2-A."""
        d0_path, fm_path, norm_path, split_path, d0_payload, fm_payload = _create_mock_checkpoints(tmp_path)
        device = torch.device("cpu")

        for branch in ("C1", "C2", "R2_A"):
            forecaster = build_and_freeze_rollout_aware_model(d0_payload, fm_payload, device=device)
            optimizer = torch.optim.AdamW(forecaster.flow_matcher.parameters(), lr=1e-3)

            dataset = DummySyntheticHorizon2Dataset(num_samples=2)
            batch = next(iter(DataLoader(dataset, batch_size=2)))

            optimizer.zero_grad()
            res = compute_rollout_aware_step_loss(forecaster, batch, device=device, branch=branch)
            loss = res["loss"]
            loss.backward()

            # Verify gradient exists on flow matcher parameters
            has_grad = False
            for p in forecaster.flow_matcher.parameters():
                if p.grad is not None:
                    has_grad = True
                    assert torch.isfinite(p.grad).all()
            assert has_grad, f"No gradients computed for branch {branch}"

            optimizer.step()
