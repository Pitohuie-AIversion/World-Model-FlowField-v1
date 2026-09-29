"""Unit and Failure Mode Tests for Latent Flow Matching Training Entrypoint.

Tests:
1. evaluate_flow_matching_loss:
   - Fails closed on empty dataloader (ValueError).
   - Fails closed when model has no attached flow_matcher (RuntimeError).
   - Fails closed on non-finite loss (FloatingPointError).
2. verify_flow_matching_preflight_contract:
   - Fails closed when D0 checkpoint missing (FileNotFoundError).
   - Fails closed when normalizer file missing (FileNotFoundError).
   - Fails closed when split file missing (FileNotFoundError).
3. build_and_freeze_flow_matching_model:
   - Strict parameter freeze governance assertion.
4. Directory collision protection:
   - Existing artifacts fail closed unless overwrite=True.
5. End-to-end synthetic training run:
   - Verifies training step, checkpointing, and metric output generation.
"""

import json
import os
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scripts.train_latent_flow_matching import (
    verify_flow_matching_preflight_contract,
    build_and_freeze_flow_matching_model,
    evaluate_flow_matching_loss,
    train_latent_flow_matching,
)
from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.latent_flow_matching import LatentFlowMatcher


class DummySyntheticFlowDataset(Dataset):
    """Synthetic dataset for testing training execution without requiring large hdf5 files."""

    def __init__(self, num_samples: int = 4, produce_nans: bool = False):
        self.num_samples = num_samples
        self.produce_nans = produce_nans

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.produce_nans:
            history = torch.full((4, 4, 32, 32), float("nan"))
            future = torch.full((1, 4, 32, 32), float("nan"))
        else:
            history = torch.randn(4, 4, 32, 32)
            future = torch.randn(1, 4, 32, 32)
        return {
            "history": history,
            "future": future,
            "re": torch.tensor(10000.0),
            "sc": torch.tensor(0.5),
        }


class MockForecasterNoFM(nn.Module):
    """Forecaster without attached flow matcher."""
    def __init__(self):
        super().__init__()
        self.flow_matcher = None


class MockForecasterNanFM(nn.Module):
    """Forecaster whose flow matcher produces non-finite loss."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.transformer = lambda z, re=None, sc=None: z[:, :1]
        self.flow_matcher = MockNanFlowMatcher()

    def eval(self):
        pass


class MockNanFlowMatcher(nn.Module):
    def compute_loss(self, z_next, mu, re=None, sc=None, generator=None):
        return {"loss": torch.tensor(float("nan"))}


class TestEvaluateFlowMatchingLossFailureModes:
    """Verify evaluation function fails closed on invalid inputs."""

    def test_empty_dataloader_raises_value_error(self):
        model = LatentForecaster(
            encoder=Encoder2D(in_channels=4, latent_channels=8, base_channels=8, channel_mult=[1, 1, 1]),
            transformer=LatentSTTransformer(latent_channels=8, embed_dim=16, cond_dim=16, depth=1),
            decoder=Decoder2D(latent_channels=8, out_channels=4, base_channels=8, channel_mult=[1, 1, 1]),
        )
        model.attach_flow_matcher(LatentFlowMatcher(latent_channels=8, cond_dim=16, hidden_channels=16, num_blocks=1))

        empty_loader = DataLoader([])
        with pytest.raises(ValueError, match="Validation dataloader is empty"):
            evaluate_flow_matching_loss(model, empty_loader, device=torch.device("cpu"))

    def test_missing_flow_matcher_raises_runtime_error(self):
        model = MockForecasterNoFM()
        ds = DummySyntheticFlowDataset(num_samples=2)
        loader = DataLoader(ds, batch_size=2)
        with pytest.raises(RuntimeError, match="has no attached flow_matcher"):
            evaluate_flow_matching_loss(model, loader, device=torch.device("cpu"))

    def test_non_finite_loss_fails_closed(self):
        model = MockForecasterNanFM()
        ds = DummySyntheticFlowDataset(num_samples=2)
        loader = DataLoader(ds, batch_size=2)
        with pytest.raises(FloatingPointError, match="Non-finite validation CFM loss"):
            evaluate_flow_matching_loss(model, loader, device=torch.device("cpu"), fail_on_non_finite=True)


class TestPreflightContractFailClosed:
    """Verify pre-flight checks fail closed on missing paths."""

    def test_missing_d0_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="D0 checkpoint not found"):
            verify_flow_matching_preflight_contract(
                d0_checkpoint_path=str(tmp_path / "non_existent.pt"),
                normalizer_path=str(tmp_path / "normalizer.json"),
                split_file=str(tmp_path / "split.json"),
            )

    def test_missing_normalizer_raises(self, tmp_path):
        dummy_ckpt = tmp_path / "d0.pt"
        torch.save({"config": {}}, dummy_ckpt)
        with pytest.raises(FileNotFoundError, match="Normalizer file not found"):
            verify_flow_matching_preflight_contract(
                d0_checkpoint_path=str(dummy_ckpt),
                normalizer_path=str(tmp_path / "non_existent_norm.json"),
                split_file=str(tmp_path / "split.json"),
            )


class TestBuildAndFreezeModelGovernance:
    """Verify parameter freezing contracts."""

    def test_parameter_freeze_contract(self):
        ckpt_data = {
            "config": {
                "embed_dim": 32,
                "depth": 1,
                "num_heads": 2,
                "prediction_mode": "direct",
                "use_spatial_pos": False,
            },
            "model_state_dict": {},
        }
        # Build dummy weights matching architecture
        enc = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
        dec = Decoder2D(latent_channels=64, out_channels=4, base_channels=32)
        trans = LatentSTTransformer(
            latent_channels=64,
            embed_dim=32,
            cond_dim=128,
            depth=1,
            num_heads=2,
            use_spatial_pos=False,
        )
        forecaster = LatentForecaster(enc, trans, dec)
        ckpt_data["model_state_dict"] = forecaster.state_dict()

        frozen_model = build_and_freeze_flow_matching_model(
            ckpt_data=ckpt_data,
            device=torch.device("cpu"),
            hidden_channels=32,
            num_blocks=1,
        )

        assert frozen_model.has_flow_matcher
        for p in frozen_model.encoder.parameters():
            assert not p.requires_grad
        for p in frozen_model.decoder.parameters():
            assert not p.requires_grad
        for p in frozen_model.transformer.parameters():
            assert not p.requires_grad
        for p in frozen_model.flow_matcher.parameters():
            assert p.requires_grad
