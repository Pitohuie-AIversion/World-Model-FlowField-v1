"""Tests for torch.compile checkpoint round-trip compatibility.

Validates the critical contract:
    compiled training → save checkpoint → uncompiled evaluation → load (strict=True)

This is a P1 requirement: checkpoints produced during compiled training MUST be
loadable by standard (uncompiled) model instances without any key mismatches.
"""

import os
import tempfile
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from src.utils.checkpoint import (
    BestCheckpointTracker,
    load_checkpoint,
    save_checkpoint,
    strip_compiled_prefix,
    _COMPILE_PREFIX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal model for round-trip tests."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z.mean(dim=(-2, -1)))


def _make_compiled_state_dict(model: nn.Module) -> OrderedDict:
    """Simulate the _orig_mod. prefix that torch.compile adds to state_dict keys."""
    sd = model.state_dict()
    compiled_sd = OrderedDict()
    for k, v in sd.items():
        compiled_sd[f"{_COMPILE_PREFIX}{k}"] = v
    return compiled_sd


def _make_nested_compiled_state_dict(model: nn.Module) -> OrderedDict:
    """Simulate double-nested _orig_mod._orig_mod. prefixes (edge case)."""
    sd = model.state_dict()
    compiled_sd = OrderedDict()
    for k, v in sd.items():
        compiled_sd[f"{_COMPILE_PREFIX}{_COMPILE_PREFIX}{k}"] = v
    return compiled_sd


# ---------------------------------------------------------------------------
# strip_compiled_prefix unit tests
# ---------------------------------------------------------------------------

class TestStripCompiledPrefix:
    """Tests for the strip_compiled_prefix utility function."""

    def test_noop_on_clean_state_dict(self):
        """No-op when state_dict has no _orig_mod. prefixes."""
        model = _TinyModel()
        original_sd = model.state_dict()
        cleaned = strip_compiled_prefix(original_sd)

        assert set(cleaned.keys()) == set(original_sd.keys())
        for k in original_sd:
            assert torch.equal(cleaned[k], original_sd[k])

    def test_strips_single_prefix(self):
        """Correctly strips single _orig_mod. prefix."""
        model = _TinyModel()
        original_keys = set(model.state_dict().keys())
        compiled_sd = _make_compiled_state_dict(model)

        # Confirm compiled keys have prefix
        for k in compiled_sd:
            assert k.startswith(_COMPILE_PREFIX)

        cleaned = strip_compiled_prefix(compiled_sd)
        assert set(cleaned.keys()) == original_keys

    def test_strips_nested_prefix(self):
        """Correctly strips double-nested _orig_mod._orig_mod. prefixes."""
        model = _TinyModel()
        original_keys = set(model.state_dict().keys())
        compiled_sd = _make_nested_compiled_state_dict(model)

        cleaned = strip_compiled_prefix(compiled_sd)
        assert set(cleaned.keys()) == original_keys

    def test_idempotent(self):
        """Applying strip twice yields the same result."""
        model = _TinyModel()
        compiled_sd = _make_compiled_state_dict(model)

        cleaned_once = strip_compiled_prefix(compiled_sd)
        cleaned_twice = strip_compiled_prefix(cleaned_once)

        assert set(cleaned_once.keys()) == set(cleaned_twice.keys())
        for k in cleaned_once:
            assert torch.equal(cleaned_once[k], cleaned_twice[k])

    def test_preserves_parameter_values(self):
        """Parameter values are unchanged after stripping."""
        model = _TinyModel()
        original_sd = model.state_dict()
        compiled_sd = _make_compiled_state_dict(model)
        cleaned = strip_compiled_prefix(compiled_sd)

        for k in original_sd:
            assert torch.equal(cleaned[k], original_sd[k]), (
                f"Value mismatch for key '{k}' after stripping"
            )

    def test_key_collision_raises(self):
        """Raises ValueError when stripping would create duplicate keys."""
        # Create a pathological state dict with both prefixed and unprefixed
        sd = OrderedDict()
        sd["weight"] = torch.tensor([1.0])
        sd["_orig_mod.weight"] = torch.tensor([2.0])

        with pytest.raises(ValueError, match="Key collision"):
            strip_compiled_prefix(sd)

    def test_returns_ordered_dict(self):
        """Always returns an OrderedDict."""
        model = _TinyModel()
        result = strip_compiled_prefix(model.state_dict())
        assert isinstance(result, OrderedDict)

        compiled = _make_compiled_state_dict(model)
        result = strip_compiled_prefix(compiled)
        assert isinstance(result, OrderedDict)


# ---------------------------------------------------------------------------
# Full round-trip tests: compile → save → load(strict=True)
# ---------------------------------------------------------------------------

class TestCompileCheckpointRoundTrip:
    """End-to-end round-trip tests for compiled model checkpoint compatibility."""

    def test_compiled_save_uncompiled_load_strict(self):
        """Core contract: checkpoint from compiled model loads into uncompiled model (strict=True)."""
        model_compiled = _TinyModel()
        # Set non-trivial weights to verify value preservation
        with torch.no_grad():
            for p in model_compiled.parameters():
                p.fill_(0.42)

        # Simulate what train_forecaster does: strip prefix before saving
        compiled_sd = _make_compiled_state_dict(model_compiled)
        clean_sd = strip_compiled_prefix(compiled_sd)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "compiled_checkpoint.pt")
            state = {"model_state_dict": clean_sd, "epoch": 5}
            save_checkpoint(state, ckpt_path)

            # Load into a fresh, uncompiled model with strict=True
            model_fresh = _TinyModel()
            loaded = load_checkpoint(ckpt_path, model_fresh, strict=True)

            assert loaded["epoch"] == 5
            # Verify all parameters match
            for (n1, p1), (n2, p2) in zip(
                model_compiled.named_parameters(), model_fresh.named_parameters()
            ):
                assert n1 == n2
                assert torch.equal(p1, p2), f"Parameter mismatch at '{n1}'"

    def test_compiled_save_compiled_load(self):
        """Compiled checkpoint can also be loaded into another compiled model."""
        model_src = _TinyModel()
        with torch.no_grad():
            for p in model_src.parameters():
                p.normal_(0, 1)

        compiled_sd = _make_compiled_state_dict(model_src)
        clean_sd = strip_compiled_prefix(compiled_sd)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "ckpt.pt")
            save_checkpoint({"model_state_dict": clean_sd}, ckpt_path)

            # Load into fresh model — simulates compiled eval loading
            model_dst = _TinyModel()
            load_checkpoint(ckpt_path, model_dst, strict=True)

            for (_, p1), (_, p2) in zip(
                model_src.named_parameters(), model_dst.named_parameters()
            ):
                assert torch.equal(p1, p2)

    def test_defense_in_depth_load_strips_prefix(self):
        """load_checkpoint strips _orig_mod. prefix even if save forgot to strip."""
        model = _TinyModel()
        compiled_sd = _make_compiled_state_dict(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Intentionally save WITHOUT stripping (simulates a bug or old checkpoint)
            ckpt_path = os.path.join(tmpdir, "raw_compiled_ckpt.pt")
            save_checkpoint({"model_state_dict": compiled_sd}, ckpt_path)

            # load_checkpoint should handle this gracefully
            model_fresh = _TinyModel()
            load_checkpoint(ckpt_path, model_fresh, strict=True)

            for (_, p1), (_, p2) in zip(
                model.named_parameters(), model_fresh.named_parameters()
            ):
                assert torch.equal(p1, p2)

    def test_best_checkpoint_tracker_with_compiled_model(self):
        """BestCheckpointTracker round-trip with stripped compiled state dict."""
        model = _TinyModel()
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.23)

        compiled_sd = _make_compiled_state_dict(model)
        clean_sd = strip_compiled_prefix(compiled_sd)

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = BestCheckpointTracker(
                save_dir=tmpdir, metric_name="vrmse_mean", mode="min", keep_top_k=2
            )
            state = {"model_state_dict": clean_sd, "epoch": 1}
            is_best = tracker.update(0.5, state, 1)
            assert is_best

            best_path = os.path.join(tmpdir, "best_vrmse_mean.pt")
            assert os.path.exists(best_path)

            # Load and verify
            model_loaded = _TinyModel()
            load_checkpoint(best_path, model_loaded, strict=True)

            for (_, p1), (_, p2) in zip(
                model.named_parameters(), model_loaded.named_parameters()
            ):
                assert torch.equal(p1, p2)

    def test_forward_output_consistency_after_round_trip(self):
        """Model forward output matches after save → load round-trip."""
        model_src = _TinyModel()
        model_src.eval()

        x = torch.randn(2, 4, 8, 8)
        with torch.no_grad():
            y_src = model_src(x)

        compiled_sd = _make_compiled_state_dict(model_src)
        clean_sd = strip_compiled_prefix(compiled_sd)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "ckpt.pt")
            save_checkpoint({"model_state_dict": clean_sd}, ckpt_path)

            model_dst = _TinyModel()
            load_checkpoint(ckpt_path, model_dst, strict=True)
            model_dst.eval()

            with torch.no_grad():
                y_dst = model_dst(x)

            assert torch.allclose(y_src, y_dst, atol=1e-7), (
                f"Forward output mismatch after round-trip: "
                f"max diff = {(y_src - y_dst).abs().max().item():.2e}"
            )


# ---------------------------------------------------------------------------
# Integration with _build_model (if torch.compile is available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not hasattr(torch, "compile"),
    reason="torch.compile not available in this PyTorch version",
)
class TestBuildModelCompileIntegration:
    """Integration tests using the actual _build_model function."""

    def test_build_model_compile_roundtrip(self):
        """_build_model with compile_model=True produces loadable checkpoints."""
        from scripts.train_forecaster import _build_model

        device = torch.device("cpu")
        model, _, _ = _build_model(
            model_type="direct_transformer",
            embed_dim=64,
            depth=1,
            num_heads=2,
            prediction_mode="direct",
            freeze_representation=False,
            repr_checkpoint="",
            init_checkpoint=None,
            split_hash="dummy",
            normalizer_hash="dummy",
            seed=42,
            expected_init_horizon=None,
            lambda_div=0.0,
            lambda_vort=0.0,
            use_condition=False,
            device=device,
            is_distributed=False,
            local_rank=0,
            global_rank=0,
            compile_model=True,
        )

        # Extract and strip state_dict (as train_forecaster now does)
        raw_sd = model.state_dict()
        clean_sd = strip_compiled_prefix(raw_sd)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "compiled_integration.pt")
            save_checkpoint({"model_state_dict": clean_sd}, ckpt_path)

            # Build an uncompiled model and load strictly
            model_eval, _, _ = _build_model(
                model_type="direct_transformer",
                embed_dim=64,
                depth=1,
                num_heads=2,
                prediction_mode="direct",
                freeze_representation=False,
                repr_checkpoint="",
                init_checkpoint=None,
                split_hash="dummy",
                normalizer_hash="dummy",
                seed=42,
                expected_init_horizon=None,
                lambda_div=0.0,
                lambda_vort=0.0,
                use_condition=False,
                device=device,
                is_distributed=False,
                local_rank=0,
                global_rank=0,
                compile_model=False,
            )

            load_checkpoint(ckpt_path, model_eval, strict=True)
