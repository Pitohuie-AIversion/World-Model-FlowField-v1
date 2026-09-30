"""Comprehensive verification test suite for World Model Core Contracts (Phase 1).

Covers:
A. StateSpec contract and fail-fast validation.
B. Context and PhysicalContext numerical equivalence (bitwise equal).
C. WorldModelBatch lossless conversion and Mapping compatibility.
D. DeterministicLatentDynamics parity with LatentSTTransformer baseline.
E. GaussianLatentDynamics parity and reproducibility.
F. FlowMatchingLatentDynamics deterministic fallback, ODE sampling, and loss.
G. Rollout regression at horizons H=1, H=4, H=8.
H. Checkpoint compatibility with existing real repository checkpoints.
"""

import os
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.contracts.state_spec import StateSpec, SHEAR_FLOW_STATE_SPEC
from src.contracts.context import Context, PhysicalContext, resolve_context
from src.contracts.batch import (
    WorldModelBatch,
    collate_world_model_batch,
    shear_flow_batch_adapter,
    collate_shear_flow_batch,
)
from src.contracts.latent_dynamics import (
    LatentDynamics,
    DeterministicLatentDynamics,
    GaussianLatentDynamics,
    FlowMatchingLatentDynamics,
)
from src.models.conditioning import PhysicalConditionEmbedding
from src.models.latent_transformer import LatentSTTransformer
from src.models.probabilistic_latent_dynamics import VarianceHead2D
from src.models.latent_flow_matching import LatentFlowMatcher
from src.models.latent_forecaster import LatentForecaster
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.utils.provenance import compute_file_sha256


# ==============================================================================
# A. StateSpec Contract Verification
# ==============================================================================

def test_state_spec_shear_flow_canonical():
    """Verify canonical shear_flow state specification."""
    spec = SHEAR_FLOW_STATE_SPEC
    assert spec.variables == ("u", "v", "p", "s")
    assert spec.num_channels == 4
    assert spec.spatial_dim == 2
    assert spec.channel_index("u") == 0
    assert spec.channel_index("v") == 1
    assert spec.channel_index("p") == 2
    assert spec.channel_index("s") == 3
    assert spec.has_variable("u") is True
    assert spec.has_variable("temperature") is False


def test_state_spec_tensor_validation():
    """Verify tensor shape validation with StateSpec."""
    spec = SHEAR_FLOW_STATE_SPEC
    # Valid tensors (B, C, H, W) and (B, L, C, H, W)
    t4 = torch.randn(2, 4, 32, 64)
    spec.validate_tensor(t4, channel_dim=1)

    t5 = torch.randn(2, 4, 4, 32, 64)
    spec.validate_tensor(t5, channel_dim=2)

    # Invalid channel count
    t_invalid = torch.randn(2, 3, 32, 64)
    with pytest.raises(ValueError, match="Tensor channel mismatch"):
        spec.validate_tensor(t_invalid, channel_dim=1)


def test_state_spec_fail_fast_invalid_construction():
    """Verify fail-fast behavior on malformed StateSpec parameters."""
    # Empty variables
    with pytest.raises(ValueError, match="must not be empty"):
        StateSpec(variables=())

    # Mismatch between variables and num_channels
    with pytest.raises(ValueError, match="channel count mismatch"):
        StateSpec(variables=("u", "v"), num_channels=4)

    # Duplicate variable names
    with pytest.raises(ValueError, match="duplicate names"):
        StateSpec(variables=("u", "v", "u"))

    # Invalid spatial dimension
    with pytest.raises(ValueError, match="must be positive"):
        StateSpec(variables=("u", "v"), spatial_dim=0)


def test_state_spec_serialization():
    """Verify round-trip serialization of StateSpec."""
    spec = StateSpec(variables=["u", "v", "p", "s"], num_channels=4, spatial_dim=2)
    data = spec.to_dict()
    reconstructed = StateSpec.from_dict(data)
    assert reconstructed == spec
    assert reconstructed.variables == spec.variables
    assert reconstructed.num_channels == spec.num_channels


# ==============================================================================
# B. Context & PhysicalContext Numerical Equivalence
# ==============================================================================

def test_context_physical_bitwise_parity():
    """Verify bitwise numerical parity between legacy conditioning and Context adapter."""
    torch.manual_seed(42)
    cond_embed = PhysicalConditionEmbedding(embed_dim=128, hidden_dim=256, use_log=True)
    cond_embed.eval()

    # Fixed physical inputs
    re = torch.tensor([500.0, 1000.0, 2500.0], dtype=torch.float32)
    sc = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)

    # 1. Legacy call: directly pass (re, sc)
    with torch.no_grad():
        out_legacy = cond_embed(re, sc)

    # 2. Context call: build Context, unpack via to_re_sc
    ctx = Context.from_re_sc(re=re, sc=sc)
    c_re, c_sc = ctx.to_re_sc()
    with torch.no_grad():
        out_contract = cond_embed(c_re, c_sc)

    # Must be bitwise equal
    assert torch.equal(out_legacy, out_contract), "Context unpack diverges from legacy conditioning"


def test_context_accessors_and_device_transfer():
    """Verify Context convenience accessors and device migration."""
    ctx = Context.from_re_sc(re=1000.0, sc=1.0)
    assert ctx.re is not None
    assert ctx.sc is not None
    assert float(ctx.re) == 1000.0
    assert float(ctx.sc) == 1.0

    # Device/dtype transfer
    ctx_moved = ctx.to(dtype=torch.float64)
    assert ctx_moved.re.dtype == torch.float64
    assert ctx_moved.sc.dtype == torch.float64


def test_resolve_context_duck_typing():
    """Verify resolve_context handling of Context, dict, and legacy kwargs."""
    # From Context object
    ctx = Context.from_re_sc(re=100.0, sc=2.0)
    assert resolve_context(ctx) is ctx

    # From legacy arguments
    resolved = resolve_context(re=100.0, sc=2.0)
    assert resolved is not None
    assert float(resolved.re) == 100.0
    assert float(resolved.sc) == 2.0

    # From dictionary
    dict_ctx = {"physical": {"re": torch.tensor(300.0), "sc": torch.tensor(1.5)}}
    res_dict = resolve_context(dict_ctx)
    assert res_dict is not None
    assert float(res_dict.re) == 300.0
    assert float(res_dict.sc) == 1.5


def test_context_legacy_equal_allowed():
    """Verify that when both Context and legacy parameters are provided with identical values, they are accepted."""
    ctx = Context.from_re_sc(re=1000.0, sc=1.0)
    resolved = resolve_context(context=ctx, re=1000.0, sc=1.0)
    assert resolved is not None
    assert float(resolved.re) == 1000.0
    assert float(resolved.sc) == 1.0

    # Also test tensor equality
    ctx_t = Context.from_re_sc(re=torch.tensor([500.0, 1000.0]), sc=torch.tensor([0.5, 1.0]))
    resolved_t = resolve_context(context=ctx_t, re=torch.tensor([500.0, 1000.0]), sc=torch.tensor([0.5, 1.0]))
    assert resolved_t is not None
    assert torch.allclose(resolved_t.re, torch.tensor([500.0, 1000.0]))
    assert torch.allclose(resolved_t.sc, torch.tensor([0.5, 1.0]))


def test_context_legacy_conflict_fails_closed():
    """Verify that divergent Context and legacy parameters raise ValueError immediately (Fail-Closed)."""
    ctx = Context.from_re_sc(re=1000.0, sc=1.0)
    with pytest.raises(ValueError, match="Context conflict.*divergent legacy re"):
        resolve_context(context=ctx, re=2000.0)

    with pytest.raises(ValueError, match="Context conflict.*divergent legacy sc"):
        resolve_context(context=ctx, sc=2.0)

    # Test when context has None but legacy provided
    ctx_no_re = Context(physical=PhysicalContext(sc=torch.tensor([1.0])))
    with pytest.raises(ValueError, match="Context physical.re is None, but legacy re"):
        resolve_context(context=ctx_no_re, re=1000.0)


def test_context_legacy_near_miss_conflict_fails_closed():
    """Verify that near-miss legacy parameters (e.g. 1000.0 vs 1000.05) fail closed strictly."""
    ctx = Context.from_re_sc(re=1000.0, sc=1.0)
    # 1000.0 vs 1000.05 is within 1e-4 relative tolerance, but violates physical condition identity
    with pytest.raises(ValueError, match="Context conflict.*divergent legacy re"):
        resolve_context(context=ctx, re=1000.05)

    with pytest.raises(ValueError, match="Context conflict.*divergent legacy sc"):
        resolve_context(context=ctx, sc=1.0001)


# ==============================================================================
# C. WorldModelBatch Contract Verification
# ==============================================================================

def test_world_model_batch_from_batch_dict_lossless():
    """Verify that WorldModelBatch captures all fields without loss or mutation."""
    b, l, h, c, ny, nx = 2, 4, 2, 4, 32, 64
    hist_t = torch.randn(b, l, c, ny, nx)
    fut_t = torch.randn(b, h, c, ny, nx)
    re_t = torch.tensor([1000.0, 2000.0])
    sc_t = torch.tensor([1.0, 1.5])
    dt_t = torch.tensor([0.1, 0.1])
    time_t = torch.arange(l + h).unsqueeze(0).repeat(b, 1).float() * 0.1

    legacy_batch = {
        "history": hist_t,
        "future": fut_t,
        "re": re_t,
        "sc": sc_t,
        "dt": dt_t,
        "time": time_t,
        "source_file": ["file_a.h5", "file_b.h5"],
        "traj_idx": torch.tensor([0, 1]),
        "start_t": torch.tensor([10, 20]),
        "cluster_id": torch.tensor([1, 2]),
    }

    w_batch = WorldModelBatch.from_batch_dict(legacy_batch, state_spec=SHEAR_FLOW_STATE_SPEC)

    # Tensor parity
    assert torch.equal(w_batch.history, hist_t)
    assert torch.equal(w_batch.future, fut_t)
    assert torch.equal(w_batch.context.re, re_t)
    assert torch.equal(w_batch.context.sc, sc_t)
    assert torch.equal(w_batch.coordinates["dt"], dt_t)
    assert torch.equal(w_batch.coordinates["time"], time_t)
    assert w_batch.metadata["source_file"] == ["file_a.h5", "file_b.h5"]
    assert w_batch.batch_size == b

    # Verify shear_flow_batch_adapter yields matching state_spec
    adapter_batch = shear_flow_batch_adapter(legacy_batch)
    assert adapter_batch.state_spec == SHEAR_FLOW_STATE_SPEC
    assert adapter_batch.boundary == "periodic"
    assert torch.equal(adapter_batch.history, hist_t)

    # Dictionary Mapping protocol backwards compatibility
    assert torch.equal(w_batch["history"], hist_t)
    assert torch.equal(w_batch["future"], fut_t)
    assert torch.equal(w_batch["re"], re_t)
    assert torch.equal(w_batch["sc"], sc_t)
    assert torch.equal(w_batch["dt"], dt_t)
    assert "history" in w_batch
    assert "re" in w_batch
    assert "unknown_key" not in w_batch

    # Round trip to_dict()
    flat_dict = w_batch.to_dict()
    assert torch.equal(flat_dict["history"], hist_t)
    assert torch.equal(flat_dict["re"], re_t)
    assert flat_dict["source_file"] == ["file_a.h5", "file_b.h5"]


def test_world_model_batch_channel_validation_fail_fast():
    """Verify that WorldModelBatch validates channel counts against StateSpec."""
    b, l, c_bad, ny, nx = 2, 4, 3, 32, 64
    hist_bad = torch.randn(b, l, c_bad, ny, nx)

    with pytest.raises(ValueError, match="history channel count mismatch"):
        WorldModelBatch(
            history=hist_bad,
            state_spec=SHEAR_FLOW_STATE_SPEC,  # expects 4 channels
        )


def test_world_model_batch_requires_explicit_state_spec():
    """Verify that WorldModelBatch cannot be instantiated without explicitly providing StateSpec."""
    hist = torch.randn(2, 4, 4, 32, 64)
    with pytest.raises(TypeError):
        # Missing required positional/keyword argument 'state_spec'
        WorldModelBatch(history=hist)  # type: ignore

    with pytest.raises(TypeError, match="state_spec must be an instance of StateSpec"):
        WorldModelBatch(history=hist, state_spec="not_a_statespec")  # type: ignore


def test_world_model_batch_single_geometry_source():
    """Verify that Geometry has a single source of truth in Context, and batch.geometry is a property."""
    geom_data = torch.ones((1, 32, 64))
    ctx = Context(geometry=geom_data)
    hist = torch.randn(1, 4, 4, 32, 64)
    batch = WorldModelBatch(history=hist, state_spec=SHEAR_FLOW_STATE_SPEC, context=ctx)

    assert batch.geometry is geom_data
    assert batch["geometry"] is geom_data
    assert "geometry" in batch
    # Verify no independent storage in __dict__
    assert "geometry" not in batch.__dict__

    # Batch with no context has None geometry
    batch_empty = WorldModelBatch(history=hist, state_spec=SHEAR_FLOW_STATE_SPEC)
    assert batch_empty.geometry is None
    assert "geometry" not in batch_empty
    with pytest.raises(KeyError):
        _ = batch_empty["geometry"]


def test_world_model_batch_single_boundary_source():
    """Verify that Boundary has a single source of truth in Context, and batch.boundary is a property."""
    ctx = Context(boundary="no_slip")
    hist = torch.randn(1, 4, 4, 32, 64)
    batch = WorldModelBatch(history=hist, state_spec=SHEAR_FLOW_STATE_SPEC, context=ctx)

    assert batch.boundary == "no_slip"
    assert batch["boundary"] == "no_slip"
    assert "boundary" in batch
    # Verify no independent storage in __dict__
    assert "boundary" not in batch.__dict__

    # from_batch_dict with conflicting boundary raises ValueError (Fail-Closed)
    with pytest.raises(ValueError, match="Boundary conflict"):
        WorldModelBatch.from_batch_dict(
            {"history": hist, "context": ctx},
            state_spec=SHEAR_FLOW_STATE_SPEC,
            boundary="periodic",
        )


def test_collate_world_model_batch():
    """Verify collate_world_model_batch as a DataLoader collate_fn."""
    samples = [
        {
            "history": torch.randn(4, 4, 16, 32),
            "future": torch.randn(1, 4, 16, 32),
            "re": torch.tensor(1000.0),
            "sc": torch.tensor(1.0),
            "dt": torch.tensor(0.1),
            "source_file": "sim0.h5",
        },
        {
            "history": torch.randn(4, 4, 16, 32),
            "future": torch.randn(1, 4, 16, 32),
            "re": torch.tensor(2000.0),
            "sc": torch.tensor(2.0),
            "dt": torch.tensor(0.1),
            "source_file": "sim1.h5",
        },
    ]

    batch = collate_world_model_batch(samples, state_spec=SHEAR_FLOW_STATE_SPEC)
    assert isinstance(batch, WorldModelBatch)
    assert batch.history.shape == (2, 4, 4, 16, 32)
    assert batch.future.shape == (2, 1, 4, 16, 32)
    assert batch.context.re.shape == (2,)

    # Adapter collate test
    adapter_collate_batch = collate_shear_flow_batch(samples)
    assert isinstance(adapter_collate_batch, WorldModelBatch)
    assert adapter_collate_batch.state_spec == SHEAR_FLOW_STATE_SPEC
    assert adapter_collate_batch.boundary == "periodic"


# ==============================================================================
# D. Deterministic LatentDynamics Parity
# ==============================================================================

def test_deterministic_latent_dynamics_parity():
    """Verify bitwise parity between DeterministicLatentDynamics and LatentSTTransformer."""
    torch.manual_seed(42)
    transformer = LatentSTTransformer(
        latent_channels=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        cond_dim=16,
        history_length=4,
    )
    transformer.eval()
    dynamics = DeterministicLatentDynamics(transformer=transformer)
    dynamics.eval()

    z_hist = torch.randn(2, 4, 8, 8, 16)
    re = torch.tensor([1000.0, 2000.0])
    sc = torch.tensor([1.0, 1.0])
    ctx = Context.from_re_sc(re=re, sc=sc)

    with torch.no_grad():
        out_raw = transformer(z_hist, re=re, sc=sc)
        out_dynamics_mean = dynamics.predict_mean(z_hist, context=ctx)
        out_dynamics_forward = dynamics(z_hist, context=ctx)
        out_dynamics_sample1 = dynamics.sample(z_hist, context=ctx, num_samples=1)

    assert torch.equal(out_raw, out_dynamics_mean)
    assert torch.equal(out_raw, out_dynamics_forward)
    assert torch.equal(out_raw, out_dynamics_sample1)

    # Multi-sample shape check
    out_samples4 = dynamics.sample(z_hist, context=ctx, num_samples=4)
    assert out_samples4.shape == (2, 4, 1, 8, 8, 16)


# ==============================================================================
# E. Gaussian LatentDynamics Parity and Reproducibility
# ==============================================================================

def test_gaussian_latent_dynamics_parity_and_reproducibility():
    """Verify GaussianLatentDynamics parity, strict variance floor, and reproducibility."""
    torch.manual_seed(42)
    transformer = LatentSTTransformer(
        latent_channels=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        cond_dim=16,
        history_length=4,
    )
    vhead = VarianceHead2D(embed_dim=32, latent_channels=8, variance_floor=1e-4)
    transformer.attach_variance_head(vhead)
    transformer.eval()

    dynamics = GaussianLatentDynamics(transformer=transformer)
    dynamics.eval()

    z_hist = torch.randn(2, 4, 8, 8, 16)
    re = torch.tensor([1000.0, 1500.0])
    sc = torch.tensor([1.0, 2.0])
    ctx = Context.from_re_sc(re=re, sc=sc)

    with torch.no_grad():
        mu_raw, var_raw = transformer.predict_distribution(z_hist, re=re, sc=sc)
        mu_dyn, var_dyn = dynamics.predict_distribution(z_hist, context=ctx)
        mu_mean = dynamics.predict_mean(z_hist, context=ctx)

    assert torch.equal(mu_raw, mu_dyn)
    assert torch.equal(mu_raw, mu_mean)
    assert torch.equal(var_raw, var_dyn)
    assert (var_dyn >= 1e-4).all()

    # Reproducibility under fixed seed
    s1 = dynamics.sample(z_hist, context=ctx, num_samples=1, seed=12345)
    s2 = dynamics.sample(z_hist, context=ctx, num_samples=1, seed=12345)
    assert torch.equal(s1, s2)

    # Multi-sample draw
    s_multi = dynamics.sample(z_hist, context=ctx, num_samples=3, seed=12345)
    assert s_multi.shape == (2, 3, 1, 8, 8, 16)


# ==============================================================================
# F. FlowMatchingLatentDynamics Verification
# ==============================================================================

def test_flow_matching_latent_dynamics_fallback_and_sampling():
    """Verify FlowMatchingLatentDynamics deterministic fallback and ODE solver integration."""
    torch.manual_seed(42)
    backbone = LatentSTTransformer(
        latent_channels=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        cond_dim=16,
        history_length=4,
    )
    flow_matcher = LatentFlowMatcher(
        latent_channels=8,
        cond_dim=16,
        hidden_channels=16,
        num_blocks=2,
        use_spatial_attn=False,
    )
    backbone.eval()
    flow_matcher.eval()

    dynamics = FlowMatchingLatentDynamics(backbone=backbone, flow_matcher=flow_matcher)
    dynamics.eval()

    z_hist = torch.randn(2, 4, 8, 8, 16)
    re = torch.tensor([1000.0, 2000.0])
    sc = torch.tensor([1.0, 1.0])
    ctx = Context.from_re_sc(re=re, sc=sc)

    # 1. Deterministic mean must be bitwise equal to backbone prediction
    with torch.no_grad():
        mu_d0 = backbone(z_hist, re=re, sc=sc)
        mu_dyn = dynamics.predict_mean(z_hist, context=ctx)
    assert torch.equal(mu_d0, mu_dyn)

    # 2. Deterministic fallback must return mu exactly
    fallback_sample = dynamics.sample(
        z_hist,
        context=ctx,
        deterministic_fallback=True,
    )
    assert torch.equal(mu_d0, fallback_sample)

    # 3. ODE sampling reproducibility with fixed seed
    s1 = dynamics.sample(
        z_hist,
        context=ctx,
        num_samples=1,
        num_steps=2,
        solver="euler",
        seed=999,
    )
    s2 = dynamics.sample(
        z_hist,
        context=ctx,
        num_samples=1,
        num_steps=2,
        solver="euler",
        seed=999,
    )
    assert torch.equal(s1, s2)
    assert s1.shape == (2, 1, 8, 8, 16)

    # 4. Multi-sample shape check
    s_multi = dynamics.sample(
        z_hist,
        context=ctx,
        num_samples=2,
        num_steps=2,
        solver="euler",
        seed=999,
    )
    assert s_multi.shape == (2, 2, 1, 8, 8, 16)


def test_flow_matching_latent_dynamics_compute_loss():
    """Verify compute_loss via FlowMatchingLatentDynamics."""
    torch.manual_seed(42)
    backbone = LatentSTTransformer(
        latent_channels=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        cond_dim=16,
        history_length=4,
    )
    flow_matcher = LatentFlowMatcher(
        latent_channels=8,
        cond_dim=16,
        hidden_channels=16,
        num_blocks=2,
        use_spatial_attn=False,
    )
    dynamics = FlowMatchingLatentDynamics(backbone=backbone, flow_matcher=flow_matcher)

    z_hist = torch.randn(2, 4, 8, 8, 16)
    z_next = torch.randn(2, 1, 8, 8, 16)
    ctx = Context.from_re_sc(re=1000.0, sc=1.0)

    loss_dict = dynamics.compute_loss(
        z_next=z_next,
        latent_history=z_hist,
        context=ctx,
    )
    assert "loss" in loss_dict
    assert torch.isfinite(loss_dict["loss"])


# ==============================================================================
# G. Rollout Regression Verification (H=1, H=4, H=8)
# ==============================================================================

@pytest.mark.parametrize("horizon", [1, 4, 8])
def test_deterministic_rollout_regression(horizon):
    """Verify that LatentDynamics.rollout matches legacy LatentForecaster autoregression bitwise."""
    torch.manual_seed(42)
    encoder = Encoder2D(in_channels=4, latent_channels=8, base_channels=8, channel_mult=[1, 1, 1])
    decoder = Decoder2D(out_channels=4, latent_channels=8, base_channels=8, channel_mult=[1, 1, 1])
    transformer = LatentSTTransformer(
        latent_channels=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        cond_dim=16,
        history_length=4,
    )
    forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder)
    forecaster.eval()

    dynamics = DeterministicLatentDynamics(transformer=transformer)
    dynamics.eval()

    q_hist = torch.randn(2, 4, 4, 16, 32)
    re = torch.tensor([1000.0, 2000.0])
    sc = torch.tensor([1.0, 1.0])
    ctx = Context.from_re_sc(re=re, sc=sc)

    with torch.no_grad():
        z_hist = encoder(q_hist)

        # 1. Roll out via LatentDynamics contract in latent space
        z_rollout_contract = dynamics.rollout(
            latent_history=z_hist,
            context=ctx,
            horizon=horizon,
            stochastic=False,
        )
        q_rollout_contract = decoder(z_rollout_contract)

        # 2. Roll out via legacy forecaster
        q_rollout_legacy = forecaster.forward_rollout(
            q_hist=q_hist,
            re=re,
            sc=sc,
            horizon=horizon,
        )

    # Both pathways must produce bitwise identical output
    assert torch.equal(q_rollout_contract, q_rollout_legacy), (
        f"Rollout mismatch at horizon H={horizon}"
    )


# ==============================================================================
# H. Checkpoint Compatibility Verification
# ==============================================================================

def test_checkpoint_compatibility_d0_deterministic():
    """Verify loading real D0 checkpoint and executing under DeterministicLatentDynamics contract."""
    d0_ckpt_path = "outputs/checkpoints/dynamics/closure_r4/ablation_E0_single_step/latent_transformer/best_vrmse_mean.pt"
    if not os.path.exists(d0_ckpt_path):
        pytest.skip(f"D0 checkpoint not present at {d0_ckpt_path}")

    ckpt = torch.load(d0_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    # Construct standard architecture matching closure_r4
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=256,
        depth=6,
        num_heads=8,
        cond_dim=128,
        history_length=4,
        use_spatial_pos=True,
        prediction_mode="direct",
    )
    # Clean compile prefixes and extract transformer submodule weights
    cleaned_sd = {}
    for k, v in state_dict.items():
        clean_k = k.replace("_orig_mod.", "")
        if clean_k.startswith("transformer."):
            cleaned_sd[clean_k.replace("transformer.", "")] = v
        elif not any(clean_k.startswith(p) for p in ("encoder.", "decoder.")):
            cleaned_sd[clean_k] = v

    transformer.load_state_dict(cleaned_sd, strict=True)
    transformer.eval()

    # Wrap in DeterministicLatentDynamics
    dynamics = DeterministicLatentDynamics(transformer=transformer)
    dynamics.eval()

    z_hist = torch.randn(1, 4, 64, 16, 32)
    ctx = Context.from_re_sc(re=1000.0, sc=1.0)

    with torch.no_grad():
        out = dynamics.predict_mean(z_hist, context=ctx)
    assert out.shape == (1, 1, 64, 16, 32)
    assert torch.isfinite(out).all()


def test_checkpoint_compatibility_g1_variance_head():
    """Verify loading real G1 variance head checkpoint under GaussianLatentDynamics contract."""
    g1_ckpt_path = "outputs/checkpoints/probabilistic/variance_head/formal_r1_seed42/best_g1_variance_head.pt"
    if not os.path.exists(g1_ckpt_path):
        pytest.skip(f"G1 checkpoint not present at {g1_ckpt_path}")

    ckpt = torch.load(g1_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("variance_head_state_dict", ckpt.get("state_dict", ckpt))

    vhead = VarianceHead2D(embed_dim=256, latent_channels=64, variance_floor=1e-4)
    vhead.load_state_dict(state_dict, strict=True)
    vhead.eval()

    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=256,
        depth=6,
        num_heads=8,
        cond_dim=128,
        history_length=4,
    )
    transformer.attach_variance_head(vhead)
    transformer.eval()

    dynamics = GaussianLatentDynamics(transformer=transformer)
    dynamics.eval()

    z_hist = torch.randn(1, 4, 64, 16, 32)
    ctx = Context.from_re_sc(re=1000.0, sc=1.0)

    with torch.no_grad():
        mu, var = dynamics.predict_distribution(z_hist, context=ctx)
        sample = dynamics.sample(z_hist, context=ctx, num_samples=1, seed=42)

    assert mu.shape == (1, 1, 64, 16, 32)
    assert var.shape == (1, 1, 64, 16, 32)
    assert (var >= 1e-4).all()
    assert sample.shape == (1, 1, 64, 16, 32)


def test_checkpoint_compatibility_latent_flow_matcher():
    """Verify loading real Flow Matching pilot checkpoint under FlowMatchingLatentDynamics."""
    fm_ckpt_path = "outputs/checkpoints/probabilistic/flow_matching_pilot3ep/best_latent_flow_matcher.pt"
    if not os.path.exists(fm_ckpt_path):
        pytest.skip(f"FM checkpoint not present at {fm_ckpt_path}")

    ckpt = torch.load(fm_ckpt_path, map_location="cpu", weights_only=False)
    # Checkpoint provenance contract must remain intact
    assert "provenance" in ckpt
    assert "d0_checkpoint" in ckpt["provenance"]
    fm_sd = ckpt.get("flow_matcher_state_dict", ckpt.get("model_state_dict"))
    assert fm_sd is not None

    flow_matcher = LatentFlowMatcher(
        latent_channels=64,
        cond_dim=128,
        hidden_channels=128,
        num_blocks=4,
        use_spatial_attn=True,
    )
    flow_matcher.load_state_dict(fm_sd, strict=True)
    flow_matcher.eval()

    backbone = LatentSTTransformer(
        latent_channels=64,
        embed_dim=256,
        depth=6,
        num_heads=8,
        cond_dim=128,
        history_length=4,
    )
    backbone.eval()

    dynamics = FlowMatchingLatentDynamics(backbone=backbone, flow_matcher=flow_matcher)
    dynamics.eval()

    z_hist = torch.randn(1, 4, 64, 16, 32)
    ctx = Context.from_re_sc(re=1000.0, sc=1.0)

    with torch.no_grad():
        # Fallback check
        fallback = dynamics.sample(z_hist, context=ctx, deterministic_fallback=True)
        assert fallback.shape == (1, 1, 64, 16, 32)

        # 1-step ODE integration test
        sample = dynamics.sample(
            z_hist,
            context=ctx,
            num_samples=1,
            num_steps=2,
            solver="euler",
            seed=42,
        )
        assert sample.shape == (1, 1, 64, 16, 32)
        assert torch.isfinite(sample).all()


def test_residual_scale_valid_legacy_checkpoint_loads():
    """Verify that a checkpoint containing a valid residual_scale loads cleanly under strict=True."""
    model1 = LatentFlowMatcher(latent_channels=64, cond_dim=128, hidden_channels=64, num_blocks=2)
    scale_val = torch.abs(torch.randn(1, 64, 1, 1)) + 0.1
    model1.set_residual_scale(scale_val)

    state_dict = model1.state_dict()
    assert "residual_scale" in state_dict

    # Fresh model without residual_scale
    model2 = LatentFlowMatcher(latent_channels=64, cond_dim=128, hidden_channels=64, num_blocks=2)
    assert model2.residual_scale is None

    # Load with strict=True must succeed
    model2.load_state_dict(state_dict, strict=True)
    assert model2.residual_scale is not None
    assert torch.allclose(model2.residual_scale, scale_val)


def test_residual_scale_wrong_shape_fails():
    """Verify that a checkpoint containing a malformed residual_scale shape fails under strict=True."""
    model = LatentFlowMatcher(latent_channels=64, cond_dim=128, hidden_channels=64, num_blocks=2)
    state_dict = model.state_dict()

    # Inject bad shape: 32 channels instead of 64
    state_dict["residual_scale"] = torch.ones(1, 32, 1, 1)
    with pytest.raises(RuntimeError, match=r"Error\(s\) in loading state_dict"):
        model.load_state_dict(state_dict, strict=True)

    # Inject completely wrong dimension (2D instead of 1D/4D)
    state_dict["residual_scale"] = torch.ones(64, 64)
    with pytest.raises(RuntimeError, match=r"Error\(s\) in loading state_dict"):
        model.load_state_dict(state_dict, strict=True)


def test_residual_scale_nonfinite_fails():
    """Verify that checkpoints containing NaN, Inf, non-positive, or None residual_scale fail under strict=True."""
    model = LatentFlowMatcher(latent_channels=64, cond_dim=128, hidden_channels=64, num_blocks=2)
    state_dict = model.state_dict()

    # Inject NaN
    bad_scale_nan = torch.ones(1, 64, 1, 1)
    bad_scale_nan[0, 0, 0, 0] = float("nan")
    state_dict["residual_scale"] = bad_scale_nan
    with pytest.raises(RuntimeError, match=r"Error\(s\) in loading state_dict"):
        model.load_state_dict(state_dict, strict=True)

    # Inject Inf
    bad_scale_inf = torch.ones(1, 64, 1, 1)
    bad_scale_inf[0, 0, 0, 0] = float("inf")
    state_dict["residual_scale"] = bad_scale_inf
    with pytest.raises(RuntimeError, match=r"Error\(s\) in loading state_dict"):
        model.load_state_dict(state_dict, strict=True)

    # Inject non-positive (0.0 or negative)
    bad_scale_zero = torch.ones(1, 64, 1, 1)
    bad_scale_zero[0, 0, 0, 0] = 0.0
    state_dict["residual_scale"] = bad_scale_zero
    with pytest.raises(RuntimeError, match=r"Error\(s\) in loading state_dict"):
        model.load_state_dict(state_dict, strict=True)

    # Inject None
    state_dict["residual_scale"] = None
    with pytest.raises(RuntimeError, match=r"Error\(s\) in loading state_dict"):
        model.load_state_dict(state_dict, strict=True)

