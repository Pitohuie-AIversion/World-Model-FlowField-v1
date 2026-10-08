"""Smoke tests for StocBench single-channel model pipeline and dynamics interfaces.

Verifies:
1. Single-channel Encoder2D / Decoder2D forward, latency reduction, with project_pressure=False.
2. Forward and sampling passes for all three latent dynamics contracts
   (Deterministic, Gaussian, Flow Matching) under default context without Re/Sc conditioning.
3. Reference ensemble dimension contract (K_ref=5000) and isolation from prediction horizon H.
4. Physical scaling roundtrip (w_phys = w_stored * 3.0) without loss.
5. Strict train/reference file and identity isolation.
"""

from pathlib import Path
import pytest
import numpy as np
import torch

from src.contracts.batch import WorldModelBatch
from src.contracts.context import Context
from src.contracts.latent_dynamics import (
    DeterministicLatentDynamics,
    GaussianLatentDynamics,
    FlowMatchingLatentDynamics,
)
from src.data.stocbench_dataset import (
    STOCBENCH_SAMPLE_DT,
    STOCBENCH_SCALING_STD,
    STOCBENCH_STATE_SPEC,
    StocBenchReferenceEnsemble,
    StocBenchTrainDataset,
    stocbench_batch_adapter,
)
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.latent_transformer import LatentSTTransformer
from src.models.probabilistic_latent_dynamics import VarianceHead2D
from src.models.latent_flow_matching import LatentFlowMatcher


# ==============================================================================
# 1. Single-Channel Encoder/Decoder Contract & Pressure Isolation
# ==============================================================================


def test_single_channel_autoencoder_smoke():
    """Verify Encoder2D and Decoder2D run forward passes on 1-channel vorticity fields."""
    b, c, h, w = 2, 1, 64, 64
    x = torch.randn(b, c, h, w, dtype=torch.float32)

    enc = Encoder2D(in_channels=1, latent_channels=64, base_channels=16, channel_mult=[1, 2, 4])
    dec = Decoder2D(
        out_channels=1,
        latent_channels=64,
        base_channels=16,
        channel_mult=[4, 2, 1],
        project_pressure=False,
    )

    z = enc(x)
    assert z.shape == (b, 64, 8, 8), f"Expected latent shape (2, 64, 8, 8), got {z.shape}"
    assert torch.isfinite(z).all(), "Latent representation contains NaN/Inf"

    x_rec = dec(z)
    assert x_rec.shape == (b, 1, 64, 64), f"Expected reconstructed shape (2, 1, 64, 64), got {x_rec.shape}"
    assert torch.isfinite(x_rec).all(), "Reconstructed field contains NaN/Inf"


def test_decoder_pressure_projection_isolated_for_single_channel():
    """Verify that project_pressure=False cleanly handles 1-channel vorticity decoding."""
    z = torch.randn(2, 64, 8, 8, dtype=torch.float32)

    dec = Decoder2D(
        out_channels=1,
        latent_channels=64,
        base_channels=16,
        channel_mult=[4, 2, 1],
        project_pressure=False,  # Single-channel vorticity isolates pressure gauge
    )
    assert dec.project_pressure is False

    q = dec(z)
    assert q.shape == (2, 1, 64, 64)
    assert torch.isfinite(q).all()


# ==============================================================================
# 2. Latent Dynamics Forward & Sampling with Default Context (No Re/Sc)
# ==============================================================================


def test_latent_dynamics_forward_default_context_smoke():
    """Verify Deterministic, Gaussian, and FlowMatching dynamics run forward with context=None."""
    b, l, c_z, hz, wz = 2, 1, 64, 8, 8
    z_hist = torch.randn(b, l, c_z, hz, wz, dtype=torch.float32)
    ctx = Context(physical=None, boundary="periodic")

    # Lightweight transformer backbone
    backbone = LatentSTTransformer(
        latent_channels=64,
        embed_dim=64,
        cond_dim=64,
        depth=2,
        num_heads=4,
    )

    # 1. Deterministic dynamics
    dyn_det = DeterministicLatentDynamics(transformer=backbone)
    z_next_det = dyn_det.predict_mean(z_hist, context=ctx)
    assert z_next_det.shape == (b, 1, c_z, hz, wz)
    assert torch.isfinite(z_next_det).all()

    z_roll_det = dyn_det.rollout(z_hist, context=ctx, horizon=3)
    assert z_roll_det.shape == (b, 3, c_z, hz, wz)
    assert torch.isfinite(z_roll_det).all()

    # 2. Gaussian probabilistic dynamics
    var_head = VarianceHead2D(embed_dim=64, latent_channels=64)
    dyn_gauss = GaussianLatentDynamics(transformer=backbone, variance_head=var_head)

    mu_g, var_g = dyn_gauss.predict_distribution(z_hist, context=ctx)
    assert mu_g.shape == (b, 1, c_z, hz, wz)
    assert var_g.shape == (b, 1, c_z, hz, wz)
    assert (var_g > 0).all(), "Variance must be strictly positive"

    samples_g = dyn_gauss.sample(z_hist, context=ctx, num_samples=3)
    assert samples_g.shape == (b, 3, 1, c_z, hz, wz)
    assert torch.isfinite(samples_g).all()

    # Verify variance across samples is non-zero (non-degenerate stochasticity)
    sample_std_g = torch.std(samples_g, dim=1)
    assert (sample_std_g > 0).any(), "Stochastic samples must not be identical"

    # 3. Residual Latent Flow Matching dynamics
    flow_matcher = LatentFlowMatcher(
        latent_channels=64,
        hidden_channels=64,
        cond_dim=64,
        num_blocks=2,
    )
    dyn_fm = FlowMatchingLatentDynamics(backbone=backbone, flow_matcher=flow_matcher)

    # Single-sample flow matching ODE step
    z_next_fm = dyn_fm.sample(z_hist, context=ctx, num_samples=1, num_steps=3)
    assert z_next_fm.shape == (b, 1, c_z, hz, wz)
    assert torch.isfinite(z_next_fm).all()

    # Multi-sample flow matching ODE step
    samples_fm = dyn_fm.sample(z_hist, context=ctx, num_samples=2, num_steps=3)
    assert samples_fm.shape == (b, 2, 1, c_z, hz, wz)
    assert torch.isfinite(samples_fm).all()


# ==============================================================================
# 3. Reference Ensemble Dimension Contract & Ground Truth Isolation
# ==============================================================================


def test_stocbench_reference_ensemble_dimension_and_latent_projection():
    """Verify reference bifurcation ensemble maps correctly to latent space while isolating K_ref from H."""
    step_file = Path("data/stocbench/incns_stoc/64/step_seed_100.npz")
    if not step_file.exists():
        pytest.skip(f"Reference file not found: {step_file}")

    ref = StocBenchReferenceEnsemble(step_file)
    assert ref.num_members == 5000, f"Expected K_ref=5000, got {ref.num_members}"
    assert ref.spatial_shape == (64, 64)

    # Statistical consistency check on ddof=0 population statistics
    stats = ref.verify_statistical_consistency(atol=1e-4)
    assert stats["max_mean_discrepancy"] < 1e-4
    assert stats["max_std_discrepancy"] < 1e-4

    # Encode initial condition with single-channel encoder
    enc = Encoder2D(in_channels=1, latent_channels=64, base_channels=16, channel_mult=[1, 2, 4])
    init_t = torch.from_numpy(ref.init_np).unsqueeze(0)  # (1, 1, 64, 64)
    z_init = enc(init_t)
    assert z_init.shape == (1, 64, 8, 8)

    # Encode a small subset of reference futures (e.g. 8 members) into latent space
    subset_futures = torch.from_numpy(ref.raw_np[:8]).squeeze(2)  # (8, 1, 64, 64)
    z_ref_subset = enc(subset_futures)
    assert z_ref_subset.shape == (8, 64, 8, 8)
    assert torch.isfinite(z_ref_subset).all()


# ==============================================================================
# 4. Physical Scaling Roundtrip Contract
# ==============================================================================


def test_stocbench_physical_scaling_roundtrip():
    """Verify that stored to physical vorticity scaling roundtrip is lossless."""
    w_stored = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=torch.float32)

    # Forward: w_phys = w_stored * 3.0
    w_phys = w_stored * STOCBENCH_SCALING_STD

    # Inverse: w_stored = w_phys / 3.0
    w_stored_rec = w_phys / STOCBENCH_SCALING_STD

    max_err = torch.max(torch.abs(w_stored - w_stored_rec)).item()
    assert max_err < 1e-6, f"Scaling roundtrip error {max_err} exceeds tolerance"


# ==============================================================================
# 5. Train / Reference Data Separation and Provenance
# ==============================================================================


def test_train_and_reference_data_isolation():
    """Verify that training dataset and evaluation reference file are strictly isolated."""
    traj_file = Path("data/stocbench/incns_stoc/64/traj_seed_42.npy")
    step_file = Path("data/stocbench/incns_stoc/64/step_seed_100.npz")
    if not traj_file.exists() or not step_file.exists():
        pytest.skip("StocBench data files not found")

    train_ds = StocBenchTrainDataset(file_path=traj_file, history_length=1, horizon=1)
    ref = StocBenchReferenceEnsemble(step_file)

    sample = train_ds[0]
    assert sample["source_file"] == "traj_seed_42.npy"
    assert "step_seed" not in sample["source_file"]

    # Trajectory data and bifurcation data have different seeds and independent files
    assert sample["history"].shape == (1, 1, 64, 64)
    assert ref.init_np.shape == (1, 64, 64)
    # Ensure they are not referencing identical memory
    assert not np.allclose(sample["history"].numpy().squeeze(), ref.init_np.squeeze())
