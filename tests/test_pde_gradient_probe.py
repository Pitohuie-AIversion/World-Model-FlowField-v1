"""Tests for PDE gradient norm and loss scale probing utilities.

Verifies:
1. Mathematical precision of gradient norm and cosine similarity computations.
2. Decoupled gradient computation arriving at trainable dynamics parameters theta.
3. Frozen encoder and decoder receive zero gradient updates during PDE backprop.
4. End-to-end gradient probe execution on mock checkpoints without requiring large weights.
"""

import json
import math
import os
import tempfile
import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.probe_pde_gradient_scales import (
    compute_decoupled_batch_gradients,
    compute_gradient_cosine_similarity,
    compute_gradient_norm,
    evaluate_on_fixed_validation_window,
    run_pde_gradient_probe,
)
from scripts.train_forecaster import LatentForecasterWrapper
from src.data.normalization import FieldNormalizer
from src.losses.navier_stokes import (
    NavierStokesMomentumResidualLoss,
    NavierStokesPDELoss,
    TracerAdvectionDiffusionResidualLoss,
)
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_transformer import LatentSTTransformer


def test_gradient_norm_and_cosine_similarity_math():
    """Verify compute_gradient_norm and cosine similarity handle edge cases correctly."""
    # Orthogonal vectors
    v1 = [torch.tensor([1.0, 0.0, 0.0])]
    v2 = [torch.tensor([0.0, 2.0, 0.0])]

    norm1 = compute_gradient_norm(v1, v1)
    norm2 = compute_gradient_norm(v2, v2)
    assert math.isclose(norm1, 1.0, rel_tol=1e-5)
    assert math.isclose(norm2, 2.0, rel_tol=1e-5)

    cos_sim = compute_gradient_cosine_similarity(v1, v2)
    assert math.isclose(cos_sim, 0.0, abs_tol=1e-6)

    # Collinear vectors
    v3 = [torch.tensor([3.0, 0.0, 0.0])]
    cos_sim_collinear = compute_gradient_cosine_similarity(v1, v3)
    assert math.isclose(cos_sim_collinear, 1.0, rel_tol=1e-5)

    # Opposing vectors
    v4 = [torch.tensor([-1.0, 0.0, 0.0])]
    cos_sim_opposing = compute_gradient_cosine_similarity(v1, v4)
    assert math.isclose(cos_sim_opposing, -1.0, rel_tol=1e-5)

    # Non-finite gradient returns NaN
    nan_v = [torch.tensor([float("nan"), 1.0])]
    assert math.isnan(compute_gradient_norm(nan_v, nan_v))


@pytest.fixture
def mock_probe_environment(tmp_path):
    """Create a minimal self-contained environment with mock HDF5 and model weights."""
    h5_path = tmp_path / "probe_flow_Reynolds_1e3_Schmidt_1e0.hdf5"
    n_sims = 2
    t_steps = 10
    nx, ny = 32, 64

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("scalars/Reynolds", data=1000.0)
        f.create_dataset("scalars/Schmidt", data=1.0)
        f.create_dataset("dimensions/time", data=np.arange(t_steps, dtype=np.float32) * 0.05)
        f.create_dataset("t1_fields/velocity", data=np.zeros((n_sims, t_steps, nx, ny, 2), dtype=np.float32))
        f.create_dataset("t0_fields/pressure", data=np.zeros((n_sims, t_steps, nx, ny), dtype=np.float32))
        f.create_dataset("t0_fields/tracer", data=np.zeros((n_sims, t_steps, nx, ny), dtype=np.float32))

    # Split JSON
    split_file = tmp_path / "mock_split.json"
    with open(split_file, "w") as f:
        json.dump({
            "train": [{"file_path": str(h5_path), "traj_idx": 0}],
            "valid": [{"file_path": str(h5_path), "traj_idx": 1}],
        }, f)

    # Normalizer stats
    norm_file = tmp_path / "mock_stats.pt"
    torch.save({
        "mean": torch.zeros(4),
        "std": torch.ones(4),
    }, str(norm_file))

    # AE checkpoint
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    ae_file = tmp_path / "mock_ae.pt"
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
    }, str(ae_file))

    # D0 checkpoint
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=256,
        depth=6,
        num_heads=8,
        history_length=4,
        prediction_mode="direct",
    )
    wrapper = LatentForecasterWrapper(
        encoder=encoder,
        transformer=transformer,
        decoder=decoder,
        freeze_representation=True,
    )
    d0_file = tmp_path / "mock_d0.pt"
    torch.save({
        "model_state_dict": wrapper.state_dict(),
    }, str(d0_file))

    return {
        "data_root": str(tmp_path),
        "split_file": str(split_file),
        "norm_file": str(norm_file),
        "ae_ckpt": str(ae_file),
        "d0_ckpt": str(d0_file),
    }


def test_decoupled_batch_gradients_computation(mock_probe_environment):
    """Verify decoupled gradients arrive at transformer parameters and representation stays frozen."""
    env = mock_probe_environment
    device = torch.device("cpu")

    # Build model
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=256,
        depth=6,
        num_heads=8,
        history_length=4,
        prediction_mode="direct",
    )
    model = LatentForecasterWrapper(
        encoder=encoder,
        transformer=transformer,
        decoder=decoder,
        freeze_representation=True,
    )

    normalizer = FieldNormalizer()
    normalizer.load_state_dict(torch.load(env["norm_file"], weights_only=True))

    mom_loss_fn = NavierStokesMomentumResidualLoss(domain_size=(1.0, 2.0), dealias=True)
    tracer_loss_fn = TracerAdvectionDiffusionResidualLoss(domain_size=(1.0, 2.0), dealias=True)
    rollout_loss_fn = nn.MSELoss()

    # Synthetic batch
    B, L, H = 2, 4, 4
    batch = {
        "history": torch.randn(B, L, 4, 16, 32),
        "future": torch.randn(B, H, 4, 16, 32),
        "re": torch.tensor([1000.0, 1000.0]),
        "sc": torch.tensor([1.0, 1.0]),
        "dt": torch.tensor([0.05, 0.05]),
    }

    result = compute_decoupled_batch_gradients(
        model=model,
        batch=batch,
        normalizer=normalizer,
        mom_loss_fn=mom_loss_fn,
        tracer_loss_fn=tracer_loss_fn,
        rollout_loss_fn=rollout_loss_fn,
        device=device,
    )

    assert math.isfinite(result["loss_existing"])
    assert math.isfinite(result["loss_mom_raw"])
    assert math.isfinite(result["loss_tr_raw"])
    assert math.isfinite(result["norm_existing"])
    assert math.isfinite(result["norm_mom_raw"])
    assert math.isfinite(result["norm_tr_raw"])

    # Gradients genuinely arrive at transformer
    assert result["norm_existing"] > 0.0
    assert result["norm_mom_raw"] > 0.0
    assert result["norm_tr_raw"] > 0.0

    # Frozen status check
    assert result["frozen_encoder_active"] is False
    assert result["frozen_decoder_active"] is False


def test_run_pde_gradient_probe_mock_end_to_end(mock_probe_environment, tmp_path):
    """End-to-end integration test: run_pde_gradient_probe on mock checkpoints."""
    env = mock_probe_environment
    output_json = tmp_path / "mock_probe_output.json"

    result = run_pde_gradient_probe(
        split_file=env["split_file"],
        norm_file=env["norm_file"],
        ae_ckpt=env["ae_ckpt"],
        d0_ckpt=env["d0_ckpt"],
        data_root=env["data_root"],
        num_probe_batches=1,
        update_steps=2,
        history_length=4,
        horizon=4,
        candidate_weights=[(0.01, 0.01)],
        output_json=str(output_json),
        device=torch.device("cpu"),
    )

    assert os.path.exists(output_json)
    assert "q1_verification" in result
    assert "q2_loss_and_gradient_scales" in result
    assert "q3_validation_performance_comparison" in result

    q1 = result["q1_verification"]
    assert q1["gradients_finite"] is True
    assert q1["representation_frozen"] is True
    assert q1["mean_norm_mom_raw"] > 0.0
    assert q1["mean_norm_tr_raw"] > 0.0

    q2 = result["q2_loss_and_gradient_scales"]
    evals = q2["candidate_weight_evaluations"]
    assert len(evals) == 1
    assert "grad_ratio_mom" in evals[0]
    assert "grad_ratio_tr" in evals[0]

    q3 = result["q3_validation_performance_comparison"]
    assert "before" in q3["comparison"]["vrmse"]
    assert "after" in q3["comparison"]["vrmse"]
    assert len(q3["step_losses"]) == 2
