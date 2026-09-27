"""Tests for PDE residual audit contracts, dataset fail-closed validation, and training pipeline integration.

Verifies:
1. Strict manifest path validation (no fallback to valid/).
2. Strict trajectory index bounds check (no modulo remapping).
3. Zero-sample audit fail-closed behavior (no fake zero metrics).
4. Dataset enforcement of required pressure, tracer, and valid uniform time grids.
5. Trainer rejection of PDE residual losses combined with pushforward_steps > 0.
6. Trainer loss fail-closed handling when physical parameters (Re, Sc, dt) are missing.
"""

import json
import os
import tempfile
import h5py
import numpy as np
import pytest
import torch

from scripts.audit_pde_residuals import (
    audit_pde_residuals,
    precheck_audit_manifest_and_samples,
)
from scripts.train_forecaster import (
    _compute_batch_loss,
    validate_forecaster_pde_config,
)
from src.data.pipeline import create_flow_dataloaders, create_flow_datasets
from src.data.shear_flow_dataset import ShearFlowDataset
from src.losses.navier_stokes import (
    NavierStokesMomentumResidualLoss,
    TracerAdvectionDiffusionResidualLoss,
)


@pytest.fixture
def dummy_h5_file(tmp_path):
    """Create a minimal valid HDF5 file with 2 trajectories, velocity, pressure, tracer, and time."""
    file_path = tmp_path / "dummy_shear_flow_Reynolds_1e3_Schmidt_1e0.hdf5"
    n_sims = 2
    t_steps = 10
    nx, ny = 32, 64

    with h5py.File(file_path, "w") as f:
        # Scalars
        f.create_dataset("scalars/Reynolds", data=1000.0)
        f.create_dataset("scalars/Schmidt", data=1.0)

        # Dimensions
        time_arr = np.arange(t_steps, dtype=np.float32) * 0.05
        f.create_dataset("dimensions/time", data=time_arr)

        # Fields
        vel = np.zeros((n_sims, t_steps, nx, ny, 2), dtype=np.float32)
        p = np.zeros((n_sims, t_steps, nx, ny), dtype=np.float32)
        s = np.zeros((n_sims, t_steps, nx, ny), dtype=np.float32)

        f.create_dataset("t1_fields/velocity", data=vel)
        f.create_dataset("t0_fields/pressure", data=p)
        f.create_dataset("t0_fields/tracer", data=s)

    return str(file_path)


def test_audit_missing_manifest_path_fails(tmp_path):
    """Scenario 2a: Manifest path pointing to non-existent file fails immediately without fallback."""
    split_file = tmp_path / "bad_split.json"
    split_data = {
        "valid": [{"file_path": "non_existent_dir/non_existent_file.hdf5", "traj_idx": 0}]
    }
    with open(split_file, "w") as f:
        json.dump(split_data, f)

    # 1. Test isolated production precheck function (clean CI environment, no model weights needed)
    with pytest.raises(FileNotFoundError, match="Strict data contract violation: specified manifest file"):
        precheck_audit_manifest_and_samples(
            split_file=str(split_file),
            data_root=str(tmp_path),
            total_window=8,
            num_samples=1,
        )

    # 2. Integration: audit_pde_residuals must fail in precheck phase before inspecting model checkpoints
    with pytest.raises(FileNotFoundError, match="Strict data contract violation: specified manifest file"):
        audit_pde_residuals(
            split_file=str(split_file),
            norm_file="non_existent_norm.pt",
            ae_ckpt="non_existent_ae.pt",
            d0_ckpt="non_existent_d0.pt",
            data_root=str(tmp_path),
            num_samples=1,
        )


def test_audit_out_of_bounds_traj_fails(dummy_h5_file, tmp_path):
    """Scenario 2b: Trajectory index >= n_sims or negative fails immediately without taking modulo."""
    split_file = tmp_path / "out_of_bounds_split.json"
    # dummy_h5_file has n_sims=2. Request traj_idx=5.
    split_data = {
        "valid": [{"file_path": dummy_h5_file, "traj_idx": 5}]
    }
    with open(split_file, "w") as f:
        json.dump(split_data, f)

    with pytest.raises(IndexError, match="out of range"):
        precheck_audit_manifest_and_samples(
            split_file=str(split_file),
            data_root=str(tmp_path),
            total_window=8,
            num_samples=1,
        )

    with pytest.raises(IndexError, match="out of range"):
        audit_pde_residuals(
            split_file=str(split_file),
            norm_file="non_existent_norm.pt",
            ae_ckpt="non_existent_ae.pt",
            d0_ckpt="non_existent_d0.pt",
            data_root=str(tmp_path),
            num_samples=1,
        )

    # Test negative index rejection
    split_file_neg = tmp_path / "neg_traj_split.json"
    with open(split_file_neg, "w") as f:
        json.dump({"valid": [{"file_path": dummy_h5_file, "traj_idx": -1}]}, f)

    with pytest.raises(IndexError, match="out of range"):
        precheck_audit_manifest_and_samples(
            split_file=str(split_file_neg),
            data_root=str(tmp_path),
            total_window=8,
            num_samples=1,
        )


def test_audit_zero_samples_fails(tmp_path):
    """Scenario 4: When valid list is empty, audit fails closed rather than outputting zeros."""
    split_file = tmp_path / "empty_split.json"
    with open(split_file, "w") as f:
        json.dump({"valid": []}, f)

    with pytest.raises(ValueError, match="No valid trajectories found"):
        precheck_audit_manifest_and_samples(
            split_file=str(split_file),
            data_root=str(tmp_path),
            total_window=8,
            num_samples=1,
        )

    with pytest.raises(ValueError, match="No valid trajectories found"):
        audit_pde_residuals(
            split_file=str(split_file),
            norm_file="non_existent_norm.pt",
            ae_ckpt="non_existent_ae.pt",
            d0_ckpt="non_existent_d0.pt",
            data_root=str(tmp_path),
            num_samples=1,
        )


def test_audit_irregular_or_non_monotonic_time_fails(dummy_h5_file, tmp_path):
    """Scenario 4b: Audit precheck strictly rejects non-uniform or non-finite time grids across full window."""
    # 1. Non-uniform interval in evaluation window
    bad_time_h5 = tmp_path / "irregular_time.hdf5"
    with h5py.File(dummy_h5_file, "r") as src, h5py.File(bad_time_h5, "w") as dst:
        for k in src.keys():
            src.copy(k, dst)
        del dst["dimensions/time"]
        dst.create_dataset(
            "dimensions/time",
            data=np.array([0.0, 0.1, 0.2, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1], dtype=np.float32),
        )

    split_file = tmp_path / "irregular_split.json"
    with open(split_file, "w") as f:
        json.dump({"valid": [{"file_path": str(bad_time_h5), "traj_idx": 0}]}, f)

    with pytest.raises(ValueError, match="irregular intervals"):
        precheck_audit_manifest_and_samples(
            split_file=str(split_file),
            data_root=str(tmp_path),
            total_window=8,
            num_samples=1,
        )

    # 2. NaN in time grid
    nan_time_h5 = tmp_path / "nan_time.hdf5"
    with h5py.File(dummy_h5_file, "r") as src, h5py.File(nan_time_h5, "w") as dst:
        for k in src.keys():
            src.copy(k, dst)
        del dst["dimensions/time"]
        t_nan = np.arange(10, dtype=np.float32) * 0.1
        t_nan[4] = np.nan
        dst.create_dataset("dimensions/time", data=t_nan)

    split_file_nan = tmp_path / "nan_split.json"
    with open(split_file_nan, "w") as f:
        json.dump({"valid": [{"file_path": str(nan_time_h5), "traj_idx": 0}]}, f)

    with pytest.raises(ValueError, match="non-finite values"):
        precheck_audit_manifest_and_samples(
            split_file=str(split_file_nan),
            data_root=str(tmp_path),
            total_window=8,
            num_samples=1,
        )


def test_dataset_strict_pressure_and_tracer_validation(tmp_path):
    """Scenario 5a: ShearFlowDataset rejects files missing pressure or tracer when required."""
    h5_missing_p = tmp_path / "missing_p_Reynolds_1e4_Schmidt_1e-1.hdf5"
    with h5py.File(h5_missing_p, "w") as f:
        f.create_dataset("dimensions/time", data=np.arange(8, dtype=np.float32) * 0.1)
        f.create_dataset("t1_fields/velocity", data=np.zeros((1, 8, 16, 32, 2), dtype=np.float32))

    with pytest.raises(KeyError, match="require_pressure=True but pressure field is missing"):
        ShearFlowDataset(
            file_paths=[str(h5_missing_p)],
            history_length=4,
            horizon=2,
            require_pressure=True,
        )

    h5_missing_s = tmp_path / "missing_s_Reynolds_1e4_Schmidt_1e-1.hdf5"
    with h5py.File(h5_missing_s, "w") as f:
        f.create_dataset("dimensions/time", data=np.arange(8, dtype=np.float32) * 0.1)
        f.create_dataset("t1_fields/velocity", data=np.zeros((1, 8, 16, 32, 2), dtype=np.float32))
        f.create_dataset("t0_fields/pressure", data=np.zeros((1, 8, 16, 32), dtype=np.float32))

    with pytest.raises(KeyError, match="require_tracer=True but tracer field is missing"):
        ShearFlowDataset(
            file_paths=[str(h5_missing_s)],
            history_length=4,
            horizon=2,
            require_tracer=True,
        )


def test_dataset_strict_time_validation(tmp_path):
    """Scenario 6: ShearFlowDataset rejects non-existent, non-finite, non-monotonic, or non-uniform time."""
    # 1. Missing time when PDE is required
    h5_no_time = tmp_path / "no_time_Reynolds_1e4_Schmidt_1e-1.hdf5"
    with h5py.File(h5_no_time, "w") as f:
        f.create_dataset("t1_fields/velocity", data=np.zeros((1, 8, 16, 32, 2), dtype=np.float32))
        f.create_dataset("t0_fields/pressure", data=np.zeros((1, 8, 16, 32), dtype=np.float32))
        f.create_dataset("t0_fields/tracer", data=np.zeros((1, 8, 16, 32), dtype=np.float32))

    with pytest.raises(KeyError, match="requires explicit ground-truth time dataset"):
        ShearFlowDataset(
            file_paths=[str(h5_no_time)],
            history_length=4,
            horizon=2,
            require_pressure=True,
        )

    # 2. Non-finite time
    h5_nan_time = tmp_path / "nan_time_Reynolds_1e4_Schmidt_1e-1.hdf5"
    t_nan = np.arange(8, dtype=np.float32) * 0.1
    t_nan[3] = np.nan
    with h5py.File(h5_nan_time, "w") as f:
        f.create_dataset("dimensions/time", data=t_nan)
        f.create_dataset("t1_fields/velocity", data=np.zeros((1, 8, 16, 32, 2), dtype=np.float32))
        f.create_dataset("t0_fields/pressure", data=np.zeros((1, 8, 16, 32), dtype=np.float32))

    with pytest.raises(ValueError, match="contains non-finite values"):
        ShearFlowDataset(
            file_paths=[str(h5_nan_time)],
            history_length=4,
            horizon=2,
            require_pressure=True,
        )

    # 3. Non-monotonic time
    h5_non_mono = tmp_path / "non_mono_time_Reynolds_1e4_Schmidt_1e-1.hdf5"
    t_non_mono = np.array([0.0, 0.1, 0.05, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    with h5py.File(h5_non_mono, "w") as f:
        f.create_dataset("dimensions/time", data=t_non_mono)
        f.create_dataset("t1_fields/velocity", data=np.zeros((1, 8, 16, 32, 2), dtype=np.float32))
        f.create_dataset("t0_fields/pressure", data=np.zeros((1, 8, 16, 32), dtype=np.float32))

    with pytest.raises(ValueError, match="not strictly monotonically increasing"):
        ShearFlowDataset(
            file_paths=[str(h5_non_mono)],
            history_length=4,
            horizon=2,
            require_pressure=True,
        )

    # 4. Irregular / non-uniform intervals
    h5_irregular = tmp_path / "irregular_time_Reynolds_1e4_Schmidt_1e-1.hdf5"
    t_irregular = np.array([0.0, 0.1, 0.3, 0.4, 0.6, 0.7, 0.9, 1.0], dtype=np.float32)
    with h5py.File(h5_irregular, "w") as f:
        f.create_dataset("dimensions/time", data=t_irregular)
        f.create_dataset("t1_fields/velocity", data=np.zeros((1, 8, 16, 32, 2), dtype=np.float32))
        f.create_dataset("t0_fields/pressure", data=np.zeros((1, 8, 16, 32), dtype=np.float32))

    with pytest.raises(ValueError, match="irregular intervals"):
        ShearFlowDataset(
            file_paths=[str(h5_irregular)],
            history_length=4,
            horizon=2,
            require_pressure=True,
        )


def test_trainer_rejects_pde_with_pushforward():
    """Scenario 7: Real production validate_forecaster_pde_config rejects combining PDE loss with pushforward_steps > 0."""
    # 1. Momentum PDE + pushforward
    with pytest.raises(ValueError, match="cannot be combined with pushforward_steps"):
        validate_forecaster_pde_config(
            lambda_mom=0.5,
            lambda_tr=0.0,
            pushforward_steps=2,
            pushforward_mode="future",
        )

    # 2. Tracer PDE + pushforward
    with pytest.raises(ValueError, match="cannot be combined with pushforward_steps"):
        validate_forecaster_pde_config(
            lambda_mom=0.0,
            lambda_tr=0.1,
            pushforward_steps=1,
            pushforward_mode="future",
        )

    # 3. pushforward_mode='history' with pushforward_steps > 0
    with pytest.raises(NotImplementedError, match="pushforward_mode='history'"):
        validate_forecaster_pde_config(
            lambda_mom=0.0,
            lambda_tr=0.0,
            pushforward_steps=2,
            pushforward_mode="history",
        )

    # Permitted valid combinations must not raise
    validate_forecaster_pde_config(
        lambda_mom=0.0,
        lambda_tr=0.0,
        pushforward_steps=2,
        pushforward_mode="future",
    )
    validate_forecaster_pde_config(
        lambda_mom=0.1,
        lambda_tr=0.1,
        pushforward_steps=0,
        pushforward_mode="future",
    )


def test_audit_end_to_end_mock(dummy_h5_file, tmp_path):
    """End-to-end integration test: verifies audit_pde_residuals successfully executes

    all 4 operational levels and writes valid JSON metadata without requiring large pre-trained weights.
    """
    from src.models.encoder import Encoder2D
    from src.models.decoder import Decoder2D
    from src.models.latent_transformer import LatentSTTransformer
    from scripts.train_forecaster import LatentForecasterWrapper

    split_file = tmp_path / "valid_split.json"
    with open(split_file, "w") as f:
        json.dump({"valid": [{"file_path": dummy_h5_file, "traj_idx": 0}]}, f)

    norm_file = tmp_path / "mock_stats.pt"
    torch.save({
        "mean": torch.zeros(4),
        "std": torch.ones(4),
    }, str(norm_file))

    encoder = Encoder2D(in_channels=4, latent_channels=64)
    decoder = Decoder2D(latent_channels=64, out_channels=4)
    ae_file = tmp_path / "mock_ae.pt"
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
    }, str(ae_file))

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

    output_json = tmp_path / "mock_audit_out.json"

    result = audit_pde_residuals(
        split_file=str(split_file),
        norm_file=str(norm_file),
        ae_ckpt=str(ae_file),
        d0_ckpt=str(d0_file),
        data_root=str(tmp_path),
        num_samples=1,
        horizon=4,
        history_length=4,
        output_json=str(output_json),
        device=torch.device("cpu"),
    )

    assert os.path.exists(output_json)
    assert "metadata" in result
    assert "results" in result
    meta = result["metadata"]
    assert meta["sample_count"] == 1
    assert meta["ae_ckpt_sha256"] is not None
    assert meta["d0_ckpt_sha256"] is not None
    assert meta["git_commit"] != ""
    assert "dealias_protocol" in meta

    res = result["results"]
    for lvl in ["1_gt_full", "2_gt_downsampled", "3_ae_reconstruction", "4_d0_prediction"]:
        assert lvl in res
        for metric in ["div_rmse", "res_u_rmse", "res_v_rmse", "res_s_rmse"]:
            assert metric in res[lvl]
            assert np.isfinite(res[lvl][metric])


def test_compute_batch_loss_fail_closed_on_missing_params():
    """Scenario 5b: _compute_batch_loss fails closed if PDE loss is requested but Re, Sc, or dt is None."""
    pred = torch.randn(1, 2, 4, 16, 32)
    target = torch.randn(1, 2, 4, 16, 32)
    rollout_loss_fn = torch.nn.MSELoss()
    mom_fn = NavierStokesMomentumResidualLoss(domain_size=(1.0, 2.0))
    tracer_fn = TracerAdvectionDiffusionResidualLoss(domain_size=(1.0, 2.0))

    # 1. lambda_mom > 0 but re is None
    with pytest.raises(RuntimeError, match="lambda_mom > 0 requires physical Reynolds number"):
        _compute_batch_loss(
            pred=pred,
            q_future=target,
            normalizer=None,
            field_loss_space="physical",
            lambda_div=0.0,
            lambda_vort=0.0,
            rollout_loss_fn=rollout_loss_fn,
            lambda_mom=0.1,
            mom_loss_fn=mom_fn,
            re=None,
            dt=torch.tensor(0.1),
        )

    # 2. lambda_mom > 0 but dt is None
    with pytest.raises(RuntimeError, match="lambda_mom > 0 requires physical time step"):
        _compute_batch_loss(
            pred=pred,
            q_future=target,
            normalizer=None,
            field_loss_space="physical",
            lambda_div=0.0,
            lambda_vort=0.0,
            rollout_loss_fn=rollout_loss_fn,
            lambda_mom=0.1,
            mom_loss_fn=mom_fn,
            re=torch.tensor(1000.0),
            dt=None,
        )

    # 3. lambda_tr > 0 but sc is None
    with pytest.raises(RuntimeError, match="lambda_tr > 0 requires physical parameters"):
        _compute_batch_loss(
            pred=pred,
            q_future=target,
            normalizer=None,
            field_loss_space="physical",
            lambda_div=0.0,
            lambda_vort=0.0,
            rollout_loss_fn=rollout_loss_fn,
            lambda_tr=0.1,
            tracer_loss_fn=tracer_fn,
            re=torch.tensor(1000.0),
            sc=None,
            dt=torch.tensor(0.1),
        )
