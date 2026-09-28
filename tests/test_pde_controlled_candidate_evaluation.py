"""Verification and governance contract tests for evaluate_pde_controlled_candidates.py.

Verifies:
1. End-to-end full evaluation benchmark execution across all sliding windows in mock environment.
2. Cryptographic governance contract for candidate checkpoints (parent SHA256, hashes, configs).
3. Subgroup separation and aggregation consistency across distinct physical conditions (e.g. Schmidt numbers).
"""

import json
import math
import os
import h5py
import numpy as np
import pytest
import torch

from scripts.audit_pde_residuals import compute_file_sha256
from scripts.evaluate_pde_controlled_candidates import (
    build_candidate_governance_metadata,
    load_forecaster_model,
    run_full_validation_benchmark,
)
from scripts.train_forecaster import LatentForecasterWrapper
from src.data.pipeline import compute_split_hash
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_transformer import LatentSTTransformer


@pytest.fixture
def mock_candidate_eval_env(tmp_path):
    """Create a minimal multi-trajectory, multi-window mock dataset and checkpoints."""
    # Create two HDF5 files with different Sc values to test subgroup separation
    h5_sc01 = tmp_path / "shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5"
    h5_sc10 = tmp_path / "shear_flow_Reynolds_1e4_Schmidt_1e0.hdf5"
    n_sims = 2
    t_steps = 10
    nx, ny = 32, 64

    for p, sc_val in [(h5_sc01, 0.1), (h5_sc10, 1.0)]:
        with h5py.File(p, "w") as f:
            f.create_dataset("scalars/Reynolds", data=10000.0)
            f.create_dataset("scalars/Schmidt", data=sc_val)
            f.create_dataset("dimensions/time", data=np.arange(t_steps, dtype=np.float32) * 0.1)
            f.create_dataset("t1_fields/velocity", data=np.zeros((n_sims, t_steps, nx, ny, 2), dtype=np.float32))
            f.create_dataset("t0_fields/pressure", data=np.zeros((n_sims, t_steps, nx, ny), dtype=np.float32))
            f.create_dataset("t0_fields/tracer", data=np.zeros((n_sims, t_steps, nx, ny), dtype=np.float32))

    split_file = tmp_path / "mock_split.json"
    with open(split_file, "w") as f:
        json.dump({
            "train": [{"file_path": str(h5_sc01), "traj_idx": 0}],
            "valid": [
                {"file_path": str(h5_sc01), "traj_idx": 1},
                {"file_path": str(h5_sc10), "traj_idx": 0},
            ],
            "test": [{"file_path": str(h5_sc10), "traj_idx": 1}],
        }, f)

    norm_file = tmp_path / "mock_stats.pt"
    torch.save({
        "mean": torch.zeros(4),
        "std": torch.ones(4),
    }, str(norm_file))

    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
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
        history_length=2,
        prediction_mode="direct",
    )
    wrapper = LatentForecasterWrapper(
        encoder=encoder,
        transformer=transformer,
        decoder=decoder,
        freeze_representation=True,
    )

    d0_file = tmp_path / "mock_d0.pt"
    p0_file = tmp_path / "mock_p0_step50.pt"
    pde_file = tmp_path / "mock_pde_step50.pt"

    torch.save({"model_state_dict": wrapper.state_dict()}, str(d0_file))
    torch.save({"step": 50, "model_state_dict": wrapper.state_dict()}, str(p0_file))
    torch.save({"step": 50, "model_state_dict": wrapper.state_dict()}, str(pde_file))

    return {
        "data_root": str(tmp_path),
        "split_file": str(split_file),
        "norm_file": str(norm_file),
        "ae_ckpt": str(ae_file),
        "d0_ckpt": str(d0_file),
        "p0_ckpt": str(p0_file),
        "pde_ckpt": str(pde_file),
    }


def test_evaluate_pde_controlled_candidates_mock_end_to_end(mock_candidate_eval_env, tmp_path):
    """Verify end-to-end full evaluation benchmark across all sliding windows in mock environment."""
    env = mock_candidate_eval_env
    output_json = tmp_path / "candidates_full_val.json"

    result = run_full_validation_benchmark(
        d0_checkpoint=env["d0_ckpt"],
        p0_checkpoint=env["p0_ckpt"],
        pde_checkpoint=env["pde_ckpt"],
        ae_checkpoint=env["ae_ckpt"],
        split_file=env["split_file"],
        norm_file=env["norm_file"],
        data_root=env["data_root"],
        split="valid",
        history_length=2,
        horizon=2,
        valid_stride=1,
        batch_size=2,
        num_workers=0,
        device=torch.device("cpu"),
        output_json=str(output_json),
    )

    # 1. Output JSON existence and primary sections
    assert os.path.exists(output_json)
    assert "metadata" in result
    assert "governance" in result
    assert "overall_comparison" in result
    assert "subgroup_comparison" in result
    assert "raw_overall_metrics" in result
    assert "verdict" in result

    # 2. Check window count: 2 trajectories * (10 - 2 - 2 + 1) = 2 * 7 = 14 windows
    eval_proto = result["metadata"]["evaluation_protocol"]
    assert eval_proto["total_windows_evaluated"] == 14
    assert eval_proto["split"] == "valid"

    # 3. Governance verification
    gov = result["governance"]
    assert "p0_step_50" in gov
    assert "pde_step_50" in gov
    assert gov["pde_step_50"]["parent_d0_sha256"] == compute_file_sha256(env["d0_ckpt"])
    assert gov["pde_step_50"]["checkpoint_sha256"] == compute_file_sha256(env["pde_ckpt"])
    assert gov["pde_step_50"]["pde_weights"]["lambda_mom"] == 2.5e-5

    # 4. Metrics table check
    overall = result["overall_comparison"]
    assert "vrmse_standard" in overall
    assert "div_rmse" in overall
    assert "res_u_rmse" in overall
    assert "res_v_rmse" in overall
    assert "res_s_rmse" in overall

    # 5. Subgroups check
    subgroups = result["subgroup_comparison"]
    assert "Sc_0.1" in subgroups
    assert "Sc_1.0" in subgroups
    assert subgroups["Sc_0.1"]["num_windows"] == 7
    assert subgroups["Sc_1.0"]["num_windows"] == 7

    # 6. Verdict summary
    verdict = result["verdict"]
    assert verdict["total_windows_evaluated"] == 14
    assert isinstance(verdict["pde_beats_p0_on_vrmse"], bool)
    assert "trade_off_observation" in verdict


def test_build_candidate_governance_metadata(mock_candidate_eval_env):
    """Verify build_candidate_governance_metadata produces rigorous self-describing metadata."""
    env = mock_candidate_eval_env
    d0_sha = compute_file_sha256(env["d0_ckpt"])
    ckpt_data = torch.load(env["pde_ckpt"], map_location="cpu", weights_only=False)

    gov = build_candidate_governance_metadata(
        checkpoint_path=env["pde_ckpt"],
        ckpt_data=ckpt_data,
        parent_d0_sha256=d0_sha,
        split_file=env["split_file"],
        norm_file=env["norm_file"],
        history_length=4,
        horizon=12,
        seed=42,
        lambda_mom=2.5e-5,
        lambda_tr=4.0e-6,
        branch_type="PDE_experiment",
        step=50,
    )

    assert gov["step"] == 50
    assert gov["branch_type"] == "PDE_experiment"
    assert gov["parent_d0_sha256"] == d0_sha
    assert gov["checkpoint_sha256"] == compute_file_sha256(env["pde_ckpt"])
    assert gov["seed"] == 42
    assert gov["training_horizon"] == 12
    assert gov["history_length"] == 4
    assert gov["pde_weights"]["lambda_mom"] == 2.5e-5
    assert gov["pde_weights"]["lambda_tr"] == 4.0e-6
    assert gov["residual_scales"]["tracer_scale_s"] == 0.02
    assert len(gov["split_hash"]) == 64
    assert len(gov["normalizer_hash"]) == 64
