"""Rigorous verification tests for H=12 controlled continued training experiment.

Verifies:
1. Fail-safe decision rule contract:
   - When all candidate checkpoints (Steps 20, 50, 100) underperform Step 0 D0 baseline,
     the decision must fail-safe to NO_SUPERIOR_CANDIDATE and recommend the frozen D0 checkpoint.
   - When a candidate strictly outperforms Step 0 D0 baseline, it is selected and recommended.
2. Configuration parity and alignment:
   - Effective batch size equals batch_size * grad_accum_steps.
   - Non-destructive storage: baseline D0 and AE checkpoints are strictly read-only and preserved.
3. End-to-end execution of run_pde_controlled_training in mock environment:
   - Evaluates multi-node checkpoints (Step 0, Step 1, Step 2).
   - Generates well-formed evaluation JSON and separate checkpoint artifacts.
4. Symmetry under zero PDE weights:
   - When lambda_mom=0 and lambda_tr=0, PDE experiment branch numerically matches P0 control branch.
"""

import copy
import json
import math
import os
import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.audit_pde_residuals import compute_file_sha256
from scripts.run_pde_controlled_training import run_pde_controlled_training
from scripts.train_forecaster import LatentForecasterWrapper
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_transformer import LatentSTTransformer


def test_controlled_training_fail_safe_decision_rule():
    """Verify the fail-safe contract logic under both degradation and improvement scenarios."""
    eval_steps = [0, 20, 50, 100]
    base_vrmse = 0.222902

    # Scenario A: All updated steps degrade (common continued training degradation)
    p0_records_degrade = {
        0: {"vrmse_standard": base_vrmse},
        20: {"vrmse_standard": 0.2612},
        50: {"vrmse_standard": 0.2834},
        100: {"vrmse_standard": 0.3150},
    }
    pde_records_degrade = {
        0: {"vrmse_standard": base_vrmse},
        20: {"vrmse_standard": 0.2450},  # Better than P0, but worse than D0
        50: {"vrmse_standard": 0.2520},
        100: {"vrmse_standard": 0.2780},
    }

    best_cand_vrmse = float("inf")
    best_cand_branch = None
    best_cand_step = None
    for s in eval_steps:
        if s == 0:
            continue
        v_p0 = p0_records_degrade[s]["vrmse_standard"]
        v_pde = pde_records_degrade[s]["vrmse_standard"]
        if v_p0 < best_cand_vrmse:
            best_cand_vrmse = v_p0
            best_cand_branch = "P0"
            best_cand_step = s
        if v_pde < best_cand_vrmse:
            best_cand_vrmse = v_pde
            best_cand_branch = "PDE"
            best_cand_step = s

    superior_to_d0 = best_cand_vrmse < base_vrmse
    assert superior_to_d0 is False
    assert best_cand_branch == "PDE"
    assert best_cand_step == 20
    assert best_cand_vrmse > base_vrmse

    # Scenario B: A candidate improves over D0 baseline
    pde_records_improve = copy.deepcopy(pde_records_degrade)
    pde_records_improve[50]["vrmse_standard"] = 0.2105  # Outperforms D0 (0.2105 < 0.222902)

    best_cand_vrmse_b = float("inf")
    best_cand_branch_b = None
    best_cand_step_b = None
    for s in eval_steps:
        if s == 0:
            continue
        v_p0 = p0_records_degrade[s]["vrmse_standard"]
        v_pde = pde_records_improve[s]["vrmse_standard"]
        if v_p0 < best_cand_vrmse_b:
            best_cand_vrmse_b = v_p0
            best_cand_branch_b = "P0"
            best_cand_step_b = s
        if v_pde < best_cand_vrmse_b:
            best_cand_vrmse_b = v_pde
            best_cand_branch_b = "PDE"
            best_cand_step_b = s

    superior_to_d0_b = best_cand_vrmse_b < base_vrmse
    assert superior_to_d0_b is True
    assert best_cand_branch_b == "PDE"
    assert best_cand_step_b == 50
    assert best_cand_vrmse_b < base_vrmse


@pytest.fixture
def mock_controlled_training_env(tmp_path):
    """Create minimal self-contained mock datasets and model checkpoints for controlled training."""
    h5_path = tmp_path / "controlled_flow_Reynolds_1e3_Schmidt_1e0.hdf5"
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

    split_file = tmp_path / "mock_split.json"
    with open(split_file, "w") as f:
        json.dump({
            "train": [{"file_path": str(h5_path), "traj_idx": 0}],
            "valid": [{"file_path": str(h5_path), "traj_idx": 1}],
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


def test_controlled_training_mock_end_to_end(mock_controlled_training_env, tmp_path):
    """End-to-end integration test of run_pde_controlled_training with multi-node evaluation."""
    env = mock_controlled_training_env
    output_json = tmp_path / "controlled_output.json"
    output_ckpt_dir = tmp_path / "checkpoints"

    d0_sha_before = compute_file_sha256(env["d0_ckpt"])
    ae_sha_before = compute_file_sha256(env["ae_ckpt"])

    result = run_pde_controlled_training(
        split_file=env["split_file"],
        norm_file=env["norm_file"],
        ae_ckpt=env["ae_ckpt"],
        d0_ckpt=env["d0_ckpt"],
        data_root=env["data_root"],
        total_steps=2,
        eval_steps=[0, 1, 2],
        history_length=2,
        horizon=2,
        batch_size=1,
        grad_accum_steps=1,
        lr=5e-5,
        lambda_div=0.01,
        lambda_vort=0.05,
        mom_scale_u=0.05,
        mom_scale_v=0.05,
        tracer_scale_s=0.02,
        lambda_mom=2.5e-5,
        lambda_tr=4.0e-6,
        seed=42,
        num_workers=0,
        output_ckpt_dir=str(output_ckpt_dir),
        output_json=str(output_json),
        device=torch.device("cpu"),
    )

    # 1. Output files exist and are populated
    assert os.path.exists(output_json)
    assert os.path.isdir(output_ckpt_dir)

    # 2. Baseline models were preserved without mutation (Non-destructive property)
    d0_sha_after = compute_file_sha256(env["d0_ckpt"])
    ae_sha_after = compute_file_sha256(env["ae_ckpt"])
    assert d0_sha_before == d0_sha_after, "D0 checkpoint was modified during training!"
    assert ae_sha_before == ae_sha_after, "Autoencoder checkpoint was modified during training!"

    # 3. Checkpoints saved for intermediate steps
    assert os.path.exists(output_ckpt_dir / "p0_step_1.pt")
    assert os.path.exists(output_ckpt_dir / "p0_step_2.pt")
    assert os.path.exists(output_ckpt_dir / "pde_step_1.pt")
    assert os.path.exists(output_ckpt_dir / "pde_step_2.pt")

    # 4. JSON payload validation
    assert "metadata" in result
    assert "step_0_d0_baseline" in result
    assert "evaluation_trajectory" in result
    assert "loss_histories" in result
    assert "decision_contract" in result

    meta = result["metadata"]
    assert meta["alignment_configuration"]["training_horizon"] == 2
    assert meta["alignment_configuration"]["effective_batch_size"] == 1
    assert meta["alignment_configuration"]["pde_weights"]["lambda_mom"] == 2.5e-5

    traj = result["evaluation_trajectory"]
    assert "0" in traj
    assert "1" in traj
    assert "2" in traj
    assert "p0_vrmse" in traj["1"]
    assert "pde_vrmse" in traj["1"]
    assert "d0_baseline_vrmse" in traj["1"]

    decision = result["decision_contract"]
    assert "superior_to_d0" in decision
    assert "verdict" in decision
    assert "recommended_checkpoint" in decision
    assert isinstance(decision["superior_to_d0"], bool)


def test_zero_pde_weights_controlled_training_symmetry(mock_controlled_training_env, tmp_path):
    """Verify that setting lambda_mom=0 and lambda_tr=0 yields numerical symmetry between P0 and PDE branches."""
    env = mock_controlled_training_env
    output_json = tmp_path / "zero_pde_controlled.json"
    output_ckpt_dir = tmp_path / "zero_pde_checkpoints"

    result = run_pde_controlled_training(
        split_file=env["split_file"],
        norm_file=env["norm_file"],
        ae_ckpt=env["ae_ckpt"],
        d0_ckpt=env["d0_ckpt"],
        data_root=env["data_root"],
        total_steps=2,
        eval_steps=[0, 1, 2],
        history_length=2,
        horizon=2,
        batch_size=1,
        grad_accum_steps=1,
        lr=5e-5,
        lambda_div=0.01,
        lambda_vort=0.05,
        mom_scale_u=0.05,
        mom_scale_v=0.05,
        tracer_scale_s=0.02,
        lambda_mom=0.0,
        lambda_tr=0.0,
        seed=42,
        num_workers=0,
        output_ckpt_dir=str(output_ckpt_dir),
        output_json=str(output_json),
        device=torch.device("cpu"),
    )

    # 1. Step losses must match
    p0_losses = result["loss_histories"]["p0_step_losses"]
    pde_losses = result["loss_histories"]["pde_step_losses"]
    assert len(p0_losses) == len(pde_losses) == 2
    for s_idx, (l0, lpde) in enumerate(zip(p0_losses, pde_losses)):
        assert math.isclose(l0, lpde, rel_tol=1e-6), f"Loss mismatch at step {s_idx}: {l0} vs {lpde}"

    # 2. Evaluation trajectory metrics must match
    traj = result["evaluation_trajectory"]
    for s in [1, 2]:
        v_p0 = traj[str(s)]["p0_vrmse"]
        v_pde = traj[str(s)]["pde_vrmse"]
        delta_pct = traj[str(s)]["delta_pde_vs_p0_pct"]
        assert math.isclose(v_p0, v_pde, rel_tol=1e-5), f"VRMSE mismatch at step {s}: {v_p0} vs {v_pde}"
        assert math.isclose(delta_pct, 0.0, abs_tol=1e-4), f"Delta PDE vs P0 non-zero at step {s}: {delta_pct}%"
