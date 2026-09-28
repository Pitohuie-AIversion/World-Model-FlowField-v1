"""Verification and governance contract tests for evaluate_pde_controlled_candidates.py.

Verifies:
1. End-to-end full evaluation benchmark execution across all sliding windows in mock environment.
2. Cryptographic governance contract for candidate checkpoints (parent SHA256, hashes, configs).
3. Subgroup separation and aggregation consistency across distinct physical conditions (e.g. Schmidt numbers).
4. Training provenance strictly decoupled from evaluation HEAD (P1-2 regression defense).
5. use_spatial_pos preserved strictly from true model runtime semantics (P1-1 regression defense).
6. Semantic normalizer_hash strictly decoupled from file byte SHA256 (P1-3 regression defense).
7. Verdict avoids labeling error degradations as benefits (P2-1 regression defense).
"""

import inspect
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
    build_scientific_verdict,
    load_forecaster_model,
    rebuild_metadata_and_verdict_only,
    resolve_candidate_spatial_pos,
    resolve_candidate_training_provenance,
    run_full_validation_benchmark,
)
from scripts.train_forecaster import LatentForecasterWrapper
from src.data.normalization import FieldNormalizer
from src.data.pipeline import compute_split_hash
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_transformer import LatentSTTransformer
from src.utils.provenance import compute_normalizer_hash


@pytest.fixture
def mock_candidate_eval_env(tmp_path):
    """Create a minimal multi-trajectory, multi-window mock dataset and checkpoints."""
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

    torch.save({"model_state_dict": wrapper.state_dict(), "config": {"use_spatial_pos": True}}, str(d0_file))
    torch.save({"step": 50, "model_state_dict": wrapper.state_dict()}, str(p0_file))
    torch.save({"step": 50, "model_state_dict": wrapper.state_dict()}, str(pde_file))

    training_rec_file = tmp_path / "mock_training_record.json"
    with open(training_rec_file, "w") as f:
        json.dump({
            "metadata": {
                "git_commit": "1234567890abcdef1234567890abcdef12345678",
                "git_dirty": False,
                "experiment_script": "scripts/run_pde_controlled_training.py",
                "experiment_script_sha256": "fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",
            }
        }, f)

    return {
        "data_root": str(tmp_path),
        "split_file": str(split_file),
        "norm_file": str(norm_file),
        "ae_ckpt": str(ae_file),
        "d0_ckpt": str(d0_file),
        "p0_ckpt": str(p0_file),
        "pde_ckpt": str(pde_file),
        "training_rec_file": str(training_rec_file),
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
        training_record_file=env["training_rec_file"],
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

    assert os.path.exists(output_json)
    assert "metadata" in result
    assert "governance" in result
    assert "overall_comparison" in result
    assert "subgroup_comparison" in result
    assert "raw_overall_metrics" in result
    assert "verdict" in result

    eval_proto = result["metadata"]["evaluation_protocol"]
    assert eval_proto["total_windows_evaluated"] == 14
    assert eval_proto["split"] == "valid"

    gov = result["governance"]
    assert "p0_step_50" in gov
    assert "pde_step_50" in gov
    assert gov["pde_step_50"]["parent_d0_sha256"] == compute_file_sha256(env["d0_ckpt"])
    assert gov["pde_step_50"]["checkpoint_sha256"] == compute_file_sha256(env["pde_ckpt"])
    assert gov["pde_step_50"]["pde_weights"]["lambda_mom"] == 2.5e-5
    assert gov["pde_step_50"]["use_spatial_pos"] is True

    subgroups = result["subgroup_comparison"]
    assert "Sc_0.1" in subgroups
    assert "Sc_1.0" in subgroups
    assert subgroups["Sc_0.1"]["num_windows"] == 7
    assert subgroups["Sc_1.0"]["num_windows"] == 7

    verdict = result["verdict"]
    assert "continued_training_delta_vs_d0_pct" in verdict
    assert "continued_training_benefit_pct" not in verdict


def test_governance_uses_training_provenance_not_evaluation_head(mock_candidate_eval_env):
    """P1-2 Defense: training_git_commit must originate from training record, not evaluation HEAD."""
    env = mock_candidate_eval_env
    d0_sha = compute_file_sha256(env["d0_ckpt"])
    ckpt_data = torch.load(env["pde_ckpt"], map_location="cpu", weights_only=False)

    gov = build_candidate_governance_metadata(
        checkpoint_path=env["pde_ckpt"],
        ckpt_data=ckpt_data,
        parent_d0_sha256=d0_sha,
        split_file=env["split_file"],
        norm_file=env["norm_file"],
        training_record_file=env["training_rec_file"],
        history_length=4,
        horizon=12,
        seed=42,
        lambda_mom=2.5e-5,
        lambda_tr=4.0e-6,
        branch_type="PDE_experiment",
        step=50,
    )

    expected_commit = "1234567890abcdef1234567890abcdef12345678"
    assert gov["training_git_commit"] == expected_commit
    assert gov["training_provenance"]["training_git_commit"] == expected_commit
    assert gov["training_provenance"]["provenance_source"] == "controlled_training_record"


def test_governance_preserves_true_spatial_pos_semantics(mock_candidate_eval_env):
    """P1-1 Defense: use_spatial_pos must not be hardcoded to False, but resolved from config/model."""
    env = mock_candidate_eval_env
    ckpt_data_empty = {}

    pos = resolve_candidate_spatial_pos(ckpt_data_empty, parent_d0_path=env["d0_ckpt"])
    assert pos is True

    # Test explicit override in config is honored
    ckpt_explicit_false = {"config": {"use_spatial_pos": False}}
    assert resolve_candidate_spatial_pos(ckpt_explicit_false) is False


def test_normalizer_semantic_hash_is_distinct_from_file_sha(mock_candidate_eval_env):
    """P1-3 Defense: normalizer_hash must be semantic content fingerprint, distinct from file byte SHA256."""
    env = mock_candidate_eval_env
    norm = FieldNormalizer()
    norm.load_state_dict(torch.load(env["norm_file"], map_location="cpu", weights_only=True))

    sem_hash = compute_normalizer_hash(norm)
    file_sha = compute_file_sha256(env["norm_file"])

    assert len(sem_hash) == 64
    assert len(file_sha) == 64
    assert sem_hash != file_sha

    ckpt_data = torch.load(env["pde_ckpt"], map_location="cpu", weights_only=False)
    gov = build_candidate_governance_metadata(
        checkpoint_path=env["pde_ckpt"],
        ckpt_data=ckpt_data,
        parent_d0_sha256="dummy_sha",
        split_file=env["split_file"],
        norm_file=env["norm_file"],
        normalizer=norm,
        training_record_file=env["training_rec_file"],
    )

    assert gov["normalizer_hash"] == sem_hash
    assert gov["normalizer_file_sha256"] == file_sha


def test_verdict_does_not_label_positive_error_delta_as_benefit():
    """P2-1 Defense: Verdict must use neutral delta fields and not claim degradation as benefit."""
    d0_overall = {"vrmse_standard": 0.250, "div_rmse": 0.120}
    p0_overall = {"vrmse_standard": 0.260, "div_rmse": 0.130}  # Degraded relative to D0
    pde_overall = {"vrmse_standard": 0.258, "div_rmse": 0.129}  # Slightly better than P0, worse than D0

    verdict = build_scientific_verdict(
        d0_overall=d0_overall,
        p0_overall=p0_overall,
        pde_overall=pde_overall,
        total_val_windows=100,
        split="valid",
    )

    assert "continued_training_delta_vs_d0_pct" in verdict
    assert "continued_training_benefit_pct" not in verdict
    assert verdict["continued_training_delta_vs_d0_pct"] > 0.0  # Error increased by +4.0%
    assert verdict["pde_beats_p0_on_vrmse"] is True
    assert verdict["pde_beats_d0_on_vrmse"] is False
    assert "Degrades" in verdict["trade_off_observation"] or "degrades" in verdict["trade_off_observation"]
    assert verdict["status"] == "PARTIAL_OR_UNPROVEN"
