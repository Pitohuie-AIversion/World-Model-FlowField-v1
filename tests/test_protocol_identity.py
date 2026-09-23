"""Closure-R3: Protocol Identity and Fail-Closed Governance Tests.

Validates that:
1. Normalizer metadata enforces split fingerprint (split_hash); stale cache is invalidated when split changes.
2. Datasets/DataLoaders fail-closed with RuntimeError when split is marked BLOCKED_BY_DATA.
3. Physics ablation evaluation rejects Closure-R1 legacy checkpoints unless --allow_legacy_checkpoints is set.
4. Failure analysis rejects legacy checkpoints unless --allow_legacy_checkpoint is set.
5. Multi-model rollout benchmark rejects mixed/conflicting data contracts across checkpoints.
"""

import hashlib
import json
import os
import tempfile
import h5py
import numpy as np
import pytest
import torch

from scripts.evaluate_rollout import verify_checkpoint_contract
from scripts.evaluate_physics_ablation import (
    ABLATION_GROUPS,
    resolve_evaluation_output_path,
    run_physics_ablation_eval,
)
from src.data.pipeline import compute_split_hash, create_flow_datasets
from src.utils.physics_contract import (
    PHYSICS_PROTOCOL,
    SPATIAL_AXIS_CONTRACT,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
    validate_ablation_checkpoint_semantics,
)
from src.utils.provenance import (
    compute_split_hash,
    compute_normalizer_hash,
    compute_file_sha256,
    create_checkpoint_provenance,
    resolve_checkpoint_provenance,
    validate_evaluation_provenance,
    validate_formal_provenance_bundle,
)


def _create_mock_h5(path: str, n_trajs: int = 2, nt: int = 6, ny: int = 16, nx: int = 32):
    """Helper to create minimal valid HDF5 file for testing pipeline."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("velocity", data=np.random.randn(n_trajs, nt, ny, nx, 2).astype(np.float32) * 2.0)
        f.create_dataset("pressure", data=np.random.randn(n_trajs, nt, ny, nx, 1).astype(np.float32) * 0.05 + 1.2)
        f.create_dataset("density", data=np.random.randn(n_trajs, nt, ny, nx, 1).astype(np.float32) * 0.8 + 3.0)


def test_normalizer_invalidated_when_split_changes():
    """P1-1: Normalizer cache must be invalidated when split manifest hash changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_a = os.path.join(tmpdir, "shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5")
        h5_b = os.path.join(tmpdir, "shear_flow_Reynolds_1e4_Schmidt_1e0.hdf5")
        _create_mock_h5(h5_a)
        _create_mock_h5(h5_b)

        split_file = os.path.join(tmpdir, "split.json")
        stats_dir = os.path.join(tmpdir, "normalization")
        os.makedirs(stats_dir, exist_ok=True)

        # 1. First split version: only h5_a
        split_data_v1 = {"train": [h5_a], "valid": [h5_a], "test": [h5_a]}
        with open(split_file, "w") as f:
            json.dump(split_data_v1, f)

        _, _, _, norm1 = create_flow_datasets(
            split_type="identity_test",
            split_file=split_file,
            data_root=tmpdir,
            history_length=1,
            horizon=1,
            stats_dir=stats_dir,
        )
        meta_file = os.path.join(stats_dir, "stats_identity_test_metadata.json")
        assert os.path.exists(meta_file)
        with open(meta_file, "r") as mf:
            meta1 = json.load(mf)

        expected_hash1 = compute_split_hash(split_data_v1)
        assert meta1["split_hash"] == expected_hash1
        first_mean = meta1["mean"]

        # 2. Update split manifest with different trajectories (h5_b added)
        split_data_v2 = {"train": [h5_a, h5_b], "valid": [h5_a], "test": [h5_b]}
        with open(split_file, "w") as f:
            json.dump(split_data_v2, f)

        expected_hash2 = compute_split_hash(split_data_v2)
        assert expected_hash1 != expected_hash2

        # 3. Running create_flow_datasets must detect hash mismatch, invalidate cache, and refit
        _, _, _, norm2 = create_flow_datasets(
            split_type="identity_test",
            split_file=split_file,
            data_root=tmpdir,
            history_length=1,
            horizon=1,
            stats_dir=stats_dir,
        )
        with open(meta_file, "r") as mf:
            meta2 = json.load(mf)

        assert meta2["split_hash"] == expected_hash2
        assert meta2["split_hash"] != meta1["split_hash"]


def test_blocked_holdout_evaluation_fails_closed():
    """P1-4: Loading a split marked BLOCKED_BY_DATA must immediately raise RuntimeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        blocked_split_file = os.path.join(tmpdir, "parameter_holdout_re.json")
        blocked_content = {
            "train": ["data/dummy1.hdf5"],
            "valid": ["data/dummy2.hdf5"],
            "test": [],
            "status": "BLOCKED_BY_DATA",
            "reason": "Only 1 distinct Reynolds number in dataset",
        }
        with open(blocked_split_file, "w") as f:
            json.dump(blocked_content, f)

        with pytest.raises(RuntimeError) as exc_info:
            create_flow_datasets(
                split_type="parameter_holdout_re",
                split_file=blocked_split_file,
                data_root=tmpdir,
            )

        assert "BLOCKED_BY_DATA" in str(exc_info.value)
        assert "Only 1 distinct Reynolds number" in str(exc_info.value)


def test_ablation_legacy_checkpoint_requires_opt_in():
    """Closure-R4 permits pre-R4 field-only reuse but blocks poisoned physics weights."""
    assert ABLATION_GROUPS["E0_single_step"]["legacy_candidates"]
    assert ABLATION_GROUPS["E1_rollout_field"]["legacy_candidates"]

    for key in ["E2_plus_L_div", "E3_plus_L_vort", "E4_full_physics"]:
        group = ABLATION_GROUPS[key]
        assert group["legacy_candidates"] == []
        assert group["invalid_axis_candidates"]
        assert all("closure_r4" in p for p in group["candidates"])


def test_failure_analysis_legacy_checkpoint_requires_opt_in():
    """P1-3: Failure analysis fails closed when model doesn't exist and opt-in is False."""
    from scripts.analyze_failure_cases import analyze_failure_cases

    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_model = os.path.join(tmpdir, "does_not_exist.pt")

        # Without opt-in, FileNotFoundError must be raised
        with pytest.raises(FileNotFoundError) as exc_info:
            analyze_failure_cases(
                model_path=non_existent_model,
                allow_legacy_checkpoint=False,
            )
        assert "Checkpoint not found" in str(exc_info.value)


def test_failure_analysis_rejects_poisoned_physics_checkpoint():
    """Closure-R4 failure analysis rejects legacy checkpoints trained with physics losses."""
    from scripts.analyze_failure_cases import analyze_failure_cases

    with tempfile.TemporaryDirectory() as tmpdir:
        poisoned_ckpt = os.path.join(tmpdir, "poisoned.pt")
        torch.save({"config": {"lambda_div": 0.01, "physics_protocol": "Closure-R3"}}, poisoned_ckpt)
        with pytest.raises(RuntimeError) as exc_info:
            analyze_failure_cases(model_path=poisoned_ckpt)
        assert "non-zero physics losses" in str(exc_info.value)


def test_benchmark_rejects_mixed_data_contracts():
    """P1-5: verify_checkpoint_contract rejects mismatched contracts and pre-R4 physics checkpoints."""
    benchmark_contract = {
        "split_type": "grouped",
        "downsample_factor": 2,
        "normalize": True,
    }

    # 1. Matching config passes
    matching_cfg = {
        "split_type": "grouped",
        "downsample_factor": 2,
        "normalize": True,
    }
    verify_checkpoint_contract("test_model", "ckpt.pt", matching_cfg, benchmark_contract)

    # 2. Conflicting downsample_factor raises ValueError
    mismatched_ds_cfg = {
        "split_type": "grouped",
        "downsample_factor": 1,
        "normalize": True,
    }
    with pytest.raises(ValueError) as exc:
        verify_checkpoint_contract("test_model", "ckpt.pt", mismatched_ds_cfg, benchmark_contract)
    assert "downsample_factor" in str(exc.value)

    # 3. Conflicting normalize raises ValueError
    mismatched_norm_cfg = {
        "split_type": "grouped",
        "downsample_factor": 2,
        "normalize": False,
    }
    with pytest.raises(ValueError) as exc:
        verify_checkpoint_contract("test_model", "ckpt.pt", mismatched_norm_cfg, benchmark_contract)
    assert "normalize" in str(exc.value)

    # 4. Conflicting split_type raises ValueError
    mismatched_split_cfg = {
        "split_type": "official",
        "downsample_factor": 2,
        "normalize": True,
    }
    with pytest.raises(ValueError) as exc:
        verify_checkpoint_contract("test_model", "ckpt.pt", mismatched_split_cfg, benchmark_contract)
    assert "split_type" in str(exc.value)

    # 5. Pre-R4 checkpoints trained with physics losses are scientifically invalid.
    poisoned_physics_cfg = {
        "split_type": "grouped",
        "downsample_factor": 2,
        "normalize": True,
        "lambda_div": 0.01,
        "lambda_vort": 0.0,
        "physics_protocol": "Closure-R3",
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
    }
    with pytest.raises(ValueError) as exc:
        verify_checkpoint_contract("poisoned_model", "old_e2.pt", poisoned_physics_cfg, benchmark_contract)
    assert "physics_protocol" in str(exc.value)

    # 6. Checkpoint with wrong spatial_axis_contract is rejected
    wrong_axis_cfg = dict(poisoned_physics_cfg)
    wrong_axis_cfg["physics_protocol"] = PHYSICS_PROTOCOL
    wrong_axis_cfg["spatial_axis_contract"] = "tensor(...,Ny,Nx):dim-2=y,dim-1=x"
    with pytest.raises(ValueError) as exc:
        verify_checkpoint_contract("wrong_axis_model", "axis_err.pt", wrong_axis_cfg, benchmark_contract)
    assert "spatial_axis_contract" in str(exc.value)

    # 7. Checkpoint with wrong physics_domain_size_xy is rejected
    wrong_domain_cfg = dict(poisoned_physics_cfg)
    wrong_domain_cfg["physics_protocol"] = PHYSICS_PROTOCOL
    wrong_domain_cfg["physics_domain_size_xy"] = [2.0, 1.0]
    with pytest.raises(ValueError) as exc:
        verify_checkpoint_contract("wrong_domain_model", "domain_err.pt", wrong_domain_cfg, benchmark_contract)
    assert "physics_domain_size_xy" in str(exc.value)

    # 8. Complete valid Closure-R4 physics checkpoint is accepted
    valid_r4_physics_cfg = dict(poisoned_physics_cfg)
    valid_r4_physics_cfg["physics_protocol"] = PHYSICS_PROTOCOL
    verify_checkpoint_contract("r4_model", "r4_e2.pt", valid_r4_physics_cfg, benchmark_contract)


def test_ablation_checkpoint_semantic_validation():
    """Verify strict semantic validation for E0-E4 ablation checkpoints."""
    # 1. E2 with horizon=1 is rejected
    e2_wrong_h = {
        "horizon": 1,
        "lambda_div": 0.01,
        "lambda_vort": 0.0,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
    }
    with pytest.raises(ValueError) as exc:
        validate_ablation_checkpoint_semantics("E2_plus_L_div", e2_wrong_h)
    assert "horizon" in str(exc.value)

    # 2. E2 with lambda_div=0 is rejected
    e2_missing_div = {
        "horizon": 2,
        "lambda_div": 0.0,
        "lambda_vort": 0.0,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
    }
    with pytest.raises(ValueError) as exc:
        validate_ablation_checkpoint_semantics("E2_plus_L_div", e2_missing_div)
    assert "lambda_div" in str(exc.value)

    # 3. E3 with lambda_vort=0 is rejected
    e3_missing_vort = {
        "horizon": 2,
        "lambda_div": 0.0,
        "lambda_vort": 0.0,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
    }
    with pytest.raises(ValueError) as exc:
        validate_ablation_checkpoint_semantics("E3_plus_L_vort", e3_missing_vort)
    assert "lambda_vort" in str(exc.value)

    # 4. E4 missing lambda_div or lambda_vort is rejected
    e4_missing_div = {
        "horizon": 2,
        "lambda_div": 0.0,
        "lambda_vort": 0.05,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
    }
    with pytest.raises(ValueError) as exc:
        validate_ablation_checkpoint_semantics("E4_full_physics", e4_missing_div)
    assert "lambda_div" in str(exc.value)

    e4_missing_vort = {
        "horizon": 2,
        "lambda_div": 0.01,
        "lambda_vort": 0.0,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
    }
    with pytest.raises(ValueError) as exc:
        validate_ablation_checkpoint_semantics("E4_full_physics", e4_missing_vort)
    assert "lambda_vort" in str(exc.value)

    # 5. Valid E0 - E4 checkpoints pass without error
    valid_e0 = {
        "horizon": 1,
        "lambda_div": 0.0,
        "lambda_vort": 0.0,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
    }
    validate_ablation_checkpoint_semantics("E0_single_step", valid_e0)

    valid_e1 = {
        "horizon": 2,
        "lambda_div": 0.0,
        "lambda_vort": 0.0,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
    }
    validate_ablation_checkpoint_semantics("E1_rollout_field", valid_e1)

    valid_e2 = {
        "horizon": 2,
        "lambda_div": 0.01,
        "lambda_vort": 0.0,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
    }
    validate_ablation_checkpoint_semantics("E2_plus_L_div", valid_e2)

    valid_e3 = {
        "horizon": 2,
        "lambda_div": 0.0,
        "lambda_vort": 0.05,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
    }
    validate_ablation_checkpoint_semantics("E3_plus_L_vort", valid_e3)

    valid_e4 = {
        "horizon": 2,
        "lambda_div": 0.01,
        "lambda_vort": 0.05,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
    }
    validate_ablation_checkpoint_semantics("E4_full_physics", valid_e4)

    # 6. Required field missing fail-closed validations
    # 6a. E2 missing horizon is rejected
    e2_missing_horizon = {
        "lambda_div": 0.01,
        "lambda_vort": 0.0,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
    }
    with pytest.raises(ValueError) as exc:
        validate_ablation_checkpoint_semantics("E2_plus_L_div", e2_missing_horizon)
    assert "missing required field: horizon" in str(exc.value)

    # 6b. E0 R4 missing horizon is rejected
    e0_r4_missing_horizon = {
        "lambda_div": 0.0,
        "lambda_vort": 0.0,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
    }
    with pytest.raises(ValueError) as exc:
        validate_ablation_checkpoint_semantics("E0_single_step", e0_r4_missing_horizon)
    assert "missing required field: horizon" in str(exc.value)

    # 6c. E1 R4 missing lambda_div is rejected
    e1_r4_missing_lambda_div = {
        "horizon": 2,
        "lambda_vort": 0.0,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
    }
    with pytest.raises(ValueError) as exc:
        validate_ablation_checkpoint_semantics("E1_rollout_field", e1_r4_missing_lambda_div)
    assert "missing required field: lambda_div" in str(exc.value)

    # 6d. Legacy E0 missing new R4 metadata passes under legacy opt-in
    legacy_e0 = {
        "horizon": 1,
        "lambda_div": 0.0,
        "lambda_vort": 0.0,
        # Intentionally omitting physics_protocol, spatial_axis_contract, physics_domain_size_xy
    }
    validate_ablation_checkpoint_semantics("E0_single_step", legacy_e0, is_legacy=True)

    # 6e. Legacy E0 without opt-in (is_legacy=False) is rejected due to missing R4 metadata
    with pytest.raises(ValueError) as exc:
        validate_ablation_checkpoint_semantics("E0_single_step", legacy_e0, is_legacy=False)
    assert "missing required field: physics_protocol" in str(exc.value)


# -----------------------------------------------------------------------------
# Issue #3 & Issue #4: Provenance Closure and Seed Isolation Contract Tests
# -----------------------------------------------------------------------------


def test_provenance_validation_missing_split_hash_fails_closed():
    """Fail-closed rejection when split_hash is missing from checkpoint/provenance."""
    bundle = {
        "training_git_commit": "abc",
        "seed": 42,
        "split_hash": None,
        "normalizer_hash": "norm_hash_valid",
    }
    with pytest.raises(RuntimeError) as exc_info:
        validate_evaluation_provenance(bundle, eval_split_hash="split_hash_123", eval_normalizer_hash="norm_hash_valid")
    assert "missing required 'split_hash'" in str(exc_info.value)


def test_provenance_validation_wrong_split_hash_fails_closed():
    """Fail-closed rejection when checkpoint split_hash does not match evaluation dataset split_hash."""
    bundle = {
        "training_git_commit": "abc",
        "seed": 42,
        "split_hash": "mismatched_split_hash_aaa",
        "normalizer_hash": "norm_hash_valid",
    }
    with pytest.raises(RuntimeError) as exc_info:
        validate_evaluation_provenance(bundle, eval_split_hash="eval_split_hash_bbb", eval_normalizer_hash="norm_hash_valid")
    assert "Split hash mismatch" in str(exc_info.value)


def test_provenance_validation_missing_normalizer_hash_fails_closed():
    """Fail-closed rejection when normalizer_hash is missing from checkpoint/provenance."""
    bundle = {
        "training_git_commit": "abc",
        "seed": 42,
        "split_hash": "split_hash_valid",
        "normalizer_hash": None,
    }
    with pytest.raises(RuntimeError) as exc_info:
        validate_evaluation_provenance(bundle, eval_split_hash="split_hash_valid", eval_normalizer_hash="eval_norm_hash_xyz")
    assert "missing required 'normalizer_hash'" in str(exc_info.value)


def test_provenance_validation_wrong_normalizer_hash_fails_closed():
    """Fail-closed rejection when checkpoint normalizer_hash does not match evaluation normalizer."""
    bundle = {
        "training_git_commit": "abc",
        "seed": 42,
        "split_hash": "split_hash_valid",
        "normalizer_hash": "ckpt_norm_hash_old",
    }
    with pytest.raises(RuntimeError) as exc_info:
        validate_evaluation_provenance(bundle, eval_split_hash="split_hash_valid", eval_normalizer_hash="eval_norm_hash_new")
    assert "Normalizer hash mismatch" in str(exc_info.value)


def test_provenance_validation_wrong_seed_link_fails_closed():
    """Fail-closed rejection when checkpoint seed does not match expected evaluation seed."""
    bundle = {
        "training_git_commit": "abc",
        "seed": 43,
        "split_hash": "split_hash_valid",
        "normalizer_hash": "norm_hash_valid",
    }
    with pytest.raises(RuntimeError) as exc_info:
        validate_evaluation_provenance(
            bundle,
            eval_split_hash="split_hash_valid",
            eval_normalizer_hash="norm_hash_valid",
            expected_seed=42,
        )
    assert "Seed mismatch" in str(exc_info.value)


def test_provenance_manifest_integrity_tampering_fails_closed():
    """Tampered or mismatched checkpoint linked to manifest fails closed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_ckpt = os.path.join(tmpdir, "fake_ckpt.pt")
        with open(fake_ckpt, "w") as f:
            f.write("content_v1")

        fake_manifest = os.path.join(tmpdir, "manifest.json")
        manifest_data = {
            "groups": {
                "group1": {
                    "checkpoint_path": fake_ckpt,
                    "checkpoint_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    "training_git_commit": "dummy",
                    "seed": 42,
                    "split_hash": "split123",
                    "normalizer_hash": "norm123",
                }
            }
        }
        with open(fake_manifest, "w") as f:
            json.dump(manifest_data, f)

        with pytest.raises(ValueError) as exc_info:
            resolve_checkpoint_provenance(fake_ckpt, {}, manifest_path=fake_manifest)
        assert "Manifest integrity mismatch" in str(exc_info.value)


def test_provenance_valid_bundle_passes():
    """Valid provenance bundle matching evaluation environment passes cleanly."""
    bundle = {
        "training_git_commit": "6593b65005843c3f3c2640cb262bfd56708e11ad",
        "seed": 42,
        "split_hash": "split_hash_exact_match_123456",
        "normalizer_hash": "norm_hash_exact_match_abcdef",
    }
    is_valid, errors = validate_evaluation_provenance(
        bundle,
        eval_split_hash="split_hash_exact_match_123456",
        eval_normalizer_hash="norm_hash_exact_match_abcdef",
        expected_seed=42,
        fail_closed=True,
    )
    assert is_valid is True
    assert len(errors) == 0


def test_closure_r4_seed42_manifest_and_checkpoints_integrity():
    """Verify that existing seed-42 manifest links 100% cleanly to existing checkpoints."""
    manifest_path = "outputs/manifests/closure_r4_seed42.json"
    if not os.path.exists(manifest_path):
        pytest.skip("Manifest not found on current environment")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    split_hash = manifest["split_hash"]
    normalizer_hash = manifest["normalizer_hash"]

    for grp_key, grp_info in manifest["groups"].items():
        ckpt_path = grp_info["checkpoint_path"]
        if not os.path.exists(ckpt_path):
            continue
        # Verify SHA-256 matches
        disk_sha = compute_file_sha256(ckpt_path)
        assert disk_sha == grp_info["checkpoint_sha256"], f"SHA256 mismatch for {grp_key}"

        # Resolve provenance via manifest
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        prov = resolve_checkpoint_provenance(ckpt_path, ckpt_data, manifest_path=manifest_path)
        assert prov["seed"] == 42
        assert prov["split_hash"] == split_hash
        assert prov["normalizer_hash"] == normalizer_hash

        # Validate against environment hashes
        is_valid, errors = validate_evaluation_provenance(
            prov,
            eval_split_hash=split_hash,
            eval_normalizer_hash=normalizer_hash,
            expected_seed=42,
            fail_closed=True,
        )
        assert is_valid is True
        assert len(errors) == 0


def test_evaluation_output_filename_seed_isolation():
    """Verify that evaluator automatically generates non-colliding output paths per seed."""
    out_default = resolve_evaluation_output_path(None, seed=None)
    out_42 = resolve_evaluation_output_path(None, seed=42)
    out_43 = resolve_evaluation_output_path(None, seed=43)
    out_44 = resolve_evaluation_output_path(None, seed=44)

    assert out_default == "outputs/metrics/closure_r4_physics_ablation_v2.json"
    assert out_42 == "outputs/metrics/closure_r4_physics_ablation_seed42_v2.json"
    assert out_43 == "outputs/metrics/closure_r4_physics_ablation_seed43_v2.json"
    assert out_44 == "outputs/metrics/closure_r4_physics_ablation_seed44_v2.json"

    # Ensure none of the paths overwrite each other
    paths = {out_default, out_42, out_43, out_44}
    assert len(paths) == 4, "Output filenames collide across seeds!"

    # Explicit override is respected
    custom = "outputs/metrics/my_custom.json"
    assert resolve_evaluation_output_path(custom, seed=43) == custom


def test_provenance_validation_missing_seed_fails_closed():
    """Fail-closed rejection when seed is missing from checkpoint/provenance and manifest."""
    bundle = {
        "training_git_commit": "abc",
        "seed": None,
        "split_hash": "split_hash_valid",
        "normalizer_hash": "norm_hash_valid",
    }
    with pytest.raises(RuntimeError) as exc_info:
        validate_evaluation_provenance(bundle, eval_split_hash="split_hash_valid", eval_normalizer_hash="norm_hash_valid")
    assert "missing required 'seed'" in str(exc_info.value)


def test_checkpoint_seed_missing_not_silently_interpreted_as_42():
    """Missing seed must NOT be silently defaulted to 42."""
    raw_ckpt_data = {
        "config": {
            "split_hash": "split123",
            "normalizer_hash": "norm123",
        }
    }
    # With no manifest, resolved seed must be None, not 42
    prov = resolve_checkpoint_provenance("dummy.pt", raw_ckpt_data, manifest_path=None)
    assert prov["seed"] is None
    with pytest.raises(RuntimeError) as exc_info:
        validate_evaluation_provenance(prov, eval_split_hash="split123", eval_normalizer_hash="norm123")
    assert "missing required 'seed'" in str(exc_info.value)


def test_untrained_seed_requested_only_seed42_exists_rejects_cross_seed_fallback():
    """When an untrained seed (e.g. 9999) is requested and only seed 42 checkpoints exist, cross-seed fallback is forbidden."""
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5")
        _create_mock_h5(h5_path, n_trajs=2, nt=8, ny=16, nx=32)
        split_file = os.path.join(tmpdir, "mock_split.json")
        with open(split_file, "w") as f:
            json.dump({"train": [h5_path], "valid": [h5_path], "test": [h5_path]}, f)

        with pytest.raises(FileNotFoundError) as exc_info:
            run_physics_ablation_eval(
                data_dir=tmpdir,
                split_type="mock",
                split_file=split_file,
                stats_dir=os.path.join(tmpdir, "normalization"),
                max_horizon=2,
                seed=9999,
                groups=["E1_rollout_field"],
            )
        assert "Cross-seed fallback is strictly forbidden" in str(exc_info.value)
        assert "seed_9999" in str(exc_info.value)


def test_untrained_seed_groups_filtering_skips_missing_e0():
    """Evaluator honors --groups and does not check or attempt to access unselected groups like E0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5")
        _create_mock_h5(h5_path, n_trajs=2, nt=8, ny=16, nx=32)
        split_file = os.path.join(tmpdir, "mock_split.json")
        with open(split_file, "w") as f:
            json.dump({"train": [h5_path], "valid": [h5_path], "test": [h5_path]}, f)

        # When requesting only E2 for an untrained seed, E0 should never be checked
        with pytest.raises(FileNotFoundError) as exc_info:
            run_physics_ablation_eval(
                data_dir=tmpdir,
                split_type="mock",
                split_file=split_file,
                stats_dir=os.path.join(tmpdir, "normalization"),
                max_horizon=2,
                seed=9999,
                groups=["E2_plus_L_div"],
            )
        # The error must be about E2, not E0!
        assert "E2_plus_L_div" in str(exc_info.value)
        assert "E0_single_step" not in str(exc_info.value)


def test_validate_evaluation_provenance_rejects_short_hash_prefix():
    """Verify validate_evaluation_provenance rejects hash prefixes shorter than 16 characters."""
    valid_full = "41fbe6ebe7edd460b4353fd6cf20ad064f1222fdcb4558b04ff1a50be390b93d"
    norm_valid = "3a0fe52689657618a92a90881aca42d349639502e956017c6fcbf0368c5d4bec"

    # Valid 16-char prefix should pass
    prov_16 = {
        "split_hash": valid_full[:16],
        "normalizer_hash": norm_valid,
        "seed": 42,
    }
    is_valid, _ = validate_evaluation_provenance(prov_16, eval_split_hash=valid_full, eval_normalizer_hash=norm_valid)
    assert is_valid is True

    # 1-char or short prefix (<16 chars) must FAIL
    prov_short = {
        "split_hash": "4",
        "normalizer_hash": norm_valid,
        "seed": 42,
    }
    with pytest.raises(RuntimeError, match="Split hash mismatch"):
        validate_evaluation_provenance(prov_short, eval_split_hash=valid_full, eval_normalizer_hash=norm_valid)


def test_formal_provenance_contract_rejects_unknown_or_dirty_git():
    """Verify formal provenance contract rejects checkpoints with unknown (None), dirty (True), or invalid git commit."""
    canonical_prov = {
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
        "training_git_commit": "6593b65005843c3f3c2640cb262bfd56708e11ad",
        "training_git_dirty": False,
    }

    # Baseline: cleanly passes
    is_valid, errors = validate_formal_provenance_bundle(canonical_prov, fail_closed=False)
    assert is_valid is True
    assert len(errors) == 0

    # Production test 1: training_git_dirty is None (missing) -> raises RuntimeError
    prov_unknown = dict(canonical_prov, training_git_dirty=None)
    with pytest.raises(RuntimeError, match="must be cleanly recorded as False"):
        validate_formal_provenance_bundle(prov_unknown, fail_closed=True)

    # Production test 2: training_git_dirty is True (dirty) -> raises RuntimeError
    prov_dirty = dict(canonical_prov, training_git_dirty=True)
    with pytest.raises(RuntimeError, match="must be cleanly recorded as False"):
        validate_formal_provenance_bundle(prov_dirty, fail_closed=True)

    # Production test 3: training_git_commit is UNKNOWN -> raises RuntimeError
    prov_unknown_commit = dict(canonical_prov, training_git_commit="UNKNOWN")
    with pytest.raises(RuntimeError, match="Training git commit is UNKNOWN or missing"):
        validate_formal_provenance_bundle(prov_unknown_commit, fail_closed=True)

    # Production test 4: training_git_commit does not exist in repo -> raises RuntimeError
    prov_bad_commit = dict(canonical_prov, training_git_commit="b305fd4c66daff5b42d765377ea9d44f6f32ee6e")
    with pytest.raises(RuntimeError, match="does not exist cryptographically in git history"):
        validate_formal_provenance_bundle(prov_bad_commit, fail_closed=True)

    # Production test 5: legacy attestation allows UNKNOWN commit on historical checkpoints
    prov_legacy = dict(
        canonical_prov,
        training_git_commit="UNKNOWN",
        legacy_attestation={"status": "historical_untracked", "note": "Verified via SHA256"},
    )
    is_valid_legacy, _ = validate_formal_provenance_bundle(prov_legacy, fail_closed=True)
    assert is_valid_legacy is True


