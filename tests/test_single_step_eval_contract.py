"""Tests for Single-Step Prediction Evaluation Contract, Checkpoint-Split Binding, and Data Isolation."""

import os
import json
import copy
import tempfile
import pytest
import numpy as np

from scripts.run_genuine_single_step_eval import (
    verify_test_partition_membership,
    verify_checkpoint_split_binding,
    compute_metrics_from_arrays,
    verify_prediction_arrays,
)
from src.data.pipeline import compute_split_hash
from src.utils.provenance import (
    compute_file_sha256,
    compute_split_hash_from_file,
)


MANIFEST_PATH = "outputs/splits/grouped_split.json"
TRAINING_MANIFEST_PATH = "outputs/manifests/closure_r4_seed42.json"
NPZ_PATH = "outputs/figures/paper_synthesis/single_step_real_prediction_arrays.npz"
PROVENANCE_PATH = "outputs/figures/paper_synthesis/single_step_real_prediction_provenance.json"
CHECKPOINT_PATH = "outputs/checkpoints/dynamics/closure_r4/ablation_E4_full_physics/latent_transformer/best_vrmse_mean.pt"


def test_verify_test_partition_membership_accepts_test():
    """Verify that a true test trajectory in grouped_split.json is accepted."""
    entry = verify_test_partition_membership(
        manifest_path=MANIFEST_PATH,
        target_file_rel="data/test/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5",
        target_traj_idx=1,
    )
    assert entry is not None
    assert int(entry["traj_idx"]) == 1
    assert int(entry["cluster_id"]) == 1


def test_verify_test_partition_membership_rejects_train():
    """Verify that a trajectory in train partition (traj_idx=0 in the same file) is rejected fail-closed."""
    with pytest.raises(ValueError, match="does NOT belong to 'test' partition"):
        verify_test_partition_membership(
            manifest_path=MANIFEST_PATH,
            target_file_rel="data/test/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5",
            target_traj_idx=0,
        )


def test_verify_test_partition_membership_rejects_valid():
    """Verify that a trajectory in valid partition (traj_idx=3 in the same file) is rejected fail-closed."""
    with pytest.raises(ValueError, match="does NOT belong to 'test' partition"):
        verify_test_partition_membership(
            manifest_path=MANIFEST_PATH,
            target_file_rel="data/test/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5",
            target_traj_idx=3,
        )


def test_provenance_and_manifest_hashes_match():
    """Verify cryptographic hashes in provenance json match disk state."""
    assert os.path.exists(PROVENANCE_PATH), f"Missing {PROVENANCE_PATH}"
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 1. Split content hash verification
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    expected_split_content_hash = compute_split_hash(manifest)
    assert meta["split_content_hash"] == expected_split_content_hash, "Split content hash mismatch"
    assert meta["split_hash"] == expected_split_content_hash, "Split hash mismatch"

    # 2. Split file sha256 verification
    expected_file_sha256 = compute_file_sha256(MANIFEST_PATH)
    assert meta["split_manifest_file_sha256"] == expected_file_sha256, "Split manifest file SHA256 mismatch"

    # 3. Checkpoint SHA256 verification
    ckpt_path = meta["checkpoint_path"]
    expected_ckpt_hash = compute_file_sha256(ckpt_path)
    assert meta["checkpoint_sha256"] == expected_ckpt_hash, "Checkpoint sha256 mismatch"

    # 4. Trajectory and cluster identity
    assert meta["traj_idx"] == 1
    assert meta["cluster_id"] == 1
    assert meta["dataset_split"] == "test"
    assert meta["evaluation_type"] == "single_sample_diagnostic"


def test_saved_arrays_recompute_exact_metrics():
    """Verify that recomputing metrics directly from the raw numpy arrays matches provenance numbers within 1e-5."""
    assert os.path.exists(NPZ_PATH), f"Missing {NPZ_PATH}"
    assert os.path.exists(PROVENANCE_PATH), f"Missing {PROVENANCE_PATH}"
    assert verify_prediction_arrays(NPZ_PATH, PROVENANCE_PATH, atol=1e-5) is True


def test_pressure_gauge_contract():
    """Verify that pressure gauge handling strictly follows the evaluation protocol."""
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["decoder_project_pressure_at_forward"] is False
    assert meta["pressure_handling"] == "evaluation_zero_mean_gauge_post_denorm"


def test_checkpoint_split_binding_provenance_record():
    """Verify that checkpoint's training split configuration and content hash strictly match runtime grouped split."""
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["checkpoint_config_split_type"] == "grouped"
    assert meta["checkpoint_split_binding_status"] in (
        "VERIFIED_MATCH_EMBEDDED",
        "VERIFIED_MATCH_AUDITED_MANIFEST",
    )
    assert meta["checkpoint_expected_split_hash"] == meta["split_content_hash"]
    assert meta["checkpoint_split_binding_source"] == "audited_historical_manifest"
    assert "historical_attestation" in meta
    assert meta["historical_attestation"]["legacy_attestation"]["status"] == "historical_untracked"


def test_reject_split_content_hash_mismatch_even_if_split_type_grouped():
    """Negative test: Even if both split types are 'grouped', differing content hashes MUST be rejected fail-closed."""
    mock_ckpt = {
        "config": {
            "split_type": "grouped",
            "split_hash": "deadbeef00000000000000000000000000000000000000000000000000000000",
        }
    }
    with pytest.raises(ValueError, match="Data split content fingerprint mismatch"):
        verify_checkpoint_split_binding(
            checkpoint_path=CHECKPOINT_PATH,
            ckpt_data=mock_ckpt,
            manifest_path=MANIFEST_PATH,
            training_manifest_path=None,
            fail_closed=True,
        )


def test_checkpoint_missing_split_hash_and_manifest_cannot_be_verified_match():
    """Negative test: Historical checkpoint lacking split_hash and lacking audited manifest cannot be verified."""
    mock_ckpt = {
        "config": {
            "split_type": "grouped",
            # No split_hash provided
        }
    }
    # Fail-closed should raise ValueError
    with pytest.raises(ValueError, match="lacks embedded split_hash"):
        verify_checkpoint_split_binding(
            checkpoint_path=CHECKPOINT_PATH,
            ckpt_data=mock_ckpt,
            manifest_path=MANIFEST_PATH,
            training_manifest_path=None,
            fail_closed=True,
        )

    # With fail_closed=False, should return TYPE_MATCH_ONLY_UNVERIFIED and NEVER VERIFIED_MATCH
    report = verify_checkpoint_split_binding(
        checkpoint_path=CHECKPOINT_PATH,
        ckpt_data=mock_ckpt,
        manifest_path=MANIFEST_PATH,
        training_manifest_path=None,
        fail_closed=False,
    )
    assert report["binding_status"] == "TYPE_MATCH_ONLY_UNVERIFIED"
    assert "VERIFIED_MATCH" not in report["binding_status"]


def test_json_whitespace_indent_formatting_does_not_alter_content_hash():
    """Verify that canonical JSON sorting and compact serialization is invariant to whitespace / indent."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_obj = json.load(f)

    base_content_hash = compute_split_hash(manifest_obj)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f1:
        json.dump(manifest_obj, f1, indent=4)
        f1_path = f1.name

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f2:
        json.dump(manifest_obj, f2, indent=None, separators=(", ", " : "))
        f2_path = f2.name

    try:
        hash1 = compute_split_hash_from_file(f1_path)
        hash2 = compute_split_hash_from_file(f2_path)
        assert hash1 == base_content_hash
        assert hash2 == base_content_hash
    finally:
        os.unlink(f1_path)
        os.unlink(f2_path)


def test_partition_trajectory_swap_alters_content_hash_and_fails_binding():
    """Negative test: Swapping a trajectory between train and test partitions changes content hash and fails binding."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_obj = json.load(f)

    base_content_hash = compute_split_hash(manifest_obj)

    # Create tampered manifest: move 1 trajectory from test to train
    tampered = copy.deepcopy(manifest_obj)
    moved_item = tampered["test"].pop(0)
    tampered["train"].append(moved_item)

    tampered_content_hash = compute_split_hash(tampered)
    assert tampered_content_hash != base_content_hash, "Tampered partition MUST produce different content hash!"

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f_tmp:
        json.dump(tampered, f_tmp)
        tmp_path = f_tmp.name

    try:
        with pytest.raises(ValueError, match="Data split content fingerprint mismatch"):
            verify_checkpoint_split_binding(
                checkpoint_path=CHECKPOINT_PATH,
                ckpt_data={"config": {"split_type": "grouped", "split_hash": base_content_hash}},
                manifest_path=tmp_path,
                training_manifest_path=None,
                fail_closed=True,
            )
    finally:
        os.unlink(tmp_path)
