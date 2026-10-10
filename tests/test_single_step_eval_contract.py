"""Tests for Single-Step Prediction Evaluation Contract, Checkpoint-Split Binding, and Data Isolation."""

import os
import json
import copy
import tempfile
import pytest
import numpy as np
import torch

from scripts.run_genuine_single_step_eval import (
    verify_test_partition_membership,
    verify_checkpoint_split_binding,
    compute_metrics_from_arrays,
    verify_prediction_arrays,
)
from src.metrics.field import compute_vrmse
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

    # 3. Checkpoint SHA256 verification (strictly require real asset file existence)
    ckpt_path = meta["checkpoint_path"]
    assert os.path.exists(ckpt_path), f"Real checkpoint asset must exist on disk: {ckpt_path}"
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


def test_reject_split_content_hash_mismatch_even_if_split_type_grouped(tmp_path):
    """Negative test: Even if both split types are 'grouped', differing content hashes MUST be rejected fail-closed."""
    mock_ckpt_file = tmp_path / "mock_weights.pt"
    mock_ckpt_file.write_bytes(b"dummy_weights_content_for_hash")
    mock_ckpt = {
        "config": {
            "split_type": "grouped",
            "split_hash": "deadbeef00000000000000000000000000000000000000000000000000000000",
        }
    }
    with pytest.raises(ValueError, match="Data split content fingerprint mismatch"):
        verify_checkpoint_split_binding(
            checkpoint_path=str(mock_ckpt_file),
            ckpt_data=mock_ckpt,
            manifest_path=MANIFEST_PATH,
            training_manifest_path=None,
            fail_closed=True,
        )


def test_checkpoint_missing_split_hash_and_manifest_cannot_be_verified_match(tmp_path):
    """Negative test: Historical checkpoint lacking split_hash and lacking audited manifest cannot be verified."""
    mock_ckpt_file = tmp_path / "mock_weights.pt"
    mock_ckpt_file.write_bytes(b"dummy_weights_content_for_hash")
    mock_ckpt = {
        "config": {
            "split_type": "grouped",
            # No split_hash provided
        }
    }
    # Fail-closed should raise ValueError
    with pytest.raises(ValueError, match="lacks embedded split_hash"):
        verify_checkpoint_split_binding(
            checkpoint_path=str(mock_ckpt_file),
            ckpt_data=mock_ckpt,
            manifest_path=MANIFEST_PATH,
            training_manifest_path=None,
            fail_closed=True,
        )

    # With fail_closed=False, should return TYPE_MATCH_ONLY_UNVERIFIED and NEVER VERIFIED_MATCH
    report = verify_checkpoint_split_binding(
        checkpoint_path=str(mock_ckpt_file),
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


def test_partition_trajectory_swap_alters_content_hash_and_fails_binding(tmp_path):
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

    tmp_manifest = tmp_path / "tampered_manifest.json"
    with open(tmp_manifest, "w", encoding="utf-8") as f_tmp:
        json.dump(tampered, f_tmp)

    mock_ckpt_file = tmp_path / "mock_weights.pt"
    mock_ckpt_file.write_bytes(b"dummy_weights_content_for_hash")

    with pytest.raises(ValueError, match="Data split content fingerprint mismatch"):
        verify_checkpoint_split_binding(
            checkpoint_path=str(mock_ckpt_file),
            ckpt_data={"config": {"split_type": "grouped", "split_hash": base_content_hash}},
            manifest_path=str(tmp_manifest),
            training_manifest_path=None,
            fail_closed=True,
        )


# =========================================================================
# P1-1 Tests: Canonical VRMSE protocol alignment against boundary & normal fields
# =========================================================================

def test_vrmse_matches_canonical_field_metric_on_boundary_cases():
    """P1-1 contract test: Verify VRMSE adheres to canonical src.metrics.field formula on boundary conditions."""
    # Case 1: Constant zero field with 1e-4 perturbation (reviewer's counterexample)
    gt_const = np.zeros((4, 128, 256), dtype=np.float32)
    pred_pert = np.full((4, 128, 256), 1e-4, dtype=np.float32)
    gt_vort = np.zeros((128, 256), dtype=np.float32)
    pred_vort = np.zeros((128, 256), dtype=np.float32)

    metrics = compute_metrics_from_arrays(gt_const, pred_pert, gt_vort, pred_vort)
    canonical_vrmse = compute_vrmse(torch.from_numpy(pred_pert[0]), torch.from_numpy(gt_const[0]), eps=1e-6).item()

    # Canonical protocol produces ~0.1 (NOT 10000.0 from old bespoke formula)
    assert abs(metrics["vrmse_u"] - 0.1) < 1e-4
    assert abs(metrics["vrmse_u"] - canonical_vrmse) < 1e-6
    assert abs(metrics["vrmse_mean"] - 0.1) < 1e-4

    # Case 2: Low-variance field
    rng = np.random.RandomState(42)
    gt_low_var = (rng.randn(4, 128, 256) * 1e-3).astype(np.float32)
    pred_low_var = (gt_low_var + rng.randn(4, 128, 256) * 1e-4).astype(np.float32)
    metrics_low = compute_metrics_from_arrays(gt_low_var, pred_low_var, gt_vort, pred_vort)
    for c, ch in enumerate(["u", "v", "p", "s"]):
        ref = compute_vrmse(torch.from_numpy(pred_low_var[c]), torch.from_numpy(gt_low_var[c]), eps=1e-6).item()
        assert abs(metrics_low[f"vrmse_{ch}"] - ref) < 1e-5

    # Case 3: Standard normal field
    gt_normal = rng.randn(4, 128, 256).astype(np.float32)
    pred_normal = (gt_normal + rng.randn(4, 128, 256) * 0.1).astype(np.float32)
    metrics_normal = compute_metrics_from_arrays(gt_normal, pred_normal, gt_vort, pred_vort)
    for c, ch in enumerate(["u", "v", "p", "s"]):
        ref = compute_vrmse(torch.from_numpy(pred_normal[c]), torch.from_numpy(gt_normal[c]), eps=1e-6).item()
        assert abs(metrics_normal[f"vrmse_{ch}"] - ref) < 1e-5


# =========================================================================
# P1-2 Tests: Array self-verification hardening & plotting entry blocking
# =========================================================================

def test_verify_prediction_arrays_rejects_nan_in_pred_array(tmp_path):
    """P1-2 test: verify_prediction_arrays must fail-closed if prediction array contains NaN."""
    data = dict(np.load(NPZ_PATH))
    data["pred_u"] = data["pred_u"].copy()
    data["pred_u"][0, 0] = np.nan
    tampered_npz = tmp_path / "tampered.npz"
    np.savez(tampered_npz, **data)

    with pytest.raises(ValueError, match="non-finite values"):
        verify_prediction_arrays(str(tampered_npz), PROVENANCE_PATH)


def test_verify_prediction_arrays_rejects_nan_in_gt_array(tmp_path):
    """P1-2 test: verify_prediction_arrays must fail-closed if ground truth contains NaN."""
    data = dict(np.load(NPZ_PATH))
    data["gt_u"] = data["gt_u"].copy()
    data["gt_u"][0, 0] = np.nan
    tampered_npz = tmp_path / "tampered.npz"
    np.savez(tampered_npz, **data)

    with pytest.raises(ValueError, match="non-finite values"):
        verify_prediction_arrays(str(tampered_npz), PROVENANCE_PATH)


def test_verify_prediction_arrays_rejects_nan_in_stored_metrics(tmp_path):
    """P1-2 test: verify_prediction_arrays must fail-closed if stored metric contains NaN."""
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta = copy.deepcopy(meta)
    meta["sample_metrics"]["vrmse_u"] = float("nan")
    tampered_json = tmp_path / "tampered.json"
    with open(tampered_json, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    with pytest.raises(ValueError, match="Non-finite metric detected"):
        verify_prediction_arrays(NPZ_PATH, str(tampered_json))


def test_verify_prediction_arrays_rejects_tampered_error_field(tmp_path):
    """P1-2 test: verify_prediction_arrays must reject tampered err_u = 999."""
    data = dict(np.load(NPZ_PATH))
    data["err_u"] = np.full((128, 256), 999.0, dtype=np.float32)
    tampered_npz = tmp_path / "tampered.npz"
    np.savez(tampered_npz, **data)

    with pytest.raises(ValueError, match="Error field 'err_u' does not match"):
        verify_prediction_arrays(str(tampered_npz), PROVENANCE_PATH)


def test_verify_prediction_arrays_rejects_non_zero_mean_pressure(tmp_path):
    """P1-2 test: verify_prediction_arrays must reject arrays violating zero-mean gauge."""
    data = dict(np.load(NPZ_PATH))
    data["gt_p"] = data["gt_p"] + 1.0
    data["err_p"] = np.abs(data["gt_p"] - data["pred_p"])
    tampered_npz = tmp_path / "tampered.npz"
    np.savez(tampered_npz, **data)

    with pytest.raises(ValueError, match="violates zero-mean gauge"):
        verify_prediction_arrays(str(tampered_npz), PROVENANCE_PATH)


def test_verify_prediction_arrays_rejects_invalid_spatial_shape(tmp_path):
    """P1-2 test: verify_prediction_arrays must reject non-128x256 arrays."""
    data = dict(np.load(NPZ_PATH))
    data["pred_u"] = np.zeros((64, 128), dtype=np.float32)
    tampered_npz = tmp_path / "tampered.npz"
    np.savez(tampered_npz, **data)

    with pytest.raises(ValueError, match="invalid shape"):
        verify_prediction_arrays(str(tampered_npz), PROVENANCE_PATH)


def test_draw_entry_blocks_on_verification_failure(tmp_path, monkeypatch):
    """P1-2 test: draw_single_step_prediction_eval must abort and raise if arrays fail verification."""
    import scripts.generate_synthesis_figures_v2 as gen_mod

    data = dict(np.load(NPZ_PATH))
    data["err_u"] = np.full((128, 256), 999.0, dtype=np.float32)
    tampered_npz = tmp_path / "single_step_real_prediction_arrays.npz"
    np.savez(tampered_npz, **data)

    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    tampered_json = tmp_path / "single_step_real_prediction_provenance.json"
    with open(tampered_json, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    monkeypatch.setattr(gen_mod, "OUTPUT_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="Error field 'err_u' does not match"):
        gen_mod.draw_single_step_prediction_eval()


# =========================================================================
# P1-1 & P1-2 Hardened Contract Tests: Binding status, semantic identity, and vorticity derivation
# =========================================================================

def test_verify_prediction_arrays_rejects_unverified_or_mismatched_binding_status(tmp_path):
    """P1-1 negative test: Rejects TYPE_MATCH_ONLY_UNVERIFIED or CONTENT_HASH_MISMATCH binding status fail-closed."""
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 1. TYPE_MATCH_ONLY_UNVERIFIED must be rejected
    meta_unverified = copy.deepcopy(meta)
    meta_unverified["checkpoint_split_binding_status"] = "TYPE_MATCH_ONLY_UNVERIFIED"
    json_unverified = tmp_path / "unverified.json"
    with open(json_unverified, "w", encoding="utf-8") as f:
        json.dump(meta_unverified, f)

    with pytest.raises(ValueError, match="invalid or unverified"):
        verify_prediction_arrays(NPZ_PATH, str(json_unverified))

    # 2. CONTENT_HASH_MISMATCH must be rejected
    meta_mismatch = copy.deepcopy(meta)
    meta_mismatch["checkpoint_split_binding_status"] = "CONTENT_HASH_MISMATCH"
    json_mismatch = tmp_path / "mismatch.json"
    with open(json_mismatch, "w", encoding="utf-8") as f:
        json.dump(meta_mismatch, f)

    with pytest.raises(ValueError, match="invalid or unverified"):
        verify_prediction_arrays(NPZ_PATH, str(json_mismatch))

    # 3. Arbitrary unknown status must be rejected
    meta_unknown = copy.deepcopy(meta)
    meta_unknown["checkpoint_split_binding_status"] = "UNKNOWN_STATUS"
    json_unknown = tmp_path / "unknown.json"
    with open(json_unknown, "w", encoding="utf-8") as f:
        json.dump(meta_unknown, f)

    with pytest.raises(ValueError, match="invalid or unverified"):
        verify_prediction_arrays(NPZ_PATH, str(json_unknown))


def test_verify_prediction_arrays_rejects_invalid_or_empty_identity(tmp_path):
    """P1-1 negative test: Rejects null/empty trajectory identities or split fingerprint inconsistency."""
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 1. traj_idx is None
    meta_null_traj = copy.deepcopy(meta)
    meta_null_traj["traj_idx"] = None
    json_null_traj = tmp_path / "null_traj.json"
    with open(json_null_traj, "w", encoding="utf-8") as f:
        json.dump(meta_null_traj, f)
    with pytest.raises(ValueError, match="Invalid or empty traj_idx"):
        verify_prediction_arrays(NPZ_PATH, str(json_null_traj))

    # 2. cluster_id is None
    meta_null_cluster = copy.deepcopy(meta)
    meta_null_cluster["cluster_id"] = None
    json_null_cluster = tmp_path / "null_cluster.json"
    with open(json_null_cluster, "w", encoding="utf-8") as f:
        json.dump(meta_null_cluster, f)
    with pytest.raises(ValueError, match="Invalid or empty cluster_id"):
        verify_prediction_arrays(NPZ_PATH, str(json_null_cluster))

    # 3. source_file_relative is empty string
    meta_empty_src = copy.deepcopy(meta)
    meta_empty_src["source_file_relative"] = "   "
    json_empty_src = tmp_path / "empty_src.json"
    with open(json_empty_src, "w", encoding="utf-8") as f:
        json.dump(meta_empty_src, f)
    with pytest.raises(ValueError, match="Invalid or empty source_file_relative"):
        verify_prediction_arrays(NPZ_PATH, str(json_empty_src))

    # 4. checkpoint_expected_split_hash != split_content_hash
    meta_hash_drift = copy.deepcopy(meta)
    meta_hash_drift["checkpoint_expected_split_hash"] = "deadbeef" * 8
    json_hash_drift = tmp_path / "hash_drift.json"
    with open(json_hash_drift, "w", encoding="utf-8") as f:
        json.dump(meta_hash_drift, f)
    with pytest.raises(ValueError, match="split fingerprint mismatch"):
        verify_prediction_arrays(NPZ_PATH, str(json_hash_drift))


def test_verify_prediction_arrays_rejects_vorticity_offset_not_matching_velocity(tmp_path):
    """P1-2 negative test: Rejects vorticity fields that do not match velocity curl (even if err_vort is invariant).

    Reviewer's counterexample:
    Keeping all velocity fields, err_* fields, and sample metrics unchanged, but adding +10 to both gt_vort and pred_vort.
    Since (pred_vort + 10) - (gt_vort + 10) = pred_vort - gt_vort, err_vort and vorticity_rmse remain unchanged,
    but vorticity fields are no longer derived from velocity fields.
    The hardened verify_prediction_arrays MUST reject this fail-closed.
    """
    data = dict(np.load(NPZ_PATH))
    data["gt_vort"] = data["gt_vort"] + 10.0
    data["pred_vort"] = data["pred_vort"] + 10.0
    # err_vort remains exactly abs(gt_vort - pred_vort)
    data["err_vort"] = np.abs(data["gt_vort"] - data["pred_vort"])

    tampered_npz = tmp_path / "tampered_vort_offset.npz"
    np.savez(tampered_npz, **data)

    with pytest.raises(ValueError, match="does not match derived vorticity from velocity fields"):
        verify_prediction_arrays(str(tampered_npz), PROVENANCE_PATH)


# =========================================================================
# P1 Final Fix: Test Partition Membership, Manifest Integrity, & Entry Gate Tests
# =========================================================================

def test_verify_prediction_arrays_rejects_train_partition_identity(tmp_path):
    """P1 test: Rejects cached identity belonging to train partition (traj_idx=0 in same source file)."""
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta_train = copy.deepcopy(meta)
    meta_train["traj_idx"] = 0  # traj_idx=0 is valid non-negative int but belongs to train
    meta_train["cluster_id"] = 0
    json_train = tmp_path / "train_identity.json"
    with open(json_train, "w", encoding="utf-8") as f:
        json.dump(meta_train, f)

    with pytest.raises(ValueError, match="does NOT belong to 'test' partition"):
        verify_prediction_arrays(NPZ_PATH, str(json_train))


def test_verify_prediction_arrays_rejects_valid_partition_identity(tmp_path):
    """P1 test: Rejects cached identity belonging to valid partition (traj_idx=3 in same source file)."""
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta_valid = copy.deepcopy(meta)
    meta_valid["traj_idx"] = 3  # traj_idx=3 is valid non-negative int but belongs to valid
    json_valid = tmp_path / "valid_identity.json"
    with open(json_valid, "w", encoding="utf-8") as f:
        json.dump(meta_valid, f)

    with pytest.raises(ValueError, match="does NOT belong to 'test' partition"):
        verify_prediction_arrays(NPZ_PATH, str(json_valid))


def test_verify_prediction_arrays_rejects_mismatched_cluster_id(tmp_path):
    """P1 test: Rejects genuine test trajectory when cached cluster_id does not match manifest entry."""
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta_bad_cluster = copy.deepcopy(meta)
    meta_bad_cluster["cluster_id"] = 999  # Valid non-negative int, but manifest entry has cluster_id=1
    json_bad_cluster = tmp_path / "bad_cluster.json"
    with open(json_bad_cluster, "w", encoding="utf-8") as f:
        json.dump(meta_bad_cluster, f)

    with pytest.raises(ValueError, match="does not match the test manifest entry"):
        verify_prediction_arrays(NPZ_PATH, str(json_bad_cluster))


def test_verify_prediction_arrays_rejects_source_file_outside_manifest(tmp_path):
    """P1 test: Rejects valid non-empty source file path that does not exist in split manifest."""
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta_extraneous = copy.deepcopy(meta)
    meta_extraneous["source_file_relative"] = "data/test/nonexistent_shear_flow_file.hdf5"
    json_extraneous = tmp_path / "extraneous_source.json"
    with open(json_extraneous, "w", encoding="utf-8") as f:
        json.dump(meta_extraneous, f)

    with pytest.raises(ValueError, match="does NOT belong to 'test' partition"):
        verify_prediction_arrays(NPZ_PATH, str(json_extraneous))


def test_verify_prediction_arrays_rejects_missing_or_nonexistent_manifest(tmp_path):
    """P1 test: verify_prediction_arrays fails closed when manifest_path is missing or nonexistent."""
    # 1. Nonexistent manifest file path -> FileNotFoundError
    nonexistent_manifest = tmp_path / "nonexistent_manifest.json"
    with pytest.raises(FileNotFoundError, match="Formal split manifest file not found"):
        verify_prediction_arrays(NPZ_PATH, PROVENANCE_PATH, manifest_path=str(nonexistent_manifest))

    # 2. None manifest_path -> ValueError
    with pytest.raises(ValueError, match="manifest_path must be provided"):
        verify_prediction_arrays(NPZ_PATH, PROVENANCE_PATH, manifest_path=None)

    # 3. Empty string manifest_path -> ValueError
    with pytest.raises(ValueError, match="manifest_path must be provided"):
        verify_prediction_arrays(NPZ_PATH, PROVENANCE_PATH, manifest_path="")


def test_draw_entry_blocks_and_does_not_generate_figure_on_train_identity(tmp_path, monkeypatch):
    """P1 test: draw_single_step_prediction_eval refuses to create output figure on train partition cache."""
    import shutil
    import scripts.generate_synthesis_figures_v2 as gen_mod

    # Setup temporary directory with valid npz but train-identity provenance
    tampered_npz = tmp_path / "single_step_real_prediction_arrays.npz"
    shutil.copyfile(NPZ_PATH, tampered_npz)

    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["traj_idx"] = 0  # Train partition trajectory
    meta["cluster_id"] = 0
    tampered_json = tmp_path / "single_step_real_prediction_provenance.json"
    with open(tampered_json, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    monkeypatch.setattr(gen_mod, "OUTPUT_DIR", str(tmp_path))

    target_fig = tmp_path / "fig_single_step_prediction_eval.png"
    assert not target_fig.exists(), "Target figure should not exist prior to test run"

    with pytest.raises(ValueError, match="does NOT belong to 'test' partition"):
        gen_mod.draw_single_step_prediction_eval()

    # Fail-closed guarantee: Figure must NOT have been created
    assert not target_fig.exists(), "Figure must NOT be created when cache partition verification fails!"


def test_draw_entry_blocks_and_does_not_overwrite_figure_on_mismatched_cluster(tmp_path, monkeypatch):
    """P1 test: draw_single_step_prediction_eval refuses to overwrite existing figure on mismatched cluster_id."""
    import shutil
    import scripts.generate_synthesis_figures_v2 as gen_mod

    tampered_npz = tmp_path / "single_step_real_prediction_arrays.npz"
    shutil.copyfile(NPZ_PATH, tampered_npz)

    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["cluster_id"] = 999  # Mismatched cluster_id
    tampered_json = tmp_path / "single_step_real_prediction_provenance.json"
    with open(tampered_json, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    monkeypatch.setattr(gen_mod, "OUTPUT_DIR", str(tmp_path))

    target_fig = tmp_path / "fig_single_step_prediction_eval.png"
    sentinel_content = b"PRE_EXISTING_UNTOUCHED_IMAGE_SENTINEL"
    target_fig.write_bytes(sentinel_content)

    with pytest.raises(ValueError, match="does not match the test manifest entry"):
        gen_mod.draw_single_step_prediction_eval()

    # Fail-closed guarantee: Figure must NOT have been overwritten
    assert target_fig.read_bytes() == sentinel_content, "Existing figure must NOT be overwritten when verification fails!"
