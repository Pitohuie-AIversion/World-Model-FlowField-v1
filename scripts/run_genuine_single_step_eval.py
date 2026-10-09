#!/usr/bin/env python3
"""Run genuine single-step model inference on actual test dataset to produce real arrays and evaluation plot.

Provenance and Contract Protocol:
- Manifest binding: outputs/splits/grouped_split.json
- Verifies that target trajectory belongs strictly to manifest['test'] partition
- Rejects non-test trajectories fail-closed
- Data source: /root/autodl-tmp/datasets/shear_flow
- Evaluated trajectory: traj_idx=1, cluster_id=1 (bona fide test partition member)
- Window: start_t=0, split_t=4, end_t=5 (history: t=0..3, target: t=4)
- Model checkpoint: outputs/checkpoints/dynamics/closure_r4/ablation_E4_full_physics/latent_transformer/best_vrmse_mean.pt
- Autoencoder checkpoint: outputs/checkpoints/representation/best_vrmse_mean.pt
- Normalizer stats: outputs/normalization/stats_grouped.pt
- Pressure handling: decoder project_pressure=False during forward pass; zero-mean pressure gauge applied post-denormalization per evaluation protocol
- Output arrays: outputs/figures/paper_synthesis/single_step_real_prediction_arrays.npz
- Metadata: outputs/figures/paper_synthesis/single_step_real_prediction_provenance.json
- Plot: outputs/figures/paper_synthesis/fig_single_step_prediction_eval.png
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.pipeline import create_flow_datasets, compute_split_hash
from src.data.normalization import FieldNormalizer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.latent_transformer import LatentSTTransformer
from src.models.latent_forecaster import LatentForecaster
from src.utils.fft_derivatives import compute_vorticity
from src.utils.physics_contract import zero_mean_pressure_gauge, SHEAR_FLOW_DOMAIN_SIZE_XY
from src.utils.provenance import (
    compute_file_sha256,
    compute_normalizer_hash,
    compute_split_hash_from_file,
)

OUTPUT_DIR = "outputs/figures/paper_synthesis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_forecaster_model(ckpt_path: str, repr_path: str, device: torch.device) -> Tuple[LatentForecaster, Dict[str, Any]]:
    """Load trained LatentForecaster with strict checkpoint binding and decoder project_pressure=False."""
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=256,
        cond_dim=128,
        depth=6,
        num_heads=8,
        history_length=4,
        prediction_mode="direct",
        use_spatial_pos=False,
    )
    model = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder).to(device)
    state_dict = ckpt_data.get("model_state_dict", ckpt_data)
    cleaned_state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()
    return model, ckpt_data


def normalize_rel_path(path_str: str) -> str:
    """Normalize file path to canonical relative posix path starting with data/."""
    p = Path(path_str).as_posix()
    if p.startswith("./"):
        p = p[2:]
    if "data/" in p:
        p = p[p.index("data/"):]
    return p


def verify_test_partition_membership(
    manifest_path: str,
    target_file_rel: str,
    target_traj_idx: int,
) -> Dict[str, Any]:
    """Verify that a trajectory belongs strictly to manifest['test'] partition using full file path + traj_idx identity."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Split manifest not found at: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    target_norm = normalize_rel_path(target_file_rel)

    # Check if target is in test using normalized path + traj_idx identity
    target_entry = None
    for entry in manifest.get("test", []):
        entry_norm = normalize_rel_path(entry.get("file_path", ""))
        if (
            entry_norm == target_norm
            and int(entry.get("traj_idx")) == int(target_traj_idx)
        ):
            target_entry = entry
            break

    if target_entry is None:
        found_in = None
        for partition in ("train", "valid"):
            for entry in manifest.get(partition, []):
                entry_norm = normalize_rel_path(entry.get("file_path", ""))
                if (
                    entry_norm == target_norm
                    and int(entry.get("traj_idx")) == int(target_traj_idx)
                ):
                    found_in = partition
                    break
            if found_in:
                break

        raise ValueError(
            f"Trajectory ({target_norm}, traj_idx={target_traj_idx}) does NOT belong to 'test' partition! "
            f"Actual partition: {found_in or 'UNKNOWN'}. Cannot evaluate as test sample."
        )

    return target_entry


def verify_checkpoint_split_binding(
    checkpoint_path: str,
    ckpt_data: Dict[str, Any],
    manifest_path: str,
    training_manifest_path: Optional[str] = "outputs/manifests/closure_r4_seed42.json",
    fail_closed: bool = True,
) -> Dict[str, Any]:
    """Verify that checkpoint's training data split content strictly matches the runtime split manifest.

    Verification protocol:
    1. Computes runtime split content hash (canonical SHA-256 of sorted compact JSON)
       and file byte SHA-256.
    2. Retrieves expected split hash and type:
       - First from checkpoint embedded metadata (config/provenance).
       - If absent, attempts resolution via audited historical training manifest
         cryptographically bound to checkpoint file SHA-256.
       - If neither provides training split content hash:
         Marks status as 'TYPE_MATCH_ONLY_UNVERIFIED' and raises ValueError if fail_closed=True.
    3. Rejects split_type mismatch fail-closed.
    4. Rejects split content hash mismatch fail-closed.
    5. Emits explicit status:
       - 'VERIFIED_MATCH_EMBEDDED' (checkpoint internally contains matching split_hash)
       - 'VERIFIED_MATCH_AUDITED_MANIFEST' (historical checkpoint verified via audited manifest)
       - 'TYPE_MATCH_ONLY_UNVERIFIED' (never upgraded to VERIFIED_MATCH)
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Runtime split manifest not found at: {manifest_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 1. Compute runtime fingerprints
    runtime_split_file_sha256 = compute_file_sha256(manifest_path)
    runtime_split_content_hash = compute_split_hash_from_file(manifest_path)
    ckpt_sha256 = compute_file_sha256(checkpoint_path)

    # 2. Extract checkpoint metadata
    cfg = ckpt_data.get("config", {}) if isinstance(ckpt_data.get("config"), dict) else {}
    prov = ckpt_data.get("provenance", {}) if isinstance(ckpt_data.get("provenance"), dict) else {}

    ckpt_split_type = ckpt_data.get("split_type") or cfg.get("split_type")
    embedded_split_hash = (
        ckpt_data.get("split_hash") or cfg.get("split_hash") or prov.get("split_hash")
    )

    expected_split_hash = None
    expected_split_type = ckpt_split_type
    binding_source = "none"
    attestation_info = None

    if embedded_split_hash:
        expected_split_hash = embedded_split_hash
        binding_source = "checkpoint_embedded"
    elif training_manifest_path and os.path.exists(training_manifest_path):
        with open(training_manifest_path, "r", encoding="utf-8") as mf:
            manifest = json.load(mf)
        matched_group = None
        for grp_key, grp_info in manifest.get("groups", {}).items():
            if grp_info.get("checkpoint_sha256") == ckpt_sha256:
                matched_group = grp_info
                break
        if matched_group is not None:
            expected_split_hash = matched_group.get("split_hash")
            expected_split_type = matched_group.get("split_type") or expected_split_type
            binding_source = "audited_historical_manifest"
            attestation_info = {
                "manifest_path": training_manifest_path,
                "group_title": matched_group.get("title"),
                "legacy_attestation": matched_group.get("legacy_attestation"),
            }

    # 3. Check split type
    if expected_split_type != "grouped":
        if fail_closed:
            raise ValueError(
                f"Split type mismatch: checkpoint training split_type='{expected_split_type}' "
                f"does not match required 'grouped' split."
            )
        return {
            "binding_status": "SPLIT_TYPE_MISMATCH",
            "checkpoint_split_type": expected_split_type,
            "runtime_split_type": "grouped",
            "binding_source": binding_source,
        }

    # 4. Check content hash
    if expected_split_hash is None:
        status = "TYPE_MATCH_ONLY_UNVERIFIED"
        if fail_closed:
            raise ValueError(
                f"Historical checkpoint ({ckpt_sha256[:16]}) lacks embedded split_hash "
                f"and no audited training manifest match was found. "
                f"Cannot verify that training partition content matches runtime split."
            )
        return {
            "binding_status": status,
            "checkpoint_split_type": expected_split_type,
            "runtime_split_type": "grouped",
            "runtime_split_content_hash": runtime_split_content_hash,
            "runtime_split_file_sha256": runtime_split_file_sha256,
            "checkpoint_expected_split_hash": None,
            "binding_source": "none",
        }

    if expected_split_hash != runtime_split_content_hash:
        status = "CONTENT_HASH_MISMATCH"
        if fail_closed:
            raise ValueError(
                f"Data split content fingerprint mismatch! "
                f"Checkpoint expected split_hash={expected_split_hash}, "
                f"but runtime manifest content hash is {runtime_split_content_hash}. "
                f"Possible partition drift or data leakage."
            )
        return {
            "binding_status": status,
            "checkpoint_split_type": expected_split_type,
            "runtime_split_type": "grouped",
            "checkpoint_expected_split_hash": expected_split_hash,
            "runtime_split_content_hash": runtime_split_content_hash,
            "runtime_split_file_sha256": runtime_split_file_sha256,
            "binding_source": binding_source,
        }

    final_status = (
        "VERIFIED_MATCH_EMBEDDED"
        if binding_source == "checkpoint_embedded"
        else "VERIFIED_MATCH_AUDITED_MANIFEST"
    )

    return {
        "binding_status": final_status,
        "checkpoint_split_type": expected_split_type,
        "runtime_split_type": "grouped",
        "checkpoint_expected_split_hash": expected_split_hash,
        "runtime_split_content_hash": runtime_split_content_hash,
        "runtime_split_file_sha256": runtime_split_file_sha256,
        "binding_source": binding_source,
        "historical_attestation": attestation_info,
    }


from src.metrics.field import compute_vrmse


def compute_metrics_from_arrays(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    gt_vort: np.ndarray,
    pred_vort: np.ndarray,
) -> Dict[str, float]:
    """Compute deterministic single-step VRMSE and Vorticity RMSE metrics using canonical project protocols.

    VRMSE is defined canonically per The Well benchmark in src.metrics.field::compute_vrmse:
        VRMSE = sqrt( mean((pred - target)^2) / (Var(target) + 1e-6) )
    Rejects non-finite inputs fail-closed.
    """
    if not (np.isfinite(gt_arr).all() and np.isfinite(pred_arr).all()):
        raise ValueError("Field arrays contain non-finite values (NaN or Inf)")
    if not (np.isfinite(gt_vort).all() and np.isfinite(pred_vort).all()):
        raise ValueError("Vorticity arrays contain non-finite values (NaN or Inf)")

    gt_t = torch.from_numpy(gt_arr).float()
    pred_t = torch.from_numpy(pred_arr).float()

    vrmse_u = float(compute_vrmse(pred_t[0], gt_t[0], eps=1e-6).item())
    vrmse_v = float(compute_vrmse(pred_t[1], gt_t[1], eps=1e-6).item())
    vrmse_p = float(compute_vrmse(pred_t[2], gt_t[2], eps=1e-6).item())
    vrmse_s = float(compute_vrmse(pred_t[3], gt_t[3], eps=1e-6).item())
    vort_rmse = float(np.sqrt(np.mean((gt_vort - pred_vort) ** 2)))
    vrmse_mean = float((vrmse_u + vrmse_v + vrmse_p + vrmse_s) / 4.0)

    res = {
        "vrmse_u": vrmse_u,
        "vrmse_v": vrmse_v,
        "vrmse_p": vrmse_p,
        "vrmse_s": vrmse_s,
        "vrmse_mean": vrmse_mean,
        "vorticity_rmse": vort_rmse,
    }
    for k, v in res.items():
        if not np.isfinite(v):
            raise ValueError(f"Computed metric '{k}' is non-finite: {v}")
    return res


def verify_prediction_arrays(
    npz_path: str,
    provenance_path: str,
    atol: float = 1e-5,
    vort_atol: float = 1e-3,
    manifest_path: Optional[str] = "outputs/splits/grouped_split.json",
) -> bool:
    """Verify saved npz prediction arrays strictly reproduce provenance json metrics and conform to physics contract.

    Verification protocol:
    1. Checks existence of npz and provenance files fail-closed.
    2. Validates provenance JSON contains all mandatory identity fields (no default fallbacks).
    3. Validates identity status & semantic values:
       - dataset_split == "test"
       - split_type == "grouped"
       - checkpoint_split_binding_status in ("VERIFIED_MATCH_EMBEDDED", "VERIFIED_MATCH_AUDITED_MANIFEST")
       - traj_idx is valid non-negative integer
       - cluster_id is valid non-negative integer
       - source_file_relative is non-empty string
       - checkpoint_expected_split_hash == split_content_hash (and matches runtime manifest if present)
    4. Validates required array keys in npz (gt_*, pred_*, err_* for u, v, p, s, and vort).
    5. Checks strict finiteness of all raw arrays (rejects NaN / Inf fail-closed).
    6. Checks strict spatial grid dimensions: (128, 256) for Nx=128, Ny=256.
    7. Checks error fields consistency: err_* must strictly match abs(gt_* - pred_*) within atol.
    8. Checks zero-mean pressure gauge on raw spatial arrays: |mean(p)| <= 1e-5.
    9. Checks that vorticity fields are physically derived from velocity fields via
       spectral curl (omega = dv/dx - du/dy) on SHEAR_FLOW_DOMAIN_SIZE_XY within vort_atol.
    10. Recomputes metrics using canonical compute_metrics_from_arrays protocol.
    11. Checks that all stored metrics and recomputed metrics are strictly finite.
    12. Checks absolute error between recomputed and stored metrics <= atol.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing prediction arrays archive: {npz_path}")
    if not os.path.exists(provenance_path):
        raise FileNotFoundError(f"Missing provenance metadata: {provenance_path}")

    # 1. Validate provenance metadata
    with open(provenance_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required_meta_keys = [
        "dataset_split",
        "split_type",
        "traj_idx",
        "cluster_id",
        "source_file_relative",
        "sample_metrics",
        "checkpoint_split_binding_status",
        "checkpoint_expected_split_hash",
        "split_content_hash",
    ]
    for rk in required_meta_keys:
        if rk not in meta:
            raise KeyError(f"Provenance metadata missing mandatory identity field: '{rk}'")

    if meta["dataset_split"] != "test":
        raise ValueError(f"Provenance dataset_split is '{meta['dataset_split']}', expected 'test'")

    if meta.get("split_type") != "grouped":
        raise ValueError(f"Provenance split_type is '{meta.get('split_type')}', expected 'grouped'")

    # P1-1: Verify binding status is a verified success state
    binding_status = meta.get("checkpoint_split_binding_status")
    allowed_success_statuses = ("VERIFIED_MATCH_EMBEDDED", "VERIFIED_MATCH_AUDITED_MANIFEST")
    if binding_status not in allowed_success_statuses:
        raise ValueError(
            f"Checkpoint split binding status '{binding_status}' is invalid or unverified. "
            f"Must be one of {allowed_success_statuses}. Formal plotting rejects UNVERIFIED or MISMATCH states."
        )

    # P1-1: Verify identity values are valid types and non-empty / non-negative
    traj_idx = meta.get("traj_idx")
    if traj_idx is None or isinstance(traj_idx, bool) or not isinstance(traj_idx, int) or traj_idx < 0:
        raise ValueError(f"Invalid or empty traj_idx in provenance: {traj_idx}, expected non-negative integer.")

    cluster_id = meta.get("cluster_id")
    if cluster_id is None or isinstance(cluster_id, bool) or not isinstance(cluster_id, int) or cluster_id < 0:
        raise ValueError(f"Invalid or empty cluster_id in provenance: {cluster_id}, expected non-negative integer.")

    source_file_rel = meta.get("source_file_relative")
    if not isinstance(source_file_rel, str) or len(source_file_rel.strip()) == 0:
        raise ValueError(f"Invalid or empty source_file_relative in provenance: '{source_file_rel}'")

    # P1-1: Verify split fingerprint consistency
    ckpt_exp_hash = meta.get("checkpoint_expected_split_hash")
    split_cnt_hash = meta.get("split_content_hash")
    if not isinstance(ckpt_exp_hash, str) or not isinstance(split_cnt_hash, str) or ckpt_exp_hash != split_cnt_hash:
        raise ValueError(
            f"Provenance record split fingerprint mismatch: checkpoint_expected_split_hash='{ckpt_exp_hash}' "
            f"!= split_content_hash='{split_cnt_hash}'"
        )
    if "split_hash" in meta and meta["split_hash"] != split_cnt_hash:
        raise ValueError(
            f"Provenance record split fingerprint mismatch: split_hash='{meta['split_hash']}' "
            f"!= split_content_hash='{split_cnt_hash}'"
        )
    if manifest_path and os.path.exists(manifest_path):
        runtime_split_hash = compute_split_hash_from_file(manifest_path)
        if split_cnt_hash != runtime_split_hash:
            raise ValueError(
                f"Provenance record split content hash '{split_cnt_hash}' does not match "
                f"runtime manifest content hash '{runtime_split_hash}' from {manifest_path}."
            )

    stored = meta["sample_metrics"]
    if not isinstance(stored, dict):
        raise ValueError("Provenance sample_metrics must be a dictionary")

    # 2. Validate NPZ arrays
    data = np.load(npz_path)

    required_array_keys = [
        "gt_u", "gt_v", "gt_p", "gt_s", "gt_vort",
        "pred_u", "pred_v", "pred_p", "pred_s", "pred_vort",
        "err_u", "err_v", "err_p", "err_s", "err_vort"
    ]
    for ak in required_array_keys:
        if ak not in data:
            raise KeyError(f"Prediction arrays archive missing required array: '{ak}'")
        arr = data[ak]
        # Finiteness check
        if not np.isfinite(arr).all():
            raise ValueError(f"Array '{ak}' contains non-finite values (NaN or Inf)")
        # Shape check (Nx=128, Ny=256)
        if arr.shape != (128, 256):
            raise ValueError(f"Array '{ak}' has invalid shape {arr.shape}, expected (128, 256)")

    # 3. Consistency of stored error fields with abs(gt - pred)
    for ch in ["u", "v", "p", "s", "vort"]:
        expected_err = np.abs(data[f"gt_{ch}"] - data[f"pred_{ch}"])
        max_err_diff = float(np.max(np.abs(data[f"err_{ch}"] - expected_err)))
        if not np.isfinite(max_err_diff) or max_err_diff > atol:
            raise ValueError(
                f"Error field 'err_{ch}' does not match abs(gt_{ch} - pred_{ch}): "
                f"max diff = {max_err_diff} > {atol}"
            )

    # 4. Physical pressure gauge check: spatial mean of gauge pressure must be zero
    gt_p_mean = float(np.abs(np.mean(data["gt_p"])))
    pred_p_mean = float(np.abs(np.mean(data["pred_p"])))
    if gt_p_mean > 1e-5:
        raise ValueError(f"Ground-truth pressure violates zero-mean gauge: |mean(gt_p)| = {gt_p_mean} > 1e-5")
    if pred_p_mean > 1e-5:
        raise ValueError(f"Predicted pressure violates zero-mean gauge: |mean(pred_p)| = {pred_p_mean} > 1e-5")

    # 5. P1-2: Validate that vorticity fields are derived from velocity fields (omega = dv/dx - du/dy)
    gt_u_t = torch.from_numpy(data["gt_u"]).float().unsqueeze(0)
    gt_v_t = torch.from_numpy(data["gt_v"]).float().unsqueeze(0)
    pred_u_t = torch.from_numpy(data["pred_u"]).float().unsqueeze(0)
    pred_v_t = torch.from_numpy(data["pred_v"]).float().unsqueeze(0)

    derived_gt_vort = compute_vorticity(gt_u_t, gt_v_t, domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY).squeeze(0).numpy()
    derived_pred_vort = compute_vorticity(pred_u_t, pred_v_t, domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY).squeeze(0).numpy()

    gt_vort_diff = float(np.max(np.abs(data["gt_vort"] - derived_gt_vort)))
    pred_vort_diff = float(np.max(np.abs(data["pred_vort"] - derived_pred_vort)))

    if not np.isfinite(gt_vort_diff) or gt_vort_diff > vort_atol:
        raise ValueError(
            f"Stored ground-truth vorticity 'gt_vort' does not match derived vorticity from velocity fields "
            f"(omega = dv/dx - du/dy): max diff = {gt_vort_diff} > {vort_atol}"
        )
    if not np.isfinite(pred_vort_diff) or pred_vort_diff > vort_atol:
        raise ValueError(
            f"Stored predicted vorticity 'pred_vort' does not match derived vorticity from velocity fields "
            f"(omega = dv/dx - du/dy): max diff = {pred_vort_diff} > {vort_atol}"
        )

    # 6. Metric reproduction check
    gt_arr = np.stack([data["gt_u"], data["gt_v"], data["gt_p"], data["gt_s"]], axis=0)
    pred_arr = np.stack([data["pred_u"], data["pred_v"], data["pred_p"], data["pred_s"]], axis=0)
    gt_vort = data["gt_vort"]
    pred_vort = data["pred_vort"]

    recomputed = compute_metrics_from_arrays(gt_arr, pred_arr, gt_vort, pred_vort)

    for k, v in recomputed.items():
        if k not in stored:
            raise KeyError(f"Metric '{k}' present in recomputed but missing from provenance sample_metrics")
        sv = stored[k]
        if not (np.isfinite(v) and np.isfinite(sv)):
            raise ValueError(f"Non-finite metric detected for '{k}': recomputed={v}, stored={sv}")
        diff = abs(v - sv)
        if diff > atol:
            raise ValueError(f"Metric mismatch for {k}: recomputed={v}, stored={sv}, diff={diff} > {atol}")

    return True


def run_genuine_single_step_evaluation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = "/root/autodl-tmp/datasets/shear_flow"
    manifest_file = "outputs/splits/grouped_split.json"
    ckpt_path = "outputs/checkpoints/dynamics/closure_r4/ablation_E4_full_physics/latent_transformer/best_vrmse_mean.pt"
    repr_path = "outputs/checkpoints/representation/best_vrmse_mean.pt"

    assert os.path.exists(data_root), f"Data root not found: {data_root}"
    assert os.path.exists(manifest_file), f"Manifest file not found: {manifest_file}"
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    assert os.path.exists(repr_path), f"Representation checkpoint not found: {repr_path}"

    # 1. Manifest Split Hash
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    split_hash = compute_split_hash(manifest_data)
    print(f"Verified Split Hash: {split_hash}")

    # 2. Create flow datasets
    train_ds, val_ds, test_ds, normalizer = create_flow_datasets(
        data_root=data_root,
        split_type="grouped",
        history_length=4,
        horizon=1,
        downsample_factor=2,
        normalize=True,
    )

    # 3. Retrieve sample 0 from test_ds
    meta0 = test_ds.get_window_metadata(0)
    source_file_abs = meta0["source_file"]
    traj_idx = meta0["traj_idx"]
    cluster_id = meta0["cluster_id"]
    start_t = meta0["start_t"]
    split_t = meta0["split_t"]
    end_t = meta0["end_t"]

    rel_source_path = os.path.relpath(source_file_abs, data_root) if source_file_abs.startswith(data_root) else source_file_abs
    if not rel_source_path.startswith("data/"):
        rel_source_path = os.path.join("data", os.path.relpath(source_file_abs, os.path.join(data_root, "data")))

    # 4. Strict fail-closed verification of test partition membership
    test_entry = verify_test_partition_membership(manifest_file, rel_source_path, traj_idx)
    print(f"Test Partition Verified: file={rel_source_path}, traj_idx={traj_idx}, cluster_id={cluster_id}")
    assert test_entry["cluster_id"] == cluster_id, "Cluster ID mismatch!"

    # 5. Load Forecaster Model & Verify Checkpoint Config Binding
    forecaster, ckpt_data = load_forecaster_model(ckpt_path, repr_path, device)
    binding_report = verify_checkpoint_split_binding(
        checkpoint_path=ckpt_path,
        ckpt_data=ckpt_data,
        manifest_path=manifest_file,
        training_manifest_path="outputs/manifests/closure_r4_seed42.json",
        fail_closed=True,
    )
    print(
        f"Verified Checkpoint-Split Binding: {binding_report['binding_status']} "
        f"(source={binding_report['binding_source']}, expected_hash={binding_report['checkpoint_expected_split_hash'][:16]}...)"
    )
    normalizer_hash = compute_normalizer_hash(normalizer)

    # 6. Extract Sample Data
    sample = test_ds[0]
    q_hist = sample["history"].unsqueeze(0).to(device)   # (1, 4, 4, 128, 256) normalized
    q_future = sample["future"].unsqueeze(0).to(device)  # (1, 1, 4, 128, 256) normalized
    re = torch.tensor([sample["re"]], dtype=torch.float32, device=device) if "re" in sample else None
    sc = torch.tensor([sample["sc"]], dtype=torch.float32, device=device) if "sc" in sample else None

    # 7. Model Forward Pass
    with torch.no_grad():
        pred_norm = forecaster(q_hist, re=re, sc=sc, horizon=1) # (1, 1, 4, 128, 256)

    # 8. Denormalize to physical units
    pred_phys = normalizer.denormalize(pred_norm)
    gt_phys = normalizer.denormalize(q_future)

    # 9. Pressure zero-mean gauge post-denormalization per evaluation protocol
    pred_phys = zero_mean_pressure_gauge(pred_phys)
    gt_phys = zero_mean_pressure_gauge(gt_phys)

    # 10. Extract 2D fields for Step 1 (t+1)
    pred_arr = pred_phys[0, 0].cpu().numpy()  # (4, 128, 256)
    gt_arr = gt_phys[0, 0].cpu().numpy()      # (4, 128, 256)
    err_arr = np.abs(gt_arr - pred_arr)

    # 11. Compute vorticity: dv/dx - du/dy
    pred_u_t = pred_phys[:, 0, 0]
    pred_v_t = pred_phys[:, 0, 1]
    gt_u_t = gt_phys[:, 0, 0]
    gt_v_t = gt_phys[:, 0, 1]

    pred_vort = compute_vorticity(pred_u_t, pred_v_t, domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)[0].cpu().numpy()
    gt_vort = compute_vorticity(gt_u_t, gt_v_t, domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)[0].cpu().numpy()
    err_vort = np.abs(gt_vort - pred_vort)

    # 12. Save Real Arrays to .npz
    arrays_out = os.path.join(OUTPUT_DIR, "single_step_real_prediction_arrays.npz")
    np.savez_compressed(
        arrays_out,
        gt_u=gt_arr[0], gt_v=gt_arr[1], gt_p=gt_arr[2], gt_s=gt_arr[3], gt_vort=gt_vort,
        pred_u=pred_arr[0], pred_v=pred_arr[1], pred_p=pred_arr[2], pred_s=pred_arr[3], pred_vort=pred_vort,
        err_u=err_arr[0], err_v=err_arr[1], err_p=err_arr[2], err_s=err_arr[3], err_vort=err_vort
    )
    print(f"Saved real prediction arrays: {arrays_out}")

    # 13. Calculate exact metrics
    sample_metrics = compute_metrics_from_arrays(gt_arr, pred_arr, gt_vort, pred_vort)
    print(f"Sample Metrics: {sample_metrics}")

    # 14. Save full provenance JSON
    provenance = {
        "dataset_split": "test",
        "split_type": "grouped",
        "checkpoint_config_split_type": binding_report["checkpoint_split_type"],
        "checkpoint_split_binding_status": binding_report["binding_status"],
        "checkpoint_split_binding_source": binding_report["binding_source"],
        "checkpoint_expected_split_hash": binding_report["checkpoint_expected_split_hash"],
        "split_manifest_path": manifest_file,
        "split_manifest_file_sha256": binding_report["runtime_split_file_sha256"],
        "split_content_hash": binding_report["runtime_split_content_hash"],
        "split_hash": binding_report["runtime_split_content_hash"],
        "historical_attestation": binding_report.get("historical_attestation"),
        "source_file_relative": rel_source_path,
        "source_file_absolute": source_file_abs,
        "traj_idx": int(traj_idx),
        "cluster_id": int(cluster_id),
        "time_window": {
            "start_t": int(start_t),
            "split_t": int(split_t),
            "end_t": int(end_t),
            "history_indices": list(range(int(start_t), int(split_t))),
            "target_index": int(split_t)
        },
        "history_frames": 4,
        "prediction_horizon": 1,
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": compute_file_sha256(ckpt_path),
        "representation_checkpoint_path": repr_path,
        "representation_checkpoint_sha256": compute_file_sha256(repr_path),
        "normalizer_path": "outputs/normalization/stats_grouped.pt",
        "normalizer_hash": normalizer_hash,
        "pressure_handling": "evaluation_zero_mean_gauge_post_denorm",
        "decoder_project_pressure_at_forward": False,
        "evaluation_type": "single_sample_diagnostic",
        "note": "Individual test trajectory window evaluation; not to be conflated with full test set multi-seed average.",
        "sample_metrics": sample_metrics
    }
    meta_out = os.path.join(OUTPUT_DIR, "single_step_real_prediction_provenance.json")
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"Saved provenance metadata: {meta_out}")

    # 15. Verify arrays match metadata
    verify_prediction_arrays(arrays_out, meta_out)
    print("Self-verification PASSED: Saved npz arrays match provenance json metrics exactly.")

    # 16. Plot figure directly from verified arrays
    fig, axes = plt.subplots(5, 3, figsize=(12, 10), dpi=300)
    channel_data = [
        ("Streamwise Velocity u", gt_arr[0], pred_arr[0], err_arr[0], "RdBu_r"),
        ("Cross-stream Velocity v", gt_arr[1], pred_arr[1], err_arr[1], "RdBu_r"),
        ("Gauge Pressure p", gt_arr[2], pred_arr[2], err_arr[2], "viridis"),
        ("Passive Tracer s", gt_arr[3], pred_arr[3], err_arr[3], "inferno"),
        ("Vorticity w", gt_vort, pred_vort, err_vort, "bwr"),
    ]

    col_titles = ["Ground Truth (t+1)", "Model Prediction (t+1)", "Absolute Error |GT - Pred|"]

    for row_idx, (name, gt, pred, err, cmap) in enumerate(channel_data):
        vmin = min(np.min(gt), np.min(pred))
        vmax = max(np.max(gt), np.max(pred))
        if "RdBu" in cmap or "bwr" in cmap:
            bound = max(abs(vmin), abs(vmax))
            vmin, vmax = -bound, bound

        im0 = axes[row_idx, 0].imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
        axes[row_idx, 0].set_ylabel(name, fontsize=8.5, weight='bold')
        fig.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        im1 = axes[row_idx, 1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
        fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        im2 = axes[row_idx, 2].imshow(err, cmap="magma", aspect='auto', origin='lower')
        fig.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10.0, weight='bold', pad=8)

    title_str = (
        f"Single-Step Prediction Evaluation on Unseen Test Sample\n"
        f"[Source: {rel_source_path} (traj_idx={traj_idx}, cluster_id={cluster_id}) | "
        f"Sample Mean VRMSE = {sample_metrics['vrmse_mean']:.4f} | Vorticity RMSE = {sample_metrics['vorticity_rmse']:.4f}]"
    )
    plt.suptitle(title_str, fontsize=11.5, weight='bold', y=0.99)
    plt.tight_layout()
    out_fig = os.path.join(OUTPUT_DIR, "fig_single_step_prediction_eval.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Generated verified figure: {out_fig}")


if __name__ == "__main__":
    run_genuine_single_step_evaluation()
