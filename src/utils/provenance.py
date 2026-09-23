"""Scientific Machine Learning experiment provenance and protocol governance."""

import datetime
import hashlib
import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from src.utils.physics_contract import (
    PHYSICS_PROTOCOL,
    SPATIAL_AXIS_CONTRACT,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
)


def get_git_commit(project_root: Optional[str] = None) -> str:
    """Retrieve full git HEAD commit SHA, with fallback to environment or 'UNKNOWN'."""
    if "GIT_COMMIT" in os.environ:
        return os.environ["GIT_COMMIT"]

    cwd = project_root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        ret = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if ret.returncode == 0:
            return ret.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


def is_git_dirty(project_root: Optional[str] = None) -> bool:
    """Check whether git working directory has unstaged or uncommitted changes."""
    cwd = project_root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        ret = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if ret.returncode == 0:
            return len(ret.stdout.strip()) > 0
    except Exception:
        pass
    return False


def compute_file_sha256(file_path: str, chunk_size: int = 65536) -> str:
    """Compute deterministic SHA-256 fingerprint of a file on disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found for sha256 computation: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_split_hash(split_data: dict) -> str:
    """Compute deterministic SHA-256 fingerprint of split manifest content."""
    serialized = json.dumps(split_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_split_hash_from_file(split_file: str) -> str:
    """Load split manifest JSON and compute deterministic SHA-256 fingerprint."""
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, "r") as f:
        data = json.load(f)
    return compute_split_hash(data)


def compute_normalizer_hash(normalizer: Any) -> str:
    """Compute deterministic SHA-256 fingerprint of fitted FieldNormalizer state."""
    if normalizer is None:
        return "NONE"

    buffer = bytearray()
    if hasattr(normalizer, "mean") and normalizer.mean is not None:
        buffer.extend(normalizer.mean.detach().cpu().numpy().tobytes())
    if hasattr(normalizer, "std") and normalizer.std is not None:
        buffer.extend(normalizer.std.detach().cpu().numpy().tobytes())

    if len(buffer) == 0:
        return "EMPTY_NORMALIZER"
    return hashlib.sha256(buffer).hexdigest()


def create_checkpoint_provenance(
    seed: int,
    split_type: str,
    split_hash: str,
    normalizer_hash: str,
    downsample_factor: int = 2,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct canonical provenance bundle for model checkpoint."""
    return {
        "training_git_commit": get_git_commit(project_root),
        "training_git_dirty": is_git_dirty(project_root),
        "seed": seed,
        "split_type": split_type,
        "split_hash": split_hash,
        "normalizer_hash": normalizer_hash,
        "downsample_factor": downsample_factor,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
    }


def resolve_checkpoint_provenance(
    ckpt_path: str,
    ckpt_data: dict,
    manifest_path: Optional[str] = "outputs/manifests/closure_r4_seed42.json",
) -> Dict[str, Any]:
    """Extract provenance bundle from checkpoint dict, with fallback to seed-42 manifest."""
    commit = ckpt_data.get("training_git_commit") or ckpt_data.get("config", {}).get("training_git_commit")
    git_dirty = ckpt_data.get("training_git_dirty") or ckpt_data.get("config", {}).get("training_git_dirty", False)
    seed = ckpt_data.get("seed") if "seed" in ckpt_data else ckpt_data.get("config", {}).get("seed")
    split_type = ckpt_data.get("split_type") or ckpt_data.get("config", {}).get("split_type")
    split_hash = ckpt_data.get("split_hash") or ckpt_data.get("config", {}).get("split_hash")
    normalizer_hash = ckpt_data.get("normalizer_hash") or ckpt_data.get("config", {}).get("normalizer_hash")
    downsample_factor = ckpt_data.get("downsample_factor") or ckpt_data.get("config", {}).get("downsample_factor", 2)
    physics_protocol = ckpt_data.get("physics_protocol") or ckpt_data.get("config", {}).get("physics_protocol")
    spatial_axis_contract = ckpt_data.get("spatial_axis_contract") or ckpt_data.get("config", {}).get("spatial_axis_contract")
    physics_domain_size_xy = ckpt_data.get("physics_domain_size_xy") or ckpt_data.get("config", {}).get("physics_domain_size_xy")

    # If missing split_hash, normalizer_hash, or seed in state dict, consult manifest
    if (split_hash is None or normalizer_hash is None or seed is None) and manifest_path and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as mf:
                manifest = json.load(mf)
            ckpt_sha = compute_file_sha256(ckpt_path)
            for grp_key, grp_info in manifest.get("groups", {}).items():
                manifest_path_match = os.path.abspath(grp_info.get("checkpoint_path", "")) == os.path.abspath(ckpt_path)
                manifest_sha_match = grp_info.get("checkpoint_sha256") == ckpt_sha
                if manifest_path_match or manifest_sha_match:
                    # Enforce SHA-256 integrity when matching by path
                    manifest_expected_sha = grp_info.get("checkpoint_sha256")
                    if manifest_expected_sha and manifest_expected_sha != ckpt_sha:
                        raise ValueError(
                            f"Manifest integrity mismatch for {ckpt_path}: "
                            f"file SHA256 is {ckpt_sha}, but manifest expected {manifest_expected_sha}"
                        )
                    commit = commit or grp_info.get("training_git_commit")
                    seed = seed if seed is not None else grp_info.get("seed")
                    split_type = split_type or grp_info.get("split_type", "grouped")
                    split_hash = split_hash or grp_info.get("split_hash")
                    normalizer_hash = normalizer_hash or grp_info.get("normalizer_hash")
                    downsample_factor = downsample_factor or grp_info.get("downsample_factor", 2)
                    physics_protocol = physics_protocol or grp_info.get("physics_protocol", PHYSICS_PROTOCOL)
                    spatial_axis_contract = spatial_axis_contract or grp_info.get("spatial_axis_contract", SPATIAL_AXIS_CONTRACT)
                    physics_domain_size_xy = physics_domain_size_xy or grp_info.get("physics_domain_size_xy", list(SHEAR_FLOW_DOMAIN_SIZE_XY))
                    break
        except ValueError:
            raise
        except Exception:
            pass

    return {
        "training_git_commit": commit or "UNKNOWN",
        "training_git_dirty": git_dirty,
        "seed": seed,  # Note: DO NOT fallback to 42. If seed is missing, keep None to fail closed!
        "split_type": split_type,
        "split_hash": split_hash,
        "normalizer_hash": normalizer_hash,
        "downsample_factor": downsample_factor,
        "physics_protocol": physics_protocol,
        "spatial_axis_contract": spatial_axis_contract,
        "physics_domain_size_xy": physics_domain_size_xy,
    }


def validate_evaluation_provenance(
    ckpt_provenance: Dict[str, Any],
    eval_split_hash: str,
    eval_normalizer_hash: str,
    expected_seed: Optional[int] = None,
    fail_closed: bool = True,
) -> Tuple[bool, List[str]]:
    """Enforces fail-closed protocol identity validation between training and evaluation.

    Args:
        ckpt_provenance: Checkpoint provenance bundle.
        eval_split_hash: Evaluation dataset split hash.
        eval_normalizer_hash: Evaluation normalizer hash.
        expected_seed: Optional expected seed to verify.
        fail_closed: If True, raises RuntimeError upon violation.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    ckpt_split_hash = ckpt_provenance.get("split_hash")
    ckpt_norm_hash = ckpt_provenance.get("normalizer_hash")
    ckpt_seed = ckpt_provenance.get("seed")

    if not ckpt_split_hash:
        errors.append("Checkpoint or manifest is missing required 'split_hash'")
    elif ckpt_split_hash != eval_split_hash:
        errors.append(
            f"Split hash mismatch: checkpoint/manifest has {ckpt_split_hash[:12]}..., "
            f"but evaluation environment has {eval_split_hash[:12]}..."
        )

    if not ckpt_norm_hash:
        errors.append("Checkpoint or manifest is missing required 'normalizer_hash'")
    elif ckpt_norm_hash != eval_normalizer_hash:
        errors.append(
            f"Normalizer hash mismatch: checkpoint/manifest has {ckpt_norm_hash[:12]}..., "
            f"but evaluation environment has {eval_normalizer_hash[:12]}..."
        )

    if ckpt_seed is None:
        errors.append("Checkpoint or manifest is missing required 'seed'")
    elif expected_seed is not None and ckpt_seed != expected_seed:
        errors.append(
            f"Seed mismatch: checkpoint/manifest has seed={ckpt_seed}, but expected {expected_seed}"
        )

    if errors and fail_closed:
        raise RuntimeError(
            "Evaluation failed-closed due to provenance / protocol identity contract violations:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return len(errors) == 0, errors


def validate_init_checkpoint_contract(
    init_ckpt: Dict[str, Any],
    ckpt_path: Optional[str] = None,
    manifest_path: Optional[str] = "outputs/manifests/closure_r4_seed42.json",
    requested_model_type: str = "latent_transformer",
    current_split_hash: Optional[str] = None,
    current_normalizer_hash: Optional[str] = None,
    expected_seed: Optional[int] = 42,
    expected_horizon: Optional[int] = 2,
    expected_lambda_div: Optional[float] = 0.01,
    expected_lambda_vort: Optional[float] = 0.05,
    expected_protocol: Optional[str] = PHYSICS_PROTOCOL,
    expected_axis_contract: Optional[str] = SPATIAL_AXIS_CONTRACT,
    expected_domain_size: Optional[Tuple[float, float]] = SHEAR_FLOW_DOMAIN_SIZE_XY,
    expected_prediction_mode: Optional[str] = "direct",
    expected_use_condition: Optional[bool] = True,
    fail_closed: bool = True,
) -> Tuple[bool, List[str]]:
    """Enforces fail-closed semantic contract validation for warm-start parent checkpoints.

    Validates that a checkpoint provided for warm-start initialization adheres strictly
    to the scientific protocol and expected parameter specifications.

    Args:
        init_ckpt: Loaded checkpoint dictionary.
        ckpt_path: Optional path to checkpoint file on disk.
        manifest_path: Optional path to manifest for fallback resolution.
        requested_model_type: Requested architecture type.
        current_split_hash: Current dataset split hash.
        current_normalizer_hash: Current normalizer hash.
        expected_seed: Expected training seed (e.g. 42).
        expected_horizon: Expected parent prediction horizon (e.g. 2).
        expected_lambda_div: Expected incompressibility weight (e.g. 0.01).
        expected_lambda_vort: Expected vorticity penalty weight (e.g. 0.05).
        expected_protocol: Expected physics protocol ('Closure-R4').
        expected_axis_contract: Expected spatial axis orientation contract.
        expected_domain_size: Expected physical domain bounds (1.0, 2.0).
        expected_prediction_mode: Expected prediction mode ('direct').
        expected_use_condition: Expected conditioning flag (True).
        fail_closed: If True, raises ValueError on violation.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    # Resolve full provenance bundle if ckpt_path is available
    prov = {}
    if ckpt_path:
        try:
            prov = resolve_checkpoint_provenance(ckpt_path, init_ckpt, manifest_path=manifest_path)
        except Exception:
            pass

    # 1. Model architecture
    model_type = init_ckpt.get("model_type") or init_ckpt.get("config", {}).get("model_type")
    if model_type and model_type != requested_model_type:
        errors.append(f"Model type mismatch: parent has '{model_type}', requested '{requested_model_type}'")

    # 2. Split hash
    split_hash = init_ckpt.get("split_hash") or init_ckpt.get("config", {}).get("split_hash") or prov.get("split_hash")
    if current_split_hash and current_split_hash != "UNKNOWN_SPLIT" and split_hash and split_hash != "UNKNOWN_SPLIT":
        if split_hash != current_split_hash:
            errors.append(f"Split contract violation: parent has {split_hash[:12]}..., current has {current_split_hash[:12]}...")

    # 3. Normalizer hash
    normalizer_hash = init_ckpt.get("normalizer_hash") or init_ckpt.get("config", {}).get("normalizer_hash") or prov.get("normalizer_hash")
    if current_normalizer_hash and current_normalizer_hash != "NONE" and normalizer_hash and normalizer_hash != "NONE":
        if normalizer_hash != current_normalizer_hash:
            errors.append(f"Normalizer contract violation: parent has {normalizer_hash[:12]}..., current has {current_normalizer_hash[:12]}...")

    # 4. Seed
    seed = init_ckpt.get("seed") if "seed" in init_ckpt else init_ckpt.get("config", {}).get("seed")
    if seed is None:
        seed = prov.get("seed")
    if expected_seed is not None and seed is not None and seed != expected_seed:
        errors.append(f"Seed contract violation: parent checkpoint has seed={seed}, expected {expected_seed}")

    # 5. Horizon
    horizon = init_ckpt.get("horizon") if "horizon" in init_ckpt else init_ckpt.get("config", {}).get("horizon")
    if expected_horizon is not None and horizon is not None and horizon != expected_horizon:
        errors.append(f"Horizon contract violation: parent checkpoint has horizon={horizon}, expected {expected_horizon}")

    # 6. Loss penalties (lambda_div and lambda_vort)
    lambda_div = init_ckpt.get("lambda_div") if "lambda_div" in init_ckpt else init_ckpt.get("config", {}).get("lambda_div")
    if expected_lambda_div is not None and lambda_div is not None and abs(float(lambda_div) - float(expected_lambda_div)) > 1e-4:
        errors.append(f"Loss parameter mismatch (lambda_div): parent has {lambda_div}, expected {expected_lambda_div}")

    lambda_vort = init_ckpt.get("lambda_vort") if "lambda_vort" in init_ckpt else init_ckpt.get("config", {}).get("lambda_vort")
    if expected_lambda_vort is not None and lambda_vort is not None and abs(float(lambda_vort) - float(expected_lambda_vort)) > 1e-4:
        errors.append(f"Loss parameter mismatch (lambda_vort): parent has {lambda_vort}, expected {expected_lambda_vort}")

    # 7. Protocol and Physics contracts
    physics_protocol = init_ckpt.get("physics_protocol") or init_ckpt.get("config", {}).get("physics_protocol") or prov.get("physics_protocol")
    if expected_protocol is not None and physics_protocol and physics_protocol != expected_protocol:
        errors.append(f"Physics protocol mismatch: parent has '{physics_protocol}', expected '{expected_protocol}'")

    spatial_axis_contract = init_ckpt.get("spatial_axis_contract") or init_ckpt.get("config", {}).get("spatial_axis_contract") or prov.get("spatial_axis_contract")
    if expected_axis_contract is not None and spatial_axis_contract and spatial_axis_contract != expected_axis_contract:
        errors.append(f"Axis contract mismatch: parent has '{spatial_axis_contract}', expected '{expected_axis_contract}'")

    domain_size = init_ckpt.get("physics_domain_size_xy") or init_ckpt.get("config", {}).get("physics_domain_size_xy") or prov.get("physics_domain_size_xy")
    if expected_domain_size is not None and domain_size and list(domain_size) != list(expected_domain_size):
        errors.append(f"Domain size mismatch: parent has {domain_size}, expected {expected_domain_size}")

    # 8. Prediction mode and condition
    prediction_mode = init_ckpt.get("prediction_mode") or init_ckpt.get("config", {}).get("prediction_mode")
    if expected_prediction_mode is not None and prediction_mode and prediction_mode != expected_prediction_mode:
        errors.append(f"Prediction mode mismatch: parent has '{prediction_mode}', expected '{expected_prediction_mode}'")

    use_condition = init_ckpt.get("use_condition") if "use_condition" in init_ckpt else init_ckpt.get("config", {}).get("use_condition")
    if expected_use_condition is not None and use_condition is not None and use_condition != expected_use_condition:
        errors.append(f"Conditioning mismatch: parent has use_condition={use_condition}, expected {expected_use_condition}")

    if errors and fail_closed:
        raise ValueError(
            "Init checkpoint semantic contract violation:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return len(errors) == 0, errors
