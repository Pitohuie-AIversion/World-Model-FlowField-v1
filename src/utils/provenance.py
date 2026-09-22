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
    seed = ckpt_data.get("seed") if "seed" in ckpt_data else ckpt_data.get("config", {}).get("seed")
    split_type = ckpt_data.get("split_type") or ckpt_data.get("config", {}).get("split_type")
    split_hash = ckpt_data.get("split_hash") or ckpt_data.get("config", {}).get("split_hash")
    normalizer_hash = ckpt_data.get("normalizer_hash") or ckpt_data.get("config", {}).get("normalizer_hash")
    downsample_factor = ckpt_data.get("downsample_factor") or ckpt_data.get("config", {}).get("downsample_factor", 2)
    physics_protocol = ckpt_data.get("physics_protocol") or ckpt_data.get("config", {}).get("physics_protocol")
    spatial_axis_contract = ckpt_data.get("spatial_axis_contract") or ckpt_data.get("config", {}).get("spatial_axis_contract")
    physics_domain_size_xy = ckpt_data.get("physics_domain_size_xy") or ckpt_data.get("config", {}).get("physics_domain_size_xy")

    # If missing split_hash or normalizer_hash in state dict, consult manifest
    if (split_hash is None or normalizer_hash is None) and manifest_path and os.path.exists(manifest_path):
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
                    seed = seed if seed is not None else grp_info.get("seed", 42)
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
        "seed": seed if seed is not None else 42,
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

    if expected_seed is not None and ckpt_seed != expected_seed:
        errors.append(
            f"Seed mismatch: checkpoint/manifest has seed={ckpt_seed}, but expected {expected_seed}"
        )

    if errors and fail_closed:
        raise RuntimeError(
            "Evaluation failed-closed due to provenance / protocol identity contract violations:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return len(errors) == 0, errors
