"""Checkpointing and artifact persistence utilities."""

import os
import shutil
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

# --- torch.compile compatibility -------------------------------------------
# torch.compile wraps modules inside an OptimizedModule, which prepends
# ``_orig_mod.`` to every key in state_dict().  To guarantee that checkpoints
# produced during compiled training can be loaded by **uncompiled** evaluation
# scripts with ``strict=True``, we strip these prefixes at save-time.
_COMPILE_SEGMENT = "_orig_mod"
_COMPILE_PREFIX = "_orig_mod."


def strip_compiled_prefix(state_dict: Dict[str, Any]) -> OrderedDict:
    """Remove ``_orig_mod`` module segments injected by ``torch.compile``.

    ``torch.compile`` wraps modules in an ``OptimizedModule``, injecting
    ``_orig_mod`` into parameter paths in ``state_dict()``. Depending on where
    compile was applied, this segment may appear at the start of a key (when
    compiling the entire model, e.g. ``_orig_mod.transformer.weight``) or inside
    a sub-module path (when compiling sub-modules individually, e.g.
    ``transformer._orig_mod.weight`` or ``transformer.block._orig_mod.weight``).

    This function removes all dot-separated ``_orig_mod`` segments from keys.
    If no keys carry ``_orig_mod``, the original dict is returned unchanged.
    This function is idempotent and raises a ValueError on key collisions.

    Args:
        state_dict: Model state dictionary possibly containing compiled keys.

    Returns:
        OrderedDict with clean, uncompiled key names.
    """
    needs_strip = any(_COMPILE_SEGMENT in k.split(".") for k in state_dict)
    if not needs_strip:
        return OrderedDict(state_dict)

    cleaned = OrderedDict()
    for key, value in state_dict.items():
        parts = key.split(".")
        new_parts = [p for p in parts if p != _COMPILE_SEGMENT]
        new_key = ".".join(new_parts)

        if new_key in cleaned:
            raise ValueError(
                f"Key collision after stripping '{_COMPILE_SEGMENT}' segments: "
                f"both '{key}' and an earlier key map to '{new_key}'. "
                f"This indicates an unexpected state_dict structure."
            )
        cleaned[new_key] = value
    return cleaned


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None,
):
    """Save model checkpoint safely."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    if is_best and best_filepath:
        os.makedirs(os.path.dirname(best_filepath), exist_ok=True)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load checkpoint into model and optional optimizer.

    Returns the loaded checkpoint dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    map_location = device or torch.device("cpu")
    checkpoint = torch.load(filepath, map_location=map_location)

    # Handle model state dict — strip _orig_mod. prefixes for compile compat
    if "model_state_dict" in checkpoint:
        sd = strip_compiled_prefix(checkpoint["model_state_dict"])
        model.load_state_dict(sd, strict=strict)
    elif "state_dict" in checkpoint:
        sd = strip_compiled_prefix(checkpoint["state_dict"])
        model.load_state_dict(sd, strict=strict)
    else:
        sd = strip_compiled_prefix(checkpoint)
        model.load_state_dict(sd, strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


class BestCheckpointTracker:
    """Tracks and saves top-K checkpoints based on a validation metric."""

    def __init__(
        self,
        save_dir: str,
        metric_name: str = "vrmse_mean",
        mode: str = "min",
        keep_top_k: int = 3,
    ):
        self.save_dir = save_dir
        self.metric_name = metric_name
        self.mode = mode
        self.keep_top_k = keep_top_k
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.checkpoints: List[Tuple[float, str]] = []  # [(score, path), ...]

        os.makedirs(save_dir, exist_ok=True)

    def is_better(self, score: float, best_score: float) -> bool:
        if self.mode == "min":
            return score < best_score
        return score > best_score

    def update(self, score: float, state_dict: dict, step_or_epoch: int) -> bool:
        """Records checkpoint and returns True if this is a new absolute best."""
        is_best = self.is_better(score, self.best_score)
        if is_best:
            self.best_score = score

        ckpt_path = os.path.join(
            self.save_dir,
            f"checkpoint_step_{step_or_epoch}_{self.metric_name}_{score:.4f}.pt",
        )
        best_path = os.path.join(self.save_dir, f"best_{self.metric_name}.pt")

        save_checkpoint(state_dict, ckpt_path, is_best=is_best, best_filepath=best_path)

        self.checkpoints.append((score, ckpt_path))
        # Sort checkpoints
        reverse = self.mode == "max"
        self.checkpoints.sort(key=lambda x: x[0], reverse=reverse)

        # Remove older checkpoints exceeding keep_top_k
        while len(self.checkpoints) > self.keep_top_k:
            _, old_path = self.checkpoints.pop()
            if os.path.exists(old_path) and old_path != best_path:
                try:
                    os.remove(old_path)
                except OSError:
                    pass

        return is_best


POST_SPATIAL_POS_INTRO_COMMIT = "0f2ed248d360acf28b55370f59ecfc756931047a"

KNOWN_SPATIAL_POS_PRE_COMMITS = {
    "6593b65005843c3f3c2640cb262bfd56708e11ad",  # Parent H8 saved long-best (ep 11)
    "6593b650",
    "0a8a4da137e3f73c4f696cc81f527c32a8ba8405",  # H16 short-best (ep 8)
    "0a8a4da1",
    "02298bc6110f6cfc721c5f35dcf3f084ba68fe0d",  # H16 long-best (ep 12)
    "02298bc6",
}
KNOWN_SPATIAL_POS_POST_COMMITS = {
    "09e49d949430c6aeb851be97193b2bf8a1b6377e",  # H12 extension runner
    "09e49d9",
    "f9f5815e5a8d9be37d7d0830d1b3b08b7a32866f",  # H12 extension runner with Beff guards
    "f9f5815",
}


def resolve_spatial_pos_config(
    checkpoint_data: Dict[str, Any],
    allow_unverified_fallback: bool = False,
    default_if_unverified: bool = False,
) -> bool:
    """Resolve whether spatial positional encoding was active during model training.

    Ensures strict backward compatibility and fail-closed validation:
    1. If `use_spatial_pos` is explicitly specified in `config` as a boolean, return it.
    2. If absent from `config`:
       - Check `training_git_commit` / `commit_sha`.
       - If commit is in KNOWN_SPATIAL_POS_PRE_COMMITS -> return False.
       - If commit is in KNOWN_SPATIAL_POS_POST_COMMITS -> return True.
       - Otherwise, attempt `git merge-base --is-ancestor 0f2ed24 <commit>`:
         * returncode == 0: strictly descendant -> return True.
         * returncode == 1: strictly NOT descendant (prior commit) -> return False.
         * other non-zero (returncode > 1 or error): git error or missing commit object.
           Must raise RuntimeError rather than guessing!
    3. If commit is missing and no explicit config:
       - If allow_unverified_fallback is True, return default_if_unverified.
       - Otherwise, raise ValueError to fail closed against silent inductive bias drift.
    """
    cfg = checkpoint_data.get("config", {}) if isinstance(checkpoint_data, dict) else {}
    if "use_spatial_pos" in cfg and isinstance(cfg["use_spatial_pos"], bool):
        return cfg["use_spatial_pos"]

    # Fallback to commit-based inference
    commit = (
        checkpoint_data.get("training_git_commit")
        or checkpoint_data.get("commit_sha")
        or cfg.get("training_git_commit")
        or cfg.get("commit_sha")
    )
    if commit:
        commit_str = str(commit).strip()
        if any(commit_str.startswith(c) for c in KNOWN_SPATIAL_POS_PRE_COMMITS):
            return False
        if any(commit_str.startswith(c) for c in KNOWN_SPATIAL_POS_POST_COMMITS):
            return True

        # Rigorous git ancestry query with strict return code interpretation:
        # 0 = ancestor, 1 = not ancestor, >1 = git error (object missing, corrupted, etc.)
        try:
            import subprocess
            res = subprocess.run(
                ["git", "merge-base", "--is-ancestor", POST_SPATIAL_POS_INTRO_COMMIT, commit_str],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                return True
            elif res.returncode == 1:
                return False
            else:
                stderr_msg = res.stderr.strip() if res.stderr else f"exit code {res.returncode}"
                raise RuntimeError(
                    f"Git ancestry check failed for commit '{commit_str}' against introduction commit "
                    f"'{POST_SPATIAL_POS_INTRO_COMMIT[:8]}': {stderr_msg}. Cannot verify spatial position encoding."
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Git ancestry check timed out for commit '{commit_str}'.")

    # If provenance is completely absent or unverified:
    if allow_unverified_fallback:
        return default_if_unverified

    raise ValueError(
        "Cannot resolve 'use_spatial_pos' for checkpoint: explicit 'use_spatial_pos' is missing from config "
        "and no verifiable git commit provenance is present. Formal model evaluation cannot proceed by guessing."
    )


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse softplus: y = softplus^{-1}(x) = log(exp(x) - 1)."""
    # For large x (x >= 20.0), softplus(x) ~= x, so inverse_softplus(x) ~= x
    # For smaller x, use log(expm1(x))
    threshold = 20.0
    return torch.where(x >= threshold, x, torch.log(torch.expm1(torch.clamp(x, min=1e-12))))


def compute_g1_bias_init_from_g0(
    v_g0: torch.Tensor,
    variance_floor: float = 1e-4,
    min_margin: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute numerically stable bias initialization for G1 variance head matching G0.

    Ensures that for sigma^2 = softplus(a) + variance_floor:
    When linear weights W_var = 0, a = b_c:
    pred_variance = softplus(b_c) + variance_floor == effective_g0.

    Handles boundary condition where v_g0 <= variance_floor:
    Clamps effective_g0 = max(v_g0, variance_floor + min_margin), ensuring
    argument to inverse_softplus is strictly positive, eliminating NaN and -inf.

    Returns:
        (b_init, effective_g0): The bias vector and the effective G0 variance vector.
    """
    effective_g0 = torch.clamp(v_g0, min=variance_floor + min_margin)
    target_softplus = effective_g0 - variance_floor
    b_init = inverse_softplus(target_softplus)
    return b_init, effective_g0
