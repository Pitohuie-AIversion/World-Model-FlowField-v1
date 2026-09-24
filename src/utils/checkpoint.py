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
