"""Checkpointing and artifact persistence utilities."""

import os
import shutil
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


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

    # Handle model state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)

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
