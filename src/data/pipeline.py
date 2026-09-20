"""Reproducible Data Pipeline for The Well shear_flow V1.

Provides end-to-end DataLoader creation with:
- Grouped Split (Zero-IC-Leakage) or Official Split
- Channel-wise normalization fitted strictly on training data
- Sliding historical and future window generation
- Fully deterministic batch sequencing across epochs
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader

from src.data.normalization import FieldNormalizer
from src.data.shear_flow_dataset import ShearFlowDataset
from src.utils.reproducibility import seed_everything


def seed_worker(worker_id: int):
    """Worker init function to guarantee deterministic behavior in multi-worker DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    import random
    import numpy as np
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def fit_normalizer_on_dataset(
    dataset: ShearFlowDataset,
    max_samples: int = 500,
) -> FieldNormalizer:
    """Fit a FieldNormalizer instance using only samples from the training dataset.

    Computes channel-wise mean and std across physical states q = [u, v, p, s].
    """
    normalizer = FieldNormalizer()
    collected = []
    step = max(1, len(dataset) // max_samples)
    for i in range(0, len(dataset), step):
        sample = dataset[i]
        # history: (L, 4, Ny, Nx), future: (H, 4, Ny, Nx)
        full_seq = torch.cat([sample["history"], sample["future"]], dim=0) # (L + H, 4, Ny, Nx)
        collected.append(full_seq)

    stacked = torch.stack(collected, dim=0) # (N_sub, L+H, 4, Ny, Nx)
    # Fit across (N_sub, L+H, Ny, Nx)
    mean = stacked.mean(dim=(0, 1, 3, 4), keepdim=True).squeeze(1) # (1, 4, 1, 1)
    std = stacked.std(dim=(0, 1, 3, 4), keepdim=True).squeeze(1)
    std = torch.clamp(std, min=1e-6)

    normalizer.register_buffer("mean", mean)
    normalizer.register_buffer("std", std)
    return normalizer


def create_flow_dataloaders(
    split_type: str = "grouped",
    split_file: Optional[str] = None,
    history_length: int = 4,
    horizon: int = 1,
    stride: int = 1,
    downsample_factor: int = 1,
    batch_size: int = 4,
    num_workers: int = 0,
    normalize: bool = True,
    normalizer: Optional[FieldNormalizer] = None,
    stats_dir: str = "outputs/normalization",
    preload_to_memory: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[FieldNormalizer]]:
    """Create reproducible PyTorch DataLoaders for train, valid, and test sets.

    Args:
        split_type: 'grouped' (zero-leakage) or 'official'.
        split_file: Optional path to split JSON registry. Defaults to 'outputs/splits/{split_type}_split.json'.
        history_length: Historical sequence length L (default: 4).
        horizon: Prediction horizon H (default: 1).
        stride: Temporal window stride (default: 1).
        downsample_factor: Spatial downsampling factor (default: 1 for 256x512, 2 for 128x256).
        batch_size: Mini-batch size.
        num_workers: DataLoader workers.
        normalize: Whether to apply channel-wise normalization.
        normalizer: Optional pre-fitted normalizer.
        stats_dir: Directory to cache fitted normalizer statistics.
        preload_to_memory: Whether to cache HDF5 files in memory.
        seed: Random seed for deterministic data loading.

    Returns:
        (train_loader, valid_loader, test_loader, normalizer)
    """
    seed_everything(seed)

    if split_file is None:
        split_file = f"outputs/splits/{split_type}_split.json"

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file {split_file} not found. Please run scripts/build_splits.py first.")

    with open(split_file, "r") as f:
        split_data = json.load(f)

    # Instantiate datasets without normalizer first to calculate train statistics if needed
    if split_type == "grouped":
        train_trajs = split_data["train"]
        valid_trajs = split_data["valid"]
        test_trajs = split_data["test"]

        raw_train_ds = ShearFlowDataset(
            trajectories=train_trajs,
            history_length=history_length,
            horizon=horizon,
            stride=stride,
            normalizer=None,
            preload_to_memory=preload_to_memory,
            downsample_factor=downsample_factor,
        )
    elif split_type == "official":
        train_files = split_data["train"]
        valid_files = split_data["valid"]
        test_files = split_data["test"]

        raw_train_ds = ShearFlowDataset(
            file_paths=train_files,
            history_length=history_length,
            horizon=horizon,
            stride=stride,
            normalizer=None,
            preload_to_memory=preload_to_memory,
            downsample_factor=downsample_factor,
        )
    else:
        raise ValueError(f"Unsupported split_type: {split_type}. Choose 'grouped' or 'official'.")

    # Fit or load normalizer strictly from training data
    fitted_normalizer = None
    if normalize:
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, f"stats_{split_type}.pt")

        if normalizer is not None:
            fitted_normalizer = normalizer
        elif os.path.exists(stats_path):
            state = torch.load(stats_path, weights_only=True)
            fitted_normalizer = FieldNormalizer()
            fitted_normalizer.load_state_dict(state)
        else:
            fitted_normalizer = fit_normalizer_on_dataset(raw_train_ds)
            torch.save(fitted_normalizer.state_dict(), stats_path)

    # Attach normalizer to train dataset
    raw_train_ds.normalizer = fitted_normalizer
    train_dataset = raw_train_ds

    # Build validation and test datasets
    if split_type == "grouped":
        valid_dataset = ShearFlowDataset(
            trajectories=valid_trajs,
            history_length=history_length,
            horizon=horizon,
            stride=stride,
            normalizer=fitted_normalizer,
            preload_to_memory=preload_to_memory,
            downsample_factor=downsample_factor,
        )
        test_dataset = ShearFlowDataset(
            trajectories=test_trajs,
            history_length=history_length,
            horizon=horizon,
            stride=stride,
            normalizer=fitted_normalizer,
            preload_to_memory=preload_to_memory,
            downsample_factor=downsample_factor,
        )
    else:
        valid_dataset = ShearFlowDataset(
            file_paths=valid_files,
            history_length=history_length,
            horizon=horizon,
            stride=stride,
            normalizer=fitted_normalizer,
            preload_to_memory=preload_to_memory,
            downsample_factor=downsample_factor,
        )
        test_dataset = ShearFlowDataset(
            file_paths=test_files,
            history_length=history_length,
            horizon=horizon,
            stride=stride,
            normalizer=fitted_normalizer,
            preload_to_memory=preload_to_memory,
            downsample_factor=downsample_factor,
        )

    # Deterministic DataLoaders with PyTorch Generator
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=torch.cuda.is_available(),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, valid_loader, test_loader, fitted_normalizer
