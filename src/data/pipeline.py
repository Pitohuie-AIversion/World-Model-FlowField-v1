"""Reproducible Data Pipeline for The Well shear_flow V1.

Provides end-to-end DataLoader creation with:
- Grouped Split (Zero-IC-Leakage) or Official Split
- Channel-wise normalization fitted strictly on training data
- Sliding historical and future window generation
- Fully deterministic batch sequencing across epochs
"""

import json
import os
import time
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
    Samples single frame per item to be strictly invariant to horizon H.
    """
    normalizer = FieldNormalizer()
    collected = []
    step = max(1, len(dataset) // max_samples)
    for i in range(0, len(dataset), step):
        sample = dataset[i]
        # Use first history frame q_0 in R^(4, Ny, Nx) to guarantee horizon H-invariance
        collected.append(sample["history"][0])

    stacked = torch.stack(collected, dim=0) # (N_sub, 4, Ny, Nx)
    # Fit across (N_sub, Ny, Nx)
    mean = stacked.mean(dim=(0, 2, 3), keepdim=True) # (1, 4, 1, 1)
    std = stacked.std(dim=(0, 2, 3), keepdim=True)
    std = torch.clamp(std, min=1e-6)

    normalizer.register_buffer("mean", mean)
    normalizer.register_buffer("std", std)
    return normalizer


def create_flow_datasets(
    split_type: str = "grouped",
    split_file: Optional[str] = None,
    data_root: Optional[str] = None,
    history_length: int = 4,
    horizon: int = 1,
    stride: int = 1,
    train_stride: Optional[int] = None,
    valid_stride: Optional[int] = None,
    test_stride: Optional[int] = None,
    downsample_factor: int = 1,
    normalize: bool = True,
    normalizer: Optional[FieldNormalizer] = None,
    stats_dir: str = "outputs/normalization",
    preload_to_memory: bool = False,
    seed: int = 42,
) -> Tuple[ShearFlowDataset, ShearFlowDataset, ShearFlowDataset, Optional[FieldNormalizer]]:
    """Create reproducible PyTorch Datasets for train, valid, and test sets.

    Supports:
        - grouped (zero-leakage trajectory clusters)
        - official (The Well designated files)
        - parameter_holdout_re (Reynolds OOD generalization)
        - parameter_holdout_sc (Schmidt OOD generalization)
        - parameter_holdout (combined)

    Args:
        split_type: Name of split ('grouped', 'official', 'parameter_holdout_re', 'parameter_holdout_sc', etc.)
        split_file: Optional path to split JSON registry.
        data_root: Optional base directory for dataset HDF5 files to resolve relative paths.
        history_length: Historical sequence length L (default: 4).
        horizon: Prediction horizon H (default: 1).
        stride: Default temporal window stride (default: 1).
        train_stride: Optional stride override for training set.
        valid_stride: Optional stride override for validation set.
        test_stride: Optional stride override for test set.
        downsample_factor: Spatial downsampling factor (default: 1 for 256x512, 2 for 128x256).
        normalize: Whether to apply channel-wise normalization.
        normalizer: Optional pre-fitted normalizer.
        stats_dir: Directory to cache fitted normalizer statistics.
        preload_to_memory: Whether to cache HDF5 files in memory.
        seed: Random seed for deterministic initialization.

    Returns:
        (train_dataset, valid_dataset, test_dataset, normalizer)
    """
    seed_everything(seed)

    if split_file is None:
        cand1 = f"outputs/splits/{split_type}_split.json"
        cand2 = f"outputs/splits/{split_type}.json"
        if os.path.exists(cand1):
            split_file = cand1
        elif os.path.exists(cand2):
            split_file = cand2
        else:
            split_file = cand1

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file {split_file} not found. Please run scripts/build_splits.py first.")

    with open(split_file, "r") as f:
        split_data = json.load(f)

    t_stride = train_stride if train_stride is not None else stride
    v_stride = valid_stride if valid_stride is not None else stride
    te_stride = test_stride if test_stride is not None else stride

    train_entries = split_data.get("train", [])
    valid_entries = split_data.get("valid", [])
    test_entries = split_data.get("test", [])

    is_dict_trajs = len(train_entries) > 0 and isinstance(train_entries[0], dict)

    if is_dict_trajs:
        raw_train_ds = ShearFlowDataset(
            trajectories=train_entries,
            data_root=data_root,
            history_length=history_length,
            horizon=horizon,
            stride=t_stride,
            normalizer=None,
            preload_to_memory=preload_to_memory,
            downsample_factor=downsample_factor,
        )
    else:
        raw_train_ds = ShearFlowDataset(
            file_paths=train_entries,
            data_root=data_root,
            history_length=history_length,
            horizon=horizon,
            stride=t_stride,
            normalizer=None,
            preload_to_memory=preload_to_memory,
            downsample_factor=downsample_factor,
        )

    fitted_normalizer = None
    if normalize:
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, f"stats_{split_type}.pt")
        meta_path = os.path.join(stats_dir, f"stats_{split_type}_metadata.json")

        valid_cached = False
        if normalizer is not None:
            fitted_normalizer = normalizer
            valid_cached = True
        elif os.path.exists(stats_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as mf:
                    meta = json.load(mf)
                if (
                    meta.get("fit_protocol") == "trajectory-reference-v2"
                    and meta.get("downsample_factor") == downsample_factor
                ):
                    state = torch.load(stats_path, weights_only=True)
                    fitted_normalizer = FieldNormalizer()
                    fitted_normalizer.load_state_dict(state)
                    valid_cached = True
            except Exception:
                valid_cached = False

        if not valid_cached:
            # Build an invariant reference dataset with L=1, H=1, stride=1 to fit normalizer
            # This guarantees statistics are 100% horizon-invariant and stride-invariant
            if is_dict_trajs:
                ref_norm_ds = ShearFlowDataset(
                    trajectories=train_entries,
                    data_root=data_root,
                    history_length=1,
                    horizon=1,
                    stride=1,
                    normalizer=None,
                    preload_to_memory=False,
                    downsample_factor=downsample_factor,
                )
            else:
                ref_norm_ds = ShearFlowDataset(
                    file_paths=train_entries,
                    data_root=data_root,
                    history_length=1,
                    horizon=1,
                    stride=1,
                    normalizer=None,
                    preload_to_memory=False,
                    downsample_factor=downsample_factor,
                )
            fitted_normalizer = fit_normalizer_on_dataset(ref_norm_ds)
            torch.save(fitted_normalizer.state_dict(), stats_path)

            metadata = {
                "fit_protocol": "trajectory-reference-v2",
                "split_type": split_type,
                "downsample_factor": downsample_factor,
                "channels": ["u", "v", "p", "s"],
                "mean": fitted_normalizer.mean.view(-1).tolist() if fitted_normalizer.mean is not None else None,
                "std": fitted_normalizer.std.view(-1).tolist() if fitted_normalizer.std is not None else None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(meta_path, "w") as mf:
                json.dump(metadata, mf, indent=2)

    raw_train_ds.normalizer = fitted_normalizer
    train_dataset = raw_train_ds

    if is_dict_trajs:
        valid_dataset = ShearFlowDataset(
            trajectories=valid_entries,
            data_root=data_root,
            history_length=history_length,
            horizon=horizon,
            stride=v_stride,
            normalizer=fitted_normalizer,
            preload_to_memory=preload_to_memory,
            downsample_factor=downsample_factor,
        )
        test_dataset = ShearFlowDataset(
            trajectories=test_entries,
            data_root=data_root,
            history_length=history_length,
            horizon=horizon,
            stride=te_stride,
            normalizer=fitted_normalizer,
            preload_to_memory=preload_to_memory,
            downsample_factor=downsample_factor,
        )
    else:
        valid_dataset = ShearFlowDataset(
            file_paths=valid_entries,
            data_root=data_root,
            history_length=history_length,
            horizon=horizon,
            stride=v_stride,
            normalizer=fitted_normalizer,
            preload_to_memory=preload_to_memory,
            downsample_factor=downsample_factor,
        )
        test_dataset = ShearFlowDataset(
            file_paths=test_entries,
            data_root=data_root,
            history_length=history_length,
            horizon=horizon,
            stride=te_stride,
            normalizer=fitted_normalizer,
            preload_to_memory=preload_to_memory,
            downsample_factor=downsample_factor,
        )

    return train_dataset, valid_dataset, test_dataset, fitted_normalizer


def create_flow_dataloaders(
    split_type: str = "grouped",
    split_file: Optional[str] = None,
    data_root: Optional[str] = None,
    history_length: int = 4,
    horizon: int = 1,
    stride: int = 1,
    train_stride: Optional[int] = None,
    valid_stride: Optional[int] = None,
    test_stride: Optional[int] = None,
    downsample_factor: int = 1,
    batch_size: int = 4,
    num_workers: int = 0,
    normalize: bool = True,
    normalizer: Optional[FieldNormalizer] = None,
    stats_dir: str = "outputs/normalization",
    preload_to_memory: bool = False,
    is_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    return_sampler: bool = False,
) -> Union[
    Tuple[DataLoader, DataLoader, DataLoader, Optional[FieldNormalizer]],
    Tuple[DataLoader, DataLoader, DataLoader, Optional[FieldNormalizer], Optional[torch.utils.data.distributed.DistributedSampler]],
]:
    """Create reproducible PyTorch DataLoaders with optional DDP DistributedSampler support."""
    train_ds, valid_ds, test_ds, fitted_normalizer = create_flow_datasets(
        split_type=split_type,
        split_file=split_file,
        data_root=data_root,
        history_length=history_length,
        horizon=horizon,
        stride=stride,
        train_stride=train_stride,
        valid_stride=valid_stride,
        test_stride=test_stride,
        downsample_factor=downsample_factor,
        normalize=normalize,
        normalizer=normalizer,
        stats_dir=stats_dir,
        preload_to_memory=preload_to_memory,
        seed=seed,
    )

    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers if not preload_to_memory else 0,
        worker_init_fn=seed_worker,
        generator=g if train_sampler is None else None,
        pin_memory=torch.cuda.is_available(),
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if not preload_to_memory else 0,
        worker_init_fn=seed_worker,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if not preload_to_memory else 0,
        worker_init_fn=seed_worker,
        pin_memory=torch.cuda.is_available(),
    )

    if return_sampler:
        return train_loader, valid_loader, test_loader, fitted_normalizer, train_sampler
    return train_loader, valid_loader, test_loader, fitted_normalizer
