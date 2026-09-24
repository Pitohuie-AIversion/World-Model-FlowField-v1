"""Data package for World-Model-FlowField-v1.

Exports core dataset loaders, normalizers, and split management utilities.
"""

from src.data.shear_flow_dataset import (
    ShearFlowDataset,
    generate_window_indices,
    parse_shear_flow_filename,
)
from src.data.normalization import FieldNormalizer
from src.data.splits import SplitManager
from src.data.pipeline import (
    create_flow_datasets,
    create_flow_dataloaders,
    fit_normalizer_on_dataset,
    compute_split_hash,
)

__all__ = [
    "ShearFlowDataset",
    "generate_window_indices",
    "parse_shear_flow_filename",
    "FieldNormalizer",
    "SplitManager",
    "create_flow_datasets",
    "create_flow_dataloaders",
    "fit_normalizer_on_dataset",
    "compute_split_hash",
]
