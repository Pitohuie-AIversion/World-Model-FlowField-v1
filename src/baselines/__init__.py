"""Baselines package."""

from src.baselines.persistence import PersistenceBaseline
from src.baselines.fno import FNO2D
from src.baselines.pde_transformer import PDETransformer

__all__ = ["PersistenceBaseline", "FNO2D", "PDETransformer"]
