"""Losses package for World-Model-FlowField-v1.

Exports field reconstruction losses, multi-step rollout losses,
and spectral physical conservation penalties (divergence and vorticity).
"""

from src.losses.field import FieldLoss
from src.losses.rollout import RolloutLoss
from src.losses.divergence import DivergenceLoss
from src.losses.vorticity import VorticityLoss

__all__ = [
    "FieldLoss",
    "RolloutLoss",
    "DivergenceLoss",
    "VorticityLoss",
]
