"""Reproducibility utilities for deterministic experiments across PyTorch, NumPy, and Python."""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic_cuda: bool = False):
    """Sets random seed across all libraries for deterministic execution.

    Args:
        seed: Random seed integer.
        deterministic_cuda: If True, enforces torch.backends.cudnn.deterministic = True.
                            Note that this may slightly reduce throughput.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
