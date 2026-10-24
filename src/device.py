"""Pick PyTorch device: CUDA > MPS > CPU."""
from __future__ import annotations

import torch


def pick_device(*, prefer_mps: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ray_shard_device() -> torch.device:
    """Ray workers: one model per process; avoid stacking many models on MPS."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
