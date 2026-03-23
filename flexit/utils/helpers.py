"""Helper utilities for transformer models."""

import copy
from typing import Any

import torch
from torch import nn


def clone_module(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Create n deep copies of a module.

    Args:
        module: Module to clone
        n: Number of copies

    Returns:
        ModuleList of n independent copies
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Total number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.0,
    no_decay_bias: bool = True,
    no_decay_norm: bool = True,
) -> list[dict[str, Any]]:
    """
    Create parameter groups with different weight decay settings.

    Typically, biases and normalization parameters should not have weight decay.

    Args:
        model: PyTorch model
        weight_decay: Weight decay value for regularized parameters
        no_decay_bias: If True, exclude bias from weight decay
        no_decay_norm: If True, exclude normalization layers from weight decay

    Returns:
        List of parameter group dicts for optimizer
    """
    decay = set()
    no_decay = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if this is a bias parameter
        if (no_decay_bias and 'bias' in name) or (
            no_decay_norm
            and any(norm_type in name.lower() for norm_type in ['norm', 'ln', 'bn', 'gn'])
        ):
            no_decay.add(name)
        else:
            decay.add(name)

    # Verify all parameters are assigned
    param_dict = dict(model.named_parameters())
    assert len(decay) + len(no_decay) == len(param_dict)

    return [
        {
            'params': [param_dict[name] for name in sorted(decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [param_dict[name] for name in sorted(no_decay)],
            'weight_decay': 0.0,
        },
    ]


def get_device(device: str | torch.device | None = None) -> torch.device:
    """
    Get torch device.

    Args:
        device: Device string or object. If None, auto-detect (cuda if available)

    Returns:
        torch.device object
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if isinstance(device, str):
        return torch.device(device)
    return device


def move_to_device(batch: Any, device: torch.device) -> Any:
    """
    Recursively move batch to device.

    Handles dict, list, tuple, and tensor structures.

    Args:
        batch: Batch data (tensor, dict, list, or tuple)
        device: Target device

    Returns:
        Batch moved to device
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    else:
        return batch


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f'{hours}h')
    if minutes > 0:
        parts.append(f'{minutes}m')
    parts.append(f'{secs}s')

    return ' '.join(parts)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
