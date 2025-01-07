import copy

import torch
import torch.nn as nn


def clone(module: nn.Module, n_clones: int) -> nn.ModuleList:
    """
    Produce n_clones identical layers.

    Args:
        module (nn.Module): Module to clone.
        n_clones (int): Number of clones.

    Returns:
        nn.ModuleList: List of cloned modules.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_clones)])


def subsequent_mask(size: int) -> torch.Tensor:
    """
    Create a mask to hide future positions.

    Args:
        size (int): Size of the mask.

    Returns:
        torch.Tensor: Subsequent mask tensor.
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0
