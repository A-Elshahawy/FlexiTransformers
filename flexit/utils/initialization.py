"""Weight initialization utilities."""

import math

from torch import nn


def xavier_uniform_init(module: nn.Module, gain: float = 1.0) -> None:
    """
    Apply Xavier/Glorot uniform initialization.

    Args:
        module: Module to initialize
        gain: Scaling factor
    """
    if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight, gain=gain)


def xavier_normal_init(module: nn.Module, gain: float = 1.0) -> None:
    """
    Apply Xavier/Glorot normal initialization.

    Args:
        module: Module to initialize
        gain: Scaling factor
    """
    if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
        nn.init.xavier_normal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight, gain=gain)


def kaiming_uniform_init(
    module: nn.Module, mode: str = 'fan_in', nonlinearity: str = 'relu'
) -> None:
    """
    Apply Kaiming/He uniform initialization.

    Args:
        module: Module to initialize
        mode: 'fan_in' or 'fan_out'
        nonlinearity: 'relu' or 'leaky_relu'
    """
    if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)


def kaiming_normal_init(
    module: nn.Module, mode: str = 'fan_in', nonlinearity: str = 'relu'
) -> None:
    """
    Apply Kaiming/He normal initialization.

    Args:
        module: Module to initialize
        mode: 'fan_in' or 'fan_out'
        nonlinearity: 'relu' or 'leaky_relu'
    """
    if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)


def normal_init(module: nn.Module, mean: float = 0.0, std: float = 0.02) -> None:
    """
    Apply normal initialization.

    Common for transformer models (GPT-2/3 use std=0.02).

    Args:
        module: Module to initialize
        mean: Mean of normal distribution
        std: Standard deviation
    """
    if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
        nn.init.normal_(module.weight, mean=mean, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=mean, std=std)


def scaled_init(module: nn.Module, d_model: int, n_layers: int) -> None:
    """
    Apply scaled initialization for transformers.

    Scales weights by 1/sqrt(2*n_layers) as in GPT-2/3.

    Args:
        module: Module to initialize
        d_model: Model dimension
        n_layers: Number of layers
    """
    std = 0.02 / math.sqrt(2 * n_layers)

    if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)


def init_bert_weights(module: nn.Module) -> None:
    """
    Initialize weights as in BERT.

    - Normal(0, 0.02) for most weights
    - Zeros for biases
    - Ones for LayerNorm weights, zeros for LayerNorm biases

    Args:
        module: Module to initialize
    """
    if isinstance(module, nn.Linear | nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def init_weights(
    module: nn.Module,
    method: str = 'xavier',
    std: float = 0.02,
    d_model: int | None = None,
    n_layers: int | None = None,
) -> None:
    """
    Initialize module weights using specified method.

    Args:
        module: Module to initialize
        method: Initialization method ('xavier', 'kaiming', 'normal', 'scaled', 'bert')
        std: Standard deviation for normal init
        d_model: Model dimension (required for scaled init)
        n_layers: Number of layers (required for scaled init)
    """
    if method == 'xavier':
        xavier_uniform_init(module)
    elif method == 'kaiming':
        kaiming_uniform_init(module)
    elif method == 'normal':
        normal_init(module, std=std)
    elif method == 'scaled':
        if d_model is None or n_layers is None:
            raise ValueError('scaled init requires d_model and n_layers')
        scaled_init(module, d_model, n_layers)
    elif method == 'bert':
        init_bert_weights(module)
    else:
        raise ValueError(f'Unknown initialization method: {method}')
