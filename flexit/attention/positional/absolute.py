import math
from typing import Literal

import torch
from torch import Tensor, nn

from .base import PositionalEncoding


class SinusoidalPE(PositionalEncoding):
    """Standard sinusoidal positional encoding (Vaswani et al.)"""

    @property
    def injection_point(self) -> Literal['embedding']:
        return 'embedding'

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def apply_to_embedding(self, x: Tensor) -> Tensor:
        """x: [batch, seq_len, d_model]"""
        x = x + self.pe[:, : x.size(1)]  # type: ignore
        return self.dropout(x)


class LearnedPE(PositionalEncoding):
    """Learned positional embeddings."""

    @property
    def injection_point(self) -> Literal['embedding']:
        return 'embedding'

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def apply_to_embedding(self, x: Tensor) -> Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.pe(positions)
        return self.dropout(x)
