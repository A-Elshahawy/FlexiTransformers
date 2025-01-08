import math

import torch
import torch.nn as nn


class SinaoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Args:
        d_model (int): Model dimension.
        dropout (float): Dropout probability.
        max_len (int): Maximum sequence length. Default: 5000.
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super(SinaoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
