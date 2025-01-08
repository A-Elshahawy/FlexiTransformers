import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Args:
        d_model (int): Model dimension.
        dropout (float): Dropout probability.
        max_len (int): Maximum sequence length. Default: 5000.
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super(SinusoidalPositionalEncoding, self).__init__()
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


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for self-attention.

    Args:
        d_model (int): Model dimension.
        max_len (int): Maximum sequence length. Default: 5000.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(2 * max_len - 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device)
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(0) + self.max_len - 1
        return x + self.embedding(rel_pos)


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary positional encoding (RoPE).

    Args:
        d_model (int): Model dimension.
        max_len (int): Maximum sequence length. Default: 5000.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(RotaryPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.freqs = self._compute_frequencies()

    def _compute_frequencies(self) -> torch.Tensor:
        freqs = 1.0 / (
            10000 ** (torch.arange(0, self.d_model, 2, dtype=torch.float) / self.d_model)
        )
        return freqs

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(pos, self.freqs.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = torch.cos(emb).unsqueeze(0)
        sin = torch.sin(emb).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin


class ALiBiPositionalEncoding(nn.Module):
    """
    Attention with Linear Biases (ALiBi) positional encoding.

    Args:
        num_heads (int): Number of attention heads.
        max_len (int): Maximum sequence length. Default: 5000.
    """

    def __init__(self, num_heads: int, max_len: int = 5000) -> None:
        super(ALiBiPositionalEncoding, self).__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        self.slopes = self._compute_slopes()

    def _compute_slopes(self) -> torch.Tensor:
        slopes = torch.pow(
            2,
            torch.arange(1, self.num_heads + 1, dtype=torch.float)
            * -(math.log2(10000) / self.num_heads),
        )
        return slopes

    def forward(self, attention_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len, device=attention_scores.device).float()
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(0)
        biases = -torch.abs(rel_pos) * self.slopes.view(-1, 1, 1).to(attention_scores.device)
        return attention_scores + biases
