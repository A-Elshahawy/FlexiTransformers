"""
Positional Encoding Implementations

This module implements various positional encoding methods used in transformer models,
including absolute, rotary, and ALiBi positional encodings.
"""

import math

import torch
import torch.nn as nn


class AbsolutePositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Args:
        d_model (int): Model dimension.
        dropout (float): Dropout probability.
        max_len (int): Maximum sequence length. Default: 5000.

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        pe (torch.Tensor): Positional encoding tensor.

    Methods:
        forward: Add positional encoding to input tensor.
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        """
        Initialize absolute positional encoding.

        Args:
            d_model (int): Model dimension.
            dropout (float): Dropout probability.
            max_len (int): Maximum sequence length.
        """
        super(AbsolutePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """
    Implementation of rotary positional encoding as described in
    `RoFormer: Enhanced Transformer with Rotary Position Embedding <https://arxiv.org/abs/2104.09864>`_.

    Args:
        d_features (int): Dimension of features to apply rotary encoding to.
        base (int): Base for frequency bands calculation. Default: 10000.

    Attributes:
        d_features (int): Dimension of features.
        base (int): Base for frequency bands calculation.
        cos_cache (torch.Tensor): Cached cosine values.
        sin_cache (torch.Tensor): Cached sine values.

    Methods:
        _build_cache: Build cache for rotary encoding.
        _negative_half: Create negative half of features.
        forward: Apply rotary positional encoding to input tensor.
    """

    def __init__(self, d_features: int, base: int = 10_000) -> None:
        """
        Initialize rotary positional encoding.

        Args:
            d_features (int): Dimension of features.
            base (int): Base for frequency bands calculation.
        """
        super().__init__()
        self.d_features = d_features
        self.base = base
        self.cos_cache: torch.Tensor | None = None
        self.sin_cache: torch.Tensor | None = None

    def _build_cache(self, x: torch.Tensor) -> None:
        """
        Build cache for rotary encoding.

        Args:
            x (torch.Tensor): Input tensor.
        """
        if self.cos_cache is not None and x.shape[0] <= self.cos_cache.shape[0]:
            return
        seq_len = x.shape[0]
        device = x.device
        # Computing frequency bands for rotation
        self.inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.d_features, 2).float().to(device) / self.d_features)
        )
        seq_idx = torch.arange(seq_len, device=device).float()
        idx_theta = torch.einsum('n,d->nd', seq_idx, self.inv_freq)

        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.cos_cache = idx_theta2.cos()[:, None, None, :].to(device)
        self.sin_cache = idx_theta2.sin()[:, None, None, :].to(device)

    def _negative_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create negative half of features.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with negative half of features.
        """
        d_2 = self.d_features // 2

        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the rotary positional encoding of the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The sin and cos of the rotary positional encoding.
        """
        self._build_cache(x)
        x_rope, x_pass = x[..., : self.d_features], x[..., self.d_features :]
        neg_half_x = self._negative_half(x_rope)
        assert self.cos_cache is not None and self.sin_cache is not None
        x_rope = (x_rope * self.cos_cache[: x.shape[0]]) + (
            neg_half_x * self.sin_cache[: x.shape[0]]
        )

        return torch.cat((x_rope, x_pass), dim=-1)


class ALiBiPositionalEncoding(nn.Module):
    """
    Implements ALiBi (Attention with Linear Biases) positional encoding.

    Args:
        num_heads (int): Number of attention heads.
        max_len (int): Maximum sequence length. Default: 5000.

    Attributes:
        num_heads (int): Number of attention heads.
        max_len (int): Maximum sequence length.
        slopes (torch.Tensor): Slopes for each attention head.

    Methods:
        _get_slopes: Calculate slopes for each attention head.
        forward: Generate attention biases for ALiBi.
    """

    def __init__(self, num_heads: int, max_len: int = 5000) -> None:
        """
        Initialize ALiBi positional encoding.

        Args:
            num_heads (int): Number of attention heads.
            max_len (int): Maximum sequence length.
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        self.slopes = self._get_slopes(num_heads)

    def _get_slopes(self, n_heads: int) -> torch.Tensor:
        """
        Calculate slopes for each attention head.

        Args:
            n_heads (int): Number of attention heads.

        Returns:
            torch.Tensor: Slopes for each attention head.
        """
        n = 2 ** math.floor(math.log2(n_heads))
        m_0 = 2.0 ** (-8.0 / n)
        m = torch.pow(m_0, torch.arange(1, 1 + n))

        if n < n_heads:
            m_hat_0 = 2.0 ** (-4.0 / n)
            m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
            m = torch.cat([m, m_hat])

        return m

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate attention biases for ALiBi.

        Args:
            seq_len (int): Sequence length.
            device (torch.device): Device to store the biases.

        Returns:
            torch.Tensor: Attention biases for ALiBi.
        """
        positions = torch.arange(seq_len, device=device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        rel_pos = -torch.abs(rel_pos)
        return self.slopes.to(device).unsqueeze(-1).unsqueeze(-1) * rel_pos
