import math
from abc import ABC, abstractmethod
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from pos_embeddings import ALiBiPositionalEncoding, RotaryPositionalEncoding
from utils import clone


class AbstractAttention(ABC, nn.Module):
    """
    Abstract base class for attention mechanisms.
    """

    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1) -> None:
        """
        Initialize the attention mechanism.

        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Model dimension.
            dropout (float): Dropout probability.
        """
        super(AbstractAttention, self).__init__()
        assert d_model % n_heads == 0, 'incompatible `d_model` and `n_heads`'
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linears = clone(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    @abstractmethod
    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        dropout: nn.Dropout | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Scaled Dot-Product Attention.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (Optional[torch.Tensor]): Mask tensor. Default: None.
            dropout (Optional[nn.Dropout]): Dropout layer. Default: None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        raise NotImplementedError

    def forward(
        self,
        query: torch.torch.Tensor,
        key: torch.torch.Tensor,
        value: torch.torch.Tensor,
        mask: torch.torch.Tensor | None = None,
    ) -> torch.torch.Tensor:
        """
        Forward pass for multi-headed attention.

        Args:
            query (torch.torch.Tensor): Query tensor.
            key (torch.torch.Tensor): Key tensor.
            value (torch.torch.Tensor): Value tensor.
            mask (Optional[torch.torch.Tensor]): Mask tensor. Default: None.

        Returns:
            torch.torch.Tensor: Output tensor.
        """
        if mask is not None:
            mask = mask.unsqueeze(1).to(query.device)
        nbatches = query.size(0)

        # Linear projections
        query, key, value = [
            lin(x).view(nbatches, -1, self.n_heads, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value), strict=False)
        ]

        # Apply attention
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concatenate and apply final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_heads * self.d_k)
        return self.linears[-1](x)


class AbsoluteMultiHeadedAttention(AbstractAttention):
    """
    Implements multi-headed attention.

    Args:
        n_heads (int): Number of attention heads.
        d_model (int): Model dimension.
        dropout (float): Dropout probability.
    """

    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1) -> None:
        super(AbsoluteMultiHeadedAttention, self).__init__(n_heads, d_model, dropout)

    @override
    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    @override
    def forward(self, query, key, value, mask=None):
        return super().forward(query, key, value, mask)


class RelativeGlobalAttention(AbstractAttention):
    """
    Relative global attention as described in [1]_

    Parameters
    ----------
    d_model : int
        Dimensionality of the model.
    n_heads : int
        Number of attention heads.
    max_len : int, optional
        Maximum sequence length. Default: 1024.
    dropout : float, optional
        Dropout probability. Default: 0.1.

    References
    ----------
    .. [1] https://arxiv.org/abs/1803.02155
    """

    def __init__(
        self, n_heads: int, d_model: int, max_len: int = 1024, dropout: float = 0.1
    ) -> None:
        super().__init__(n_heads, d_model, dropout)
        self.max_len = max_len
        self.Er = nn.Parameter(torch.randn(max_len, self.d_k))

    def _skew(self, qe_r: torch.Tensor) -> torch.Tensor:
        """Apply skewing operation for relative attention."""
        padded = F.pad(qe_r, (1, 0))  # Pad one column to the left
        batch_size, n_heads, num_rows, num_cols = padded.shape
        reshaped = padded.view(batch_size, n_heads, num_cols, num_rows)
        # Only keep the relevant part of the matrix
        return reshaped[:, :, 1 : num_rows + 1, :]

    def _get_relative_embeddings(self, seq_len: int) -> torch.Tensor:
        """Get relative embeddings with proper offset handling."""
        start = max(0, self.max_len - seq_len)
        end = self.max_len
        return self.Er[start:end, :][:seq_len, :]

    @override
    def attention(self, query, key, value, mask=None, dropout=None):
        seq_len = query.size(2)
        if seq_len > self.max_len:
            raise ValueError('sequence length exceeds model capacity')

        # Get the relevant part of positional encodings
        er_t = self._get_relative_embeddings(seq_len).transpose(0, 1)
        qe_r = torch.matmul(query, er_t)
        s_rel = self._skew(qe_r)

        qk_t = torch.matmul(query, key.transpose(-2, -1))

        # Ensure s_rel has the same size as qk_t
        if s_rel.size() != qk_t.size():
            s_rel = F.pad(s_rel, (0, qk_t.size(-1) - s_rel.size(-1)))

        scores = (qk_t + s_rel) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        return super().forward(query, key, value, mask)


class RotaryMultiHeadAttention(AbstractAttention):
    def __init__(self, n_heads, d_model, rope_percentage: float = 0.5, dropout=0.1):
        super().__init__(n_heads, d_model, dropout)

        d_rope = int(rope_percentage * self.d_k)

        self.query_rotary_pe = RotaryPositionalEncoding(d_rope)
        self.key_rotary_pe = RotaryPositionalEncoding(d_rope)

    @override
    def attention(self, query, key, value, mask=None, dropout=None):
        # * Apply Rotation onto Q & K
        d_k = query.size(-1)
        query = self.query_rotary_pe(query)
        key = self.key_rotary_pe(key)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    @override
    def forward(
        self,
        query: torch.torch.Tensor,
        key: torch.torch.Tensor,
        value: torch.torch.Tensor,
        mask: torch.torch.Tensor | None = None,
    ) -> torch.torch.Tensor:
        return super().forward(query, key, value, mask)


class ALiBiMultiHeadAttention(AbstractAttention):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__(n_heads, d_model, dropout)

        self.pe = ALiBiPositionalEncoding(self.n_heads)

    @override
    def attention(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        tgt_len, src_len = scores.size(-2), scores.size(-1)
        alibi_bias = self.pe(max(tgt_len, src_len), scores.device)

        # Slice to match exact dimensions and expand batch
        alibi_bias = alibi_bias[:, :tgt_len, :src_len].unsqueeze(0)
        alibi_bias = alibi_bias.expand(scores.size(0), -1, -1, -1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    @override
    def forward(
        self,
        query: torch.torch.Tensor,
        key: torch.torch.Tensor,
        value: torch.torch.Tensor,
        mask: torch.torch.Tensor | None = None,
    ) -> torch.torch.Tensor:
        return super().forward(query, key, value, mask)
