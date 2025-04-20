"""
Attention Mechanisms

This module implements various attention mechanisms used in transformer models,
including absolute, relative, rotary, and ALiBi attention implementations.
"""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .pos_embeddings import ALiBiPositionalEncoding, RotaryPositionalEncoding
from .utils import clone


class AbstractAttention(ABC, nn.Module):
    """
    Abstract base class for attention mechanisms.

    Args:
        n_heads (int): Number of attention heads.
        d_model (int): Model dimension.
        dropout (float): Dropout probability.

    Attributes:
        d_k (int): Dimension per head.
        n_heads (int): Number of attention heads.
        linears (nn.ModuleList): Linear layers for query, key, value, and output projections.
        attn (torch.Tensor): Attention weights (stored for visualization).
        dropout (nn.Dropout): Dropout layer.

    Methods:
        attention: Abstract method to compute attention scores.
        forward: Forward pass for multi-headed attention.
    """

    def __init__(
        self, n_heads: int, d_model: int, dropout: float = 0.1, mask_eps: float = -1e9
    ) -> None:
        """
        Initialize the attention mechanism.

        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Model dimension.
            dropout (float): Dropout probability.

        Raises:
            AssertionError: If d_model is not divisible by n_heads.
        """
        super(AbstractAttention, self).__init__()
        assert d_model % n_heads == 0, 'incompatible `d_model` and `n_heads`'
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linears = clone(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.mask_eps: float = mask_eps

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
        Compute attention scores and output.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (Optional[torch.Tensor]): Mask tensor. Default: None.
            dropout (Optional[nn.Dropout]): Dropout layer. Default: None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for multi-headed attention.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (Optional[torch.Tensor]): Mask tensor. Default: None.

        Returns:
            torch.Tensor: Output tensor after applying multi-head attention.
        """
        if mask is not None:
            mask = mask.unsqueeze(1).to(query.device)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.n_heads, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))  # noqa: B905
        ]

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_heads * self.d_k)
        return self.linears[-1](x)


class AbsoluteMultiHeadedAttention(AbstractAttention):
    """
    Implements standard multi-headed attention with absolute positional encoding.

    This class is based on the original Transformer paper:
    "Attention is All You Need" by Vaswani et al.
    Link: https://arxiv.org/abs/1706.03762

    Args:
        n_heads (int): Number of attention heads.
        d_model (int): Model dimension.
        dropout (float): Dropout probability.

    Attributes:
        Inherits all attributes from AbstractAttention.

    Methods:
        attention: Computes scaled dot-product attention.
        forward: Implements forward pass using parent class.
    """

    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1) -> None:
        """
        Initialize absolute multi-headed attention.

        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Model dimension.
            dropout (float): Dropout probability.
        """
        super(AbsoluteMultiHeadedAttention, self).__init__(n_heads, d_model, dropout)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        dropout: nn.Dropout | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (Optional[torch.Tensor]): Mask tensor. Default: None.
            dropout (Optional[nn.Dropout]): Dropout layer. Default: None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """

        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -self.mask_eps)
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for absolute multi-headed attention.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (Optional[torch.Tensor]): Mask tensor. Default: None.

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        return super().forward(query, key, value, mask)


class RotaryMultiHeadAttention(AbstractAttention):
    """
    Implements multi-headed attention with rotary positional encoding.
    Based on the paper "RoFormer: Enhanced Transformer with Rotary Positional Embedding"
    by Jianlin Su, et al. https://arxiv.org/abs/2104.09817

    Args:
        n_heads (int): Number of attention heads.
        d_model (int): Model dimension.
        rope_percentage (float): Percentage of dimensions to apply rotary encoding to.
        dropout (float): Dropout probability.

    Attributes:
        query_rotary_pe (RotaryPositionalEncoding): Rotary positional encoding for queries.
        key_rotary_pe (RotaryPositionalEncoding): Rotary positional encoding for keys.

    Methods:
        attention: Computes attention with rotary positional encoding.
        forward: Implements forward pass using parent class.
    """

    def __init__(
        self, n_heads: int, d_model: int, rope_percentage: float = 0.5, dropout: float = 0.1
    ) -> None:
        """
        Initialize rotary multi-headed attention.

        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Model dimension.
            rope_percentage (float): Percentage of dimensions to apply rotary encoding to.
            dropout (float): Dropout probability.
        """

        super().__init__(n_heads, d_model, dropout)

        d_rope = int(rope_percentage * self.d_k)

        self.query_rotary_pe = RotaryPositionalEncoding(d_rope)
        self.key_rotary_pe = RotaryPositionalEncoding(d_rope)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        dropout: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention with rotary positional encoding.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (Optional[torch.Tensor]): Mask tensor. Default: None.
            dropout (Optional[nn.Dropout]): Dropout layer. Default: None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """

        # * Apply Rotation onto Q & K
        d_k = query.size(-1)
        query = self.query_rotary_pe(query)
        key = self.key_rotary_pe(key)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -self.mask_eps)
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for rotary multi-headed attention.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (Optional[torch.Tensor]): Mask tensor. Default: None.

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        return super().forward(query, key, value, mask)


class ALiBiMultiHeadAttention(AbstractAttention):
    """Multi-head attention mechanism with ALiBi positional encoding.

    ALiBi is an efficient positional encoding scheme for transformer models that
    relies on linear biases for attention scores. This implementation is useful
    for long-range dependencies in sequence-to-sequence tasks.

    Args:
        n_heads (int): Number of attention heads.
        d_model (int): Model dimension.
        dropout (float): Dropout probability.

    Attributes:
        pe (ALiBiPositionalEncoding): ALiBi positional encoding.

    Methods:
        attention: Computes attention with ALiBi positional encoding.
        forward: Implements forward pass using parent class.
    """

    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1) -> None:
        """Initialize the ALiBi multi-head attention mechanism.

        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Dimensionality of the input and output vectors.
            dropout (float): Dropout probability. Default: 0.1.
        """
        super().__init__(n_heads, d_model, dropout)

        self.pe = ALiBiPositionalEncoding(self.n_heads)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        dropout: nn.Dropout | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ALiBi multi-head attention.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (Optional[torch.Tensor]): Mask tensor. Default: None.
            dropout (Optional[nn.Dropout]): Dropout layer. Default: None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        tgt_len, src_len = scores.size(-2), scores.size(-1)
        alibi_bias = self.pe(max(tgt_len, src_len), scores.device)

        # Slice to match exact dimensions and expand batch
        alibi_bias = alibi_bias[:, :tgt_len, :src_len].unsqueeze(0)
        alibi_bias = alibi_bias.expand(scores.size(0), -1, -1, -1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -self.mask_eps)
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for the ALiBi multi-head attention mechanism.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (Optional[torch.Tensor]): Mask tensor. Default: None.

        Returns:
            torch.Tensor: Output tensor after applying ALiBi attention.
        """
        return super().forward(query, key, value, mask)


class RelativeGlobalAttention(AbstractAttention):
    """
    Implements multi-headed attention with relative positional encoding.
    Reference:
    - "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    by Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, and
    Ruslan Salakhutdinov.
    (https://arxiv.org/abs/1906.08875)
    - "Rethinking Positional Encoding in Language Models"
    by Ziheng Lin, Mingxuan Wang, and Hang Li.
    (https://arxiv.org/abs/2006.15509)
    Args:
        n_heads (int): Number of attention heads.
        d_model (int): Model dimension.
        max_len (int): Maximum sequence length.
        dropout (float): Dropout probability.
        chunk_size (int): Chunk size for memory-efficient computation.

    Attributes:
        Er (nn.Parameter): Relative positional encoding parameters.
        position_cache (torch.Tensor): Cached position indices.

    Methods:
        _get_relative_positions: Get relative positions for sequence.
        _chunked_attention: Compute attention in chunks for memory efficiency.
        attention: Computes attention with relative positional encoding.
        forward: Implements forward pass using parent class.
    """

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        max_len: int = 1024,
        dropout: float = 0.1,
        chunk_size: int = 128,
    ) -> None:
        """
        Initialize relative global attention.

        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Model dimension.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout probability.
            chunk_size (int): Chunk size for memory-efficient computation.
        """

        super().__init__(n_heads, d_model, dropout)
        self.max_len = max_len
        self.chunk_size = chunk_size
        self.Er = nn.Parameter(torch.randn(2 * max_len - 1, self.d_k))
        nn.init.normal_(self.Er, mean=0, std=0.02)

        # Cache for position indices
        self.register_buffer('position_cache', None)
        self._last_seq_lens: tuple[int, int] = (0, 0)

    def _get_relative_positions(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        """
        Get relative positions for sequence.

        Args:
            q_len (int): Query sequence length.
            k_len (int): Key sequence length.
            device (torch.device): Device for tensor creation.

        Returns:
            torch.Tensor: Relative position indices.
        """

        if self._last_seq_lens != (q_len, k_len):
            positions = (
                torch.arange(q_len, device=device)[:, None]
                - torch.arange(k_len, device=device)[None, :]
            )
            positions = positions.clamp(-self.max_len + 1, self.max_len - 1)
            self.position_cache = positions + self.max_len - 1
            self._last_seq_lens = (q_len, k_len)

        return self.position_cache

    def _chunked_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention in chunks for memory efficiency.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            positions (torch.Tensor): Position indices.

        Returns:
            torch.Tensor: Output tensor after applying chunked attention.
        """
        batch_size, n_heads, q_len, d_k = query.size()
        outputs = []

        for chunk_start in range(0, q_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, q_len)
            q_chunk = query[:, :, chunk_start:chunk_end]
            pos_chunk = positions[chunk_start:chunk_end]

            # Get relative embeddings for chunk
            Er_chunk = self.Er[pos_chunk]

            # Compute attention scores for chunk
            QEr = torch.einsum('bhqd,qkd->bhqk', q_chunk, Er_chunk)
            QK_t = torch.matmul(q_chunk, key.transpose(-2, -1))

            chunk_scores = (QK_t + QEr) / math.sqrt(self.d_k)
            chunk_output = torch.matmul(chunk_scores.softmax(dim=-1), value)
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=2)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        dropout: nn.Dropout | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention with relative positional encoding.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (Optional[torch.Tensor]): Mask tensor. Default: None.
            dropout (Optional[nn.Dropout]): Dropout layer. Default: None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """

        q_len, k_len = query.size(2), key.size(2)
        positions = self._get_relative_positions(q_len, k_len, query.device)

        if q_len > self.chunk_size:
            output = self._chunked_attention(query, key, value, positions)
            # Approximate attention weights for visualization
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
            p_attn = scores.softmax(dim=-1)
        else:
            Er = self.Er[positions]
            QEr = torch.einsum('bhqd,qkd->bhqk', query, Er)
            QK_t = torch.matmul(query, key.transpose(-2, -1))

            scores = (QK_t + QEr) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            p_attn = scores.softmax(dim=-1)
            if dropout is not None:
                p_attn = dropout(p_attn)

            output = torch.matmul(p_attn, value)

        return output, p_attn
