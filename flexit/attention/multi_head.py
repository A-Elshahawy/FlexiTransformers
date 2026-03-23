import math

import torch
from torch import Tensor, nn

from .positional import PositionalEncoding


class MultiHeadAttention(nn.Module):
    """
    Unified multi-head attention with pluggable positional encoding.

    Supports:
    - Self-attention (q=k=v)
    - Cross-attention (q from decoder, k,v from encoder)
    - All PE types via plugin
    - Optional KV cache for inference
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        pe: PositionalEncoding | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.pe = pe

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
        kv_cache: dict | None = None,
        position_offset: int = 0,
    ) -> Tensor:
        """
        Args:
            query: [batch, q_len, d_model]
            key: [batch, k_len, d_model]
            value: [batch, v_len, d_model]
            mask: [batch, 1, q_len, k_len] or broadcastable
            kv_cache: Optional cache dict for inference
            position_offset: Position offset for cached generation

        Returns:
            output: [batch, q_len, d_model]
        """
        batch_size, q_len, _ = query.shape

        # Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape NEW tokens to heads: [batch, n_heads, seq_len, head_dim]
        k_new_len = k.size(1)
        q = q.view(batch_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, k_new_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, k_new_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply QK-level PE (RoPE) to NEW Q and K only, before cache concat
        if self.pe is not None and self.pe.injection_point == 'qk':
            k_offset = 0 if kv_cache is None else position_offset
            q, k = self.pe.apply_to_qk(q, k, q_offset=position_offset, k_offset=k_offset)

        # Handle KV cache (keys already rotated; concat on dim=2 = seq dim)
        if kv_cache is not None:
            if 'k' in kv_cache:
                k = torch.cat([kv_cache['k'], k], dim=2)
                v = torch.cat([kv_cache['v'], v], dim=2)
            kv_cache['k'] = k
            kv_cache['v'] = v

        k_len = k.size(2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply score-level PE (ALiBi, Relative)
        if self.pe is not None and self.pe.injection_point == 'scores':
            scores = self.pe.apply_to_scores(
                scores, q_len, k_len, q_offset=position_offset, query=q
            )

        # Apply mask (support both 3-D [B,1,S] and 4-D [B,1,q,k] masks)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [B, 1, q, k]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn = self.dropout(scores.softmax(dim=-1))

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape back: [batch, q_len, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)

        return self.out_proj(out)
