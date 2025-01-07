import math

import torch
import torch.nn as nn


def attention(
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
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Implements multi-headed attention.

    Args:
        n_heads (int): Number of attention heads.
        d_model (int): Model dimension.
        dropout (float): Dropout probability.
    """

    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1) -> None:
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

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
            torch.Tensor: Output tensor.
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Linear projections
        query, key, value = [
            lin(x).view(nbatches, -1, self.n_heads, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value), strict=False)
        ]

        # Apply attention
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concatenate and apply final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_heads * self.d_k)
        return self.linears[-1](x)
