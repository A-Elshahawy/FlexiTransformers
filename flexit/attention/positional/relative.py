"""Relative positional encodings (score-level)."""

from typing import Literal

import torch
from torch import Tensor, nn

from .base import PositionalEncoding


class RelativePE(PositionalEncoding):
    """
    Shaw et al. (2018) relative position encoding.

    Adds query-dependent relative position biases to attention scores.
    For each (query position, key position) pair the bias is:
        q_i · r_{clip(j - i)}
    where r is a learned embedding of size ``head_dim``.

    Reference: https://arxiv.org/abs/1803.02155
    """

    @property
    def injection_point(self) -> Literal['scores']:
        return 'scores'

    def __init__(self, head_dim: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        # One embedding per relative distance in [-(max_seq_len-1), max_seq_len-1]
        self.rel_emb = nn.Embedding(2 * max_seq_len - 1, head_dim)

    def _rel_idx(self, q_len: int, k_len: int, device: torch.device) -> Tensor:
        """Return relative position indices [q_len, k_len] in embedding range."""
        q_pos = torch.arange(q_len, device=device).unsqueeze(1)  # [q, 1]
        k_pos = torch.arange(k_len, device=device).unsqueeze(0)  # [1, k]
        rel = (k_pos - q_pos).clamp(-(self.max_seq_len - 1), self.max_seq_len - 1)
        return rel + (self.max_seq_len - 1)  # shift to [0, 2*max-2]

    def apply_to_scores(
        self,
        scores: Tensor,
        q_len: int = 0,
        k_len: int = 0,
        q_offset: int = 0,
        query: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            scores: [batch, n_heads, q_len, k_len]
            query:  [batch, n_heads, q_len, head_dim]
        """
        if query is None:
            return scores

        B, H, q_len_s, k_len_s = scores.shape
        q_len = q_len or q_len_s
        k_len = k_len or k_len_s

        idx = self._rel_idx(q_len, k_len, scores.device)  # [q, k]
        rel = self.rel_emb(idx)  # [q, k, head_dim]

        # bias[b,h,q,k] = query[b,h,q,:] · rel[q,k,:]
        bias = torch.einsum('bhqd,qkd->bhqk', query, rel)
        return scores + bias


class RelativePEWithBias(PositionalEncoding):
    """
    T5-style relative position bias.

    A learned scalar bias for each relative position bucket is added directly
    to the attention scores, independent of the query content.
    """

    @property
    def injection_point(self) -> Literal['scores']:
        return 'scores'

    def __init__(self, head_dim: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        # Scalar bias per relative position (head_dim unused but kept for API symmetry)
        self.rel_bias = nn.Embedding(2 * max_seq_len - 1, 1)

    def _rel_idx(self, q_len: int, k_len: int, device: torch.device) -> Tensor:
        q_pos = torch.arange(q_len, device=device).unsqueeze(1)
        k_pos = torch.arange(k_len, device=device).unsqueeze(0)
        rel = (k_pos - q_pos).clamp(-(self.max_seq_len - 1), self.max_seq_len - 1)
        return rel + (self.max_seq_len - 1)

    def apply_to_scores(
        self,
        scores: Tensor,
        q_len: int = 0,
        k_len: int = 0,
        q_offset: int = 0,
        query: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            scores: [batch, n_heads, q_len, k_len]
        """
        B, H, q_len_s, k_len_s = scores.shape
        q_len = q_len or q_len_s
        k_len = k_len or k_len_s

        idx = self._rel_idx(q_len, k_len, scores.device)  # [q, k]
        bias = self.rel_bias(idx).squeeze(-1)  # [q, k]
        return scores + bias.unsqueeze(0).unsqueeze(0)  # broadcast [1,1,q,k]
