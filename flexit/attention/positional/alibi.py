import math
from typing import Literal

import torch
from torch import Tensor

from .base import PositionalEncoding


class ALiBiPE(PositionalEncoding):
    """Attention with Linear Biases (ALiBi)."""

    @property
    def injection_point(self) -> Literal['scores']:
        return 'scores'

    def __init__(self, n_heads: int, max_len: int = 2048) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.max_len = max_len

        slopes = self._get_slopes(n_heads)
        self.register_buffer('slopes', slopes)  # [n_heads]

        # Pre-build bias cache
        self._bias_cache: Tensor | None = None
        self._cache_len: int = 0

    def _get_slopes(self, n_heads: int) -> Tensor:
        n = 2 ** math.floor(math.log2(n_heads))
        m_0 = 2.0 ** (-8.0 / n)
        m = torch.pow(m_0, torch.arange(1, n + 1))

        if n < n_heads:
            m_hat_0 = 2.0 ** (-4.0 / n)
            m_hat = torch.pow(m_hat_0, torch.arange(1, 2 * (n_heads - n) + 1, 2))
            m = torch.cat([m, m_hat])

        return m

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        if (
            seq_len <= self._cache_len
            and self._bias_cache is not None
            and self._bias_cache.device == device
        ):
            return

        self._cache_len = max(seq_len, self.max_len)
        positions = torch.arange(self._cache_len, device=device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq, seq]
        rel_pos = -torch.abs(rel_pos).float()

        # [n_heads, seq, seq]
        self._bias_cache = self.slopes.to(device)[:, None, None] * rel_pos[None, :, :]  # type: ignore

    def apply_to_scores(
        self,
        scores: Tensor,
        q_len: int,
        k_len: int,
        q_offset: int = 0,
        query: Tensor | None = None,
    ) -> Tensor:
        """scores: [batch, n_heads, q_len, k_len]"""
        max_len = max(q_offset + q_len, k_len)
        self._build_cache(max_len, scores.device)

        bias = self._bias_cache[:, q_offset : q_offset + q_len, :k_len]  # type: ignore # [n_heads, q_len, k_len]

        return scores + bias.unsqueeze(0)  # broadcast batch
