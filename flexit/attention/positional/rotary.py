from typing import Literal

import torch
from torch import Tensor

from .base import PositionalEncoding


class RotaryPE(PositionalEncoding):
    """Rotary Position Embedding (RoPE)."""

    @property
    def injection_point(self) -> Literal['embedding', 'qk', 'scores']:
        return 'qk'

    def __init__(
        self,
        dim: int,
        max_len: int = 2048,
        base: int = 10000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Cache
        self._cos_cache: Tensor | None = None
        self._sin_cache: Tensor | None = None
        self._cache_len: int = 0

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        if (
            seq_len <= self._cache_len
            and self._cos_cache is not None
            and self._cos_cache.device == device
        ):
            return

        self._cache_len = max(seq_len, self.max_len)
        # Always build in float32 for numerical precision, cast to input dtype at apply time
        t = torch.arange(self._cache_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))  # type: ignore
        emb = torch.cat([freqs, freqs], dim=-1)

        self._cos_cache = emb.cos()[None, None, :, :]  # [1, 1, seq, dim]
        self._sin_cache = emb.sin()[None, None, :, :]

    def _rotate_half(self, x: Tensor) -> Tensor:
        x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_to_qk(
        self,
        q: Tensor,
        k: Tensor,
        q_offset: int = 0,
        k_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """
        q, k: [batch, n_heads, seq_len, head_dim]
        """
        q_len, k_len = q.size(2), k.size(2)
        max_len = max(q_offset + q_len, k_offset + k_len)

        self._build_cache(max_len, q.device)

        # Get relevant slices and cast to input dtype for FP16/BF16 compatibility
        cos_q = self._cos_cache[:, :, q_offset : q_offset + q_len, : self.dim].to(q.dtype)  # type: ignore
        sin_q = self._sin_cache[:, :, q_offset : q_offset + q_len, : self.dim].to(q.dtype)  # type: ignore
        cos_k = self._cos_cache[:, :, k_offset : k_offset + k_len, : self.dim].to(q.dtype)  # type: ignore
        sin_k = self._sin_cache[:, :, k_offset : k_offset + k_len, : self.dim].to(q.dtype)  # type: ignore

        # Apply rotation (only to first `dim` dimensions)
        q_rope = q[..., : self.dim]
        k_rope = k[..., : self.dim]

        q_out = q_rope * cos_q + self._rotate_half(q_rope) * sin_q
        k_out = k_rope * cos_k + self._rotate_half(k_rope) * sin_k

        # Concat with non-rotated dims if any
        if self.dim < q.size(-1):
            q_out = torch.cat([q_out, q[..., self.dim :]], dim=-1)
            k_out = torch.cat([k_out, k[..., self.dim :]], dim=-1)

        return q_out, k_out
