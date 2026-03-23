from abc import ABC, abstractmethod
from typing import Literal

from torch import Tensor, nn


class PositionalEncoding(ABC, nn.Module):
    """
    Abstract base class for positional encodings.

    Injection points:
    - "embedding": Applied to embeddings before attention (Sinusoidal)
    - "qk": Applied to Q, K after projection (RoPE)
    - "scores": Applied to attention scores (ALiBi, Relative)
    """

    @property
    @abstractmethod
    def injection_point(self) -> Literal['embedding', 'qk', 'scores']:
        """Where this positional encoding is applied."""
        ...

    def apply_to_embedding(self, x: Tensor) -> Tensor:
        """Apply to embeddings. Override for embedding-level PE."""
        return x

    def apply_to_qk(
        self,
        q: Tensor,
        k: Tensor,
        q_offset: int = 0,
        k_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Apply to Q and K. Override for QK-level PE."""
        return q, k

    def apply_to_scores(
        self, scores: Tensor, q_len: int, k_len: int, q_offset: int = 0, query: Tensor | None = None
    ) -> Tensor:
        """Apply to attention scores. Override for score-level PE."""
        return scores
