import math

from torch import Tensor, nn

from ..attention.positional import PositionalEncoding


class Embeddings(nn.Module):
    """Token embeddings with optional scaling."""

    def __init__(self, vocab_size: int, d_model: int, scale: bool = True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.scale = math.sqrt(d_model) if scale else 1.0

    def forward(self, x: Tensor) -> Tensor:
        return self.embed(x) * self.scale


class EmbeddingWithPE(nn.Module):
    """Embeddings + positional encoding (for embedding-level PE)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        pe: PositionalEncoding | None = None,
        scale: bool = True,
    ):
        super().__init__()
        self.embed = Embeddings(vocab_size, d_model, scale)
        self.pe = pe

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        if self.pe is not None and self.pe.injection_point == 'embedding':
            x = self.pe.apply_to_embedding(x)
        return x
