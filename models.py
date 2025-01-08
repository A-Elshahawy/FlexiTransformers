from copy import deepcopy
from enum import Enum, auto

import torch
import torch.nn as nn

from attention import MultiHeadedAttention
from core import Decoder, Encoder, EncoderDecoder, Generator
from layers import DecoderLayer, Embeddings, EncoderLayer, PositionwiseFeedForward
from pos_embeddings import (
    ALiBiPositionalEncoding,
    RelativePositionalEncoding,
    RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
)


class PositionalEncoding(Enum):
    SINUSOIDAL = auto()
    RELATIVE = auto()
    ROTARY = auto()
    ALIBI = auto()


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    n_layers: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    n_heads: int = 8,
    dropout: float = 0.1,
    positional_encoding: str = 'sinusoidal',
) -> 'EncoderDecoder':
    """Construct a transformer model from hyperparameters."""
    c = deepcopy

    pos_encodings = {
        'sinusoidal': SinusoidalPositionalEncoding,
        'relative': RelativePositionalEncoding,
        'rotary': RotaryPositionalEncoding,
        'alibi': ALiBiPositionalEncoding,
    }

    if positional_encoding not in pos_encodings:
        raise ValueError(f'Unknown positional encoding: {positional_encoding}')

    position = pos_encodings[positional_encoding](d_model, dropout)
    attn = MultiHeadedAttention(n_heads, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_layers)
    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n_layers)

    src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
    generator = Generator(d_model, tgt_vocab)

    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class BaseTransformer(nn.Module):
    """Base transformer model with configurable positional encoding."""

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        n_layers: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        n_heads: int = 8,
        dropout: float = 0.1,
        pos_encoding: PositionalEncoding = PositionalEncoding.SINUSOIDAL,
    ) -> None:
        super().__init__()
        pos_encoding_map = {
            PositionalEncoding.SINUSOIDAL: 'sinusoidal',
            PositionalEncoding.RELATIVE: 'relative',
            PositionalEncoding.ROTARY: 'rotary',
            PositionalEncoding.ALIBI: 'alibi',
        }

        self.model = make_model(
            src_vocab,
            tgt_vocab,
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
            positional_encoding=pos_encoding_map[pos_encoding],
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(src, tgt, src_mask, tgt_mask, **kwargs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.encode(src, src_mask, **kwargs)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.model.decode(memory, src_mask, tgt, tgt_mask, **kwargs)

    def generator(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.generator(x)


class OptimusTransformer(BaseTransformer):
    """Standard transformer with sinusoidal positional encoding."""

    pass


class BumblebeeTransformer(BaseTransformer):
    """Transformer variant with ALiBi positional encoding."""

    def __init__(self, src_vocab: int, tgt_vocab: int, **kwargs) -> None:
        super().__init__(src_vocab, tgt_vocab, pos_encoding=PositionalEncoding.ALIBI, **kwargs)


class MegatronTransformer(BaseTransformer):
    """Transformer variant with relative positional encoding."""

    def __init__(self, src_vocab: int, tgt_vocab: int, **kwargs) -> None:
        super().__init__(src_vocab, tgt_vocab, pos_encoding=PositionalEncoding.RELATIVE, **kwargs)


class MirageTransformer(BaseTransformer):
    """Transformer variant with rotary positional encoding."""

    def __init__(self, src_vocab: int, tgt_vocab: int, **kwargs) -> None:
        super().__init__(src_vocab, tgt_vocab, pos_encoding=PositionalEncoding.ROTARY, **kwargs)
