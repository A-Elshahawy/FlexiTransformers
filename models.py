from copy import deepcopy
from typing import Literal

import torch
import torch.nn as nn

from attention import (
    AbsoluteMultiHeadedAttention,
    ALiBiMultiHeadAttention,
    RelativeGlobalAttention,
    RotaryMultiHeadAttention,
)
from core import Decoder, Encoder, EncoderDecoder, Generator
from layers import DecoderLayer, Embeddings, EncoderLayer, PositionwiseFeedForward
from pos_embeddings import AbsolutePositionalEncoding


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    n_layers: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    n_heads: int = 8,
    dropout: float = 0.1,
    device: str = 'cpu',
    positional_encoding: str = 'absolute',
) -> 'EncoderDecoder':
    """Construct a transformer model from hyperparameters."""
    c = deepcopy

    src_embed = nn.Sequential(Embeddings(d_model, src_vocab))
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab))

    pos_encoding_config = {
        'absolute': (
            AbsoluteMultiHeadedAttention(n_heads, d_model),
            AbsolutePositionalEncoding(d_model, dropout),
        ),
        'alibi': (ALiBiMultiHeadAttention(n_heads, d_model, dropout), None),
        'relative': (RelativeGlobalAttention(n_heads, d_model, dropout=dropout), None),
        'rotary': (RotaryMultiHeadAttention(n_heads, d_model, dropout=dropout), None),
    }

    if positional_encoding not in pos_encoding_config:
        raise ValueError(f'Unknown positional encoding: {positional_encoding}')

    attn, position = pos_encoding_config[positional_encoding]

    if position:
        src_embed.append(c(position))
        tgt_embed.append(c(position))

    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_layers)
    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n_layers)
    generator = Generator(d_model, tgt_vocab)
    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator, device)

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
        device: str = 'cpu',
        n_layers: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        n_heads: int = 8,
        dropout: float = 0.1,
        pos_encoding: Literal['absolute', 'relative', 'rotary', 'alibi'] = 'absolute',
    ) -> None:
        super().__init__()

        self.model = make_model(
            src_vocab,
            tgt_vocab,
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
            device=device,
            positional_encoding=pos_encoding,
        )

    def to(self, device):
        super().to(device)
        self.device = device
        self.model.to(device)
        return self

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
        super().__init__(src_vocab, tgt_vocab, pos_encoding='alibi', **kwargs)


class MegatronTransformer(BaseTransformer):
    """Transformer variant with relative positional encoding."""

    def __init__(self, src_vocab: int, tgt_vocab: int, **kwargs) -> None:
        super().__init__(src_vocab, tgt_vocab, pos_encoding='relative', **kwargs)


class MirageTransformer(BaseTransformer):
    """Transformer variant with rotary positional encoding."""

    def __init__(self, src_vocab: int, tgt_vocab: int, **kwargs) -> None:
        super().__init__(src_vocab, tgt_vocab, pos_encoding='rotary', **kwargs)
