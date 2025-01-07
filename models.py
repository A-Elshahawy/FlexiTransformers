import copy

import torch.nn as nn

from .attention import MultiHeadedAttention
from .core import Decoder, Encoder, EncoderDecoder, Generator
from .embeddings import Embeddings, Sinaoidal_Positional_Encoding
from .layers import DecoderLayer, EncoderLayer, Position_wise_Feed_Forward


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    n_layers: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    n_heads: int = 8,
    dropout: float = 0.1,
    positional_encoding: str = 'sinusoidal',
) -> EncoderDecoder:
    """
    Construct a transformer model from hyperparameters.

    Args:
        src_vocab (int): Size of the source vocabulary.
        tgt_vocab (int): Size of the target vocabulary.
        n_layers (int): Number of encoder/decoder layers. Default: 6.
        d_model (int): Model dimension. Default: 512.
        d_ff (int): Feed-forward dimension. Default: 2048.
        n_heads (int): Number of attention heads. Default: 8.
        dropout (float): Dropout probability. Default: 0.1.
        positional_encoding (str): Type of positional encoding.
            Options: "sinusoidal", "relative", "rotary", "alibi". Default: "sinusoidal".

    Returns:
        EncoderDecoder: A transformer model.
    """
    c = copy.deepcopy

    # Create positional encoding
    match positional_encoding:
        case 'sinusoidal':
            position = Sinaoidal_Positional_Encoding(d_model, dropout)
        # case "relative":
        #     position = Relative_Positional_Encoding(d_model)
        # case "rotary":
        #     position = Rotary_Positional_Encoding(d_model)
        # case "alibi":
        # position = Alibi_Positional_Encoding(d_model)
        case _:
            raise ValueError(f'Unknown positional encoding: {positional_encoding}')

    # Create attention and feed-forward layers
    attn = MultiHeadedAttention(n_heads, d_model)
    ff = Position_wise_Feed_Forward(d_model, d_ff, dropout)

    # Create encoder and decoder
    encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_layers)
    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n_layers)

    # Create embeddings
    src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))

    # Create generator
    generator = Generator(d_model, tgt_vocab)

    # Initialize model
    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    # Initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class OptimusTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, **kwargs):
        super(OptimusTransformer, self).__init__()
        self.model = make_model(src_vocab, tgt_vocab, positional_encoding='sinusoidal', **kwargs)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model(src, tgt, src_mask, tgt_mask)


class BumblebeeTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, **kwargs):
        super(BumblebeeTransformer, self).__init__()
        self.model = make_model(src_vocab, tgt_vocab, positional_encoding='alibi', **kwargs)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model(src, tgt, src_mask, tgt_mask)


class MegatronTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, **kwargs):
        super(MegatronTransformer, self).__init__()
        self.model = make_model(src_vocab, tgt_vocab, positional_encoding='relative', **kwargs)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model(src, tgt, src_mask, tgt_mask)


class MirageTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, **kwargs):
        super(MirageTransformer, self).__init__()
        self.model = make_model(src_vocab, tgt_vocab, positional_encoding='rotary', **kwargs)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model(src, tgt, src_mask, tgt_mask)
