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

# Constants for positional encoding types
POSITIONAL_ENCODING_TYPES = {
    'sinusoidal': SinusoidalPositionalEncoding,
    'relative': RelativePositionalEncoding,
    'rotary': RotaryPositionalEncoding,
    'alibi': ALiBiPositionalEncoding,
}


class PositionalEncoding(Enum):
    """Enumeration for different types of positional encodings."""

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
) -> EncoderDecoder:
    """
    Construct a transformer model from hyperparameters.

    Args:
        src_vocab (int): Size of the source vocabulary.
        tgt_vocab (int): Size of the target vocabulary.
        n_layers (int): Number of layers in the encoder and decoder.
        d_model (int): Dimensionality of the model.
        d_ff (int): Dimensionality of the feed-forward network.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        positional_encoding (str): Type of positional encoding to use.

    Returns:
        EncoderDecoder: The constructed transformer model.

    Raises:
        ValueError: If the specified positional encoding is not supported.
    """
    c = deepcopy

    if positional_encoding not in POSITIONAL_ENCODING_TYPES:
        raise ValueError(
            f'Unknown positional encoding: {positional_encoding}. '
            f'Supported types are: {list(POSITIONAL_ENCODING_TYPES.keys())}'
        )

    position = POSITIONAL_ENCODING_TYPES[positional_encoding](d_model, dropout)
    attn = MultiHeadedAttention(n_heads, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_layers)
    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n_layers)

    src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
    generator = Generator(d_model, tgt_vocab)

    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    # Initialize parameters with Xavier uniform initialization
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
        """
        Initialize the BaseTransformer.

        Args:
            src_vocab (int): Size of the source vocabulary.
            tgt_vocab (int): Size of the target vocabulary.
            n_layers (int): Number of layers in the encoder and decoder.
            d_model (int): Dimensionality of the model.
            d_ff (int): Dimensionality of the feed-forward network.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            pos_encoding (PositionalEncoding): Type of positional encoding to use.
        """
        super().__init__()

        pos_encoding_map = {
            PositionalEncoding.SINUSOIDAL: 'sinusoidal',
            PositionalEncoding.RELATIVE: 'relative',
            PositionalEncoding.ROTARY: 'rotary',
            PositionalEncoding.ALIBI: 'alibi',
        }

        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
        self._n_layers = n_layers
        self._d_model = d_model
        self._d_ff = d_ff
        self._n_heads = n_heads
        self._dropout = dropout
        self._pos_encoding = pos_encoding

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

    @property
    def src_vocab(self) -> int:
        """Get the size of the source vocabulary."""
        return self._src_vocab

    @src_vocab.setter
    def src_vocab(self, value: int) -> None:
        """Set the size of the source vocabulary."""
        if value <= 0:
            raise ValueError('Source vocabulary size must be a positive integer.')
        self._src_vocab = value
        # Reinitialize the model with the new vocabulary size
        self._reinitialize_model()

    @property
    def tgt_vocab(self) -> int:
        """Get the size of the target vocabulary."""
        return self._tgt_vocab

    @tgt_vocab.setter
    def tgt_vocab(self, value: int) -> None:
        """Set the size of the target vocabulary."""
        if value <= 0:
            raise ValueError('Target vocabulary size must be a positive integer.')
        self._tgt_vocab = value
        # Reinitialize the model with the new vocabulary size
        self._reinitialize_model()

    @property
    def n_layers(self) -> int:
        """Get the number of layers in the encoder and decoder."""
        return self._n_layers

    @n_layers.setter
    def n_layers(self, value: int) -> None:
        """Set the number of layers in the encoder and decoder."""
        if value <= 0:
            raise ValueError('Number of layers must be a positive integer.')
        self._n_layers = value
        # Reinitialize the model with the new number of layers
        self._reinitialize_model()

    @property
    def d_model(self) -> int:
        """Get the dimensionality of the model."""
        return self._d_model

    @d_model.setter
    def d_model(self, value: int) -> None:
        """Set the dimensionality of the model."""
        if value <= 0:
            raise ValueError('Model dimensionality must be a positive integer.')
        self._d_model = value
        # Reinitialize the model with the new dimensionality
        self._reinitialize_model()

    @property
    def d_ff(self) -> int:
        """Get the dimensionality of the feed-forward network."""
        return self._d_ff

    @d_ff.setter
    def d_ff(self, value: int) -> None:
        """Set the dimensionality of the feed-forward network."""
        if value <= 0:
            raise ValueError('Feed-forward dimensionality must be a positive integer.')
        self._d_ff = value
        # Reinitialize the model with the new feed-forward dimensionality
        self._reinitialize_model()

    @property
    def n_heads(self) -> int:
        """Get the number of attention heads."""
        return self._n_heads

    @n_heads.setter
    def n_heads(self, value: int) -> None:
        """Set the number of attention heads."""
        if value <= 0:
            raise ValueError('Number of attention heads must be a positive integer.')
        self._n_heads = value
        # Reinitialize the model with the new number of attention heads
        self._reinitialize_model()

    @property
    def dropout(self) -> float:
        """Get the dropout rate."""
        return self._dropout

    @dropout.setter
    def dropout(self, value: float) -> None:
        """Set the dropout rate."""
        if not 0 <= value < 1:
            raise ValueError('Dropout rate must be in the range [0, 1).')
        self._dropout = value
        # Reinitialize the model with the new dropout rate
        self._reinitialize_model()

    @property
    def pos_encoding(self) -> PositionalEncoding:
        """Get the type of positional encoding."""
        return self._pos_encoding

    @pos_encoding.setter
    def pos_encoding(self, value: PositionalEncoding) -> None:
        """Set the type of positional encoding."""
        if not isinstance(value, PositionalEncoding):
            raise ValueError('Positional encoding must be an instance of PositionalEncoding.')
        self._pos_encoding = value
        # Reinitialize the model with the new positional encoding
        self._reinitialize_model()

    def _reinitialize_model(self) -> None:
        """Reinitialize the model with updated hyperparameters."""
        pos_encoding_map = {
            PositionalEncoding.SINUSOIDAL: 'sinusoidal',
            PositionalEncoding.RELATIVE: 'relative',
            PositionalEncoding.ROTARY: 'rotary',
            PositionalEncoding.ALIBI: 'alibi',
        }
        self.model = make_model(
            self._src_vocab,
            self._tgt_vocab,
            n_layers=self._n_layers,
            d_model=self._d_model,
            d_ff=self._d_ff,
            n_heads=self._n_heads,
            dropout=self._dropout,
            positional_encoding=pos_encoding_map[self._pos_encoding],
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer model.

        Args:
            src (torch.Tensor): Source input tensor.
            tgt (torch.Tensor): Target input tensor.
            src_mask (torch.Tensor): Source mask tensor.
            tgt_mask (torch.Tensor): Target mask tensor.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Output tensor from the transformer model.
        """
        return self.model(src, tgt, src_mask, tgt_mask, **kwargs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode the source input.

        Args:
            src (torch.Tensor): Source input tensor.
            src_mask (torch.Tensor): Source mask tensor.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Encoded source tensor.
        """
        return self.model.encode(src, src_mask, **kwargs)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Decode the target input using the encoded source.

        Args:
            memory (torch.Tensor): Encoded source tensor.
            src_mask (torch.Tensor): Source mask tensor.
            tgt (torch.Tensor): Target input tensor.
            tgt_mask (torch.Tensor): Target mask tensor.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Decoded target tensor.
        """
        return self.model.decode(memory, src_mask, tgt, tgt_mask, **kwargs)

    def generator(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate output from the transformer model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Generated output tensor.
        """
        return self.model.generator(x)


class OptimusTransformer(BaseTransformer):
    """Standard transformer with sinusoidal positional encoding."""

    pass


class BumblebeeTransformer(BaseTransformer):
    """Transformer variant with ALiBi positional encoding."""

    def __init__(self, src_vocab: int, tgt_vocab: int, **kwargs) -> None:
        """
        Initialize the BumblebeeTransformer.

        Args:
            src_vocab (int): Size of the source vocabulary.
            tgt_vocab (int): Size of the target vocabulary.
            **kwargs: Additional arguments.
        """
        super().__init__(src_vocab, tgt_vocab, pos_encoding=PositionalEncoding.ALIBI, **kwargs)


class MegatronTransformer(BaseTransformer):
    """Transformer variant with relative positional encoding."""

    def __init__(self, src_vocab: int, tgt_vocab: int, **kwargs) -> None:
        """
        Initialize the MegatronTransformer.

        Args:
            src_vocab (int): Size of the source vocabulary.
            tgt_vocab (int): Size of the target vocabulary.
            **kwargs: Additional arguments.
        """
        super().__init__(src_vocab, tgt_vocab, pos_encoding=PositionalEncoding.RELATIVE, **kwargs)


class MirageTransformer(BaseTransformer):
    """Transformer variant with rotary positional encoding."""

    def __init__(self, src_vocab: int, tgt_vocab: int, **kwargs) -> None:
        """
        Initialize the MirageTransformer.

        Args:
            src_vocab (int): Size of the source vocabulary.
            tgt_vocab (int): Size of the target vocabulary.
            **kwargs: Additional arguments.
        """
        super().__init__(src_vocab, tgt_vocab, pos_encoding=PositionalEncoding.ROTARY, **kwargs)
