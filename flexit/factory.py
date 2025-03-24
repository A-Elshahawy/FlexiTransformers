"""
Transformer Factory

This module implements a factory class for creating different types of transformer models
based on configuration parameters.
"""

from copy import deepcopy

import torch
import torch.nn as nn

from .attention import (
    AbsoluteMultiHeadedAttention,
    ALiBiMultiHeadAttention,
    RelativeGlobalAttention,
    RotaryMultiHeadAttention,
)
from .configs import ModelConfig
from .core import Decoder, Encoder, EncoderDecoder, Generator
from .layers import DecoderLayer, Embeddings, EncoderLayer, PositionwiseFeedForward
from .models_heads import BertHead
from .pos_embeddings import AbsolutePositionalEncoding


class TransformerFactory:
    """
    Factory class for creating transformer models.

    Args:
        config (ModelConfig): Model configuration.

    Methods:
        create_model: Create transformer model based on configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize transformer factory.

        Args:
            config (ModelConfig): Model configuration.
        """

        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validates the model configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        c = self.config
        # Basic validations
        if c.d_model <= 0 or c.d_model % c.n_heads != 0:
            raise ValueError(
                f'Invalid d_model ({c.d_model}): must be positive \
                    and a multiple of n_heads ({c.n_heads}).'
            )
        if c.d_ff <= 0:
            raise ValueError(f'Invalid d_ff: {c.d_ff} (must be positive).')
        if not (0 <= c.dropout <= 1):
            raise ValueError(f'Invalid dropout: {c.dropout} (must be between 0 and 1).')

        # Model type validations
        model_validations = {
            'encoder-decoder': {
                'conditions': [c.src_vocab is not None, c.tgt_vocab is not None],
                'message': 'src_vocab and tgt_vocab are required for encoder-decoder',
            },
            'encoder-only': {
                'conditions': [c.src_vocab is not None],
                'message': 'src_vocab is required for encoder-only',
            },
            'decoder-only': {
                'conditions': [c.tgt_vocab is not None],
                'message': 'tgt_vocab is required for decoder-only',
            },
        }
        if c.model_type not in model_validations:
            raise ValueError(f'Unsupported model_type: {c.model_type}')

        validation = model_validations[c.model_type]
        if not all(validation['conditions']):
            raise ValueError(validation['message'])

    def _init_weights(self, model: nn.Module) -> None:
        """
        Initialize model weights.

        Args:
            model (nn.Module): Model to initialize.
        """
        for p in model.parameters():
            if p.dim() > 1:
                init_fn = getattr(nn.init, f'{self.config.init_method}_')
                init_fn(p)

    def _get_attention_mechanism(self) -> tuple[nn.Module | None, nn.Module | None]:
        """
        Get attention mechanism based on configuration.

        Returns:
            tuple[nn.Module | None, nn.Module | None]: Attention mechanism and positional encoding.
        """
        c = self.config
        mechanisms = {
            'absolute': (
                AbsoluteMultiHeadedAttention(c.n_heads, c.d_model, dropout=c.dropout),
                AbsolutePositionalEncoding(c.d_model, c.dropout),
            ),
            'alibi': (
                ALiBiMultiHeadAttention(c.n_heads, c.d_model, dropout=c.dropout),
                None,
            ),
            'relative': (
                RelativeGlobalAttention(c.n_heads, c.d_model, dropout=c.dropout),
                None,
            ),
            'rotary': (
                RotaryMultiHeadAttention(c.n_heads, c.d_model, dropout=c.dropout),
                None,
            ),
        }
        if c.pe_type not in mechanisms:
            raise ValueError(f'Unknown positional encoding type: {c.pe_type}')
        return mechanisms[c.pe_type]

    def _get_embedding(self, vocab_size: int | None) -> nn.Module:
        """
        Get embedding layer based on vocabulary size.

        Args:
            vocab_size (int | None): Vocabulary size.

        Returns:
            nn.Module: Embedding layer.

        Raises:
            ValueError: If vocabulary size is None.
        """
        # Get positional encoding if available
        if vocab_size is None:
            raise ValueError('Vocabulary size cannot be None')

        _, position = self._get_attention_mechanism()
        if position:
            embed = nn.Sequential(Embeddings(self.config.d_model, vocab_size))
        else:
            embed = nn.Sequential(Embeddings(self.config.d_model, vocab_size), nn.Identity())
        return embed

    def __create_encoder_decoder(self) -> nn.Module:
        """
        Create encoder-decoder model.

        Returns:
            nn.Module: Encoder-decoder model.
        """
        c = self.config
        copy = deepcopy
        attention, _ = self._get_attention_mechanism()

        if c.src_vocab is None:
            raise ValueError('src_vocab must be defined for encoder-decoder models.')
        if c.tgt_vocab is None:
            raise ValueError('tgt_vocab must be defined for encoder-decoder models.')

        src_embed = self._get_embedding(c.src_vocab)
        tgt_embed = self._get_embedding(c.tgt_vocab)

        ff = PositionwiseFeedForward(c.d_model, c.d_ff, c.dropout, activation=c.ff_activation)

        if c.tgt_vocab is None:
            raise ValueError('tgt_vocab must be defined for encoder-decoder models.')

        generator = Generator(c.d_model, c.tgt_vocab)

        # Determine layer counts for encoder and decoder.
        n_enc = c.n_layers[0] if isinstance(c.n_layers, tuple) else c.n_layers
        n_dec = c.n_layers[1] if isinstance(c.n_layers, tuple) else c.n_layers

        encoder = Encoder(
            EncoderLayer(c.d_model, copy(attention), copy(ff), c.pre_norm, c.dropout), n_enc
        )

        decoder = Decoder(
            DecoderLayer(
                c.d_model,
                copy(attention),
                copy(attention),
                copy(ff),
                c.pre_norm,
                c.dropout,
            ),
            n_dec,
        )
        model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)
        self._init_weights(model)
        return model

    def __create_encoder_only(self) -> nn.Module:
        """
        Create encoder-only model.

        Returns:
            nn.Module: Encoder-only model.
        """
        c = self.config
        copy = deepcopy
        attn, _ = self._get_attention_mechanism()
        embed = self._get_embedding(c.src_vocab)

        n_layers = c.n_layers if isinstance(c.n_layers, int) else c.n_layers[0]
        encoder = Encoder(
            EncoderLayer(
                c.d_model,
                copy(attn),
                copy(
                    PositionwiseFeedForward(
                        c.d_model, c.d_ff, c.dropout, activation=c.ff_activation
                    )
                ),
                c.pre_norm,
                c.dropout,
            ),
            n_layers,
        )

        if c.src_vocab is None:
            raise ValueError('src_vocab must be defined for encoder-only models.')
        if c.num_classes is None:
            raise ValueError("""num_classes must be defined for classification tasks
                                    in encoder-only models.""")

        bert_head = BertHead(c.d_model, c.num_classes, c.pre_norm, c.dropout, c.ff_activation)
        model = EncoderOnly(embed, encoder, bert_head, c)
        self._init_weights(model)
        return model

    def __create_decoder_only(self) -> nn.Module:
        """
        Create decoder-only model.

        Returns:
            nn.Module: Decoder-only model.
        """
        c = self.config
        copy = deepcopy
        attn, _ = self._get_attention_mechanism()
        embed = self._get_embedding(c.tgt_vocab)

        n_layers = c.n_layers if isinstance(c.n_layers, int) else c.n_layers[1]
        decoder = Decoder(
            DecoderLayer(
                c.d_model,
                copy(attn),
                None,
                copy(
                    PositionwiseFeedForward(
                        c.d_model, c.d_ff, c.dropout, activation=c.ff_activation
                    )
                ),
                c.pre_norm,
                c.dropout,
            ),
            n_layers,
        )

        if c.tgt_vocab is None:
            raise ValueError('tgt_vocab must be defined for decoder-only models.')

        generator = Generator(c.d_model, c.tgt_vocab)
        model = DecoderOnly(embed, decoder, generator)
        self._init_weights(model)
        return model

    def create_model(self) -> nn.Module:
        """
        Create transformer model based on configuration.

        Returns:
            nn.Module: Created transformer model.
        """
        creators = {
            'encoder-decoder': self.__create_encoder_decoder,
            'encoder-only': self.__create_encoder_only,
            'decoder-only': self.__create_decoder_only,
        }
        return creators[self.config.model_type]()


class EncoderOnly(nn.Module):
    """
    Encoder-only transformer architecture.

    Args:
        embed (nn.Module): Embedding layer.
        encoder (nn.Module): Encoder module.
        head (nn.Module): Classification head.
        config (ModelConfig): Model configuration.

    Methods:
        forward: Forward pass through encoder and classification head.
    """

    def __init__(
        self, embed: nn.Module, encoder: nn.Module, head: nn.Module, config: ModelConfig
    ) -> None:
        """
        Initialize encoder-only model.

        Args:
            embed (nn.Module): Embedding layer.
            encoder (nn.Module): Encoder module.
            head (nn.Module): Classification head.
            config (ModelConfig): Model configuration.
        """
        super().__init__()
        self.embed = embed
        self.encoder = encoder
        self.config = config
        self.generator = head

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and classification head.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        src_embedded = self.embed(src)
        encoder_output = self.encoder(src_embedded, src_mask)
        return self.generator(encoder_output)


class DecoderOnly(nn.Module):
    """
    Decoder-only transformer architecture.

    Args:
        embed (nn.Module): Embedding layer.
        decoder (nn.Module): Decoder module.
        generator (nn.Module): Generator module.

    Methods:
        forward: Forward pass through decoder.


    The implementation allows for both:
    1. A simplified interface (tgt, tgt_mask) for decoder-only use
    2. The full interface (src, tgt, src_mask, tgt_mask) for compatibility
    """

    def __init__(self, embed: nn.Module, decoder: nn.Module, generator: nn.Module) -> None:
        """
        Initialize decoder-only model.

        Args:
            embed (nn.Module): Embedding layer.
            decoder (nn.Module): Decoder module.
            generator (nn.Module): Generator module.
        """
        super().__init__()
        self.embed = embed
        self.decoder = decoder
        self.generator = generator

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for the decoder-only model with flexible parameter handling.


        Args:
            tgt (torch.Tensor): Target sequence input
            tgt_mask (torch.Tensor | None): Target sequence mask
            src (torch.Tensor | None): Source sequence (unused, kept for interface compatibility)
            src_mask (torch.Tensor | None): Source mask (unused, kept for interface compatibility)

        Returns:
            torch.Tensor: Output tensor

        Notes:
            - The src and src_mask parameters are included for interface compatibility
                but are not used in the decoder-only architecture
            - This implementation allows for both simplified and full interface usage:
                model(tgt, tgt_mask) or model(src, tgt, src_mask, tgt_mask)
        """

        tgt_embedded = self.embed(tgt)
        decoder_output = self.decoder(tgt_embedded, None, None, tgt_mask)
        return decoder_output
