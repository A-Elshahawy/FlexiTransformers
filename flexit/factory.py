"""
Transformer Factory

This module implements a factory class for creating different types of transformer models
based on configuration parameters.
"""

from collections.abc import Callable
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

    This factory creates different transformer architectures based on configuration.
    Supported architectures: encoder-decoder, encoder-only, decoder-only
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize transformer factory.

        Args:
            config (ModelConfig): Model configuration containing all parameters.
        """
        self.config = config
        self._validate_config()

        # Register model creation methods
        self._model_creators: dict[str, Callable[[], nn.Module]] = {
            'encoder-decoder': self._create_encoder_decoder,
            'encoder-only': self._create_encoder_only,
            'decoder-only': self._create_decoder_only,
        }

    def _validate_config(self) -> None:
        """Validates the model configuration."""
        validators = [self._validate_basic_params, self._validate_model_type_params]

        for validator in validators:
            validator()

    def _validate_basic_params(self) -> None:
        """Validate basic model parameters."""
        c = self.config

        if c.d_model <= 0 or c.d_model % c.n_heads != 0:
            raise ValueError(
                f'Invalid d_model ({c.d_model}): must be positive '
                f'and a multiple of n_heads ({c.n_heads}).'
            )
        if c.d_ff <= 0:
            raise ValueError(f'Invalid d_ff: {c.d_ff} (must be positive).')
        if not (0 <= c.dropout <= 1):
            raise ValueError(f'Invalid dropout: {c.dropout} (must be between 0 and 1).')

    def _validate_model_type_params(self) -> None:
        """Validate model-type specific parameters."""
        c = self.config

        validations = {
            'encoder-decoder': {
                'conditions': [c.src_vocab is not None, c.tgt_vocab is not None],
                'message': 'src_vocab and tgt_vocab are required for encoder-decoder models',
            },
            'encoder-only': {
                'conditions': [c.src_vocab is not None],
                'message': 'src_vocab is required for encoder-only models',
            },
            'decoder-only': {
                'conditions': [c.tgt_vocab is not None],
                'message': 'tgt_vocab is required for decoder-only models',
            },
        }

        if c.model_type not in validations:
            raise ValueError(f'Unsupported model_type: {c.model_type}')

        validation = validations[c.model_type]
        if not all(validation['conditions']):
            raise ValueError(validation['message'])

    def _init_weights(self, model: nn.Module) -> None:
        """Initialize model weights with proper scaling."""
        method = self.config.init_method or 'xavier_uniform'
        gain = 1.0

        def init_func(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                self._init_linear_layer(module, method, gain)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm | nn.BatchNorm1d | nn.BatchNorm2d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        model.apply(init_func)

    def _init_linear_layer(self, module: nn.Linear, method: str, gain: float) -> None:
        """Initialize linear layer with specified method."""
        init_methods = {
            'xavier_uniform': lambda: nn.init.xavier_uniform_(module.weight, gain=gain),
            'xavier_normal': lambda: nn.init.xavier_normal_(module.weight, gain=gain),
            'kaiming_uniform': lambda: nn.init.kaiming_uniform_(module.weight, nonlinearity='relu'),
            'kaiming_normal': lambda: nn.init.kaiming_normal_(module.weight, nonlinearity='relu'),
            'orthogonal': lambda: nn.init.orthogonal_(module.weight, gain=int(gain)),
            'zero': lambda: nn.init.zeros_(module.weight),
            'one': lambda: nn.init.ones_(module.weight),
        }

        if method in init_methods:
            init_methods[method]()
        else:
            raise ValueError(f'Unknown initialization method: {method}')

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def _get_attention_mechanism(self) -> tuple[nn.Module, nn.Module | None]:
        """
        Get attention mechanism based on configuration.

        Returns:
            Tuple of (attention_module, positional_encoding_module)
        """
        c = self.config
        mechanisms = {
            'absolute': self._create_absolute_attention,
            'alibi': self._create_alibi_attention,
            'relative': self._create_relative_attention,
            'rotary': self._create_rotary_attention,
        }

        if c.pe_type not in mechanisms:
            raise ValueError(f'Unknown positional encoding type: {c.pe_type}')

        return mechanisms[c.pe_type]()

    def _create_absolute_attention(self) -> tuple[nn.Module, nn.Module]:
        attention = AbsoluteMultiHeadedAttention(
            self.config.n_heads, self.config.d_model, dropout=self.config.dropout
        )
        position = AbsolutePositionalEncoding(self.config.d_model, self.config.dropout)
        return attention, position

    def _create_alibi_attention(self) -> tuple[nn.Module, None]:
        attention = ALiBiMultiHeadAttention(
            self.config.n_heads, self.config.d_model, dropout=self.config.dropout
        )
        return attention, None

    def _create_relative_attention(self) -> tuple[nn.Module, None]:
        attention = RelativeGlobalAttention(
            self.config.n_heads, self.config.d_model, dropout=self.config.dropout
        )
        return attention, None

    def _create_rotary_attention(self) -> tuple[nn.Module, None]:
        attention = RotaryMultiHeadAttention(
            self.config.n_heads, self.config.d_model, dropout=self.config.dropout
        )
        return attention, None

    def _get_embedding(self, vocab_size: int) -> nn.Module:
        """
        Get embedding layer with optional positional encoding.

        Args:
            vocab_size: Vocabulary size for embedding layer.

        Returns:
            Embedding module (with positional encoding if applicable).
        """
        if vocab_size is None:
            raise ValueError('Vocabulary size cannot be None')

        embed = Embeddings(self.config.d_model, vocab_size)
        _, position = self._get_attention_mechanism()

        if position:
            return nn.Sequential(embed, position)
        return embed

    def _get_feed_forward(self) -> PositionwiseFeedForward:
        """Create feed-forward network."""
        return PositionwiseFeedForward(
            self.config.d_model,
            self.config.d_ff,
            self.config.dropout,
            activation=self.config.ff_activation,
        )

    def _get_layer_counts(self) -> tuple[int, int]:
        """Get encoder and decoder layer counts."""
        n_layers = self.config.n_layers
        if isinstance(n_layers, int):
            return n_layers, n_layers
        elif isinstance(n_layers, tuple | list) and len(n_layers) == 2:
            return int(n_layers[0]), int(n_layers[1])
        else:
            raise ValueError(f'n_layers must be int or 2-tuple, got {type(n_layers)}: {n_layers}')

    def _create_encoder_decoder(self) -> nn.Module:
        """Create encoder-decoder transformer model."""
        c = self.config
        copy = deepcopy

        if c.src_vocab is None or c.tgt_vocab is None:
            raise ValueError('src_vocab and tgt_vocab must be defined for encoder-decoder models')
        # Get components
        attention, _ = self._get_attention_mechanism()
        if attention is None:
            raise ValueError('Attention mechanism cannot be None')

        src_embed = self._get_embedding(c.src_vocab)
        tgt_embed = self._get_embedding(c.tgt_vocab)
        ff = self._get_feed_forward()
        generator = Generator(c.d_model, c.tgt_vocab)

        # Get layer counts
        n_enc, n_dec = self._get_layer_counts()

        # Create encoder
        encoder = Encoder(
            EncoderLayer(c.d_model, copy(attention), copy(ff), c.pre_norm, c.dropout), n_enc
        )

        # Create decoder
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

        # Build model
        model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)
        self._init_weights(model)
        return model

    def _create_encoder_only(self) -> nn.Module:
        """Create encoder-only transformer model."""
        c = self.config
        copy = deepcopy

        if c.src_vocab is None:
            raise ValueError('src_vocab must be defined for encoder-only models')
        # Get components
        attention, _ = self._get_attention_mechanism()
        embed = self._get_embedding(c.src_vocab)
        ff = self._get_feed_forward()

        if attention is None:
            raise ValueError('Attention mechanism cannot be None')
        if c.src_vocab is None:
            raise ValueError('src_vocab must be defined for encoder-only models')
        if c.num_classes is None:
            raise ValueError('num_classes must be defined for encoder-only models')

        # Create encoder
        n_enc, _ = self._get_layer_counts()
        encoder = Encoder(
            EncoderLayer(c.d_model, copy(attention), copy(ff), c.pre_norm, c.dropout), n_enc
        )

        # Create classification head
        bert_head = BertHead(c.d_model, c.num_classes, c.pre_norm, c.dropout, c.ff_activation)

        # Build model
        model = EncoderOnly(embed, encoder, bert_head, c)
        self._init_weights(model)
        return model

    def _create_decoder_only(self) -> nn.Module:
        """Create decoder-only transformer model."""
        c = self.config
        copy = deepcopy

        if c.tgt_vocab is None:
            raise ValueError('tgt_vocab must be defined for decoder-only models')
        # Get components
        attention, _ = self._get_attention_mechanism()
        embed = self._get_embedding(c.tgt_vocab)
        ff = self._get_feed_forward()
        generator = Generator(c.d_model, c.tgt_vocab)

        if attention is None:
            raise ValueError('Attention mechanism cannot be None')
        if c.tgt_vocab is None:
            raise ValueError('tgt_vocab must be defined for decoder-only models')

        # Create decoder
        _, n_dec = self._get_layer_counts()
        decoder = Decoder(
            DecoderLayer(
                c.d_model,
                copy(attention),
                None,  # No cross-attention in decoder-only
                copy(ff),
                c.pre_norm,
                c.dropout,
            ),
            n_dec,
        )

        # Build model
        model = DecoderOnly(embed, decoder, generator)
        self._init_weights(model)
        return model

    def create_model(self) -> nn.Module:
        """
        Create transformer model based on configuration.

        Returns:
            Created transformer model.

        Raises:
            ValueError: If model_type is not supported.
        """
        if self.config.model_type not in self._model_creators:
            raise ValueError(f'Unsupported model_type: {self.config.model_type}')

        return self._model_creators[self.config.model_type]()

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'TransformerFactory':
        """Create factory instance from configuration."""
        return cls(config)

    @classmethod
    def create(cls, config: ModelConfig) -> nn.Module:
        """Convenience method to create model directly."""
        factory = cls(config)
        return factory.create_model()


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
