"""
Core Transformer Components

This module implements the core components of transformer models,
including encoder, decoder, generator, and the encoder-decoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Self

from .layers import LayerNorm
from .utils import clone


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.

    Args:
        encoder (nn.Module): Encoder module.
        decoder (nn.Module): Decoder module.
        src_embed (nn.Module): Source embedding module.
        tgt_embed (nn.Module): Target embedding module.
        generator (nn.Module): Generator module.
        device (str): Device for computation ('cpu' or 'gpu'). Default is 'cpu'.

    Attributes:
        encoder (nn.Module): Encoder module.
        decoder (nn.Module): Decoder module.
        src_embed (nn.Module): Source embedding module.
        tgt_embed (nn.Module): Target embedding module.
        generator (nn.Module): Generator module.
        device (str): Device for computation.

    Methods:
        forward: Forward pass for the encoder-decoder model.
        encode: Encode the source sequence.
        decode: Decode the target sequence using encoder memory.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        generator: nn.Module,
        device: str = 'cpu',
    ) -> None:
        """
        Initialize encoder-decoder model.

        Args:
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
            src_embed (nn.Module): Source embedding module.
            tgt_embed (nn.Module): Target embedding module.
            generator (nn.Module): Generator module.
            device (str): Device for computation.
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.device = device

    def to(self, device: str) -> Self:
        """
        Move model to specified device.

        Args:
            device (str): Target device.

        Returns:
            self: Module instance.
        """
        super().to(device)
        self.device = device
        return self

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the encoder-decoder model.

        Args:
            src (torch.Tensor): Source sequence.
            tgt (torch.Tensor): Target sequence.
            src_mask (torch.Tensor): Source mask.
            tgt_mask (torch.Tensor): Target mask.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode the source sequence.

        Args:
            src (torch.Tensor): Source sequence.
            src_mask (torch.Tensor): Source mask.

        Returns:
            torch.Tensor: Encoder output.
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode the target sequence using encoder memory.

        Args:
            memory (torch.Tensor): Encoder output.
            src_mask (torch.Tensor): Source mask.
            tgt (torch.Tensor): Target sequence.
            tgt_mask (torch.Tensor): Target mask.

        Returns:
            torch.Tensor: Decoder output.
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    Implements the generator (linear + softmax).

    Args:
        d_model (int): Model dimension.
        vocab (int): Vocabulary size.
        pre_norm (bool): Use pre-normalization. Default is True.

    Attributes:
        proj (nn.Linear): Linear projection layer.
        norm (LayerNorm): Layer normalization.
        pre_norm (bool): Use pre-normalization.

    Methods:
        forward: Forward pass for generator.
    """

    def __init__(self, d_model: int, vocab: int) -> None:
        """
        Initialize generator.

        Args:
            d_model (int): Model dimension.
            vocab (int): Vocabulary size.
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for generator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying linear projection and softmax.
        """
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers.

    Args:
        layer (nn.Module): Encoder layer module.
        n_layers (int): Number of layers.

    Attributes:
        layers (nn.ModuleList): List of encoder layers.
        norm (LayerNorm): Layer normalization.
        pre_norm (bool): Use pre-normalization.

    Methods:
        forward: Forward pass through encoder layers.
    """

    def __init__(self, layer: nn.Module, n_layers: int) -> None:
        """
        Initialize encoder.

        Args:
            layer (nn.Module): Encoder layer module.
            n_layers (int): Number of layers.
        """
        super(Encoder, self).__init__()
        self.layers = clone(layer, n_layers)
        self.norm = LayerNorm(layer.size)
        self.pre_norm = layer.pre_norm

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Pass the input (and mask) through each layer in turn.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Encoder output.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x if self.pre_norm else self.norm(x)


class Decoder(nn.Module):
    """
    Core decoder is a stack of N layers.

    Args:
        layer (nn.Module): Decoder layer module.
        n_layers (int): Number of layers.

    Attributes:
        layers (nn.ModuleList): List of decoder layers.
        norm (LayerNorm): Layer normalization.
        pre_norm (bool): Use pre-normalization.

    Methods:
        forward: Forward pass through decoder layers.
        forward_cross_attention: Forward pass with cross-attention.
        forward_self_attention: Forward pass with self-attention.
    """

    def __init__(self, layer: nn.Module, n_layers: int) -> None:
        """
        Initialize decoder.

        Args:
            layer (nn.Module): Decoder layer module.
            n_layers (int): Number of layers.
        """
        super().__init__()
        self.layers = clone(layer, n_layers)
        self.norm = LayerNorm(layer.size)
        self.pre_norm = layer.pre_norm

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None,
        src_mask: torch.Tensor | None,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Router method to appropriate forward implementation.

        Args:
            x (torch.Tensor): Input tensor.
            memory (torch.Tensor | None): Encoder memory.
            src_mask (torch.Tensor | None): Source mask.
            tgt_mask (torch.Tensor): Target mask.

        Returns:
            torch.Tensor: Decoder output.
        """

        if memory is not None:
            return self.forward_cross_attention(x, memory, src_mask, tgt_mask)
        return self.forward_self_attention(x, tgt_mask)

    def forward_cross_attention(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for encoder-decoder with cross-attention.

        Args:
            x (torch.Tensor): Input tensor.
            memory (torch.Tensor): Encoder memory.
            src_mask (torch.Tensor): Source mask.
            tgt_mask (torch.Tensor): Target mask.

        Returns:
            torch.Tensor: Decoder output.
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x if self.pre_norm else self.norm(x)

    def forward_self_attention(
        self,
        x: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for decoder-only with self-attention.

        Args:
            x (torch.Tensor): Input tensor.
            tgt_mask (torch.Tensor): Target mask.

        Returns:
            torch.Tensor: Decoder output.
        """
        for layer in self.layers:
            x = layer(x, None, None, tgt_mask)
        return x if self.pre_norm else self.norm(x)
