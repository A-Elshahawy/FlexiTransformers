"""
Transformer Layers

This module implements fundamental layers used in transformer models,
including layer normalization, sublayer connections, feed-forward networks,
and encoder/decoder layers.
"""

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import clone


class LayerNorm(nn.Module):
    """
    Construct a Layer Normalization module.

    Args:
        features (int): Number of features in the input.
        eps (float): A small value to avoid division by zero. Default: 1e-6.
        bias (bool): If True, use bias in normalization. Default: True.

    Attributes:
        a_2 (nn.Parameter): Scaling parameter.
        b_2 (nn.Parameter): Bias parameter.
        eps (float): Small value for numerical stability.

    Methods:
        forward: Forward pass through layer normalization.
    """

    def __init__(self, features: int, eps: float = 1e-6, bias: bool = True) -> None:
        """
        Initialize layer normalization.

        Args:
            features (int): Number of features.
            eps (float): Small value for numerical stability.
            bias (bool): Use bias parameter.
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        norm = self.a_2 * (x - mean) / (std + self.eps)
        return norm + self.b_2 if self.b_2 is not None else norm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer normalization.

    Args:
        size (int): Size of the input features.
        pre_norm (bool): Use pre-normalization.
        dropout (float): Dropout probability.

    Attributes:
        norm (NormalizationBlock): Normalization block.
        pre_norm (bool): Use pre-normalization.

    Methods:
        forward: Forward pass with residual connection.
    """

    def __init__(self, size: int, pre_norm: bool, dropout: float) -> None:
        """
        Initialize sublayer connection.

        Args:
            size (int): Size of input features.
            pre_norm (bool): Use pre-normalization.
            dropout (float): Dropout probability.
        """
        super(SublayerConnection, self).__init__()
        self.norm = NormalizationBlock(size, dropout)
        self.pre_norm = pre_norm

    def forward(
        self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply residual connection to any sublayer with the same size.

        Args:
            x (torch.Tensor): Input tensor.
            sublayer (Callable): A sublayer function to apply.

        Returns:
            torch.Tensor: Output tensor after applying the residual connection.
        """
        if self.pre_norm:
            return self.norm.pre_normalization(x, sublayer)
        return self.norm.post_normalization(x, sublayer)


class NormalizationBlock(nn.Module):
    """
    Normalization block for pre and post normalization.

    Args:
        size (int): Size of the input features.
        dropout (float): Dropout probability.

    Attributes:
        norm (LayerNorm): Layer normalization.
        dropout (nn.Dropout): Dropout layer.

    Methods:
        pre_normalization: Pre-normalization forward pass.
        post_normalization: Post-normalization forward pass.
    """

    def __init__(self, size: int, dropout: float) -> None:
        """
        Initialize normalization block.

        Args:
            size (int): Size of input features.
            dropout (float): Dropout probability.
        """
        super(NormalizationBlock, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def pre_normalization(
        self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Pre-normalization forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            sublayer (Callable): Sublayer function.

        Returns:
            torch.Tensor: Output tensor."""
        return x + self.dropout(sublayer(self.norm(x)))

    def post_normalization(
        self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Post-normalization forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            sublayer (Callable): Sublayer function.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.norm(x + self.dropout(sublayer(x)))


class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network.

    Args:
        d_model (int): Model dimension.
        d_ff (int): Feed-forward dimension.
        dropout (float): Dropout probability.
        activation (Union[str, Callable[[Tensor], Tensor]]): Activation function.
                Can be a string or callable. Default: 'relu'.
        bias (bool): If True, use bias in Linear layers. Default: True.

    Attributes:
        w_1 (nn.Linear): First linear layer.
        w_2 (nn.Linear): Second linear layer.
        dropout (nn.Dropout): Dropout layer.
        activation: Activation function.

    Methods:
        forward: Forward pass through feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str | Callable = 'relu',
        bias: bool = True,
    ) -> None:
        """
        Initialize position-wise feed-forward network.

        Args:
            d_model (int): Model dimension.
            d_ff (int): Feed-forward dimension.
            dropout (float): Dropout probability.
            activation (Union[str, Callable]): Activation function.
            bias (bool): Use bias in linear layers.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w_2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.activation_fns = {
            'relu': F.relu,
            'gelu': F.gelu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'leaky_relu': F.leaky_relu,
            'silu': F.silu,
            'elu': F.elu,
            'selu': F.selu,
        }

        # Handle activation function
        if isinstance(activation, str):
            if activation not in self.activation_fns:
                raise ValueError(
                    f'Unsupported activation function: {activation}. '
                    f'Supported activations are: {list(self.activation_fns.keys())}'
                )
            self.activation = self.activation_fns[activation]
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class EncoderLayer(nn.Module):
    """
    Encoder layer consisting of self-attention and feed-forward layers.

    Args:
        size (int): Size of the input features.
        self_attn (nn.Module): Self attention module.
        feed_forward (nn.Module): Feed-forward module.
        pre_norm (bool): Use pre-normalization.
        dropout (float): Dropout probability.

    Attributes:
        self_attn (nn.Module): Self attention module.
        feed_forward (nn.Module): Feed-forward module.
        sublayer (nn.ModuleList): List of sublayer connections.
        size (int): Size of input features.
        pre_norm (bool): Use pre-normalization.

    Methods:
        forward: Forward pass through encoder layer.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: nn.Module,
        pre_norm: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize the encoder layer.

        Args:
            size (int): Size of the input features.
            self_attn (nn.Module): Self attention module.
            feed_forward (nn.Module): Feed-forward module.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(size, pre_norm, dropout), 2)
        self.size = size
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder layer.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    """
    Decoder layer consisting of self-attention, source-attention, and feed-forward layers.

    Args:
        size (int): Size of the input features.
        self_attn (nn.Module): Self attention module.
        src_attn (nn.Module): Source attention module.
        feed_forward (nn.Module): Feed-forward module.
        pre_norm (bool): Use pre-normalization.
        dropout (float): Dropout probability.

    Attributes:
        size (int): Size of input features.
        self_attn (nn.Module): Self attention module.
        src_attn (nn.Module): Source attention module.
        feed_forward (nn.Module): Feed-forward module.
        sublayer (nn.ModuleList): List of sublayer connections.
        pre_norm (bool): Use pre-normalization.

    Methods:
        forward: Forward pass through decoder layer.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: nn.Module | None,
        feed_forward: nn.Module,
        pre_norm: bool = True,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize decoder layer.

        Args:
            size (int): Size of input features.
            self_attn (nn.Module): Self attention module.
            src_attn (nn.Module): Source attention module.
            feed_forward (nn.Module): Feed-forward module.
            pre_norm (bool): Use pre-normalization.
            dropout (float): Dropout probability.
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        n_sublayers = 3 if src_attn is not None else 2
        self.sublayer = clone(SublayerConnection(size, pre_norm, dropout), n_sublayers)
        self.pre_norm = pre_norm

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None,
        src_mask: torch.Tensor | None,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the decoder layer.

        Args:
            x (torch.Tensor): Input tensor.
            memory (torch.Tensor): Memory tensor from the encoder.
            src_mask (torch.Tensor): Source mask tensor.
            tgt_mask (torch.Tensor): Target mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        if self.src_attn is not None and memory is not None:
            x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
            x = self.sublayer[2](x, self.feed_forward)
        else:
            x = self.sublayer[1](x, self.feed_forward)
        return x


class Embeddings(nn.Module):
    """
    Implements token embeddings.

    Args:
        d_model (int): Model dimension.
        vocab (int): Vocabulary size.

    Attributes:
        lut (nn.Embedding): Embedding layer.
        d_model (int): Model dimension.

    Methods:
        forward: Forward pass through embedding layer.
    """

    def __init__(self, d_model: int, vocab: int) -> None:
        """
        Initialize token embeddings.

        Args:
            d_model (int): Model dimension.
            vocab (int): Vocabulary size.
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through embedding layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Embedded tensor.
        """
        return self.lut(x.to(self.lut.weight.device)) * math.sqrt(self.d_model)
