import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import clone


class LayerNorm(nn.Module):
    """
    Construct a Layer Normalization module.

    Args:
        features (int): Number of features in the input.
        eps (float): A small value to avoid division by zero. Default: 1e-6.
    """

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer normalization.

    Args:
        size (int): Size of the input features.
        dropout (float): Dropout probability.
    """

    def __init__(self, size: int, dropout: float) -> None:
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

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
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network.

    Args:
        d_model (int): Model dimension.
        d_ff (int): Feed-forward dimension.
        dropout (float): Dropout probability.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, dropout: float):
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
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder layer.

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
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: nn.Module,
        feed_forward: nn.Module,
        dropout: float,
    ) -> None:
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(size, dropout), 3)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor
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
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Embeddings(nn.Module):
    """
    Implements token embeddings.

    Args:
        d_model (int): Model dimension.
        vocab (int): Vocabulary size.
    """

    def __init__(self, d_model: int, vocab: int) -> None:
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lut(x.to(self.lut.weight.device)) * math.sqrt(self.d_model)
