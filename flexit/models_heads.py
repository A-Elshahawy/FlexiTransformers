"""
Model Heads

This module implements various model heads for different transformer architectures,
including decoding strategies and classification heads.
"""

from collections.abc import Callable

import torch
import torch.nn as nn

from .layers import LayerNorm
from .utils import subsequent_mask


class DecoderStrategy:
    """
    Base class for decoder strategies.

    This class defines the interface for decoding strategies used in transformer models.

    Methods:
        decode: Static method to perform decoding using a specific strategy.
    """

    @staticmethod
    def decode(
        model: nn.Module, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, start_symbol: int
    ) -> torch.Tensor:
        raise NotImplementedError


class EncoderDecoderStrategy(DecoderStrategy):
    """
    Decoding strategy for encoder-decoder models.

    This strategy uses the encoder-decoder architecture for decoding.

    Methods:
        decode: Perform decoding using encoder-decoder architecture.
    """

    @staticmethod
    def decode(
        model: nn.Module, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, start_symbol: int
    ) -> torch.Tensor:
        """
        Perform decoding using the specified strategy.

        Args:
            model: Transformer model.
            src (torch.Tensor): Source sequence.
            src_mask (torch.Tensor): Source mask.
            max_len (int): Maximum length for decoding.
            start_symbol (int): Start symbol for decoding.

        Returns:
            torch.Tensor: Decoded sequence.
        """
        memory = model.encode(src, src_mask)
        ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
        for _ in range(max_len - 1):
            out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        return ys


class DecoderOnlyStrategy(DecoderStrategy):
    """
    Decoding strategy for decoder-only models.

    This strategy uses the decoder-only architecture for decoding.

    Methods:
        decode: Perform decoding using decoder-only architecture.
    """

    @staticmethod
    def decode(
        model: nn.Module,
        src: torch.Tensor,
        src_mask: torch.Tensor | None,
        max_len: int,
        start_symbol: int,
    ) -> torch.Tensor:
        """
        Perform decoding using decoder-only architecture.

        Args:
            model (nn.Module): Transformer model.
            src (torch.Tensor): Source sequence.
            src_mask (torch.Tensor | None): Source mask.
            max_len (int): Maximum length for decoding.
            start_symbol (int): Start symbol for decoding.

        Returns:
            torch.Tensor: Decoded sequence.
        """
        device = src.device if src is not None else next(model.parameters()).device
        if src is None:
            ys = torch.tensor([[start_symbol]], device=device).type_as(next(model.parameters()))
        else:
            ys = src.clone().to(device)

        for _ in range(max_len - 1):
            tgt_mask = subsequent_mask(ys.size(1)).type_as(ys)
            out = model(ys, tgt_mask)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat(
                [ys, torch.full((1, 1), next_word.item(), device=device).type_as(ys)], dim=1
            )
        return ys


def greedy_decode(
    model: nn.Module, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, start_symbol: int
) -> torch.Tensor:
    """
    Greedy decoding function.

    This function selects the appropriate decoding strategy based on the model type.

    Args:
        model: Transformer model.
        src (torch.Tensor): Source sequence.
        src_mask (torch.Tensor): Source mask.
        max_len (int): Maximum length for decoding.
        start_symbol (int): Start symbol for decoding.

    Returns:
        torch.Tensor: Decoded sequence.

    Raises:
        ValueError: If model type is not supported.
    """
    strategies = {'encoder-decoder': EncoderDecoderStrategy, 'decoder-only': DecoderOnlyStrategy}
    strategy = strategies.get(model.model_type)
    if not strategy:
        raise ValueError(f'Unsupported model type: {model.model_type}')
    return strategy.decode(model, src, src_mask, max_len, start_symbol)


class BertHead(nn.Module):
    """
    BERT-style classification head for encoder-only models.

    This implementation follows the standard BERT approach:
    1. Takes the [CLS] token representation (first token)
    2. Applies a transformation with LayerNorm
    3. Projects to the target number of classes

    Args:
        d_model (int): Hidden dimension of the transformer model
        num_classes (int): Number of output classes
        dropout (float, optional): Dropout probability. Default: 0.1
        activation (callable, optional): Activation function. Default: torch.tanh

    Attributes:
        dense (nn.Linear): Linear layer for transformation.
        activation: Activation function.
        norm (LayerNorm): Layer normalization.
        dropout (nn.Dropout): Dropout layer.
        classifier (nn.Linear): Classification layer.

    Methods:
        forward: Forward pass through the classification head.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        pre_norm: bool = True,
        dropout: float = 0.1,
        activation: Callable | str = nn.functional.gelu,
    ) -> None:
        """
        Initialize BERT classification head.

        Args:
            d_model (int): Hidden dimension of the transformer model.
            num_classes (int): Number of output classes.
            pre_norm (bool): Use pre-normalization.
            dropout (float): Dropout probability.
            activation: Activation function.
        """
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = (
            activation if callable(activation) else getattr(nn.functional, activation.lower())
        )
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for BERT classification head.

        Args:
            hidden_states (torch.Tensor): Output from the transformer encoder.
                Expected shape: [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Classification logits with shape [batch_size, num_classes]
        """
        cls_token = hidden_states[:, 0]

        x = self.dense(cls_token)
        x = self.activation(x)
        x = self.dropout(x)
        if self.pre_norm:
            x = self.norm(x)
        logits = self.classifier(x)

        return logits
