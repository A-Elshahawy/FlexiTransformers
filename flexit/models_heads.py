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
        model: nn.Module,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int,
        start_symbol: int,
        end_symbol: int | None = None,
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
        model: nn.Module,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int,
        start_symbol: int,
        end_symbol: int | None = None,
    ) -> torch.Tensor:
        """
        Perform batched decoding using the encoder-decoder architecture.

        Args:
            model: Transformer model.
            src (torch.Tensor): Source sequence with shape [batch_size, seq_len].
            src_mask (torch.Tensor): Source mask with shape [batch_size, 1, seq_len].
            max_len (int): Maximum length for decoding.
            start_symbol (int): Start symbol token ID.
            end_symbol (int, optional): End symbol token ID for early stopping.

        Returns:
            torch.Tensor: Decoded sequences with shape [batch_size, seq_len].
        """
        batch_size = src.size(0)
        device = src.device

        # Encode the source sequence once
        memory = model.encode(src, src_mask)

        # Initialize with start token for each batch item
        ys = torch.full((batch_size, 1), start_symbol, device=device).type_as(src)

        # Track which sequences have completed (generated end token)
        completed = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            # Create appropriate target mask for current target length
            tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data).to(device)

            # Decode next tokens
            out = model.decode(memory, src_mask, ys, tgt_mask)
            logits = model.generator(out[:, -1])

            # Get most probable token for each sample in batch
            _, next_word = torch.max(logits, dim=1)
            next_word = next_word.unsqueeze(1)

            # Append new tokens
            ys = torch.cat([ys, next_word], dim=1)

            # Check for end token if specified
            if end_symbol is not None:
                # Mark sequences that produced the end token
                completed = completed | (next_word.squeeze(1) == end_symbol)
                # If all sequences have produced end token, we can stop
                if completed.all():
                    break

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
        end_symbol: int | None = None,
    ) -> torch.Tensor:
        """
        Perform batched decoding using decoder-only architecture.

        Args:
            model (nn.Module): Transformer model.
            src (torch.Tensor): Initial sequence or None. Shape [batch_size, seq_len].
            src_mask (torch.Tensor | None): Source mask or None.
            max_len (int): Maximum length for decoding.
            start_symbol (int): Start symbol for decoding.
            end_symbol (int, optional): End symbol token ID for early stopping.

        Returns:
            torch.Tensor: Decoded sequences with shape [batch_size, seq_len].
        """
        device = src.device if src is not None else next(model.parameters()).device

        # Initialize sequence
        if src is None:
            batch_size = 1
            ys = torch.full((batch_size, 1), start_symbol, device=device).type_as(
                next(model.parameters())
            )
        else:
            batch_size = src.size(0)
            ys = src.clone().to(device)

        # Track which sequences have completed
        completed = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            # Create causal mask for autoregressive generation
            tgt_mask = subsequent_mask(ys.size(1)).type_as(ys).to(device)

            # Forward pass through the model
            out = model(ys, tgt_mask)
            logits = model.generator(out[:, -1])

            # Select next token
            _, next_word = torch.max(logits, dim=1)
            next_word = next_word.unsqueeze(1)

            # Append to sequences
            ys = torch.cat([ys, next_word], dim=1)

            # Check for early stopping
            if end_symbol is not None:
                completed = completed | (next_word.squeeze(1) == end_symbol)
                if completed.all():
                    break

        return ys


def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int | None = None,
) -> torch.Tensor:
    """
    Perform greedy decoding based on model type.

    This function selects the appropriate decoding strategy based on the model type.

    Args:
        model: Transformer model.
        src (torch.Tensor): Source sequence.
        src_mask (torch.Tensor): Source mask.
        max_len (int): Maximum length for decoding.
        start_symbol (int): Start symbol for decoding.
        end_symbol (int, optional): End symbol for early stopping.

    Returns:
        torch.Tensor: Decoded sequence.

    Raises:
        ValueError: If model type is not supported.
    """
    strategies = {'encoder-decoder': EncoderDecoderStrategy, 'decoder-only': DecoderOnlyStrategy}
    strategy = strategies.get(getattr(model, 'model_type', 'encoder-decoder'))
    if not strategy:
        raise ValueError(f'Unsupported model type: {getattr(model, "model_type", "unknown")}')
    return strategy.decode(model, src, src_mask, max_len, start_symbol, end_symbol)


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
