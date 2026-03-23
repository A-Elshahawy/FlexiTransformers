"""Classification heads for transformer models."""

from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.normalization import LayerNorm


class BertHead(nn.Module):
    """
    BERT-style classification head.

    Takes the [CLS] token (position 0), applies a dense projection + activation +
    optional norm + dropout, then projects to ``num_classes``.

    Args:
        d_model:     Hidden dimension of the encoder.
        num_classes: Number of output classes.
        dropout:     Dropout probability (default 0.1).
        pre_norm:    Apply LayerNorm before the classifier projection (default True).
        activation:  Activation function name or callable (default ``"gelu"``).
    """

    _ACT: ClassVar = {
        'gelu': F.gelu,
        'relu': F.relu,
        'tanh': torch.tanh,
        'silu': F.silu,
    }

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float = 0.1,
        pre_norm: bool = True,
        activation: str | nn.Module = 'gelu',
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        if isinstance(activation, str):
            self.activation = self._ACT.get(activation, F.gelu)
        else:
            self.activation = activation
        self.norm = LayerNorm(d_model) if pre_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: ``[batch, seq_len, d_model]``

        Returns:
            logits: ``[batch, num_classes]``
        """
        cls = hidden_states[:, 0]  # [B, d_model]
        x = self.activation(self.dense(cls))
        x = self.dropout(self.norm(x))
        return self.classifier(x)


class SequenceClassificationHead(nn.Module):
    """
    Simple mean-pooled or max-pooled classification head.

    Useful when there is no dedicated [CLS] token.

    Args:
        d_model:     Hidden dimension.
        num_classes: Number of output classes.
        pooling:     ``"mean"`` or ``"max"`` (default ``"mean"``).
        dropout:     Dropout probability (default 0.1).
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        pooling: str = 'mean',
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if pooling not in ('mean', 'max'):
            raise ValueError(f"pooling must be 'mean' or 'max', got {pooling!r}")
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: ``[batch, seq_len, d_model]``
            mask:          ``[batch, seq_len]`` boolean mask (True = keep). Optional.

        Returns:
            logits: ``[batch, num_classes]``
        """
        if self.pooling == 'mean':
            if mask is not None:
                mask_f = mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(1)
        else:  # max
            if mask is not None:
                hidden_states = hidden_states.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            pooled = hidden_states.max(1).values

        return self.classifier(self.dropout(pooled))
