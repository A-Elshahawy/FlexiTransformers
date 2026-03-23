"""Token-level classification head (e.g. NER, POS tagging)."""

import torch
import torch.nn as nn


class TokenClassificationHead(nn.Module):
    """
    Per-token classification head.

    Projects every position in the sequence independently to ``num_classes``.
    Suitable for tasks like NER, POS tagging, or masked language modelling
    when each token needs its own label.

    Args:
        d_model:     Hidden dimension.
        num_classes: Number of per-token output classes.
        dropout:     Dropout probability applied before projection (default 0.1).
    """

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: ``[batch, seq_len, d_model]``

        Returns:
            logits: ``[batch, seq_len, num_classes]``
        """
        return self.classifier(self.dropout(hidden_states))
