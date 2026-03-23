"""Language model head — projects hidden states to vocabulary logits."""

import torch
import torch.nn as nn


class LMHead(nn.Module):
    """
    Language model head: linear projection from ``d_model`` to ``vocab_size``.

    Supports optional weight tying with an embedding layer.

    Args:
        d_model:    Hidden dimension.
        vocab_size: Output vocabulary size.
        bias:       Whether to use a bias term (default ``False``).
    """

    def __init__(self, d_model: int, vocab_size: int, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: ``[batch, seq_len, d_model]``

        Returns:
            logits: ``[batch, seq_len, vocab_size]``
        """
        return self.proj(hidden_states)

    def tie_weights(self, embedding: nn.Embedding) -> None:
        """Tie projection weights to an embedding matrix (weight tying)."""
        self.proj.weight = embedding.weight
