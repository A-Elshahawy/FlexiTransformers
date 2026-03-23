from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        bias: bool = True,
    ):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.activation: Callable[..., Any] = self._get_activation(activation)

    def _get_activation(self, name: str) -> Callable[..., Any]:
        activations = {
            'relu': F.relu,
            'gelu': F.gelu,
            'silu': F.silu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'selu': F.selu,
        }
        if name not in activations:
            raise ValueError(
                f'Unknown activation: {name!r}. '
                f'Valid options: {list(activations)}. '
                "For 'geglu'/'swiglu' use GLUFeedForward or set via create_feedforward()."
            )
        return activations[name]  # type: ignore[return-value, no-any-return]

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.dropout(self.activation(self.w1(x))))


def create_feedforward(
    d_model: int,
    d_ff: int,
    dropout: float = 0.1,
    activation: str = 'gelu',
    bias: bool = True,
) -> nn.Module:
    """Factory: returns GLUFeedForward for geglu/swiglu, FeedForward otherwise."""
    if activation in ('geglu', 'swiglu'):
        return GLUFeedForward(d_model, d_ff, dropout, activation, bias=False)
    return FeedForward(d_model, d_ff, dropout, activation, bias)


class GLUFeedForward(nn.Module):
    """Gated Linear Unit FFN (e.g., SwiGLU in LLaMA)."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'silu',
        bias: bool = False,
    ):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        _act_map = {'silu': F.silu, 'swiglu': F.silu, 'geglu': F.gelu}
        self.activation: Callable[..., Any] = _act_map.get(activation, F.silu)  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.dropout(self.activation(self.w_gate(x)) * self.w1(x)))
