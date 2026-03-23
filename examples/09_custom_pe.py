"""
Custom positional encoding plugin via register_pe.

Imports demonstrated:
    from flexit.attention.positional.base import PositionalEncoding
    from flexit import register_pe, create_pe, create_model, ModelConfig
"""

from typing import Literal

import torch
import torch.nn as nn

from flexit import ModelConfig, create_model, create_pe, register_pe
from flexit.attention.positional.base import PositionalEncoding


# --- Example 1: NoPE — identity (no position information) ---
class NoPE(PositionalEncoding):
    """No positional encoding — useful as a baseline."""

    @property
    def injection_point(self) -> Literal['embedding', 'qk', 'scores']:
        return 'embedding'  # or 'qk' or 'scores', doesn't matter since it's identity

    def apply_to_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return x  # identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


register_pe('nope', NoPE)

cfg = ModelConfig(
    model_type='decoder-only',
    vocab_size=1_000,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=256,
    pe_type='nope',  # type: ignore
)
pe_obj = create_pe(cfg)
print(f'NoPE class : {pe_obj.__class__.__name__}')

model = create_model(cfg)
out = model(tgt=torch.randint(0, 1_000, (1, 8)))
print(f'NoPE model : {tuple(out.shape)}')


# --- Example 2: Sinusoidal with learnable scale ---
class ScaledSinusoidal(PositionalEncoding):
    """Sinusoidal PE with a learnable per-dimension scale factor."""

    @property
    def injection_point(self) -> Literal['embedding', 'qk', 'scores']:
        return 'embedding'

    def __init__(self, d_model: int = 64, max_len: int = 512) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        # Precompute sinusoidal table
        import math

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, T, D]

    def apply_to_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.pe[:, : x.size(1)] * self.scale)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_to_embedding(x)


register_pe('scaled_sin', ScaledSinusoidal)

cfg2 = ModelConfig(
    model_type='decoder-only',
    vocab_size=1_000,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=256,
    pe_type='scaled_sin',  # type: ignore
)
model2 = create_model(cfg2)
out2 = model2(tgt=torch.randint(0, 1_000, (2, 16)))
print(f'ScaledSinusoidal model : {tuple(out2.shape)}')
print(
    f'Learnable scale params : {sum(p.numel() for p in model2.parameters() if p.requires_grad):,}'
)
