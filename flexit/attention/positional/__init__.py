from typing import TYPE_CHECKING

from .absolute import LearnedPE, SinusoidalPE
from .alibi import ALiBiPE
from .base import PositionalEncoding
from .relative import RelativePE, RelativePEWithBias
from .rotary import RotaryPE

if TYPE_CHECKING:
    from ...config import ModelConfig

# Registry
PE_REGISTRY: dict[str, type[PositionalEncoding]] = {
    'absolute': SinusoidalPE,
    'learned': LearnedPE,
    'rotary': RotaryPE,
    'alibi': ALiBiPE,
    'relative': RelativePE,
    'relative_bias': RelativePEWithBias,
}


def create_pe(config: 'ModelConfig') -> PositionalEncoding | None:
    """Factory function to create positional encoding from config."""
    if config.pe_type == 'none':
        return None

    pe_cls = PE_REGISTRY.get(config.pe_type)
    if pe_cls is None:
        raise ValueError(f'Unknown PE type: {config.pe_type}')

    if config.pe_type == 'absolute' or config.pe_type == 'learned':
        return pe_cls(config.d_model, config.max_seq_len, config.dropout)
    elif config.pe_type == 'rotary':
        head_dim = config.d_model // config.n_heads
        rope_dim = int(head_dim * config.rope_percentage)
        return pe_cls(rope_dim, config.max_seq_len, config.rope_base)
    elif config.pe_type == 'alibi':
        return pe_cls(config.n_heads, config.max_seq_len)
    elif config.pe_type in ('relative', 'relative_bias'):
        head_dim = config.d_model // config.n_heads
        return pe_cls(head_dim, config.max_seq_len)

    # Custom type registered via register_pe() — try no-arg construction
    try:
        return pe_cls()
    except TypeError as exc:
        raise ValueError(
            f"Cannot auto-instantiate custom PE '{config.pe_type}'. "
            'Custom PE classes must be no-arg constructors, or call create_pe manually.'
        ) from exc


def register_pe(name: str, cls: type[PositionalEncoding]) -> None:
    """Register a custom positional encoding."""
    PE_REGISTRY[name] = cls
