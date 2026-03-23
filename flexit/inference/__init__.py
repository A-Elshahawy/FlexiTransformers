"""Inference utilities: greedy and sampling-based decoding strategies."""

from .decoding import (
    DecoderOnlyStrategy,
    DecoderStrategy,
    EncoderDecoderStrategy,
    greedy_decode,
)
from .sampling import (
    sample_decode,
    temperature_sample,
    top_k_sample,
    top_p_sample,
)

__all__ = [
    # Greedy
    'DecoderOnlyStrategy',
    'DecoderStrategy',
    'EncoderDecoderStrategy',
    'greedy_decode',
    # Sampling
    'sample_decode',
    'temperature_sample',
    'top_k_sample',
    'top_p_sample',
]
