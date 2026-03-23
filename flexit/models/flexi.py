"""High-level convenience constructors for common transformer architectures."""

from __future__ import annotations

from typing import Any

from ..config import ModelConfig
from .base import BaseModel


def FlexiTransformer(
    model_type: str | None = None,
    *,
    # Old-style vocab params kept for backwards compat
    src_vocab: int | None = None,
    tgt_vocab: int | None = None,
    # New-style vocab params
    vocab_size: int | None = None,
    src_vocab_size: int | None = None,
    tgt_vocab_size: int | None = None,
    # Architecture
    d_model: int = 512,
    n_heads: int = 8,
    d_ff: int | None = None,
    # Old-style layer counts
    n_enc: int | None = None,
    n_dec: int | None = None,
    # New-style layer count (overrides n_enc/n_dec)
    n_layers: int | tuple[int, int] | None = None,
    **kwargs: Any,
) -> BaseModel:
    """
    General-purpose transformer constructor with automatic model-type inference.

    Model type is inferred from vocabulary arguments if *model_type* is omitted:
      - src_vocab + tgt_vocab  →  encoder-decoder
      - src_vocab only         →  encoder-only
      - tgt_vocab / vocab_size →  decoder-only

    Old-style parameter names (``src_vocab``, ``tgt_vocab``, ``n_enc``, ``n_dec``)
    are accepted alongside the new-style names for backwards compatibility.
    """
    # Translate old-style vocab params
    if src_vocab is not None and src_vocab_size is None:
        src_vocab_size = src_vocab
    if tgt_vocab is not None and tgt_vocab_size is None:
        tgt_vocab_size = tgt_vocab

    # Infer model_type from vocab args
    if model_type is None:
        if src_vocab_size is not None and tgt_vocab_size is not None:
            model_type = 'encoder-decoder'
        elif src_vocab_size is not None:
            model_type = 'encoder-only'
        elif tgt_vocab_size is not None or vocab_size is not None:
            model_type = 'decoder-only'
        else:
            raise ValueError(
                'Cannot infer model_type. Provide model_type or at least one '
                'of: vocab_size, src_vocab_size / src_vocab, tgt_vocab_size / tgt_vocab.'
            )

    # Resolve n_layers
    if n_layers is None:
        if n_enc is not None and n_dec is not None:
            n_layers = (n_enc, n_dec)
        elif n_enc is not None:
            n_layers = n_enc
        elif n_dec is not None:
            n_layers = n_dec
        else:
            n_layers = 6

    if d_ff is None:
        d_ff = d_model * 4

    config = ModelConfig(
        model_type=model_type,  # type: ignore[arg-type]
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        vocab_size=vocab_size,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        **kwargs,
    )
    from ..factory.model_factory import (
        create_model,  # deferred to break circular import
    )

    return create_model(config)


TransformerModel = FlexiTransformer


def FlexiBERT(
    vocab_size: int,
    d_model: int = 768,
    n_heads: int = 12,
    n_layers: int = 12,
    d_ff: int = 3072,
    num_classes: int | None = None,
    **kwargs: Any,
) -> BaseModel:
    """
    BERT-style encoder-only transformer.

    Defaults match BERT-base: d_model=768, 12 heads, 12 layers, d_ff=3072,
    absolute positional encoding, post-norm, GELU activation.
    Pass ``num_classes`` to add a classification head.
    """
    kwargs.setdefault('pe_type', 'absolute')
    kwargs.setdefault('pre_norm', False)
    kwargs.setdefault('ff_activation', 'gelu')
    kwargs.setdefault('norm_type', 'layernorm')

    config = ModelConfig(
        model_type='encoder-only',
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        num_classes=num_classes,
        **kwargs,
    )
    from ..factory.model_factory import (
        create_model,  # deferred to break circular import
    )

    return create_model(config)


def FlexiGPT(
    vocab_size: int,
    d_model: int = 768,
    n_heads: int = 12,
    n_layers: int = 12,
    d_ff: int = 3072,
    **kwargs: Any,
) -> BaseModel:
    """
    GPT-style decoder-only transformer.

    Defaults match GPT-2 small: d_model=768, 12 heads, 12 layers, d_ff=3072,
    RoPE positional encoding, pre-norm (RMSNorm), SwiGLU activation.
    """
    kwargs.setdefault('pe_type', 'rotary')
    kwargs.setdefault('pre_norm', True)
    kwargs.setdefault('norm_type', 'rmsnorm')
    kwargs.setdefault('ff_activation', 'swiglu')
    kwargs.setdefault('ff_bias', False)
    kwargs.setdefault('attention_bias', False)

    config = ModelConfig(
        model_type='decoder-only',
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        **kwargs,
    )
    from ..factory.model_factory import (
        create_model,  # deferred to break circular import
    )

    return create_model(config)
