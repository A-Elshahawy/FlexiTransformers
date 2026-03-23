"""Tests for attention mechanisms and all positional encoding types."""

import pytest
import torch

from flexit import (
    ALiBiPE,
    LearnedPE,
    ModelConfig,
    RelativePE,
    RelativePEWithBias,
    RotaryPE,
    SinusoidalPE,
    create_model,
    create_pe,
    register_pe,
)
from flexit.attention import MultiHeadAttention
from flexit.core import FeedForward, GLUFeedForward

B, S, D, H = 2, 8, 64, 4
V = 100


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------


class TestMultiHeadAttention:
    def test_self_attention_output_shape(self) -> None:
        attn = MultiHeadAttention(D, H)
        x = torch.randn(B, S, D)
        out = attn(x, x, x, mask=None)
        assert out.shape == (B, S, D)

    def test_cross_attention_output_shape(self) -> None:
        attn = MultiHeadAttention(D, H)
        q = torch.randn(B, S, D)
        kv = torch.randn(B, S * 2, D)
        out = attn(q, kv, kv, mask=None)
        assert out.shape == (B, S, D)

    def test_causal_mask_applied(self) -> None:
        """Output at position i should not depend on positions > i."""
        attn = MultiHeadAttention(D, H)
        attn.eval()
        x = torch.randn(1, 4, D)
        from flexit.utils import subsequent_mask

        mask = subsequent_mask(4)
        out_masked = attn(x, x, x, mask=mask)
        # Perturb future token and check first position is unchanged
        x2 = x.clone()
        x2[0, 2:] = torch.randn(2, D)
        out_masked2 = attn(x2, x2, x2, mask=mask)
        assert torch.allclose(out_masked[0, 0], out_masked2[0, 0], atol=1e-5)


# ---------------------------------------------------------------------------
# Positional Encodings — standalone
# ---------------------------------------------------------------------------


class TestSinusoidalPE:
    def test_output_shape(self) -> None:
        pe = SinusoidalPE(D, max_len=32)
        x = torch.randn(B, S, D)
        out = pe.apply_to_embedding(x)
        assert out.shape == (B, S, D)

    def test_deterministic(self) -> None:
        pe = SinusoidalPE(D, max_len=32)
        pe.eval()
        x = torch.randn(B, S, D)
        assert torch.allclose(pe.apply_to_embedding(x), pe.apply_to_embedding(x))


class TestLearnedPE:
    def test_output_shape(self) -> None:
        pe = LearnedPE(D, max_len=32)
        x = torch.randn(B, S, D)
        assert pe.apply_to_embedding(x).shape == (B, S, D)


class TestRotaryPE:
    def test_apply_to_qk_shape(self) -> None:
        head_dim = D // H
        pe = RotaryPE(head_dim, max_len=32)  # parameter is max_len
        q = torch.randn(B, H, S, head_dim)
        k = torch.randn(B, H, S, head_dim)
        q2, k2 = pe.apply_to_qk(q, k)
        assert q2.shape == q.shape
        assert k2.shape == k.shape


class TestALiBiPE:
    def test_apply_to_scores_shape(self) -> None:
        pe = ALiBiPE(H, max_len=32)
        scores = torch.zeros(B, H, S, S)
        out = pe.apply_to_scores(scores, q_len=S, k_len=S)
        assert out.shape == (B, H, S, S)


class TestRelativePE:
    def test_apply_to_scores_shape(self) -> None:
        head_dim = D // H
        pe = RelativePE(head_dim, max_seq_len=32)
        q = torch.randn(B, H, S, head_dim)
        scores = torch.zeros(B, H, S, S)
        out = pe.apply_to_scores(scores, q_len=S, k_len=S, query=q)
        assert out.shape == (B, H, S, S)


class TestRelativePEWithBias:
    def test_apply_to_scores_shape(self) -> None:
        head_dim = D // H
        pe = RelativePEWithBias(head_dim, max_seq_len=32)
        q = torch.randn(B, H, S, head_dim)
        scores = torch.zeros(B, H, S, S)
        out = pe.apply_to_scores(scores, q_len=S, k_len=S, query=q)
        assert out.shape == (B, H, S, S)


# ---------------------------------------------------------------------------
# create_pe factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'pe_type,extra',
    [
        ('absolute', {}),
        ('learned', {}),
        ('rotary', {}),
        ('alibi', {}),
        ('relative', {}),
        ('relative_bias', {}),
        ('none', {}),
    ],
)
def test_create_pe_factory(pe_type: str, extra: dict) -> None:  # type: ignore[override]
    cfg = ModelConfig(
        model_type='decoder-only',
        vocab_size=V,
        d_model=D,
        n_heads=H,
        d_ff=128,
        pe_type=pe_type,  # type: ignore[arg-type]
        **extra,
    )
    pe = create_pe(cfg)
    if pe_type == 'none':
        assert pe is None
    else:
        assert pe is not None


def test_register_pe_custom() -> None:
    from typing import Literal

    from flexit.attention.positional.base import PositionalEncoding

    class MyPE(PositionalEncoding):
        @property
        def injection_point(self) -> Literal['embedding', 'qk', 'scores']:
            return 'embedding'

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    register_pe('my_pe', MyPE)
    from flexit.attention.positional import PE_REGISTRY

    assert 'my_pe' in PE_REGISTRY


# ---------------------------------------------------------------------------
# FeedForward activations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'act', ['relu', 'gelu', 'silu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu']
)
def test_feedforward_activations(act: str) -> None:
    ff = FeedForward(D, 128, activation=act)
    x = torch.randn(B, S, D)
    assert ff(x).shape == (B, S, D)


def test_feedforward_unknown_activation() -> None:
    with pytest.raises(ValueError, match='Unknown activation'):
        FeedForward(D, 128, activation='mish')


@pytest.mark.parametrize('act', ['geglu', 'swiglu'])
def test_glu_feedforward(act: str) -> None:
    ff = GLUFeedForward(D, 128, activation=act)
    x = torch.randn(B, S, D)
    assert ff(x).shape == (B, S, D)


@pytest.mark.parametrize('pe_type', ['absolute', 'rotary', 'alibi', 'relative', 'none'])
def test_decoder_only_all_pe_types(pe_type: str) -> None:  # type: ignore[override]
    cfg = ModelConfig(
        model_type='decoder-only',
        vocab_size=V,
        d_model=D,
        n_heads=H,
        n_layers=2,
        d_ff=128,
        pe_type=pe_type,  # type: ignore[arg-type]
    )
    model = create_model(cfg)
    x = torch.randint(0, V, (B, S))
    out = model(tgt=x)
    assert out.shape == (B, S, V)
