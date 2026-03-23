"""Tests for greedy decoding strategies."""

import pytest
import torch

from flexit import (
    EncoderDecoderStrategy,
    ModelConfig,
    create_model,
    greedy_decode,
)
from flexit.inference.decoding import DecoderOnlyStrategy

B, S, V = 1, 6, 50
BOS, EOS = 1, 2


def _dec_cfg() -> ModelConfig:
    return ModelConfig(
        model_type='decoder-only', vocab_size=V, d_model=64, n_heads=4, n_layers=2, d_ff=128
    )


def _enc_dec_cfg() -> ModelConfig:
    return ModelConfig(
        model_type='encoder-decoder', vocab_size=V, d_model=64, n_heads=4, n_layers=(2, 2), d_ff=128
    )


# ---------------------------------------------------------------------------
# DecoderOnlyStrategy
# ---------------------------------------------------------------------------


class TestDecoderOnlyStrategy:
    def test_output_shape_no_prompt(self) -> None:
        model = create_model(_dec_cfg())
        out = DecoderOnlyStrategy.decode(
            model=model,
            src=None,
            src_mask=None,
            max_len=S,
            start_symbol=BOS,
        )
        # starts with BOS then generates up to max_len-1 more
        assert out.shape[0] == 1
        assert out.shape[1] <= S

    def test_output_starts_with_bos(self) -> None:
        model = create_model(_dec_cfg())
        out = DecoderOnlyStrategy.decode(
            model=model,
            src=None,
            src_mask=None,
            max_len=S,
            start_symbol=BOS,
        )
        assert out[0, 0].item() == BOS

    def test_stops_at_eos(self) -> None:
        model = create_model(_dec_cfg())

        # Force the model to always predict EOS by hooking generator
        class AlwaysEOS(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                logits = torch.full((x.size(0), V), -1e9)
                logits[:, EOS] = 1e9
                return logits

        model.generator = AlwaysEOS()

        out = DecoderOnlyStrategy.decode(
            model=model,
            src=None,
            src_mask=None,
            max_len=10,
            start_symbol=BOS,
            end_symbol=EOS,
        )
        # Should stop after first generated token (EOS)
        assert out.shape[1] == 2  # BOS + EOS

    def test_with_prompt(self) -> None:
        model = create_model(_dec_cfg())
        prompt = torch.randint(1, V, (B, 3))
        out = DecoderOnlyStrategy.decode(
            model=model,
            src=prompt,
            src_mask=None,
            max_len=S,
            start_symbol=BOS,
        )
        assert out.shape[0] == B
        assert out.shape[1] >= 3


# ---------------------------------------------------------------------------
# EncoderDecoderStrategy
# ---------------------------------------------------------------------------


class TestEncoderDecoderStrategy:
    def test_output_shape(self) -> None:
        model = create_model(_enc_dec_cfg())
        src = torch.randint(1, V, (B, S))
        out = EncoderDecoderStrategy.decode(
            model=model,
            src=src,
            src_mask=None,
            max_len=S,
            start_symbol=BOS,
        )
        assert out.shape[0] == B
        assert out.shape[1] <= S

    def test_output_starts_with_bos(self) -> None:
        model = create_model(_enc_dec_cfg())
        src = torch.randint(1, V, (B, S))
        out = EncoderDecoderStrategy.decode(
            model=model,
            src=src,
            src_mask=None,
            max_len=S,
            start_symbol=BOS,
        )
        assert out[0, 0].item() == BOS


# ---------------------------------------------------------------------------
# greedy_decode dispatcher
# ---------------------------------------------------------------------------


class TestGreedyDecode:
    def test_dispatches_decoder_only(self) -> None:
        model = create_model(_dec_cfg())
        out = greedy_decode(model, src=None, src_mask=None, max_len=S, start_symbol=BOS)
        assert out.shape[0] == 1

    def test_dispatches_encoder_decoder(self) -> None:
        model = create_model(_enc_dec_cfg())
        src = torch.randint(1, V, (B, S))
        out = greedy_decode(model, src=src, src_mask=None, max_len=S, start_symbol=BOS)
        assert out.shape[0] == B

    def test_unsupported_model_type_raises(self) -> None:
        model = create_model(_dec_cfg())
        model.model_type = 'unknown-type'
        with pytest.raises(ValueError, match='Unsupported model type'):
            greedy_decode(model, src=None, src_mask=None, max_len=S, start_symbol=BOS)
