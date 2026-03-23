"""Tests for ModelConfig validation and defaults."""

import pytest

from flexit import ModelConfig

# ---------------------------------------------------------------------------
# Valid configs
# ---------------------------------------------------------------------------


def test_decoder_only_minimal() -> None:
    cfg = ModelConfig(model_type='decoder-only', vocab_size=100)
    assert cfg.d_model == 512
    assert cfg.attention_dropout == cfg.dropout


def test_encoder_only_minimal() -> None:
    cfg = ModelConfig(model_type='encoder-only', vocab_size=100)
    assert cfg.vocab_size == 100


def test_encoder_decoder_unified_vocab() -> None:
    cfg = ModelConfig(model_type='encoder-decoder', vocab_size=200)
    assert cfg.src_vocab_size == 200
    assert cfg.tgt_vocab_size == 200


def test_encoder_decoder_separate_vocab() -> None:
    cfg = ModelConfig(model_type='encoder-decoder', src_vocab_size=300, tgt_vocab_size=400)
    assert cfg.src_vocab_size == 300
    assert cfg.tgt_vocab_size == 400


def test_encoder_decoder_n_layers_int_expands_to_tuple() -> None:
    cfg = ModelConfig(model_type='encoder-decoder', vocab_size=100, n_layers=4)
    assert cfg.n_layers == (4, 4)


def test_encoder_decoder_n_layers_tuple() -> None:
    cfg = ModelConfig(model_type='encoder-decoder', vocab_size=100, n_layers=(3, 6))
    assert cfg.n_layers == (3, 6)


def test_src_vocab_size_promoted_to_vocab_size_for_encoder_only() -> None:
    cfg = ModelConfig(model_type='encoder-only', src_vocab_size=500)
    assert cfg.vocab_size == 500


def test_ff_dropout_defaults_to_dropout() -> None:
    cfg = ModelConfig(model_type='decoder-only', vocab_size=50, dropout=0.2)
    assert cfg.ff_dropout == 0.2


# ---------------------------------------------------------------------------
# Invalid configs
# ---------------------------------------------------------------------------


def test_d_model_not_divisible_by_n_heads() -> None:
    with pytest.raises(ValueError, match='d_model'):
        ModelConfig(model_type='decoder-only', vocab_size=100, d_model=100, n_heads=8)


def test_no_vocab_raises() -> None:
    with pytest.raises(ValueError):
        ModelConfig(model_type='decoder-only')


def test_encoder_decoder_missing_tgt_vocab() -> None:
    with pytest.raises(ValueError):
        ModelConfig(model_type='encoder-decoder', src_vocab_size=100)


def test_rope_percentage_out_of_range() -> None:
    with pytest.raises(ValueError, match='rope_percentage'):
        ModelConfig(
            model_type='decoder-only', vocab_size=100, pe_type='rotary', rope_percentage=1.5
        )


def test_dropout_out_of_range() -> None:
    with pytest.raises(ValueError, match='dropout'):
        ModelConfig(model_type='decoder-only', vocab_size=100, dropout=1.0)


def test_non_encoder_decoder_tuple_n_layers_raises() -> None:
    with pytest.raises(ValueError):
        ModelConfig(model_type='decoder-only', vocab_size=100, n_layers=(2, 2))


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_to_dict_round_trip() -> None:
    cfg = ModelConfig(model_type='decoder-only', vocab_size=100, d_model=128, n_heads=4)
    d = cfg.to_dict()
    cfg2 = ModelConfig.from_dict(d)
    assert cfg.d_model == cfg2.d_model
    assert cfg.n_layers == cfg2.n_layers


def test_repr_contains_model_type() -> None:
    cfg = ModelConfig(model_type='encoder-only', vocab_size=100)
    assert 'encoder-only' in repr(cfg)
