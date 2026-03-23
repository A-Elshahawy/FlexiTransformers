"""Tests for all three model architectures — forward pass shapes, save/load."""

import tempfile
from typing import Any

import torch

import flexit
from flexit import ModelConfig, create_model

# Tiny config shared by most tests
B, S, V = 2, 8, 100  # batch, seq_len, vocab


def _dec_only_cfg(**kw: Any) -> ModelConfig:
    return ModelConfig(
        model_type='decoder-only', vocab_size=V, d_model=64, n_heads=4, n_layers=2, d_ff=128, **kw
    )


def _enc_only_cfg(**kw: Any) -> ModelConfig:
    return ModelConfig(
        model_type='encoder-only', vocab_size=V, d_model=64, n_heads=4, n_layers=2, d_ff=128, **kw
    )


def _enc_dec_cfg(**kw: Any) -> ModelConfig:
    return ModelConfig(
        model_type='encoder-decoder',
        vocab_size=V,
        d_model=64,
        n_heads=4,
        n_layers=(2, 2),
        d_ff=128,
        **kw,
    )


# ---------------------------------------------------------------------------
# Decoder-Only
# ---------------------------------------------------------------------------


class TestDecoderOnly:
    def test_forward_shape(self) -> None:
        model = create_model(_dec_only_cfg())
        x = torch.randint(0, V, (B, S))
        out = model(tgt=x)
        assert out.shape == (B, S, V)

    def test_return_hidden(self) -> None:
        model = create_model(_dec_only_cfg())
        x = torch.randint(0, V, (B, S))
        logits, hidden = model(tgt=x, return_hidden=True)
        assert logits.shape == (B, S, V)
        assert hidden.shape == (B, S, 64)

    def test_num_parameters_positive(self) -> None:
        model = create_model(_dec_only_cfg())
        assert model.num_parameters() > 0

    def test_len_equals_num_parameters(self) -> None:
        model = create_model(_dec_only_cfg())
        assert len(model) == model.num_parameters()

    def test_repr(self) -> None:
        model = create_model(_dec_only_cfg())
        r = repr(model)
        assert 'decoder-only' in r
        assert 'params=' in r

    def test_model_type_attribute(self) -> None:
        model = create_model(_dec_only_cfg())
        assert model.model_type == 'decoder-only'

    def test_save_load_round_trip(self) -> None:
        model = create_model(_dec_only_cfg())
        model.eval()
        x = torch.randint(0, V, (B, S))
        with torch.no_grad():
            out_before = model(tgt=x)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        model.save(path)

        from flexit import DecoderOnlyModel

        loaded = DecoderOnlyModel.load(path)
        loaded.eval()
        with torch.no_grad():
            out_after = loaded(tgt=x)

        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_kv_cache_init(self) -> None:
        model = create_model(_dec_only_cfg())
        cache = model.init_kv_cache()
        assert isinstance(cache, list)
        assert len(cache) == 2  # n_layers


# ---------------------------------------------------------------------------
# Encoder-Only
# ---------------------------------------------------------------------------


class TestEncoderOnly:
    def test_forward_shape_no_classification(self) -> None:
        model = create_model(_enc_only_cfg())
        x = torch.randint(0, V, (B, S))
        # forward(input_ids, mask=None)
        out = model(x)
        assert out.shape == (B, S, 64)

    def test_forward_shape_with_classification(self) -> None:
        model = create_model(_enc_only_cfg(num_classes=5))
        x = torch.randint(0, V, (B, S))
        out = model(x)
        assert out.shape == (B, 5)

    def test_model_type_attribute(self) -> None:
        model = create_model(_enc_only_cfg())
        assert model.model_type == 'encoder-only'

    def test_save_load_round_trip(self) -> None:
        model = create_model(_enc_only_cfg())
        model.eval()
        x = torch.randint(0, V, (B, S))
        with torch.no_grad():
            out_before = model(x)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        model.save(path)

        from flexit import EncoderOnlyModel

        loaded = EncoderOnlyModel.load(path)
        loaded.eval()
        with torch.no_grad():
            assert torch.allclose(out_before, loaded(x), atol=1e-5)


# ---------------------------------------------------------------------------
# Encoder-Decoder
# ---------------------------------------------------------------------------


class TestEncoderDecoder:
    def test_encode_decode_shapes(self) -> None:
        model = create_model(_enc_dec_cfg())
        src = torch.randint(0, V, (B, S))
        tgt = torch.randint(0, V, (B, S))

        memory = model.encode(src, src_mask=None)
        assert memory.shape == (B, S, 64)

        out = model.decode(tgt, memory, tgt_mask=None, memory_mask=None)
        assert out.shape == (B, S, 64)

    def test_forward_shape(self) -> None:
        model = create_model(_enc_dec_cfg())
        src = torch.randint(0, V, (B, S))
        tgt = torch.randint(0, V, (B, S))
        out = model(src=src, tgt=tgt)
        assert out.shape == (B, S, V)

    def test_model_type_attribute(self) -> None:
        model = create_model(_enc_dec_cfg())
        assert model.model_type == 'encoder-decoder'

    def test_save_load_round_trip(self) -> None:
        model = create_model(_enc_dec_cfg())
        model.eval()
        src = torch.randint(0, V, (B, S))
        tgt = torch.randint(0, V, (B, S))
        with torch.no_grad():
            out_before = model(src=src, tgt=tgt)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        model.save(path)

        from flexit import EncoderDecoderModel

        loaded = EncoderDecoderModel.load(path)
        loaded.eval()
        with torch.no_grad():
            assert torch.allclose(out_before, loaded(src=src, tgt=tgt), atol=1e-5)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


class TestConvenienceConstructors:
    def test_flexigpt(self) -> None:
        model = flexit.FlexiGPT(vocab_size=V, d_model=64, n_heads=4, n_layers=2, d_ff=128)
        x = torch.randint(0, V, (B, S))
        assert model(tgt=x).shape == (B, S, V)

    def test_flexibert_no_head(self) -> None:
        model = flexit.FlexiBERT(vocab_size=V, d_model=64, n_heads=4, n_layers=2, d_ff=128)
        x = torch.randint(0, V, (B, S))
        # No classification head → returns [B, S, d_model]
        assert model(x).shape == (B, S, 64)

    def test_flexibert_classification(self) -> None:
        model = flexit.FlexiBERT(
            vocab_size=V, d_model=64, n_heads=4, n_layers=2, d_ff=128, num_classes=3
        )
        x = torch.randint(0, V, (B, S))
        assert model(x).shape == (B, 3)

    def test_flexitransformer_enc_dec(self) -> None:
        model = flexit.FlexiTransformer(
            src_vocab=V, tgt_vocab=V, d_model=64, n_heads=4, n_enc=2, n_dec=2, d_ff=128
        )
        src = torch.randint(0, V, (B, S))
        tgt = torch.randint(0, V, (B, S))
        assert model(src=src, tgt=tgt).shape == (B, S, V)

    def test_flexitransformer_decoder_only_inferred(self) -> None:
        model = flexit.FlexiTransformer(vocab_size=V, d_model=64, n_heads=4, n_layers=2, d_ff=128)
        x = torch.randint(0, V, (B, S))
        assert model(tgt=x).shape == (B, S, V)

    def test_flexitransformer_encoder_only_inferred(self) -> None:
        # src_vocab only → encoder-only
        model = flexit.FlexiTransformer(src_vocab=V, d_model=64, n_heads=4, n_layers=2, d_ff=128)
        x = torch.randint(0, V, (B, S))
        # encoder-only forward: model(input_ids) → [B, S, d_model]
        assert model(x).shape == (B, S, 64)

    def test_transformermodel_alias(self) -> None:
        assert flexit.TransformerModel is flexit.FlexiTransformer
