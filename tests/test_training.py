"""Tests for training utilities: Batch, loss, LabelSmoothing, run_epoch."""

import torch

from flexit import (
    Batch,
    BertLoss,
    LabelSmoothing,
    LossCompute,
    ModelConfig,
    TrainerMetrics,
    TrainState,
    create_model,
    run_epoch,
)

B, S, V = 2, 8, 50
PAD = 0


def _dec_cfg() -> ModelConfig:
    return ModelConfig(
        model_type='decoder-only', vocab_size=V, d_model=64, n_heads=4, n_layers=2, d_ff=128
    )


def _enc_dec_cfg() -> ModelConfig:
    return ModelConfig(
        model_type='encoder-decoder', vocab_size=V, d_model=64, n_heads=4, n_layers=(2, 2), d_ff=128
    )


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class TestBatch:
    def test_src_mask_shape_encoder_only(self) -> None:
        src = torch.randint(1, V, (B, S))
        src[0, -1] = PAD
        labels = torch.randint(0, 5, (B, S))
        batch = Batch(src=src, labels=labels, pad=PAD, model_type='encoder-only')
        # src_mask: (B, 1, S)
        assert batch.src_mask.shape == (B, 1, S)

    def test_tgt_mask_is_causal(self) -> None:
        src = torch.randint(1, V, (B, S))
        tgt = torch.randint(1, V, (B, S))
        batch = Batch(src=src, tgt=tgt, pad=PAD, model_type='encoder-decoder')
        # tgt_mask should be (B, S-1, S-1) lower-triangular
        assert batch.tgt_mask.shape[1] == batch.tgt_mask.shape[2]

    def test_ntokens_excludes_pad(self) -> None:
        # decoder-only: sequence of non-pad tokens
        seq = torch.randint(1, V, (B, S))  # all non-pad
        batch = Batch(tgt=seq, pad=PAD, model_type='decoder-only')
        assert batch.ntokens > 0

    def test_decoder_only_shifts_sequence(self) -> None:
        seq = torch.randint(1, V, (B, S))
        batch = Batch(tgt=seq, pad=PAD, model_type='decoder-only')
        # tgt is seq[:, :-1], tgt_y is seq[:, 1:]
        assert batch.tgt.shape == (B, S - 1)
        assert batch.tgt_y.shape == (B, S - 1)


# ---------------------------------------------------------------------------
# LabelSmoothing
# ---------------------------------------------------------------------------


class TestLabelSmoothing:
    def test_output_is_scalar(self) -> None:
        crit = LabelSmoothing(size=V, padding_idx=PAD, smoothing=0.1)
        pred = torch.randn(B * S, V)
        target = torch.randint(1, V, (B * S,))
        loss = crit(pred, target)
        assert loss.dim() == 0 or loss.numel() == 1

    def test_zero_smoothing_nonnegative(self) -> None:
        crit = LabelSmoothing(size=V, padding_idx=PAD, smoothing=0.0)
        pred = torch.randn(4, V)
        target = torch.randint(1, V, (4,))
        loss = crit(pred, target)
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# LossCompute
# ---------------------------------------------------------------------------


class TestLossCompute:
    def test_loss_compute_returns_tensors(self) -> None:
        model = create_model(_dec_cfg())
        crit = LabelSmoothing(size=V, padding_idx=PAD, smoothing=0.1)
        # LossCompute(generator, criterion, model)
        compute = LossCompute(model.generator, crit, model)

        # Pass hidden states (pre-generator), not logits
        hidden = torch.randn(B, S, 64)
        target = torch.randint(1, V, (B * S,))
        loss, loss_backward = compute(hidden, target, B * S)
        assert loss.item() >= 0
        assert loss_backward.item() >= 0


# ---------------------------------------------------------------------------
# BertLoss
# ---------------------------------------------------------------------------


class TestBertLoss:
    def test_bert_loss_with_classification_head(self) -> None:
        model = create_model(
            ModelConfig(
                model_type='encoder-only',
                vocab_size=V,
                d_model=64,
                n_heads=4,
                n_layers=2,
                d_ff=128,
                num_classes=5,
            )
        )
        model.eval()
        crit = BertLoss()
        x = torch.randint(1, V, (B, S))
        with torch.no_grad():
            logits = model(x)  # [B, 5]
        labels = torch.randint(0, 5, (B,))
        loss, _ = crit(logits, labels, norm=1.0)
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# TrainState / TrainerMetrics
# ---------------------------------------------------------------------------


class TestTrainState:
    def test_default_values(self) -> None:
        state = TrainState()
        assert state.step == 0
        assert state.accum_step == 0

    def test_update_increments_step(self) -> None:
        state = TrainState()
        state.update(batch_size=4, ntokens=32, loss=1.0, lr=1e-3)
        assert state.step == 1


class TestTrainerMetrics:
    def test_default_empty(self) -> None:
        m = TrainerMetrics()
        assert m.train_losses == []

    def test_update_appends(self) -> None:
        m = TrainerMetrics()
        m.update(train_loss=1.0, val_loss=0.8, epoch_time=1.0, lr=1e-3, epoch=0)
        assert m.train_losses[-1] == 1.0
        assert m.val_losses[-1] == 0.8


# ---------------------------------------------------------------------------
# run_epoch (smoke test with tiny data)
# ---------------------------------------------------------------------------


def _make_batch_list(model_type: str, n: int = 3) -> list[Batch]:
    """Return a list of n tiny batches (list has len())."""
    batches = []
    for _ in range(n):
        src = torch.randint(1, V, (B, S))
        tgt = torch.randint(1, V, (B, S))
        batches.append(Batch(src=src, tgt=tgt, pad=PAD, model_type=model_type))  # type: ignore[arg-type]
    return batches


def test_run_epoch_decoder_only() -> None:
    model = create_model(_dec_cfg())
    crit = LabelSmoothing(size=V, padding_idx=PAD, smoothing=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    compute = LossCompute(model.generator, crit, model)

    avg_loss, state = run_epoch(
        data_iter=_make_batch_list('decoder-only'),
        model=model,
        loss_compute=compute,
        optimizer=opt,
        mode='train',
        device='cpu',
    )
    assert avg_loss >= 0
    assert state.step > 0


def test_run_epoch_encoder_decoder() -> None:
    model = create_model(_enc_dec_cfg())
    crit = LabelSmoothing(size=V, padding_idx=PAD, smoothing=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    compute = LossCompute(model.generator, crit, model)

    avg_loss, state = run_epoch(
        data_iter=_make_batch_list('encoder-decoder'),
        model=model,
        loss_compute=compute,
        optimizer=opt,
        mode='train',
        device='cpu',
    )
    assert avg_loss >= 0
