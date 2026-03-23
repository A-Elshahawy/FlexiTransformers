"""
Encoder-Decoder (T5/BART-style) seq2seq training loop.

Imports demonstrated:
    from flexit import ModelConfig, create_model
    from flexit import Batch, LabelSmoothing, LossCompute, run_epoch
"""

import torch

from flexit import (
    Batch,
    LabelSmoothing,
    LossCompute,
    ModelConfig,
    create_model,
    run_epoch,
)

SRC_V, TGT_V = 6_000, 6_000

model = create_model(
    ModelConfig(
        model_type='encoder-decoder',
        src_vocab_size=SRC_V,
        tgt_vocab_size=TGT_V,
        d_model=128,
        n_heads=4,
        n_layers=(2, 2),
        d_ff=512,
        pe_type='rotary',
        tie_word_embeddings=False,
    )
)
print(f'Model: {model}\n')

criterion = LabelSmoothing(size=TGT_V, padding_idx=0, smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98))
loss_fn = LossCompute(model.generator, criterion, model)  # type: ignore


def make_batches(n: int = 4) -> list[Batch]:
    batches = []
    for _ in range(n):
        src = torch.randint(1, SRC_V, (2, 20))
        tgt = torch.randint(1, TGT_V, (2, 20))
        batches.append(Batch(src=src, tgt=tgt, pad=0, model_type='encoder-decoder'))
    return batches


# Training epoch
avg_loss, state = run_epoch(
    data_iter=make_batches(),  # type: ignore
    model=model,
    loss_compute=loss_fn,
    optimizer=optimizer,
    mode='train',
    device='cpu',
)
print(f'Train avg loss : {avg_loss:.4f}  (steps={state.step})')

# Eval epoch (no grad, no optimizer step)
avg_loss_eval, _ = run_epoch(
    data_iter=make_batches(),  # type: ignore
    model=model,
    loss_compute=loss_fn,
    optimizer=None,
    mode='eval',
    device='cpu',
)
print(f'Eval  avg loss : {avg_loss_eval:.4f}')

# --- Manual forward pass ---
model.eval()
src = torch.randint(1, SRC_V, (1, 10))
tgt = torch.randint(1, TGT_V, (1, 6))
with torch.no_grad():
    logits = model(src=src, tgt=tgt)
print(f'\nManual forward : src={tuple(src.shape)} tgt={tuple(tgt.shape)} -> {tuple(logits.shape)}')
