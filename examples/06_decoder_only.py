"""
Decoder-only (GPT-style) language model — training and text generation.

Imports demonstrated:
    from flexit import ModelConfig, create_model
    from flexit import Batch, LabelSmoothing, LossCompute, run_epoch
    from flexit import greedy_decode
"""

import torch

from flexit import (
    Batch,
    LabelSmoothing,
    LossCompute,
    ModelConfig,
    create_model,
    greedy_decode,
    run_epoch,
)

LM_V = 4_000

model = create_model(
    ModelConfig(
        model_type='decoder-only',
        vocab_size=LM_V,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        pe_type='rotary',
        ff_activation='swiglu',
    )
)
print(f'Model: {model}\n')

criterion = LabelSmoothing(size=LM_V, padding_idx=0, smoothing=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = LossCompute(model.generator, criterion, model)  # type: ignore


def make_batches(n: int = 4) -> list[Batch]:
    batches = []
    for _ in range(n):
        seq = torch.randint(1, LM_V, (2, 24))
        batches.append(Batch(tgt=seq, pad=0, model_type='decoder-only'))
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

# --- Greedy generation ---
model.eval()
prompt = torch.tensor([[1, 42, 77]])  # [1, 3] token ids
with torch.no_grad():
    generated = greedy_decode(
        model,
        src=prompt,
        src_mask=None,
        max_len=20,
        start_symbol=1,
        end_symbol=2,
    )
print(f'Generated tokens : {generated[0].tolist()}')

# --- Manual next-token prediction ---
x = torch.randint(0, LM_V, (1, 8))
with torch.no_grad():
    logits = model(tgt=x)  # [1, 8, LM_V]
next_token = logits[0, -1].argmax()
print(f'Next-token prediction : token {next_token.item()}')
