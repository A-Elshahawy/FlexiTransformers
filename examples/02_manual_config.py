"""
Manual model creation via ModelConfig + create_model.

Imports demonstrated:
    from flexit import ModelConfig, create_model
"""

import torch

from flexit import ModelConfig, create_model

# --- Decoder-only (GPT-style) ---
config = ModelConfig(
    model_type='decoder-only',
    vocab_size=10_000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    pe_type='rotary',
    ff_activation='swiglu',
    norm_type='rmsnorm',
    pre_norm=True,
    dropout=0.1,
    init_method='scaled',
    tie_word_embeddings=True,
)
model = create_model(config)
print(f'Config  : {config}')
print(f'Model   : {model}')

x = torch.randint(0, 10_000, (1, 32))
logits = model(tgt=x)
print(f'Output  : {tuple(logits.shape)}\n')

# --- Encoder-only (BERT-style) ---
bert_config = ModelConfig(
    model_type='encoder-only',
    vocab_size=30_000,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    pe_type='absolute',
    ff_activation='gelu',
    norm_type='layernorm',
    pre_norm=False,
)
bert = create_model(bert_config)
print(f'BERT    : {bert}')

src = torch.randint(0, 30_000, (2, 64))
hidden = bert(src)
print(f'Output  : {tuple(hidden.shape)}\n')

# --- Encoder-Decoder (T5-style) ---
t5_config = ModelConfig(
    model_type='encoder-decoder',
    src_vocab_size=32_000,
    tgt_vocab_size=32_000,
    d_model=512,
    n_heads=8,
    n_layers=(6, 6),
    d_ff=2048,
    pe_type='relative_bias',
    tie_word_embeddings=False,
)
t5 = create_model(t5_config)
print(f'T5      : {t5}')

src = torch.randint(0, 32_000, (2, 40))
tgt = torch.randint(0, 32_000, (2, 30))
logits = t5(src=src, tgt=tgt)
print(f'Output  : {tuple(logits.shape)}')
