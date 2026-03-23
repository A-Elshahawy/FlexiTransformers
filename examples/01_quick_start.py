"""
Quick start with FlexiTransformers convenience constructors.

Imports demonstrated:
    from flexit import FlexiGPT, FlexiBERT, FlexiTransformer
"""

import torch

from flexit import FlexiBERT, FlexiGPT, FlexiTransformer

# GPT-style decoder-only (RoPE, SwiGLU, RMSNorm, pre-norm by default)
gpt = FlexiGPT(vocab_size=10_000, d_model=256, n_heads=4, n_layers=4, d_ff=1024)
print(f'FlexiGPT     : {gpt}')

x = torch.randint(0, 10_000, (2, 16))
logits = gpt(tgt=x)
print(f'  output     : {tuple(logits.shape)}')  # (2, 16, 10000)

# BERT-style encoder-only (absolute PE, GELU, LayerNorm, post-norm by default)
bert = FlexiBERT(vocab_size=10_000, d_model=256, n_heads=4, n_layers=4, d_ff=1024)
print(f'\nFlexiBERT    : {bert}')

src = torch.randint(0, 10_000, (2, 16))
hidden = bert(src)
print(f'  output     : {tuple(hidden.shape)}')  # (2, 16, 256)

# Encoder-Decoder (T5-style) inferred from src_vocab + tgt_vocab
t5 = FlexiTransformer(
    src_vocab=8_000,
    tgt_vocab=8_000,
    d_model=256,
    n_heads=4,
    n_enc=3,
    n_dec=3,
    d_ff=1024,
)
print(f'\nFlexiTransformer (enc-dec): {t5}')

src = torch.randint(0, 8_000, (2, 20))
tgt = torch.randint(0, 8_000, (2, 15))
logits = t5(src=src, tgt=tgt)
print(f'  output     : {tuple(logits.shape)}')  # (2, 15, 8000)
