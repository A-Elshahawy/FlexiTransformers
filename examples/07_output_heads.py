"""
Output heads — LMHead with weight tying, BertHead, pooled heads, token heads.

Imports demonstrated:
    from flexit import LMHead, BertHead, SequenceClassificationHead, TokenClassificationHead
    from flexit import ModelConfig, create_model
"""

import torch

from flexit import (
    BertHead,
    LMHead,
    ModelConfig,
    SequenceClassificationHead,
    TokenClassificationHead,
    create_model,
)

D, V, NC = 128, 4_000, 10

# --- LMHead with weight tying ---
enc = create_model(
    ModelConfig(
        model_type='encoder-only',
        vocab_size=V,
        d_model=D,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        pe_type='absolute',
    )
)
lm_head = LMHead(D, V, bias=False)
lm_head.tie_weights(enc.embed.embed.embed)  # share embedding matrix

src = torch.randint(0, V, (2, 16))
hidden = enc(src)  # [2, 16, D]
vocab_logits = lm_head(hidden)
print(f'LMHead output (tied)         : {tuple(vocab_logits.shape)}')  # (2, 16, 4000)

# Verify tying — same weight object
tied = lm_head.proj.weight.data_ptr() == enc.embed.embed.embed.weight.data_ptr()
print(f'Weights shared               : {tied}')

# --- BertHead (CLS-token classification) ---
bert_head = BertHead(D, num_classes=NC, pre_norm=True, activation='gelu')
seq_logits = bert_head(hidden)
print(f'BertHead logits              : {tuple(seq_logits.shape)}')  # (2, 10)

# Supported activations
for act in ('gelu', 'relu', 'tanh', 'silu'):
    h = BertHead(D, num_classes=NC, activation=act)(hidden)
    print(f'  BertHead(activation={act:<8}) : {tuple(h.shape)}')

# --- SequenceClassificationHead ---
mean_head = SequenceClassificationHead(D, num_classes=NC, pooling='mean')
max_head = SequenceClassificationHead(D, num_classes=NC, pooling='max')
print(f'SequenceHead mean-pool       : {tuple(mean_head(hidden).shape)}')
print(f'SequenceHead max-pool        : {tuple(max_head(hidden).shape)}')

# --- TokenClassificationHead ---
tok_head = TokenClassificationHead(D, num_classes=NC)
tok_logits = tok_head(hidden)
print(f'TokenClassificationHead      : {tuple(tok_logits.shape)}')  # (2, 16, 10)
