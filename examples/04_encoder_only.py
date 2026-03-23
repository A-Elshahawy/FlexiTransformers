"""
Encoder-only (BERT-style) models — classification and token labelling.

Imports demonstrated:
    from flexit import ModelConfig, create_model
    from flexit import BertHead, SequenceClassificationHead, TokenClassificationHead
"""

import torch

from flexit import (
    BertHead,
    ModelConfig,
    SequenceClassificationHead,
    TokenClassificationHead,
    create_model,
)

VOCAB, D, NC = 5_000, 128, 5
src = torch.randint(0, VOCAB, (2, 32))

# --- Built-in classification head via num_classes ---
clf_model = create_model(
    ModelConfig(
        model_type='encoder-only',
        vocab_size=VOCAB,
        d_model=D,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        pe_type='absolute',
        num_classes=NC,
    )
)
logits = clf_model(src)
print(f'Built-in classification head : {tuple(logits.shape)}')  # (2, 5)

# --- Encoder backbone (no head) ---
enc = create_model(
    ModelConfig(
        model_type='encoder-only',
        vocab_size=VOCAB,
        d_model=D,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        pe_type='absolute',
    )
)
hidden = enc(src)  # [B, S, D]
print(f'Encoder hidden states        : {tuple(hidden.shape)}')

# --- BertHead: CLS-token -> dense -> act -> norm -> dropout -> classifier ---
bert_head = BertHead(D, num_classes=NC, pre_norm=True, activation='gelu')
seq_logits = bert_head(hidden)
print(f'BertHead (CLS-pool)          : {tuple(seq_logits.shape)}')  # (2, 5)

# --- SequenceClassificationHead: mean or max pooling ---
mean_head = SequenceClassificationHead(D, num_classes=NC, pooling='mean')
max_head = SequenceClassificationHead(D, num_classes=NC, pooling='max')
print(f'Mean-pool head               : {tuple(mean_head(hidden).shape)}')
print(f'Max-pool head                : {tuple(max_head(hidden).shape)}')

# With a padding mask (True = real token)
mask = torch.ones(2, 32, dtype=torch.bool)
mask[0, 20:] = False  # first example has 20 real tokens
print(f'Mean-pool (masked)           : {tuple(mean_head(hidden, mask=mask).shape)}')

# --- TokenClassificationHead: per-position ---
tok_head = TokenClassificationHead(D, num_classes=NC)
tok_logits = tok_head(hidden)
print(f'Token classification head    : {tuple(tok_logits.shape)}')  # (2, 32, 5)
