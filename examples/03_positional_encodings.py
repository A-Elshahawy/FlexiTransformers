"""
All supported positional encoding types.

Imports demonstrated:
    from flexit import ModelConfig, create_model
    from flexit import SinusoidalPE, LearnedPE, RotaryPE, ALiBiPE, RelativePE, RelativePEWithBias
    from flexit import create_pe
"""

import torch

from flexit import (
    ALiBiPE,
    LearnedPE,
    ModelConfig,
    RelativePE,
    RelativePEWithBias,
    RotaryPE,
    SinusoidalPE,
    create_model,
    create_pe,
)

B, S, V = 2, 16, 1_000

# --- Run each PE type end-to-end through a decoder-only model ---
pe_types = ['absolute', 'learned', 'rotary', 'alibi', 'relative', 'relative_bias', 'none']

print('End-to-end decoder-only pass:')
for pe in pe_types:
    cfg = ModelConfig(
        model_type='decoder-only',
        vocab_size=V,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        pe_type=pe,  # type: ignore
    )
    m = create_model(cfg)
    out = m(tgt=torch.randint(0, V, (B, S)))
    print(f'  pe_type={pe:<14} -> {tuple(out.shape)}')

# --- Instantiate PE objects directly ---
print('\nDirect PE construction:')

sin_pe = SinusoidalPE(d_model=64, max_len=512, dropout=0.0)
x = torch.zeros(B, S, 64)
out = sin_pe.apply_to_embedding(x)
print(f'  SinusoidalPE      : {tuple(out.shape)}')

lrn_pe = LearnedPE(d_model=64, max_len=512, dropout=0.0)
out = lrn_pe.apply_to_embedding(x)
print(f'  LearnedPE         : {tuple(out.shape)}')

rope = RotaryPE(dim=16, max_len=512)
q = torch.randn(B, 4, S, 16)
k = torch.randn(B, 4, S, 16)
q2, k2 = rope.apply_to_qk(q, k)
print(f'  RotaryPE q        : {tuple(q2.shape)}')

alibi = ALiBiPE(n_heads=4, max_len=512)
scores = torch.zeros(B, 4, S, S)
out = alibi.apply_to_scores(scores, q_len=S, k_len=S)
print(f'  ALiBiPE scores    : {tuple(out.shape)}')

rel = RelativePE(head_dim=16, max_seq_len=512)
q_full = torch.randn(B, 4, S, 16)
out = rel.apply_to_scores(scores, query=q_full)
print(f'  RelativePE scores : {tuple(out.shape)}')

rel_b = RelativePEWithBias(head_dim=16, max_seq_len=512)
out = rel_b.apply_to_scores(scores)
print(f'  RelativePEWithBias: {tuple(out.shape)}')

# --- create_pe factory ---
print('\ncreate_pe factory:')
cfg = ModelConfig(
    model_type='decoder-only',
    vocab_size=V,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=256,
    pe_type='rotary',
)
pe_obj = create_pe(cfg)
print(f"  create_pe('rotary') -> {pe_obj.__class__.__name__}")

cfg_none = ModelConfig(
    model_type='decoder-only',
    vocab_size=V,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=256,
    pe_type='none',
)
pe_none = create_pe(cfg_none)
print(f"  create_pe('none')   -> {pe_none}")
