"""
Saving and loading models.

Imports demonstrated:
    from flexit import create_model, ModelConfig
    from flexit import DecoderOnlyModel, EncoderOnlyModel, EncoderDecoderModel
"""

import os
import tempfile

import torch

from flexit import (
    DecoderOnlyModel,
    EncoderOnlyModel,
    ModelConfig,
    create_model,
)

# --- Save / load a decoder-only model ---
lm = create_model(
    ModelConfig(
        model_type='decoder-only',
        vocab_size=4_000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        pe_type='rotary',
    )
)

with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    path = f.name

lm.save(path)
loaded = DecoderOnlyModel.load(path, map_location='cpu')
os.unlink(path)

lm.eval()
loaded.eval()
x = torch.randint(0, 4_000, (1, 8))
with torch.no_grad():
    before = lm(tgt=x)
    after = loaded(tgt=x)

print(f'Decoder-only weights match : {torch.allclose(before, after, atol=1e-5)}')

# --- Save / load an encoder-only model ---
enc = create_model(
    ModelConfig(
        model_type='encoder-only',
        vocab_size=5_000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        pe_type='absolute',
    )
)

with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    path = f.name

enc.save(path)
loaded_enc = EncoderOnlyModel.load(path, map_location='cpu')
os.unlink(path)

enc.eval()
loaded_enc.eval()
src = torch.randint(0, 5_000, (1, 16))
with torch.no_grad():
    h1 = enc(src)
    h2 = loaded_enc(src)

print(f'Encoder-only weights match : {torch.allclose(h1, h2, atol=1e-5)}')

# --- Inspect saved checkpoint manually ---
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    path = f.name

lm.save(path)
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
os.unlink(path)

print(f'\nCheckpoint keys            : {list(checkpoint.keys())}')
print(f'Config in checkpoint       : {checkpoint["config"]}')
print(f'State dict keys (first 3)  : {list(checkpoint["state_dict"].keys())[:3]}')
