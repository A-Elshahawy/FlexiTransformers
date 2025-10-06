# FlexiTransformers

[![FlexiTransformers Logo](docs/_static/new_logo.png)](https://github.com/A-Elshahawy/flexitransformers)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://badge.fury.io/py/flexitransformers.svg)](https://pypi.org/project/flexitransformers/0.2.1/) [![Python 3.10+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1%2B-red.svg)](https://pytorch.org/) [![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://a-elshahawy.github.io/FlexiTransformers/)[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) ![mypy](https://img.shields.io/badge/mypy-type%20checked-blue) ![pre-commit](https://img.shields.io/badge/pre--commit-enabled-success)

A modular transformer framework for educational purposes, enabling flexible experimentation with encoder-decoder, encoder-only (BERT-style), and decoder-only (GPT-style) architectures.

> **Note:** This library is primarily designed for educational purposes and research experimentation. For production use cases, consider mature frameworks like Hugging Face Transformers.

## Features

| Feature                        | Support                                                       |
| ------------------------------ | ------------------------------------------------------------- |
| **Model Types**          | Encoder-Decoder, Encoder-Only, Decoder-Only                   |
| **Attention Mechanisms** | Absolute, ALiBi, Relative (Transformer-XL), Rotary (RoFormer) |
| **Positional Encodings** | Absolute (sinusoidal), ALiBi, Rotary                          |
| **Normalization**        | Pre-norm, Post-norm                                           |
| **Training Utilities**   | Built-in Trainer, Callbacks, Learning rate scheduling         |
| **Custom Architectures** | Full configuration control                                    |

## Installation

**Requirements:**

* Python 3.11+
* PyTorch 2.0.1+

### Via pip

```bash
pip install flexitransformers
```

### From source

```bash
git clone https://github.com/A-Elshahawy/flexitransformers.git
cd flexitransformers
pip install -e .
```

**Import the library as `flexit` in your code.**

## Quick Start

### 1. Encoder-Decoder (Seq2Seq Translation)

```python
import torch
from flexit.models import FlexiTransformer
from flexit.utils import subsequent_mask

# Define model configuration
model = FlexiTransformer(
    model_type='encoder-decoder',
    src_vocab=10000,
    tgt_vocab=10000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    dropout=0.1,
    pe_type='absolute'  # or 'alibi', 'rotary'
)

# Create sample data
batch_size, seq_len = 32, 64
src = torch.randint(0, 10000, (batch_size, seq_len))
tgt = torch.randint(0, 10000, (batch_size, seq_len))

# Create masks (assuming 0 is padding)
src_mask = (src != 0).unsqueeze(-2)
tgt_mask = (tgt != 0).unsqueeze(-2) & subsequent_mask(tgt.size(-1))

# Forward pass
output = model(src, tgt, src_mask, tgt_mask)
print(f"Output shape: {output.shape}")  # [32, 64, 512]
```

### 2. Encoder-Only (BERT-style Classification)

```python
from flexit.models import FlexiBERT

# BERT-style model for binary classification
model = FlexiBERT(
    src_vocab=30000,
    num_classes=2,
    d_model=768,
    n_heads=12,
    n_layers=12,
    pe_type='alibi',  # ALiBi works well for BERT-style models
    dropout=0.1
)

# Input data
input_ids = torch.randint(0, 30000, (32, 128))
attention_mask = (input_ids != 0).unsqueeze(-2)

# Get classification logits
logits = model(input_ids, attention_mask)
print(f"Logits shape: {logits.shape}")  # [32, 2]
```

### 3. Decoder-Only (GPT-style Language Model)

```python
from flexit.models import FlexiGPT

# GPT-style autoregressive model
model = FlexiGPT(
    tgt_vocab=50000,
    d_model=768,
    n_heads=12,
    n_layers=12,
    pe_type='rotary',  # Rotary embeddings work well for GPT-style
    dropout=0.1
)

# Input sequence
input_ids = torch.randint(0, 50000, (32, 128))
tgt_mask = subsequent_mask(input_ids.size(-1))

# Forward pass
output = model(input_ids, tgt_mask)
print(f"Output shape: {output.shape}")  # [32, 128, 768]
```

## Training

### Basic Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from flexit.train import Trainer, Batch
from flexit.loss import LossCompute
from flexit.callbacks import CheckpointCallback, EarlyStoppingCallback

# Prepare your data
train_loader = DataLoader(your_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(your_val_dataset, batch_size=64)

# Setup training components
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
loss_compute = LossCompute(
    generator=model.generator,
    criterion=criterion,
    model=model,
    grad_clip=1.0
)

optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98))
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min((step + 1) ** -0.5, (step + 1) * 4000 ** -1.5)
)

# Initialize trainer with callbacks
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_compute,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    callbacks=[
        CheckpointCallback(save_best=True, keep_last=3),
        EarlyStoppingCallback(patience=5, min_delta=0.001)
    ]
)

# Train the model
metrics = trainer.fit(epochs=20)
print(metrics.to_dict())
```

### Custom Batch Handling

```python
from flexit.train import Batch

# For decoder-only models (GPT-style)
batch = Batch(
    tgt=sequence_tensor,  # [batch_size, seq_len]
    model_type='decoder-only',
    pad=0
)

# For encoder-only models (BERT-style)
batch = Batch(
    src=input_tensor,
    labels=label_tensor,
    model_type='encoder-only',
    pad=0
)

# For encoder-decoder models
batch = Batch(
    src=source_tensor,
    tgt=target_tensor,
    model_type='encoder-decoder',
    pad=0
)
```

## Advanced Configuration

### Comparing Attention Mechanisms

```python
# Experiment with different attention types
configs = {
    'absolute': {'pe_type': 'absolute'},
    'alibi': {'pe_type': 'alibi'},
    'rotary': {'pe_type': 'rotary', 'rope_percentage': 0.5},
    'relative': {'pe_type': 'relative', 'max_len': 1024}
}

for name, config in configs.items():
    model = FlexiTransformer(
        model_type='decoder-only',
        tgt_vocab=10000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        **config
    )
    # Train and evaluate each variant
```

### Asymmetric Encoder-Decoder

```python
# Different layer counts for encoder/decoder
model = FlexiTransformer(
    model_type='encoder-decoder',
    src_vocab=10000,
    tgt_vocab=10000,
    d_model=512,
    n_heads=8,
    n_layers=(12, 6),  # 12 encoder layers, 6 decoder layers
    dropout=0.1
)
```

### Custom Initialization

```python
model = FlexiTransformer(
    src_vocab=10000,
    tgt_vocab=10000,
    init_method='kaiming_uniform',  # or 'xavier_uniform', 'orthogonal'
    ff_activation='gelu',  # or 'relu', 'silu'
    pre_norm=True  # Pre-layer normalization (like GPT)
)
```

## Architecture Variants

### Available Model Classes

* **`FlexiTransformer`** : Base class, fully customizable
* **`FlexiBERT`** : Encoder-only, optimized for classification
* **`FlexiGPT`** : Decoder-only, optimized for generation
* **`TransformerModel`** : Standard encoder-decoder

### Configuration Options

```python
from flexit.configs import ModelConfig

config = ModelConfig(
    model_type='encoder-decoder',  # or 'encoder-only', 'decoder-only'
    src_vocab=10000,
    tgt_vocab=10000,
    d_model=512,          # Model dimension
    d_ff=2048,            # Feed-forward dimension
    n_heads=8,            # Attention heads
    n_layers=6,           # Number of layers (or tuple for asymmetric)
    dropout=0.1,
    pe_type='absolute',   # 'absolute', 'alibi', 'relative', 'rotary'
    pre_norm=True,        # Pre-norm vs post-norm
    ff_activation='relu', # 'relu', 'gelu', 'silu', etc.
    init_method='xavier_uniform'
)
```

## API Reference

**Full documentation:** [https://a-elshahawy.github.io/FlexiTransformers/](https://a-elshahawy.github.io/FlexiTransformers/)

### Key Modules

* **`flexit.models`** : Model classes (`FlexiTransformer`, `FlexiBERT`, `FlexiGPT`)
* **`flexit.attention`** : Attention mechanisms (Absolute, ALiBi, Relative, Rotary)
* **`flexit.train`** : Training utilities (`Trainer`, `Batch`, `LossCompute`)
* **`flexit.callbacks`** : Training callbacks (`CheckpointCallback`, `EarlyStoppingCallback`)
* **`flexit.configs`** : Configuration classes (`ModelConfig`)
* **`flexit.loss`** : Loss functions (`LabelSmoothing`, `BertLoss`)

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes with tests
4. Run tests and type checking (`mypy`, `ruff`)
5. Submit a pull request

For major changes, open an issue first to discuss the proposed changes.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use FlexiTransformers in your research, please cite:

```bibtex
@software{flexitransformers2024,
  author = {Elshahawy, Ahmed},
  title = {FlexiTransformers: A Modular Transformer Framework},
  year = {2024},
  url = {https://github.com/A-Elshahawy/flexitransformers}
}
```

## References

This library implements concepts from:

* Vaswani et al. (2017) - "Attention is All You Need"
* Press et al. (2021) - "Train Short, Test Long: Attention with Linear Biases" (ALiBi)
* Su et al. (2021) - "RoFormer: Enhanced Transformer with Rotary Position Embedding"
* Dai et al. (2019) - "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

## Contact

**Ahmed Elshahawy**

* GitHub: [@A-Elshahawy](https://github.com/A-Elshahawy)
* LinkedIn: [Ahmed Elshahawy](https://www.linkedin.com/in/ahmed-elshahawy-a42149218/)
* Email: ahmedelshahawy078@gmail.com

---

**Links:**

* [Documentation](https://a-elshahawy.github.io/FlexiTransformers/)
* [PyPI Package](https://pypi.org/project/flexitransformers/)
* [GitHub Repository](https://github.com/A-Elshahawy/flexitransformers)
