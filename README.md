# FlexiTransformers: A Modular Transformer Library

FlexiTransformers is a flexible and modular library for building and experimenting with Transformer models. It supports various attention mechanisms and positional encoding strategies, making it ideal for research, experimentation, and production use cases. The library is designed to be easily extensible, allowing users to customize and extend the Transformer architecture to suit their needs.

---

[&lt;]()

## Features

- **Multiple Attention Mechanisms**:

  - **Absolute Multi-Headed Attention**: Standard self-attention with absolute positional encoding.
  - **Relative Global Attention**: Implements relative positional encoding for better handling of sequence positions.
  - **Rotary Multi-Head Attention**: Uses rotary positional embeddings for improved performance on long sequences.
  - **ALiBi Multi-Head Attention**: Implements Attention with Linear Biases (ALiBi) for efficient handling of sequence lengths.
- **Positional Encodings**:

  - **Absolute Positional Encoding**: Standard sinusoidal positional encoding.
  - **Rotary Positional Encoding**: Implements rotary embeddings for better handling of relative positions.
  - **ALiBi Positional Encoding**: Adds linear biases to attention scores for efficient sequence modeling.
- **Modular Architecture**:

  - Easily configurable encoder-decoder architecture.
  - Support for custom layers, attention mechanisms, and positional encodings.
- **Training Utilities**:

  - Batch handling and masking utilities.
  - Learning rate scheduling and gradient accumulation.
  - Progress tracking and logging during training.
- **Extensible**:

  - Designed to be easily extended with new attention mechanisms, positional encodings, and other components.

---

## Installation

To install FlexiTransformers, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/FlexiTransformers.git
cd FlexiTransformers
pip install -r requirements.txt
```

---

## Quick Start

### Creating a Model

You can create a Transformer model using the `make_model` function from the `models` module. Here's an example of creating a model with absolute positional encoding:

```python
from models import make_model

# Create a model with absolute positional encoding
model = make_model(
    src_vocab=10000,  # Source vocabulary size
    tgt_vocab=10000,  # Target vocabulary size
    n_layers=6,       # Number of encoder/decoder layers
    d_model=512,      # Model dimension
    n_heads=8,        # Number of attention heads
    dropout=0.1,      # Dropout rate
    positional_encoding='absolute'  # Positional encoding type
)
```

### Training the Model

To train the model, use the provided training utilities. Here's an example of running a training epoch:

```python
from training import run_epoch, TrainState
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

# Define optimizer and scheduler
optimizer = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_step(step, model_size=512, factor=1.0, warmup=4000))

# Run training epoch
train_state = TrainState()
loss, train_state = run_epoch(
    data_iter,          # Data iterator
    model,              # Model to train
    loss_compute,       # Loss computation function
    optimizer,          # Optimizer
    scheduler,          # Learning rate scheduler
    mode='train',       # Training mode
    train_state=train_state  # Training state tracker
)
```

---

## Components

### Core Modules

- **EncoderDecoder**: The main Transformer architecture, combining an encoder and decoder.
- **Encoder**: Implements the encoder stack with multiple layers.
- **Decoder**: Implements the decoder stack with multiple layers.
- **Generator**: Linear layer followed by softmax for output generation.

### Attention Mechanisms

- **AbsoluteMultiHeadedAttention**: Standard multi-headed self-attention with absolute positional encoding.
- **RelativeGlobalAttention**: Implements relative positional encoding for better sequence modeling.
- **RotaryMultiHeadAttention**: Uses rotary positional embeddings for improved performance on long sequences.
- **ALiBiMultiHeadAttention**: Implements Attention with Linear Biases (ALiBi) for efficient sequence handling.

### Positional Encodings

- **AbsolutePositionalEncoding**: Standard sinusoidal positional encoding.
- **RotaryPositionalEncoding**: Implements rotary embeddings for relative positional encoding.
- **ALiBiPositionalEncoding**: Adds linear biases to attention scores for efficient sequence modeling.

### Training Utilities

- **Batch**: Handles batching of source and target sequences with masking.
- **TrainState**: Tracks training progress (steps, tokens, etc.).
- **run_epoch**: Runs a single training or evaluation epoch.
- **lr_step**: Learning rate scheduling function.

### Utilities

- **clone**: Utility function for cloning modules.
- **subsequent_mask**: Creates a mask to hide future positions in the target sequence.

---

## Examples

### Using Relative Global Attention

To create a model with relative global attention, specify `positional_encoding='relative'` when calling `make_model`:

```python
from models import make_model

# Create a model with relative global attention
model = make_model(
    src_vocab=10000,
    tgt_vocab=10000,
    n_layers=6,
    d_model=512,
    n_heads=8,
    dropout=0.1,
    positional_encoding='relative'
)
```

### Using ALiBi Positional Encoding

To use ALiBi positional encoding, specify `positional_encoding='alibi'`:

```python
from models import make_model

# Create a model with ALiBi positional encoding
model = make_model(
    src_vocab=10000,
    tgt_vocab=10000,
    n_layers=6,
    d_model=512,
    n_heads=8,
    dropout=0.1,
    positional_encoding='alibi'
)
```

---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and ensure tests pass.
4. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- This library is inspired by the original Transformer paper: [Attention is All You Need](https://arxiv.org/abs/1706.03762).
- Special thanks to the authors of ALiBi, Rotary Positional Embeddings, and other positional encoding techniques for their contributions to the field.

---

## Contact

* You can find me on :[
  ](https://www.linkedin.com/in/ahmed-elshahawy-a42149218/)
* [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ahmed-elshahawy-a42149218/)
* [![Gmail](https://img.shields.io/badge/Gmail-Email-red?style=flat&logo=gmail)](mailto:ahmedelshahawy078@gmail.com)

For questions or feedback, please open an issue on GitHub or contact the maintainers directly.

---

Happy modeling with FlexiTransformers! 🚀
