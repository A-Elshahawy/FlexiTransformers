[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://badge.fury.io/py/flexitransformers.svg)](https://pypi.org/project/flexitransformers/) [![Documentation Status](https://readthedocs.org/projects/flexitransformers/badge/?version=latest)](https://flexitransformers.readthedocs.io/en/latest/?badge=latest) [![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**FlexiTransformers** is a Python library designed to provide a flexible and extensible foundation for building and experimenting with various Transformer architectures. Whether you're interested in classic Encoder-Decoder models, Encoder-only models like BERT, or Decoder-only models like GPT, FlexiTransformers offers the building blocks and abstractions to get you started quickly and customize deeply.

## ✨ Features

- **Modular Architecture:**  Components like attention mechanisms, positional embeddings, and layers are designed to be easily swappable and extensible.
- **Versatile Model Types:** Supports Encoder-Decoder, Encoder-only (BERT-style), and Decoder-only (GPT-style) models within a unified framework.
- **Multiple Attention Mechanisms:** Includes implementations of standard Multi-Head Attention, Rotary Position Embedding (RoPE) Attention, ALiBi Attention, and Relative Global Attention.
- **Flexible Positional Embeddings:** Offers Absolute Positional Encoding, Rotary Positional Encoding, and ALiBi Positional Encoding.
- **Config-Driven Models:** Define model architectures through intuitive `ModelConfig` dataclasses, making it easy to manage and reproduce experiments.
- **Training Utilities:** Provides a `Trainer` class with built-in support for callbacks like checkpointing and early stopping, simplifying the training loop.
- **Rich Progress Monitoring:** Leverages `rich` library for visually appealing and informative training progress bars and summaries.
- **Extensible Callbacks:** Implement custom behaviors during training using the `Callback` base class, for logging, visualization, and more.
- **Easy Inference:**  Includes `predict` methods for straightforward model inference and generation.
- **Clear and Commented Code:**  Designed for readability and ease of understanding, making it suitable for both research and educational purposes.

## 🛠️ Installation

You can install FlexiTransformers via pip:

```bash
pip install flexitransformers
```

**Optional dependencies:**

* **For enhanced console output and progress bars,** **rich** **is recommended and will be installed automatically with** **pip install flexitransformers**.

## 🚀 Usage

**Here are basic examples to get you started with FlexiTransformers.**

### 1. Building a Transformer Model (Encoder-Decoder)

```python
from flexit.models import FlexiTransformer

# Define model configuration
config = {
    'src_vocab': 32000,  # Source vocabulary size
    'tgt_vocab': 32000,  # Target vocabulary size
    'd_model': 512,      # Model dimension
    'n_heads': 8,        # Number of attention heads
    'n_layers': 6,       # Number of encoder and decoder layers
    'dropout': 0.1,      # Dropout probability
    'pe_type': 'absolute' # Positional encoding type ('absolute', 'alibi', 'rotary', 'relative')
}

# Create a FlexiTransformer model
model = FlexiTransformer(**config)

print(model) # Print model configuration
```

### 2. Building a BERT-style Model (Encoder-only)

```python
from flexit.models import FlexiBERT

# Define BERT-style model configuration
config_bert = {
    'src_vocab': 32000,    # Vocabulary size
    'num_classes': 2,      # Number of classes for classification
    'd_model': 768,        # Model dimension
    'n_heads': 12,         # Number of attention heads
    'n_layers': 12,        # Number of encoder layers
    'dropout': 0.1,        # Dropout probability
    'pe_type': 'alibi'     # Positional encoding type
}

# Create a FlexiBERT model
bert_model = FlexiBERT(**config_bert)

print(bert_model)
```

### 3. Building a GPT-style Model (Decoder-only)

```python
from flexit.models import FlexiGPT

# Define GPT-style model configuration
config_gpt = {
    'tgt_vocab': 32000,    # Vocabulary size
    'd_model': 512,        # Model dimension
    'n_heads': 8,          # Number of attention heads
    'n_layers': 12,        # Number of decoder layers
    'dropout': 0.1,        # Dropout probability
    'pe_type': 'rotary'    # Positional encoding type
}

# Create a FlexiGPT model
gpt_model = FlexiGPT(**config_gpt)

print(gpt_model)
```

### 4. Training a Model with the **Trainer**

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from flexit.models import FlexiTransformer
from flexit.train import Trainer, Batch, LossCompute, LabelSmoothing, lr_step
from flexit.callbacks import CheckpointCallback, EarlyStoppingCallback

# 1. Define Model and Config (as in example 1)
config = {
    'src_vocab': 32000,
    'tgt_vocab': 32000,
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 2, # Reduced layers for quick example
    'dropout': 0.1,
    'pe_type': 'absolute'
}
model = FlexiTransformer(**config)

# 2. Create Dummy DataLoaders
src_data = torch.randint(1, config['src_vocab'], (100, 32)) # 100 samples, sequence length 32
tgt_data = torch.randint(1, config['tgt_vocab'], (100, 32))
train_dataset = TensorDataset(src_data, tgt_data)
train_dataloader = DataLoader(train_dataset, batch_size=32)
val_dataloader = DataLoader(train_dataset, batch_size=32) # Using same data for simplicity

# 3. Define Loss, Optimizer, Scheduler
criterion = LabelSmoothing(size=config['tgt_vocab'], padding_idx=0, smoothing=0.1)
loss_compute = LossCompute(model.generator, criterion)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                        lambda step: lr_step(step, config['d_model'], factor=1.0, warmup=4000))


# 4. Define Callbacks
checkpoint_callback = CheckpointCallback(save_best=True, keep_last=2, checkpoint_dir="checkpoints")
early_stopping_callback = EarlyStoppingCallback(patience=3)

# 5. Initialize and Run Trainer
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  loss_fn=loss_compute,
                  train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader,
                  callbacks=[checkpoint_callback, early_stopping_callback])

metrics = trainer.fit(epochs=5) # Train for 5 epochs
print(metrics.to_dict()) # Access training metrics
```

### 5. Inference/Prediction

```python
# Assuming you have a trained 'model' and input data 'src_input' and 'src_mask_input'
# For Encoder-Decoder models:
predictions = model.predict(src=src_input, src_mask=src_mask_input, max_len=50, start_symbol=1)

# For Encoder-only models (classification):
class_probabilities = model(src_input, src_mask_input)
predicted_classes = torch.argmax(class_probabilities, dim=-1)

# For Decoder-only models (text generation - input can be None for initial prompt):
generated_sequence = model.predict(src=None, max_len=100, start_symbol=2)
```

## 📚 Modules and Components

**FlexiTransformers is structured into the following key modules:**

* **attention.py**: Contains implementations of various attention mechanisms (**AbsoluteMultiHeadedAttention**, **RotaryMultiHeadAttention**, **ALiBiMultiHeadAttention**, **RelativeGlobalAttention**).
* **callbacks.py**: Defines callback classes for training events (**Callback**, **CheckpointCallback**, **EarlyStoppingCallback**).
* **configs.py**: Includes **ModelConfig** **dataclass for model configuration and** **ConfigDescriptor** **for managing configurations.**
* **core.py**: Implements core transformer components (**EncoderDecoder**, **Encoder**, **Decoder**, **Generator**).
* **factory.py**: Provides **TransformerFactory** **for creating models based on** **ModelConfig**.
* **layers.py**: Defines fundamental layers like **LayerNorm**, **SublayerConnection**, **PositionwiseFeedForward**, **EncoderLayer**, **DecoderLayer**, and **Embeddings**.
* **loss.py**: Contains loss function implementations (**LabelSmoothing**, **LossCompute**, **BertLoss**).
* **models.py**: Defines base and flexible transformer model classes (**BaseTransformer**, **FlexiTransformer**, **FlexiBERT**, **FlexiGPT**, **TransformerModel**).
* **models_heads.py**: Includes decoding strategies (**greedy_decode**) and model heads like **BertHead**.
* **pos_embeddings.py**: Implements positional encoding layers (**AbsolutePositionalEncoding**, **RotaryPositionalEncoding**, **ALiBiPositionalEncoding**).
* **train.py**: Provides training utilities like **Batch**, **TrainState**, **run_epoch**, and the **Trainer** **class.**
* **utils.py**: Includes utility functions such as **clone** **and** **subsequent_mask**.

## ⚙️ Customization

**FlexiTransformers is designed for customization at various levels:**

* **Configuration:** **Modify** **ModelConfig** **to adjust model hyperparameters, attention types, positional encodings, and more.**
* **Layers:** **Extend or replace existing layers in** **layers.py** **to experiment with new architectures.**
* **Attention Mechanisms:** **Implement custom attention mechanisms in** **attention.py** **by subclassing** **AbstractAttention**.
* **Callbacks:** **Create custom callbacks in** **callbacks.py** **to add specific logging, visualization, or control logic to your training process.**
* **Training Loop:** **Customize the** **run_epoch** **function or the** **Trainer** **class for advanced training procedures.**

## 🤝 Contributing

**Contributions are welcome! If you'd like to contribute to FlexiTransformers, please:**

* **Fork the repository.**
* **Create a new branch for your feature or bug fix.**
* **Make your changes and ensure tests pass (if applicable).**
* **Submit a pull request with a clear description of your changes.**

**Please follow the existing code style and conventions.**

## 📜 License

**FlexiTransformers is released under the** [MIT License](https://opensource.org/license/MIT).

## 🙏 Acknowledgments

**This library draws inspiration from various resources and implementations of Transformer models, including:**

* **The "Attention is All You Need" paper and its associated implementations.**
* **Hugging Face's Transformers library.**

**We acknowledge and appreciate the work of the open-source community in making these resources available.**

---

**Let's build amazing things with FlexiTransformers!** **🚀**
