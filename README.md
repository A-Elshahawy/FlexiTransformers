# FlexiTransformers: Modular Transformer Framework

![FlexiTransformers Logo](docs/_static/logo.png)

---

[
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://badge.fury.io/py/flexitransformers.svg?cache=0)](https://pypi.org/project/flexitransformers/0.1.5/) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1%2B-red.svg)](https://pytorch.org/) [![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://a-elshahawy.github.io/FlexiTransformers/) [![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) ![mypy](https://img.shields.io/badge/mypy-type%20checked-blue) ![pre-commit](https://img.shields.io/badge/pre--commit-enabled-success)

---

**Build, Experiment, and Innovate with Transformers. FlexiTransformers is a Python library designed for maximum flexibility in constructing and training transformer models. Craft encoder-decoder, encoder-only (BERT-style), and decoder-only (GPT-style) architectures with ease, choosing from a variety of attention mechanisms and configurations to power your NLP projects.**

## ✨ Introduction

FlexiTransformers empowers you to explore the vast landscape of transformer architectures with a modular and highly customizable framework. Whether you're tackling machine translation, text classification, language generation, or venturing into novel transformer applications, this library provides the essential building blocks and the freedom to tailor models to your precise needs.

***`This library is primarily designed for educational purposes.`*** It aims to provide a flexible implementation of transformer models to help users understand the underlying concepts and mechanisms of transformer architecture.

**Why FlexiTransformers?**

* **Unparalleled Flexibility:** Seamlessly switch between encoder-decoder, encoder-only, and decoder-only architectures within a unified framework. No more wrestling with different codebases for different model types.
* **Attention Mechanism Playground:**  Experiment with cutting-edge attention mechanisms like Absolute, ALiBi, Relative, and Rotary Attention, directly impacting your model's performance and efficiency.
* **Positional Encoding Choice:** Select the optimal positional encoding strategy for your task – Absolute, ALiBi, or Rotary – or even opt for no positional encoding for specific experiments.
* **Granular Customization:** Fine-tune every aspect of your transformer, from the number of layers and attention heads to hidden dimensions and activation functions, through a clear and intuitive configuration system.
* **Integrated Training Utilities:** Leverage built-in training loops, learning rate schedulers, and powerful callback mechanisms (Checkpointing, Early Stopping) to streamline your model development process.
* **Clear and Documented API:** Dive into well-documented code and examples, making it easy to understand, use, and extend the library for your research or projects.

FlexiTransformers is your versatile toolkit for pushing the boundaries of transformer models, whether you are a researcher exploring novel architectures, a developer building NLP applications, or a practitioner seeking efficient and customizable solutions.

FlexiTransformers is designed to provide a highly modular and customizable framework for building transformer-based models. Whether you're interested in sequence-to-sequence tasks, classification, or language modeling, FlexiTransformers offers the building blocks and flexibility to construct the right architecture.

### Prerequisites

* **Python:** 3.10 or higher
* **PyTorch:** 2.0.1 or higher (Install instructions available at [pytorch.org](https://pytorch.org/get-started/locally/))
* **Required Packages:** Ensure you have the following packages installed:

  ```bash
  pip install rich>=13.9.4 torch>=2.0.1
  ```

  These packages are essential for FlexiTransformers to function correctly.

Ensure you have PyTorch installed correctly, ideally with GPU support for faster training.

### Installation via pip

The easiest way to install FlexiTransformers is using pip:

```bash
pip install flexitransformers
```

**For the latest, development version, install directly from GitHub:**

```bash
pip install git+https://github.com/A-Elshahawy/flexitransformers.git
```

**After installation, you will import the library as** **flexit** **in your Python code.**

### Cloning from GitHub

**Alternatively, you can clone the repository directly from GitHub for development or to access the latest version:**

```bash
git clone https://github.com/A-Elshahawy/flexitransformers.git
cd flexitransformers
pip install .
```

**This will install FlexiTransformers in editable mode, and you will import the library as** **`flexit`**.

## 🚀 Quick Start: Usage Examples

**Get hands-on with FlexiTransformers through these practical examples.** **Remember to import the library as** **flexit**.

### 1. Building an Encoder-Decoder Transformer for Translation

```python
import torch
import flexit.models as models # Import as flexit.models
from flexit.utils import subsequent_mask # Import as flexit.utils

# ⚙️ Configuration for Encoder-Decoder Model
config = {
    'model_type': 'encoder-decoder',
    'src_vocab': 10000,  # Source vocabulary size
    'tgt_vocab': 10000,  # Target vocabulary size
    'd_model': 512,      # Model dimension
    'n_heads': 8,        # Number of attention heads
    'n_layers': 6,       # Number of layers in encoder and decoder
    'dropout': 0.1,      # Dropout rate
}

# 🏗️ Initialize the FlexiTransformer model
transformer_model = models.FlexiTransformer(**config) # Use flexit.models

# 📊 Example Input Data (Replace with your actual data loaders)
src_input = torch.randint(0, 10000, (64, 32))  # Batch size 64, sequence length 32
tgt_input = torch.randint(0, 10000, (64, 32))
src_mask = (src_input != 0).unsqueeze(-2)      # Assuming 0 is padding token
tgt_mask = (tgt_input != 0).unsqueeze(-2) & subsequent_mask(tgt_input.size(-1))

# ➡️ Forward Pass
output = transformer_model(src_input, tgt_input, src_mask, tgt_mask)

print("Output shape:", output.shape) # Expected: [64, 32, 10000]
```

### 2. Creating a BERT-style Classifier (Encoder-Only)

```python
import torch
import flexit.models as models # Import as flexit.models

# ⚙️ Configuration for BERT-style Encoder
config = {
    'model_type': 'encoder-only',
    'src_vocab': 10000,  # Vocabulary size
    'num_classes': 2,    # Number of classification classes (e.g., binary)
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6,
    'dropout': 0.1,
    'pe_type': 'alibi',  # ALiBi Positional Encoding (BERT-style)
}

# 🏗️ Initialize FlexiBERT Model
bert_model = models.FlexiBERT(**config) # Use flexit.models

# 📊 Example Input
input_ids = torch.randint(0, 10000, (64, 32))  # Batch size 64, sequence length 32
attention_mask = (input_ids != 0).unsqueeze(-2)

# ➡️ Forward Pass
logits = bert_model(input_ids, attention_mask)

print("Logits shape:", logits.shape) # Expected: [64, 2]
```

### 3. Building a GPT-style Language Model (Decoder-Only)

```python
import torch
import flexit.models as models # Import as flexit.models
from flexit.utils import subsequent_mask # Import as flexit.utils

# ⚙️ Configuration for GPT-style Decoder
config = {
    'model_type': 'decoder-only',
    'tgt_vocab': 10000,  # Vocabulary size
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6,
    'dropout': 0.1,
    'pe_type': 'rotary', # Rotary Positional Encoding (GPT-style)
}

# 🏗️ Initialize FlexiGPT Model
gpt_model = models.FlexiGPT(**config) # Use flexit.models

# 📊 Example Input
input_sequence = torch.randint(0, 10000, (64, 32)) # Batch size 64, sequence length 32
tgt_mask = subsequent_mask(input_sequence.size(-1))

# ➡️ Forward Pass
output = gpt_model(input_sequence, tgt_mask)

print("Output shape:", output.shape) # Expected: [64, 32, 10000]
```

### 4. Training Your Model

**FlexiTransformers includes a** **Trainer** **class to simplify the training process:**

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset # Example Data
import flexit.models as models # Import as flexit.models
import flexit.train as train # Import as flexit.train
import flexit.loss as loss # Import as flexit.loss


# ⚙️ Model Configuration (using Encoder-Decoder from Example 1)
config = { ... } # Load your config from example 1
model = models.FlexiTransformer(**config) # Use flexit.models

# 📊 Example Data Loaders (Replace with your datasets)
src_data = torch.randint(0, 10000, (1000, 32)) # Example source data
tgt_data = torch.randint(0, 10000, (1000, 32)) # Example target data
dataset = TensorDataset(src_data, tgt_data)
train_loader = DataLoader(dataset, batch_size=64)
val_loader = DataLoader(dataset, batch_size=64) # Optional validation loader

# 🎯 Loss Function and Optimizer
criterion = loss.LabelSmoothing(size=config['tgt_vocab'], padding_idx=0, smoothing=0.1) # Use flexit.loss
loss_compute = train.LossCompute(generator=model.generator, criterion=criterion) # Use flexit.train
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0) # Example

# 🏋️ Trainer Initialization
trainer = train.Trainer( # Use flexit.train
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_compute,
    train_dataloader=train_loader,
    val_dataloader=val_loader, # Optional validation loader
)

# 🚄 Start Training
metrics = trainer.fit(epochs=5) # Train for 5 epochs
print(metrics.to_dict())
```

**For more advanced examples and training configurations, explore the** [examples directory](https://www.google.com/url?sa=E&q=link-to-examples-if-available) **(coming soon).**

## 📚 API Reference

**Delve deeper into FlexiTransformers with the API documentation available at:**

[FlexiTransformers Documentation](https://www.google.com/url?sa=E&q=https%3A%2F%2Fa-elshahawy.github.io%2FFlexiTransformers%2F)

**Key modules and classes include (Note the import path** **flexit**):

* **flexit.models**:

  * **FlexiTransformer**: Base class for flexible transformer models.
* **FlexiBERT**: Pre-configured BERT-style encoder-only model.
* **FlexiGPT**: Pre-configured GPT-style decoder-only model.
* **flexit.factory**:

  * **TransformerFactory**: Creates model instances based on **ModelConfig**.
* **flexit.configs**:

  * **ModelConfig**: Dataclass for model configuration parameters.
* **flexit.train**:

  * **Trainer**: Class for training transformer models with callbacks and utilities.
* **Batch**: Data batching and handling class.
* **LossCompute**: Loss calculation and gradient handling.
* **flexit.callbacks**:

  * **CheckpointCallback**: Saves model checkpoints during training.
* **EarlyStoppingCallback**: Implements early stopping based on validation loss.
* **flexit.attention**: Contains various attention mechanisms:

  * **AbsoluteMultiHeadedAttention**, **ALiBiMultiHeadAttention**, **RelativeGlobalAttention**, **RotaryMultiHeadAttention**
* **flexit.pos_embeddings**: Positional encoding implementations:

  * **AbsolutePositionalEncoding**, **ALiBiPositionalEncoding**, **RotaryPositionalEncoding**
* **flexit.loss**: Loss functions and utilities.

  * **LabelSmoothing**, **LossCompute**, **BertLoss**

**Explore the full documentation for detailed parameter descriptions, method signatures, and module explanations.**

## 🤝 Contributing

**We warmly welcome contributions to FlexiTransformers! Join us in making this library even more powerful and versatile. Here's how you can contribute:**

* **Fork the repository** **on GitHub to your personal account.**
* **Create a dedicated branch** **for your contribution:**

  ```bash
  git checkout -b feature/your-new-feature
  ```

  OR

```bash
git checkout -b bugfix/fix-issue-xyz
```

* **Develop your changes** **, adhering to the project's coding style and best practices. Consider using code linters like** **ruff** **and type checkers like** **mypy**.
* **Write comprehensive tests** **to ensure your additions are robust and function as expected.**
* **Document your code** **clearly, following the project's documentation standards.**
* **Submit a Pull Request** **to the main repository, detailing the changes you've implemented and their purpose.**

**For significant enhancements or architectural changes, it's recommended to first open an issue to discuss your proposal with the maintainers.**

## 📜 License and Credits

 **FlexiTransformers is released under the** **MIT License**, promoting open and collaborative development.

 **This library is built upon the groundbreaking work in transformer architectures and leverages the power of PyTorch. We extend our sincere gratitude to the open-source community for their invaluable contributions.**

**Developed and maintained by** [Ahmed Elshahawy](https://www.linkedin.com/in/ahmed-elshahawy-a42149218/).

## 🔗 Additional Resources

* **GitHub Repository:** [https://github.com/A-Elshahawy/flexitransformers](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2FA-Elshahawy%2Fflexitransformers) **- Report issues, suggest features, and participate in discussions.**
* **Documentation:** [https://a-elshahawy.github.io/FlexiTransformers/](https://www.google.com/url?sa=E&q=https%3A%2F%2Fa-elshahawy.github.io%2FFlexiTransformers%2F) **- In-depth API documentation, tutorials, and guides.**
* **PyPI Release:** [https://pypi.org/project/flexitransformers/0.1.0/](https://www.google.com/url?sa=E&q=https%3A%2F%2Fpypi.org%2Fproject%2Fflexitransformers%2F0.1.0%2F) **- Download the latest release.**

---

## 🙏 Acknowledgments

**This library draws inspiration from various resources and implementations of Transformer models, including:**

* **The "Attention is All You Need" paper and its associated implementations.**
* **Hugging Face's Transformers library.**

**We acknowledge and appreciate the work of the open-source community in making these resources available.**

---

## Contact

  You can find me on:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ahmed-elshahawy-a42149218/)  [![Gmail](https://img.shields.io/badge/Gmail-Email-red?style=flat&logo=gmail)](mailto:ahmedelshahawy078@gmail.com)

---

Get ready to unlock your creativity with FlexiTransformers! If you have any questions, feedback, or need help along the way, please don't hesitate to reach out through GitHub issues.

**Together, let's create something amazing with FlexiTransformers! 🚀**
