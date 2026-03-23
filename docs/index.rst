FlexiTransformers Documentation
================================

.. image:: _static/logo_1.png
   :alt: FlexiTransformers Logo
   :align: center

|

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://badge.fury.io/py/flexitransformers.svg
   :target: https://pypi.org/project/flexitransformers/
   :alt: PyPI version

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
   :alt: Code Style: Ruff

.. image:: https://img.shields.io/badge/mypy-type%20checked-blue
   :alt: mypy

A modular transformer framework supporting encoder-decoder, encoder-only (BERT),
and decoder-only (GPT) architectures with a pluggable positional encoding system.

.. mermaid::

   graph TD
       A[FlexiTransformers] --> B[Architectures]
       A --> C[Components]
       A --> D[Training]

       B --> B1[Encoder-Decoder]
       B --> B2[Encoder-Only]
       B --> B3[Decoder-Only]

       C --> C1[MultiHeadAttention]
       C --> C2[Positional Encodings]
       C --> C3[Layers & Blocks]

       C2 --> C2a[Sinusoidal / Learned]
       C2 --> C2b[Rotary - RoPE]
       C2 --> C2c[ALiBi]
       C2 --> C2d[Relative / Relative+Bias]

       D --> D1[Trainer]
       D --> D2[Callbacks]
       D --> D3[Loss Functions]

Features
--------

- **3 architectures** — Encoder-Decoder, Encoder-Only (BERT), Decoder-Only (GPT)
- **6 positional encodings** — Sinusoidal, Learned, RoPE, ALiBi, Relative, Relative+Bias
- **Pluggable PE system** — ``register_pe("name", MyPE)`` to add custom encodings
- **KV cache** — efficient autoregressive inference with per-layer caching
- **Sampling strategies** — greedy, temperature, top-k, nucleus (top-p)
- **Training utilities** — ``Trainer``, callbacks, ``LabelSmoothing``, ``run_epoch``
- **Fully typed** — mypy-checked, ruff-formatted, pre-commit enforced

Installation
------------

.. code-block:: bash

   pip install flexitransformers

Quick Start
-----------

Decoder-Only (GPT-style)
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from flexit import FlexiGPT, greedy_decode
   import torch

   model = FlexiGPT(vocab_size=32000, d_model=512, n_heads=8, n_layers=6, d_ff=2048)
   src = torch.randint(0, 32000, (1, 10))
   out = greedy_decode(model, src, src_mask=None, max_len=50, start_symbol=1)
   print(out.shape)  # [1, 59]

Encoder-Only (BERT-style)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from flexit import FlexiBERT
   import torch

   model = FlexiBERT(vocab_size=32000, d_model=512, n_heads=8, n_layers=6,
                     d_ff=2048, num_classes=2)
   x = torch.randint(0, 32000, (4, 64))
   mask = (x != 0).unsqueeze(1).unsqueeze(2)
   logits = model(x, mask)
   print(logits.shape)  # [4, 2]

Advanced Config (Encoder-Decoder)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from flexit import ModelConfig, create_model

   config = ModelConfig(
       model_type='encoder-decoder',
       src_vocab_size=32000,
       tgt_vocab_size=32000,
       d_model=512,
       n_heads=8,
       n_layers=6,
       d_ff=2048,
       pe_type='rotary',
       dropout=0.1,
   )
   model = create_model(config)

.. mermaid::

   flowchart LR
       A[ModelConfig] --> B[TransformerFactory]
       B --> C{model_type}
       C -->|encoder-decoder| D[EncoderDecoderModel]
       C -->|encoder-only| E[EncoderOnlyModel]
       C -->|decoder-only| F[DecoderOnlyModel]
       D --> G[FlexiTransformer]
       E --> H[FlexiBERT]
       F --> I[FlexiGPT]

Positional Encodings
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Name
     - ``pe_type``
     - Injected at
     - Representative models
   * - Sinusoidal
     - ``"absolute"``
     - Embedding
     - Vaswani et al. 2017
   * - Learned
     - ``"learned"``
     - Embedding
     - BERT, GPT-2
   * - Rotary (RoPE)
     - ``"rotary"``
     - Q/K projections
     - LLaMA, GPT-NeoX
   * - ALiBi
     - ``"alibi"``
     - Attention scores
     - MPT, BLOOM
   * - Relative
     - ``"relative"``
     - Attention scores
     - Transformer-XL
   * - Relative+Bias
     - ``"relative_bias"``
     - Attention scores
     - T5
   * - None
     - ``"none"``
     - —
     - Ablations

Custom PE via plugin:

.. code-block:: python

   from flexit import register_pe
   from flexit.attention.positional import PositionalEncoding

   class NoPE(PositionalEncoding):
       @property
       def injection_point(self):
           return "embedding"

       def apply_to_embedding(self, x):
           return x

   register_pe("nope", NoPE)

Inference Strategies
--------------------

Greedy decoding and nucleus sampling are both available out of the box:

.. code-block:: python

   from flexit import greedy_decode, sample_decode, temperature_sample, top_k_sample, top_p_sample

   # Greedy
   out = greedy_decode(model, src, src_mask=None, max_len=50, start_symbol=1)

   # Nucleus (top-p) + temperature
   out = sample_decode(model, src, src_mask=None, max_len=50, start_symbol=1,
                       temperature=0.8, top_p=0.9)

   # Standalone logit samplers
   token = temperature_sample(logits, temperature=0.7)
   token = top_k_sample(logits, k=50, temperature=1.0)
   token = top_p_sample(logits, p=0.9, temperature=0.8)

Core Architecture
-----------------

.. mermaid::

   stateDiagram-v2
       [*] --> EmbeddingWithPE
       EmbeddingWithPE --> TransformerLayers

       state TransformerLayers {
           [*] --> EncoderStack
           EncoderStack --> [*]: Encoder-Only
           EncoderStack --> DecoderStack: Encoder-Decoder
           [*] --> DecoderStack: Decoder-Only
           DecoderStack --> [*]
       }

       TransformerLayers --> OutputHead
       OutputHead --> [*]

Three fundamental building blocks:

1. **Embedding System** — ``EmbeddingWithPE`` combines token embeddings with the
   chosen positional encoding (or passes through for QK/score-level encodings).

2. **Transformer Layers** — ``EncoderLayer``, ``CausalDecoderLayer``,
   ``CrossAttentionDecoderLayer`` — each wraps ``MultiHeadAttention`` + ``FeedForward``
   inside ``SublayerConnection`` (residual + norm, pre- or post-norm).

3. **Output Heads** — ``LMHead`` (language modeling, weight tying), ``BertHead``,
   ``SequenceClassificationHead``, ``TokenClassificationHead``.

Training Pipeline
-----------------

.. code-block:: python

   from flexit import (
       ModelConfig, create_model,
       Trainer, LossCompute, LabelSmoothing,
       CheckpointCallback, EarlyStoppingCallback,
   )
   import torch

   config = ModelConfig(model_type='decoder-only', vocab_size=32000,
                        d_model=512, n_heads=8, n_layers=6, d_ff=2048)
   model = create_model(config)

   criterion = LabelSmoothing(size=32000, padding_idx=0, smoothing=0.1)
   loss_fn = LossCompute(generator=model.generator, criterion=criterion, model=model)

   trainer = Trainer(
       model=model,
       optimizer=torch.optim.Adam(model.parameters(), lr=3e-4),
       loss_fn=loss_fn,
       train_dataloader=train_loader,
       val_dataloader=val_loader,
       callbacks=[
           CheckpointCallback(save_dir="ckpts/", monitor="val_loss"),
           EarlyStoppingCallback(patience=3, monitor="val_loss"),
       ],
   )
   metrics = trainer.fit(epochs=10)

Package Structure
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Subpackage
     - Key components
   * - ``flexit.attention``
     - ``MultiHeadAttention``
   * - ``flexit.attention.positional``
     - ``SinusoidalPE``, ``LearnedPE``, ``RotaryPE``, ``ALiBiPE``, ``RelativePE``, ``RelativePEWithBias``, ``create_pe``, ``register_pe``
   * - ``flexit.config``
     - ``ModelConfig``
   * - ``flexit.core``
     - ``Embeddings``, ``EmbeddingWithPE``, ``FeedForward``, ``GLUFeedForward``, ``Generator``, ``LayerNorm``, ``RMSNorm``
   * - ``flexit.layers``
     - ``EncoderLayer``, ``CausalDecoderLayer``, ``CrossAttentionDecoderLayer``, ``SublayerConnection``
   * - ``flexit.blocks``
     - ``Encoder``, ``CausalDecoder``, ``CrossAttentionDecoder``
   * - ``flexit.models``
     - ``FlexiGPT``, ``FlexiBERT``, ``FlexiTransformer``, ``DecoderOnlyModel``, ``EncoderOnlyModel``, ``EncoderDecoderModel``, ``BaseModel``
   * - ``flexit.models.heads``
     - ``LMHead``, ``BertHead``, ``SequenceClassificationHead``, ``TokenClassificationHead``
   * - ``flexit.factory``
     - ``TransformerFactory``, ``create_model``
   * - ``flexit.inference``
     - ``greedy_decode``, ``sample_decode``, ``temperature_sample``, ``top_k_sample``, ``top_p_sample``
   * - ``flexit.training``
     - ``Trainer``, ``Batch``, ``LabelSmoothing``, ``LossCompute``, ``BertLoss``, ``run_epoch``, ``Callback``, ``CheckpointCallback``, ``EarlyStoppingCallback``
   * - ``flexit.utils``
     - ``subsequent_mask``, ``create_causal_mask``, ``create_padding_mask``, ``create_combined_mask``, ``count_parameters``

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Package Modules

   modules

Examples
--------

See the `examples/ <https://github.com/A-Elshahawy/flexitransformers/tree/main/examples>`_
directory for end-to-end runnable scripts covering all architectures, PE types,
output heads, save/load, and custom plugins.

Contributing
------------

1. Fork the repository
2. Create a feature branch: ``git checkout -b feature/your-feature``
3. Follow existing code style (ruff, mypy, type annotations)
4. Write tests for new functionality
5. Submit a pull request with a clear description

License
-------

MIT License — see `license text <https://github.com/A-Elshahawy/flexitransformers/blob/main/LICENSE>`_.

Acknowledgments
---------------

- "Attention Is All You Need" — Vaswani et al. (2017)
- Hugging Face Transformers for architectural inspiration
- PyTorch community for foundational components
