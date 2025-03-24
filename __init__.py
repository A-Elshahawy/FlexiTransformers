# """
# FlexiTransformers: A Modular Transformer Library

# A comprehensive, composable framework for building and training transformer models
# with support for multiple attention mechanisms, positional encodings, and
# architecture variants.

# Features:
# ----------
# - Modular design: Mix and match components to create custom transformer architectures
# - Multiple attention mechanisms: Absolute, Relative, Rotary (RoPE), and ALiBi
# - Architecture support: Encoder-decoder, encoder-only, and decoder-only models
# - Training utilities: Callbacks, metrics tracking, and optimization helpers
# - Production-ready: Type hints, comprehensive documentation, and extensive testing

# Architectures:
# ----------
# - Encoder-decoder: Similar to the original Transformer architecture
# - Encoder-only: BERT-like models for bidirectional contextual representations
# - Decoder-only: GPT-like autoregressive language models

# Usage:
# ----------
# ```python
# import flexit
# from flexit import TransformerFactory

# # Create a custom transformer with specific components
# model = TransformerFactory.create(
#     architecture="encoder_decoder",
#     attention_type="rotary",
#     d_model=512,
#     n_heads=8,
# )
# ```

# For detailed examples and documentation, see the project repository
# at https://github.com/A-Elshahawy/flexitransformers
# """

# from flexit.attention import (
#     AbsoluteMultiHeadedAttention,
#     ALiBiMultiHeadAttention,
#     RelativeGlobalAttention,
#     RotaryMultiHeadAttention,
# )
# from flexit.core import EncoderDecoder, Generator
# from flexit.layers import LayerNorm, PositionwiseFeedForward, SublayerConnection
# from flexit.loss import LabelSmoothing, LossCompute
# from flexit.models_heads import greedy_decode
# from flexit.pos_embeddings import (
#     AbsolutePositionalEncoding,
#     ALiBiPositionalEncoding,
#     RotaryPositionalEncoding,
# )
# from flexit.train import (
#     Batch,
#     DummyOptimizer,
#     DummyScheduler,
#     Trainer,
#     TrainerMetrics,
#     TrainState,
#     create_progress_bar,
#     lr_step,
#     run_epoch,
# )
# from flexit.utils import clone, subsequent_mask

# __all__ = [
#     'ALiBiMultiHeadAttention',
#     'ALiBiPositionalEncoding',
#     'AbsoluteMultiHeadedAttention',
#     'AbsolutePositionalEncoding',
#     'Batch',
#     'DummyOptimizer',
#     'DummyScheduler',
#     'EncoderDecoder',
#     'Generator',
#     'LabelSmoothing',
#     'LayerNorm',
#     'LossCompute',
#     'PositionwiseFeedForward',
#     'RelativeGlobalAttention',
#     'RotaryMultiHeadAttention',
#     'RotaryPositionalEncoding',
#     'SublayerConnection',
#     'TrainState',
#     'Trainer',
#     'TrainerMetrics',
#     'clone',
#     'create_progress_bar',
#     'greedy_decode',
#     'lr_step',
#     'run_epoch',
#     'subsequent_mask',
# ]
