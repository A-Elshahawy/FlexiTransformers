"""
FlexiTransformers Module Components

This module provides the core components of the FlexiTransformers library,
organizing various transformer architecture elements including attention mechanisms,
positional encodings, and model structures.

Components:
- Attention: Multiple attention implementations (Absolute, ALiBi, Relative, Rotary)
- Callbacks: Training utilities for checkpointing and early stopping
- Configs: Configuration descriptors for model instantiation
- Core: Fundamental transformer building blocks (Encoder, Decoder)
- Layers: Basic neural network components (LayerNorm, FeedForward)
- Models: Complete transformer implementations with specialized variants
- Positional Encodings: Various embedding strategies for sequence positions
- Training: Utilities for efficient model training and evaluation

Each component is designed to be modular and composable, allowing for
flexible architecture design while maintaining interoperability.
"""

from .attention import (
    AbsoluteMultiHeadedAttention,
    AbstractAttention,
    ALiBiMultiHeadAttention,
    RelativeGlobalAttention,
    RotaryMultiHeadAttention,
)
from .callbacks import CheckpointCallback, EarlyStoppingCallback
from .configs import ConfigDescriptor, ModelConfig
from .core import Decoder, Encoder, EncoderDecoder, Generator
from .factory import DecoderOnly, EncoderOnly, TransformerFactory
from .layers import (
    DecoderLayer,
    Embeddings,
    EncoderLayer,
    LayerNorm,
    PositionwiseFeedForward,
    SublayerConnection,
)
from .loss import BertLoss, LabelSmoothing, LossCompute
from .models import FlexiBERT, FlexiGPT, FlexiTransformer, TransformerModel
from .pos_embeddings import (
    AbsolutePositionalEncoding,
    ALiBiPositionalEncoding,
    RotaryPositionalEncoding,
)
from .train import Batch, Trainer, TrainerMetrics, TrainState, lr_step, run_epoch

__version__ = '0.2.1'
