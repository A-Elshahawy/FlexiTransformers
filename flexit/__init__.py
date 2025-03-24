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

__all__ = [
    'ALiBiMultiHeadAttention',
    'AbsoluteMultiHeadedAttention',
    'BaseTransformer',
    'Batch',
    'Callback',
    'CheckpointCallback',
    'ConfigDescriptor',
    'Decoder',
    'EarlyStoppingCallback',
    'Encoder',
    'EncoderDecoder',
    'FlexiTransformer',
    'Generator',
    'LabelSmoothing',
    'LayerNorm',
    'LossCompute',
    'ModelConfig',
    'PositionwiseFeedForward',
    'RelativeGlobalAttention',
    'RotaryMultiHeadAttention',
    'SublayerConnection',
    'TrainState',
    'TransformerFactory',
    'clone',
    'greedy_decode',
    'lr_step',
    'run_epoch',
    'subsequent_mask',
]
