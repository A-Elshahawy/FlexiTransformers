"""
FlexiTransformers — modular transformer library.

Supported architectures:
    - Encoder-Decoder  (T5/BART style, seq2seq)
    - Encoder-Only     (BERT style, classification)
    - Decoder-Only     (GPT style, language modeling)
"""

from .attention import MultiHeadAttention
from .attention.positional import (
    ALiBiPE,
    LearnedPE,
    RelativePE,
    RelativePEWithBias,
    RotaryPE,
    SinusoidalPE,
    create_pe,
    register_pe,
)
from .blocks import CausalDecoder, CrossAttentionDecoder, Encoder
from .config import ModelConfig
from .core import (
    Embeddings,
    EmbeddingWithPE,
    FeedForward,
    Generator,
    GLUFeedForward,
    LayerNorm,
    RMSNorm,
    create_norm,
)
from .core.feedforward import create_feedforward
from .factory import TransformerFactory, create_model
from .inference import DecoderStrategy, EncoderDecoderStrategy, greedy_decode
from .layers import (
    CausalDecoderLayer,
    CrossAttentionDecoderLayer,
    EncoderLayer,
    SublayerConnection,
)
from .models import (
    BaseModel,
    BertHead,
    DecoderOnlyModel,
    EncoderDecoderModel,
    EncoderOnlyModel,
    FlexiBERT,
    FlexiGPT,
    FlexiTransformer,
    LMHead,
    SequenceClassificationHead,
    TokenClassificationHead,
    TransformerModel,
)
from .training import (
    Batch,
    BertLoss,
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    LabelSmoothing,
    LossCompute,
    Trainer,
    TrainerMetrics,
    TrainState,
    lr_step,
    run_epoch,
)
from .utils import (
    clone_module,
    count_parameters,
    create_causal_mask,
    create_combined_mask,
    create_padding_mask,
    subsequent_mask,
)
from .version import __author__, __copyright__, __license__, __version__

__all__ = [
    # Positional encodings
    'ALiBiPE',
    # Models
    'BaseModel',
    # Training
    'Batch',
    # Heads
    'BertHead',
    'BertLoss',
    'Callback',
    # Blocks
    'CausalDecoder',
    # Layers
    'CausalDecoderLayer',
    'CheckpointCallback',
    'CrossAttentionDecoder',
    'CrossAttentionDecoderLayer',
    'DecoderOnlyModel',
    # Inference
    'DecoderStrategy',
    'EarlyStoppingCallback',
    'EmbeddingWithPE',
    # Core
    'Embeddings',
    'Encoder',
    'EncoderDecoderModel',
    'EncoderDecoderStrategy',
    'EncoderLayer',
    'EncoderOnlyModel',
    'FeedForward',
    'FlexiBERT',
    'FlexiGPT',
    'FlexiTransformer',
    'GLUFeedForward',
    'Generator',
    'LMHead',
    'LabelSmoothing',
    'LayerNorm',
    'LearnedPE',
    'LossCompute',
    # Config
    'ModelConfig',
    # Attention
    'MultiHeadAttention',
    'RMSNorm',
    'RelativePE',
    'RelativePEWithBias',
    'RotaryPE',
    'SequenceClassificationHead',
    'SinusoidalPE',
    'SublayerConnection',
    'TokenClassificationHead',
    'TrainState',
    'Trainer',
    'TrainerMetrics',
    # Factory
    'TransformerFactory',
    'TransformerModel',
    '__author__',
    '__copyright__',
    '__license__',
    # Version
    '__version__',
    # Utils
    'clone_module',
    'count_parameters',
    'create_causal_mask',
    'create_combined_mask',
    'create_feedforward',
    'create_model',
    'create_norm',
    'create_padding_mask',
    'create_pe',
    'greedy_decode',
    'lr_step',
    'register_pe',
    'run_epoch',
    'subsequent_mask',
]
