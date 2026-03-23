"""Task-specific output heads for transformer models."""

from .classification import BertHead, SequenceClassificationHead
from .lm_head import LMHead
from .token_classification import TokenClassificationHead

__all__ = [
    'BertHead',
    'LMHead',
    'SequenceClassificationHead',
    'TokenClassificationHead',
]
