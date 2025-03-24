"""
Loss Functions

This module implements various loss functions used in transformer training,
including label smoothing and enhanced loss computation with gradient clipping.
"""

from collections.abc import Callable

import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):
    """
    Implement label smoothing.

    Args:
        size (int): Vocabulary size.
        padding_idx (int): Padding token index.
        smoothing (float): Smoothing value. Default: 0.0.

    Attributes:
        criterion (nn.KLDivLoss): Loss criterion.
        padding_idx (int): Padding token index.
        confidence (float): Confidence value.
        smoothing (float): Smoothing value.
        size (int): Vocabulary size.
        true_dist (torch.Tensor): True distribution.

    Methods:
        forward: Forward pass through label smoothing.
    """

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0) -> None:
        """
        Initialize label smoothing.

        Args:
            size (int): Vocabulary size.
            padding_idx (int): Padding token index.
            smoothing (float): Smoothing value.
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through label smoothing.

        Args:
            x (torch.Tensor): Model output logits.
            target (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: Loss value.
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class LossCompute:
    """
    Enhanced loss computation with proper normalization and gradient handling.

    Args:
        generator (nn.Module): Model's output generator
        criterion: Loss criterion (typically CrossEntropyLoss or KLDivLoss)
        grad_clip (float, optional): Maximum norm for gradient clipping
    """

    def __init__(self, generator: nn.Module, criterion: Callable, grad_clip: float = 1.0) -> None:
        """
        Initialize loss computation.

        Args:
            generator (nn.Module): Model's output generator.
            criterion: Loss criterion.
            grad_clip (float): Maximum norm for gradient clipping.
        """
        self.generator = generator
        self.criterion = criterion
        self.grad_clip = grad_clip

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, norm: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss with proper scaling and optional gradient clipping.

        Args:
            x (torch.Tensor): Model output logits
            y (torch.Tensor): Target labels
            norm (float): Batch normalization factor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (scaled_loss, raw_loss)
        """
        # Generate logits
        logits = self.generator(x)

        # Reshape tensors for loss computation
        flat_logits = logits.contiguous().view(-1, logits.size(-1))
        flat_targets = y.contiguous().view(-1)

        # Compute raw loss
        raw_loss = self.criterion(flat_logits, flat_targets)

        # Scale loss by normalization factor
        scaled_loss = raw_loss / norm

        return scaled_loss * norm, raw_loss


class BertLoss:
    """Enhanced BERT-style loss computation with proper scaling."""

    def __init__(self, grad_clip: float = 1.0) -> None:
        """
        Initialize BERT loss computation.

        Args:
            grad_clip (float): Maximum norm for gradient clipping.
        """
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.grad_clip = grad_clip

    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, norm: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss for sequence classification with proper scaling.

        Args:
            logits: Model output [batch_size, seq_len, num_classes]
            labels: Ground truth [batch_size]
            norm: Normalization factor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (scaled_loss, raw_loss)
        """

        raw_loss = self.criterion(logits, labels)

        # Scale loss appropriately
        scaled_loss = raw_loss * norm

        return scaled_loss, raw_loss
