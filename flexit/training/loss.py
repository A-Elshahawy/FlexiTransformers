"""Loss functions for transformer training."""

from collections.abc import Callable

import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):
    """Label smoothing loss for sequence-to-sequence tasks."""

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0) -> None:
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.size
        x = torch.nn.functional.log_softmax(x, dim=-1)
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
    """Loss computation with generator, normalization, and optional gradient clipping."""

    def __init__(
        self, generator: nn.Module, criterion: Callable, model: nn.Module, grad_clip: float = 1.0
    ) -> None:
        self.generator = generator
        self.criterion = criterion
        self.model = model
        self.grad_clip = grad_clip

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, norm: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.generator(x)
        flat_logits = logits.contiguous().view(-1, logits.size(-1))
        flat_targets = y.contiguous().view(-1)
        raw_loss = self.criterion(flat_logits, flat_targets)
        normalized_loss = raw_loss / norm if norm > 0 else raw_loss
        return normalized_loss, normalized_loss


class BertLoss:
    """BERT-style classification loss."""

    def __init__(self, grad_clip: float = 1.0) -> None:
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.grad_clip = grad_clip

    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, norm: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() == 3:
            logits = logits[:, 0, :]
        raw_loss = self.criterion(logits, labels)
        return raw_loss, raw_loss
