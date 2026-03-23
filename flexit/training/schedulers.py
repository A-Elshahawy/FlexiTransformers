"""Learning rate schedulers and dummy optimizer/scheduler for eval mode."""

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.optim.lr_scheduler import _LRScheduler


def lr_step(step: int, model_size: int, factor: float, warmup: int) -> float:
    """Noam learning rate schedule (Attention Is All You Need)."""
    if step == 0:
        step = 1
    return float(factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))))


def create_progress_bar() -> Progress:
    """Create a rich progress bar with training metrics."""
    return Progress(
        SpinnerColumn(),
        TextColumn('[bold blue]{task.description}'),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn('•'),
        TimeElapsedColumn(),
        TextColumn('•'),
        TimeRemainingColumn(),
        console=Console(),
    )


class DummyOptimizer(torch.optim.Optimizer):
    """No-op optimizer for evaluation mode."""

    def __init__(self) -> None:
        self.param_groups: list[dict] = [{'lr': 0}]

    def step(self, closure: object = None) -> None:  # type: ignore[override]
        pass

    def zero_grad(self, set_to_none: bool = False) -> None:
        pass


class DummyScheduler(_LRScheduler):
    """No-op scheduler for evaluation mode."""

    def __init__(self, optimizer: DummyOptimizer | None = None) -> None:
        self.optimizer = optimizer  # type: ignore[assignment]

    def step(self, epoch: object = None) -> None:  # type: ignore[override]
        pass

    def get_last_lr(self) -> list[float]:
        return [0.0]
