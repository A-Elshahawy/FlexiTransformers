"""Training state and metrics tracking."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self


class TrainState:
    """Track steps, examples, and tokens processed during training."""

    def __init__(self, save_dir: Path | None = None) -> None:
        self.step = 0
        self.accum_step = 0
        self.samples = 0
        self.tokens = 0
        self.epoch = 0
        self.start_time = time.time()
        self.tokens_per_sec = 0
        self.save_dir = save_dir

    def update(self, batch_size: int, ntokens: int, loss: float, lr: float) -> Self:
        self.step += 1
        self.samples += batch_size
        self.tokens += ntokens
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.tokens_per_sec = int(self.tokens / elapsed)
        return self

    def save(self, path: Path) -> None:
        if self.save_dir:
            metrics_path = Path(self.save_dir) / 'training_metrics.json'
            metrics = {
                'step': self.step,
                'accum_step': self.accum_step,
                'samples': self.samples,
                'tokens': self.tokens,
                'tokens_per_sec': self.tokens_per_sec,
                'epoch': self.epoch,
            }
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)


@dataclass
class TrainerMetrics:
    """Track training metrics across epochs."""

    epochs: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_times: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)

    def update(
        self, train_loss: float, val_loss: float, epoch_time: float, lr: float, epoch: int
    ) -> None:
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_times.append(epoch_time)
        self.learning_rates.append(lr)
        self.epochs.append(epoch)

    def to_dict(self) -> dict[str, Any]:
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_times': self.train_times,
            'learning_rates': self.learning_rates,
            'epochs': self.epochs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TrainerMetrics':
        metrics = cls()
        metrics.train_losses = data.get('train_losses', [])
        metrics.val_losses = data.get('val_losses', [])
        metrics.train_times = data.get('train_times', [])
        metrics.learning_rates = data.get('learning_rates', [])
        metrics.epochs = data.get('epochs', [])
        return metrics
