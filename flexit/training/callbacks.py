"""Training callbacks for checkpointing and early stopping."""

import re
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import override

if TYPE_CHECKING:
    from .trainer import Trainer


class Callback:
    """Base class for training callbacks."""

    def on_train_begin(self, trainer: 'Trainer') -> None: ...

    def on_train_end(self, trainer: 'Trainer') -> None: ...

    def on_epoch_begin(self, epoch: int, trainer: 'Trainer') -> None:
        pass

    def on_epoch_end(self, epoch: int, trainer: 'Trainer') -> None:
        pass


class CheckpointCallback(Callback):
    """Save model checkpoints, keeping only the best and last N."""

    def __init__(
        self,
        save_best: bool = True,
        keep_last: int = 3,
        checkpoint_dir: str | Path = 'checkpoints',
        filename_format: str = 'checkpoint_epoch_{epoch:03d}.pt',
        best_filename: str = 'best_model.pt',
    ) -> None:
        self.save_best = save_best
        self.keep_last = keep_last
        self.checkpoint_dir = Path(checkpoint_dir)
        self.filename_format = filename_format
        self.best_filename = best_filename
        self.best_loss = float('inf')
        self.saved_checkpoints: list[Path] = []
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @override
    def on_epoch_end(self, epoch: int, trainer: 'Trainer') -> None:
        if not trainer.metrics.val_losses:
            return

        current_loss = trainer.metrics.val_losses[-1]
        epoch_num = trainer.current_epoch

        if self.save_best and current_loss < self.best_loss:
            self.best_loss = current_loss
            best_path = self.checkpoint_dir / self.best_filename
            trainer.save_checkpoint(best_path)
            trainer.console.print(f'[green]Saved best model ({current_loss:.4f}) to {best_path}[/]')

        if self.keep_last > 0:
            filename = self.filename_format.format(epoch=epoch_num)
            path = self.checkpoint_dir / filename
            trainer.save_checkpoint(path)
            self.saved_checkpoints.append(path)
            self._cleanup_old_checkpoints(epoch_num, trainer)

    def _cleanup_old_checkpoints(self, current_epoch: int, trainer: 'Trainer') -> None:
        if len(self.saved_checkpoints) <= self.keep_last:
            return
        sorted_checkpoints = sorted(
            self.saved_checkpoints, key=lambda p: self._extract_epoch(p), reverse=True
        )
        for path in sorted_checkpoints[self.keep_last :]:
            try:
                path.unlink()
                self.saved_checkpoints.remove(path)
                trainer.console.print(f'[dim]Removed old checkpoint: {path.name}[/]')
            except Exception as e:
                trainer.console.print(f'[yellow]Error removing {path}: {e}[/]')

    def _extract_epoch(self, path: Path) -> int:
        match = re.search(r'epoch_(\d+)', path.name)
        return int(match.group(1)) if match else 0


class EarlyStoppingCallback(Callback):
    """Stop training early if validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    @override
    def on_epoch_end(self, epoch: int, trainer: 'Trainer') -> None:
        if not trainer.metrics.val_losses:
            return
        current_loss = trainer.metrics.val_losses[-1]
        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.stop_training = True
