"""
Training Callbacks

This module implements callback classes for training events,
including checkpointing and early stopping.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import override

if TYPE_CHECKING:
    from .train import Trainer


class Callback:
    """
    Base class for training callbacks.

    Defines hooks for training events that can be overridden by subclasses.

    Methods:
        on_train_begin: Called at the start of training.
        on_train_end: Called at the end of training.
        on_epoch_begin: Called at the start of each epoch.
        on_epoch_end: Called at the end of each epoch.
    """

    def on_train_begin(self, trainer: 'Trainer') -> None:
        """
        Called at the start of training.

        Args:
            trainer: Trainer instance.
        """
        ...

    def on_train_end(self, trainer: 'Trainer') -> None:
        """
        Called at the end of training.

        Args:
            trainer: Trainer instance.
        """
        ...

    def on_epoch_begin(self, epoch: int, trainer: 'Trainer') -> None:
        """
        Called at the start of each epoch.

        Args:
            epoch (int): Current epoch number.
            trainer: Trainer instance.
        """
        pass

    def on_epoch_end(self, epoch: int, trainer: 'Trainer') -> None:
        """
        Called at the end of each epoch.

        Args:
            epoch (int): Current epoch number.
            trainer: Trainer instance.
        """
        pass


class CheckpointCallback(Callback):
    """
    Callback to handle checkpointing with options to save
    only the best model and the last N checkpoints.

    Attributes:
        save_best (bool): Save best model based on validation loss.
        keep_last (int): Keep last N checkpoints.
        checkpoint_dir (Path): Directory to save checkpoints.
        filename_format (str): Format string for checkpoint names.
        best_filename (str): Filename for best model checkpoint.
        best_loss (float): Best validation loss seen so far.
        saved_checkpoints (list[Path]): List of saved checkpoint paths.

    Methods:
        on_epoch_end: Save checkpoint if conditions are met and clean up old checkpoints.
    """

    def __init__(
        self,
        save_best: bool = True,
        keep_last: int = 3,
        checkpoint_dir: str | Path = 'checkpoints',
        filename_format: str = 'checkpoint_epoch_{epoch:03d}.pt',
        best_filename: str = 'best_model.pt',
    ) -> None:
        """
        Initialize checkpoint callback.

        Args:
            save_best (bool): Save best model based on validation loss.
            keep_last (int): Keep last N checkpoints.
            checkpoint_dir (str | Path): Directory to save checkpoints.
            filename_format (str): Format string for checkpoint names.
            best_filename (str): Filename for best model checkpoint.
        """

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
        """
        Save checkpoint if conditions are met and clean up old checkpoints.

        Args:
            epoch (int): Current epoch number.
            trainer: Trainer instance.

        Returns:
            None
        """

        if not trainer.metrics.val_losses:
            return

        current_loss = trainer.metrics.val_losses[-1]
        epoch_num = trainer.current_epoch  # Use the trainer's current epoch counter

        # Save best model
        if self.save_best and current_loss < self.best_loss:
            self.best_loss = current_loss
            best_path = self.checkpoint_dir / self.best_filename
            trainer.save_checkpoint(best_path)
            trainer.console.print(f'[green]Saved best model ({current_loss:.4f}) to {best_path}[/]')

        # Save regular checkpoint
        if self.keep_last > 0:
            filename = self.filename_format.format(epoch=epoch_num)
            path = self.checkpoint_dir / filename
            trainer.save_checkpoint(path)
            self.saved_checkpoints.append(path)
            self._cleanup_old_checkpoints(epoch_num, trainer)

    def _cleanup_old_checkpoints(self, current_epoch: int, trainer: 'Trainer') -> None:
        """
        Remove excess checkpoints based on keep_last policy.

        Args:
            current_epoch (int): Current epoch number.
            trainer: Trainer instance.

        Returns:
            None

        """

        if len(self.saved_checkpoints) <= self.keep_last:
            return

        # Sort checkpoints by epoch number
        sorted_checkpoints = sorted(
            self.saved_checkpoints, key=lambda p: self._extract_epoch(p), reverse=True
        )

        # Keep only N most recent checkpoints
        for path in sorted_checkpoints[self.keep_last :]:
            try:
                path.unlink()
                self.saved_checkpoints.remove(path)
                trainer.console.print(f'[dim]Removed old checkpoint: {path.name}[/]')
            except Exception as e:
                trainer.console.print(f'[yellow]Error removing {path}: {e}[/]')

    def _extract_epoch(self, path: Path) -> int:
        """
        Extract epoch number from filename.

        Args:
            path (Path): Path to checkpoint file.

        Returns:
            int: Extracted epoch number.

        Raises:
            ValueError: If epoch number cannot be extracted.
        """
        try:
            return int(path.stem.split('_')[-1])
        except ValueError:
            return 0  # Handle best model filename


class EarlyStoppingCallback(Callback):
    """
    Callback to stop training early if validation loss doesn't improve.

    Attributes:
        patience (int): Number of epochs to wait for improvement.
        min_delta (float): Minimum change in loss to qualify as improvement.
        best_loss (float): Best validation loss seen so far.
        counter (int): Number of epochs since last improvement.

    Methods:
        on_epoch_end: Check if training should stop based on validation loss.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        """
        Initialize early stopping callback.

        Args:
            patience (int): Number of epochs to wait for improvement.
            min_delta (float): Minimum change in loss to qualify as improvement.

        Returns:
            None
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    @override
    def on_epoch_end(self, epoch: int, trainer: 'Trainer') -> None:
        """
        Check if training should stop based on validation loss.

        Args:
            epoch (int): Current epoch number.
            trainer: Trainer instance.

        Returns:
            None
        """
        current_loss = trainer.metrics.val_losses[-1]

        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.stop_training = True
