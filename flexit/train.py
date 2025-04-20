"""
This module implements training utilities including batch processing,
learning rate scheduling, training state tracking, and the main training loop.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader

from .callbacks import Callback
from .models_heads import greedy_decode
from .utils import subsequent_mask


class Batch:
    """
    Unified batch handling for all transformer architectures.
    Handles encoder-decoder, decoder-only, and encoder-only (BERT) models.

    Attributes:
        src (torch.Tensor): Source sequence.
        tgt (torch.Tensor): Target sequence.
        labels (torch.Tensor): Classification labels.
        src_mask (torch.Tensor): Source mask.
        tgt_mask (torch.Tensor): Target mask.
        ntokens (int): Number of tokens in the batch.
        model_type (Literal): Type of transformer architecture.
        device (str): Computation device.
        pad (int): Padding token ID.

    Methods:
        _validate_inputs: Validate input tensors based on model type.
        __init_encoder_only: Initialize for encoder-only models.
        __init_decoder_only: Initialize for decoder-only models.
        __init_decoder_decoder: Initialize for encoder-decoder models.
        to: Move batch to device.
    """

    def __init__(
        self,
        src: torch.Tensor | None = None,
        tgt: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        device: str = 'cpu',
        pad: int = 2,
        model_type: Literal['encoder-only', 'decoder-only', 'encoder-decoder'] = 'encoder-decoder',
    ) -> None:
        """
        Initialize batch with appropriate tensors and masks based on model type.

        Args:
            src: Input tensor (source sequences or input tokens)
            tgt: Target sequences for seq2seq tasks
            labels: Classification labels for BERT-style tasks
            device: Computation device
            pad: Padding token ID
            model_type: Type of transformer architecture
        """
        self._validate_inputs(tgt, labels, model_type)
        self.model_type = model_type
        self.device = device
        self.pad = pad

        match model_type:
            case 'encoder-decoder':
                self.__init_decoder_decoder(src, tgt)
            case 'encoder-only':
                self.__init_encoder_only(src, labels)
            case 'decoder-only':
                self.__init_decoder_only(tgt if tgt is not None else src)
            case _:
                raise ValueError(f'Invalid model type: {model_type}')

    def _validate_inputs(
        self, tgt: torch.Tensor | None, labels: torch.Tensor | None, model_type: str
    ) -> None:
        """
        Validate input tensors based on model type.
        Checks that the necessary input tensors are provided based on the model type.

        Args:
            tgt: Target sequences
            labels: Classification labels
            model_type: Type of transformer architecture

        Raises:
            ValueError: If required tensors are not provided.
        """
        if model_type == 'encoder-decoder' and tgt is None:
            raise ValueError('Target sequence required for encoder-decoder models')
        if model_type == 'encoder-only' and labels is None:
            raise ValueError('Labels required for encoder-only (BERT) models')

    def __init_encoder_only(self, src: torch.Tensor, labels: torch.Tensor) -> None:
        """Initialize for encoder-only (BERT-style) models."""
        self.src = src
        self.labels = labels
        self.src_mask = (src != self.pad).unsqueeze(-2)
        self.ntokens = (self.src != self.pad).sum()

    def __init_decoder_only(self, sequence: torch.Tensor) -> None:
        """Initialize for decoder-only (GPT Style) models using a single input sequence."""
        self.tgt = sequence[:, :-1]
        self.tgt_y = sequence[:, 1:]
        self.tgt_mask = self.make_std_mask(self.tgt, self.pad)
        self.ntokens = (self.tgt_y != self.pad).data.sum()

    def __init_decoder_decoder(self, src: torch.Tensor, tgt: torch.Tensor) -> None:
        """Initialize for encoder-decoder (Transformer) models."""

        self.src = src
        self.src_mask = (src != self.pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, self.pad)
            self.ntokens = (self.tgt_y != self.pad).data.sum()

    @staticmethod
    def make_std_mask(tgt: torch.Tensor, pad: int) -> torch.Tensor:
        """
        Create a mask to hide padding and future words.

        Args:
            tgt (torch.Tensor): Target sequence
            pad (int): Padding token index

        Returns:
            torch.Tensor: Target mask
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

    def to(self, device: str) -> 'Batch':
        """Move batch to device."""
        for attr, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))
        return self


def lr_step(step: int, model_size: int, factor: float, warmup: int) -> float:
    """
    Compute the learning rate based on the step, model size, factor, and warmup.

    Args:
        step (int): Current step.
        model_size (int): Model dimension.
        factor (float): Scaling factor.
        warmup (int): Warmup steps.

    Returns:
        float: Computed learning rate.
    """
    if step == 0:
        step = 1
    return float(factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))))


def create_progress_bar() -> Progress:
    """Create an enhanced progress bar with comprehensive metrics."""
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


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    def __init__(self, save_dir: Path | None = None) -> None:
        self.step = 0  # Steps in the current epoch
        self.accum_step = 0  # Number of gradient accumulation steps
        self.samples = 0  # total # of examples used
        self.tokens = 0  # total # of tokens processed
        self.epoch = 0  # current epoch
        self.start_time = time.time()
        self.tokens_per_sec = 0
        self.save_dir = save_dir

    def update(self, batch_size: int, ntokens: int, loss: float, lr: float) -> 'TrainState':
        """Update training state with batch statistics.

        Args:
            batch_size (int): Number of samples in the batch
            ntokens (int): Number of tokens in the batch
            loss (float): Loss value for the batch
            lr (float): Current learning rate

        Returns:
            TrainState: The updated training state instance
        """
        self.step += 1
        self.samples += batch_size
        self.tokens += ntokens

        # Calculate tokens per second
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.tokens_per_sec = int(self.tokens / elapsed)

        # You can save additional metrics here if needed
        return self

    def save(self, path: Path) -> None:
        """Save training state to a file"""
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


def run_epoch(
    data_iter: DataLoader,
    model: torch.nn.Module,
    loss_compute: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    mode: Literal['train', 'eval', 'train+log'] = 'train',
    accum_iter: int = 1,
    max_batches: int | None = None,
    train_state: TrainState | None = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: Path | None = None,
) -> tuple[float, TrainState]:
    """
    Enhanced training/evaluation pipeline with comprehensive monitoring.

    Args:
        data_iter: Data iterator
        model: Neural network model
        loss_compute: Loss computation function
        optimizer: Optimization algorithm
        scheduler: Learning rate scheduler
        mode: Training mode ('train', 'eval', 'train+log')
        max_batches: Maximum number of batches to process
        accum_iter: Gradient accumulation steps
        train_state: Training state tracker
        device: Computation device
        save_dir: Directory to save metrics

    Returns:
        tuple[float, TrainingState]: (average_loss, updated_metrics)
    """
    train_state = train_state or TrainState(save_dir)
    total_loss = 0
    tokens = 0

    # Set model mode
    model.train(mode.startswith('train'))
    torch.set_grad_enabled(mode.startswith('train'))

    progress = create_progress_bar()
    total = min(len(data_iter), max_batches) if max_batches else len(data_iter)
    try:
        with progress:
            task_id = progress.add_task(description=f'Epoch {train_state.epoch}', total=total)

            for i, batch in enumerate(data_iter):
                if max_batches and i >= max_batches:
                    break

                batch = batch.to(device)

                # Forward pass based on model architecture
                if batch.model_type == 'encoder-only':
                    out = model.forward(batch.src, batch.src_mask)
                    loss, loss_node = loss_compute(out, batch.labels, batch.ntokens)
                elif batch.model_type == 'decoder-only':
                    out = model.forward(tgt=batch.tgt, tgt_mask=batch.tgt_mask)
                    loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
                else:
                    # Standard encoder-decoder
                    out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
                    loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

                # Training step
                if mode.startswith('train'):
                    loss_node.backward()

                    if (i + 1) % accum_iter == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        train_state.accum_step += 1
                        scheduler.step()

                # Update metrics
                batch_loss = loss / batch.ntokens
                current_lr = optimizer.param_groups[0]['lr'] if mode.startswith('train') else 0

                train_state.update(
                    batch_size=batch.src.size(0) if hasattr(batch, 'src') else batch.tgt.size(0),
                    ntokens=batch.ntokens,
                    loss=batch_loss.item(),
                    lr=current_lr,
                )

                total_loss += loss
                tokens += batch.ntokens

                # Update progress display
                progress_desc = f'⚡ {mode.capitalize()} • Loss: {batch_loss:.4f}'

                if mode.startswith('train'):
                    progress_desc += f' • LR: {current_lr:.2e}'

                progress_desc += f' • Speed: {train_state.tokens_per_sec:.0f} tokens/sec'
                progress.update(task_id, advance=1, description=progress_desc)

                # Cleanup
                del loss
                del loss_node

        # Save metrics if directory provided
        if save_dir and mode.startswith('train'):
            train_state.save(Path(save_dir))

    except Exception as e:
        print(f'Error during {mode}: {e!s}')
        raise

    return total_loss / tokens, train_state


@dataclass
class TrainerMetrics:
    """Track and manage training metrics across epochs.

    Attributes:
        epochs (list[int]): List of epochs.
        train_losses (list[float]): Training losses recorded for each epoch.
        val_losses (list[float]): Validation losses recorded for each epoch.
        train_times (list[float]): Time taken for training in each epoch.
        learning_rates (list[float]): Learning rates used in each epoch.

    Methods:
        update(train_loss, val_loss, epoch_time, lr, epoch):
            Update metrics with the latest epoch data.

        to_dict() -> dict[str, Any]:
            Convert the metrics to a dictionary format.

        from_dict(data) -> TrainerMetrics:
            Create a TrainerMetrics object from a dictionary.
    """

    epochs: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_times: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)

    def update(
        self, train_loss: float, val_loss: float, epoch_time: float, lr: float, epoch: int
    ) -> None:
        """Update metrics with the latest epoch data.

        Args:
            train_loss (float): The training loss for the epoch.
            val_loss (float): The validation loss for the epoch.
            epoch_time (float): The time taken for the epoch.
            lr (float): The learning rate used for the epoch.
            epoch (int): The current epoch number.
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_times.append(epoch_time)
        self.learning_rates.append(lr)
        self.epochs.append(epoch)

    def to_dict(self) -> dict[str, Any]:
        """Convert the metrics to a dictionary format.

        Returns:
            dict: A dictionary representation of the training metrics.
        """
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_times': self.train_times,
            'learning_rates': self.learning_rates,
            'epochs': self.epochs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TrainerMetrics':
        """Create a TrainerMetrics object from a dictionary.

        Args:
            data (dict): A dictionary containing the training metrics.

        Returns:
            TrainerMetrics: An instance of TrainerMetrics populated with the provided data.
        """
        metrics = cls()
        metrics.train_losses = data.get('train_losses', [])
        metrics.val_losses = data.get('val_losses', [])
        metrics.train_times = data.get('train_times', [])
        metrics.learning_rates = data.get('learning_rates', [])
        metrics.epochs = data.get('epochs', [])
        return metrics


class DummyOptimizer(torch.optim.Optimizer):
    """Dummy optimizer for evaluation mode"""

    def __init__(self) -> None:
        self.param_groups = [{'lr': 0}]

    def step(self) -> None:
        pass

    def zero_grad(self, set_to_none: bool = False) -> None:
        pass


class DummyScheduler:
    """Dummy scheduler for evaluation mode"""

    def step(self) -> None:
        pass


class Trainer:
    """Lightweight trainer for transformer models with callback support.

    This trainer class provides a flexible infrastructure for training transformer models
    with support for various features like callbacks, checkpointing, and different model types.

    Attributes:
        device (str): Device to run training on ('cuda' or 'cpu')
        model (torch.nn.Module): The transformer model to train
        optimizer (torch.optim.Optimizer): Optimizer for model training
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        loss_fn (Callable): Loss function for training
        train_dataloader (DataLoader): DataLoader for training data
        val_dataloader (DataLoader): DataLoader for validation data
        fast_dev_run (bool): If True, runs only one batch for training and validation
        model_type (Literal['encoder-decoder', 'encoder-only', 'decoder-only']):
        Type of transformer model

        grad_accumulation_steps (int): Number of steps to accumulate gradients
        metrics (TrainerMetrics): Object to track training metrics
        current_epoch (int): Current training epoch
        train_state (TrainState): Object tracking training state
        console (Console): Rich console for pretty printing
        callbacks (list[Callback]): List of callbacks for training
        stop_training (bool): Flag to stop training

    Example:
        >>> model = TransformerModel()
        >>> optimizer = torch.optim.Adam(
        ...     model.parameters()
        ... )
        >>> scheduler = torch.optim.lr_scheduler.StepLR(
        ...     optimizer,
        ...     step_size=1,
        ... )
        >>> loss_fn = torch.nn.CrossEntropyLoss()
        >>> trainer = Trainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     loss_fn=loss_fn,
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     callbacks=[
        ...         CheckpointCallback()
        ...     ],
        ... )
        >>> metrics = (
        ...     trainer.fit(
        ...         epochs=10
        ...     )
        ... )
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: Callable,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        device: str | None = None,
        grad_accumulation_steps: int = 1,
        fast_dev_run: bool = False,
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.fast_dev_run = fast_dev_run
        self.model_type: Literal['encoder-decoder', 'encoder-only', 'decoder-only'] = (
            model.config.model_type
        )
        self.grad_accumulation_steps = grad_accumulation_steps
        self.metrics = TrainerMetrics()
        self.current_epoch = 0
        self.train_state = TrainState()
        self.console = Console()
        self.callbacks = callbacks or []
        self.stop_training = False

    def fit(self, epochs: int) -> TrainerMetrics:
        """Train the model for the specified number of epochs.
        This method handles the complete training loop including validation if a validation
        dataloader is provided. It manages callbacks, metrics tracking, and supports
        early stopping.
        Args:
            epochs (int): Number of training epochs to run.
        Returns:
            TrainerMetrics: Object containing training metrics including train/val losses,
                learning rates, and epoch times.
        Notes:
            - If `fast_dev_run` is `True`, only one batch will be used for training and validation
            - Training can be stopped early by setting stop_training to True via callbacks
            - Progress is logged to console using rich formatting
            - Callbacks are executed at the start of training, end of each epoch,and end of training
        """

        self.console.print(
            Panel('🚀 Training Starting...', title='[bold blue]Status[/]', border_style='blue')
        )
        for callback in self.callbacks:
            callback.on_train_begin(self)

        if self.fast_dev_run:
            epochs = 1

        for _ in range(epochs):
            epoch_start = time.time()
            self.train_state.epoch = self.current_epoch

            # Training
            max_train = 1 if self.fast_dev_run else None
            train_loss, self.train_state = run_epoch(
                self.train_dataloader,
                self.model,
                self.loss_fn,
                self.optimizer,
                self.scheduler,
                mode='train',
                accum_iter=self.grad_accumulation_steps,
                train_state=self.train_state,
                device=self.device,
                max_batches=max_train,
            )

            # Validation
            val_loss = 0.0
            if self.val_dataloader is not None:
                max_val = 1 if self.fast_dev_run else None
                self.model.eval()
                val_loss, _ = run_epoch(
                    self.val_dataloader,
                    self.model,
                    self.loss_fn,
                    DummyOptimizer(),
                    DummyScheduler(),
                    mode='eval',
                    train_state=self.train_state,
                    device=self.device,
                    max_batches=max_val,
                )

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            self.metrics.update(
                train_loss, val_loss, epoch_time, current_lr, int(self.current_epoch)
            )

            if not self.fast_dev_run:
                for callback in self.callbacks:
                    callback.on_epoch_end(self.current_epoch, self)

            self._log_epoch_summary(
                self.current_epoch, train_loss, val_loss, epoch_time, current_lr
            )

            self.current_epoch += 1

            if self.stop_training:
                break

        for callback in self.callbacks:
            callback.on_train_end(self)

        self.console.print(
            Panel('✨ Training Complete!', title='[bold blue]Status[/]', border_style='blue')
        )
        return self._get_clean_metrics()

    def save_checkpoint(self, path: Path) -> None:
        """Save the current training state to a checkpoint file.

        The checkpoint contains:
            - Current epoch number
            - Model state dictionary
            - Optimizer state dictionary
            - Learning rate scheduler state dictionary (if exists)
            - Training metrics
            - Model type
            - Training state including:
                - Current step
                - Accumulation step
                - Number of processed samples
                - Number of processed tokens
                - Last completed epoch

        Args:
            path (Path): Path where to save the checkpoint file.

        Returns:
            None

        Example:
            >>> trainer.save_checkpoint(
            ...     Path(
            ...         'checkpoints/model.pt'
            ...     )
            ... )
        """

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': self.metrics.to_dict(),
            'model_type': self.model_type,
            'train_state': {
                'step': self.train_state.step,
                'accum_step': self.train_state.accum_step,
                'samples': self.train_state.samples,
                'tokens': self.train_state.tokens,
                'epoch': self.current_epoch - 1,  # Last completed epoch
            },
        }
        torch.save(checkpoint, path)
        self.console.print(f'[bold blue]Checkpoint saved to {path}[/]')

    def load_checkpoint(self, path: Path | str, load_optimizer: bool = True) -> None:
        """
        Load training state from checkpoint.
        This method restores model state, optimizer state, scheduler state, metrics, and training
        state from a saved checkpoint file.

        Args:
            path (Union[Path, str]): Path to the checkpoint file.
            load_optimizer (bool, optional): Whether to load optimizer state. Defaults to True.

        Raises:
            FileNotFoundError: If checkpoint file does not exist at specified path.

        Note:
            The checkpoint file is expected to contain some or all of the following:
            - model_state_dict: State of the model
            - optimizer_state_dict: State of the optimizer (if load_optimizer=True)
            - scheduler_state_dict: State of the learning rate scheduler
            - metrics: Training metrics history
            - train_state: Training state including steps and counters
            - epoch: Last completed epoch number
        """

        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            self.console.print(f'[bold red]Checkpoint {path} not found[/]')
            raise FileNotFoundError(f'Checkpoint not found at {path}')

        self.console.print(f'[bold green]Loading checkpoint from {path}[/]')
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        # Load scheduler state
        if (
            self.scheduler
            and 'scheduler_state_dict' in checkpoint
            and checkpoint['scheduler_state_dict']
        ):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load metrics and sync epoch counter
        if 'metrics' in checkpoint:
            self.metrics = TrainerMetrics.from_dict(checkpoint['metrics'])
            if self.metrics.epochs:
                last_metric_epoch = int(self.metrics.epochs[-1])
                self.current_epoch = max(
                    self.current_epoch,  # Preserve any previous adjustment
                    last_metric_epoch + 1,  # Start after last recorded epoch
                )

        # Load training state
        if 'train_state' in checkpoint:
            train_state_data = checkpoint['train_state']
            self.train_state.step = train_state_data.get('step', 0)
            self.train_state.accum_step = train_state_data.get('accum_step', 0)
            self.train_state.samples = train_state_data.get('samples', 0)
            self.train_state.tokens = train_state_data.get('tokens', 0)

        # Final epoch sync from checkpoint
        self.current_epoch = checkpoint.get('epoch', self.current_epoch)

        # Ensure we don't repeat epochs from metrics
        if self.metrics.epochs:
            last_metric_epoch = int(self.metrics.epochs[-1])
            if self.current_epoch <= last_metric_epoch:
                self.current_epoch = last_metric_epoch + 1

        self.console.print('[bold green]Successfully loaded checkpoint.[/]')

    def evaluate(self) -> float:
        """
        Evaluate the model on the validation set.

        Runs the model in evaluation mode on the validation dataset and computes the validation loss
        No gradients are computed during evaluation.

        Returns:
            float: The validation loss value computed over the entire validation set.
        """

        self.model.eval()
        val_loss, _ = run_epoch(
            self.val_dataloader,
            self.model,
            self.loss_fn,
            DummyOptimizer(),
            DummyScheduler(),
            mode='eval',
            train_state=self.train_state,
            device=self.device,
        )
        return val_loss

    def predict(
        self,
        src: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
        max_len: int = 50,
        start_symbol: int = 0,
    ) -> torch.Tensor:
        """
        Generate predictions using model's decoding strategy.
        This method generates predictions based on the model type. For encoder-only models,
        it performs direct classification. For encoder-decoder models, it uses greedy decoding
        to generate sequences.

        Args:
            src (torch.Tensor, optional): Input tensor for prediction. Required for
                encoder-only models.
            src_mask (torch.Tensor, optional): Mask for source sequence. Required for
                encoder-decoder models.
            max_len (int): Maximum length for sequence generation. Defaults to 50.
            start_symbol (int): Starting token for sequence generation. Defaults to 0.
            torch.Tensor: Predictions from the model. For encoder-only models, returns class
                indices. For encoder-decoder models, returns generated sequences.
        Raises:
            ValueError: If src is not provided for encoder-only models, or if either src or
                src_mask is not provided for encoder-decoder models."""

        self.model.eval()

        with torch.no_grad():
            match self.model_type:
                case 'encoder-only':
                    if src is None:
                        raise ValueError('src must be provided for encoder-only models')
                    src = src.to(self.device)
                    if src_mask is None:
                        src_mask = (src != getattr(self.model.config, 'pad_token_id', 0)).unsqueeze(
                            -2
                        )
                    else:
                        src_mask = src_mask.to(self.device)
                    output = self.model(src, src_mask)
                    predictions = torch.argmax(output, dim=-1)
                    return predictions

                case 'encoder-decoder':
                    if src is None or src_mask is None:
                        raise ValueError(
                            'Both src and src_mask must be provided for encoder-decoder models'
                        )
                    src = src.to(self.device)
                    src_mask = src_mask.to(self.device)
                    return greedy_decode(self.model, src, src_mask, max_len, start_symbol)

                case _:
                    return greedy_decode(self.model, src, src_mask, max_len, start_symbol)

    def _get_clean_metrics(self) -> TrainerMetrics:
        """
        Converts and returns clean metrics from trainer metrics.
        This method processes the metrics stored in the trainer by converting PyTorch tensor
        losses to float values while preserving non-tensor losses. The cleaned metrics are
        returned in a new TrainerMetrics object.

        Returns:
            TrainerMetrics: A new metrics object containing cleaned values with:
                - train_losses: List of float losses from training
                - val_losses: List of float losses from validation
                - train_times: List of training times per epoch
                - learning_rates: List of learning rates used
                - epochs: List of epoch numbers
        """

        clean_metrics = TrainerMetrics()
        clean_metrics.train_losses = [
            float(loss) if hasattr(loss, 'item') else loss for loss in self.metrics.train_losses
        ]
        clean_metrics.val_losses = [
            float(loss) if hasattr(loss, 'item') else loss for loss in self.metrics.val_losses
        ]
        clean_metrics.train_times = self.metrics.train_times
        clean_metrics.learning_rates = self.metrics.learning_rates
        clean_metrics.epochs = self.metrics.epochs
        return clean_metrics

    def _log_epoch_summary(
        self, epoch: int, train_loss: float, val_loss: float, epoch_time: float, lr: float
    ) -> None:
        """Log epoch summary information.

        Args:
            epoch (int): Current epoch number (0-based indexing).
            train_loss (float): Training loss value for the current epoch.
            val_loss (float): Validation loss value for the current epoch.
            epoch_time (float): Time taken to complete the epoch in seconds.
            lr (float): Current learning rate.

        Note:
            This method logs the epoch summary without tracking the best model.
            If validation dataloader is not provided, validation loss will be shown as N/A.
        """

        summary = [
            f'Epoch: {epoch + 1}',
            f'Train Loss: {train_loss:.4f}',
            f'Val Loss: {val_loss:.4f}' if self.val_dataloader else 'Val Loss: N/A',
            f'Time: {epoch_time:.2f}s',
            f'LR: {lr:.2e}',
        ]
        panel = Panel('\t'.join(summary), title='Epoch Summary', border_style='blue')
        self.console.print(panel)
