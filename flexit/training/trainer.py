"""Main training loop and Trainer class."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
from rich.console import Console
from rich.panel import Panel
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..inference.decoding import greedy_decode
from .callbacks import Callback
from .schedulers import DummyOptimizer, DummyScheduler
from .state import TrainerMetrics, TrainState


def run_epoch(
    data_iter: DataLoader,
    model: torch.nn.Module,
    loss_compute: Callable,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    mode: Literal['train', 'eval'] = 'train',
    accum_iter: int = 1,
    max_batches: int | None = None,
    train_state: TrainState | None = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: Path | None = None,
) -> tuple[float, Any]:
    """Training loop with proper loss scaling and gradient accumulation."""

    train_state = train_state or TrainState(save_dir)
    total_loss = 0
    total_tokens = 0

    model.train(mode == 'train')
    torch.set_grad_enabled(mode == 'train')

    if mode == 'train' and optimizer:
        optimizer.zero_grad(set_to_none=True)

    total = min(len(data_iter), max_batches) if max_batches else len(data_iter)
    pbar = tqdm(
        total=total,
        desc=f'[{mode.upper()}] Epoch {train_state.epoch + 1}',
        bar_format='{l_bar}{bar:20}{r_bar}',
    )

    accumulated_steps = 0

    for i, batch in enumerate(data_iter):
        if max_batches and i >= max_batches:
            break

        batch = batch.to(device)

        if hasattr(batch, 'model_type'):
            if batch.model_type == 'encoder-only':
                # Encoder-only: model returns hidden states (no head) or class logits (with head)
                out = model.forward(batch.src, batch.src_mask)
                batch_size = batch.src.size(0)
                loss, loss_for_backward = loss_compute(out, batch.labels, batch_size)
            elif batch.model_type == 'decoder-only':
                # Decoder-only: LossCompute applies generator → pass hidden states
                _, hidden = model.forward(
                    tgt=batch.tgt, tgt_mask=batch.tgt_mask, return_hidden=True
                )
                loss, loss_for_backward = loss_compute(hidden, batch.tgt_y, batch.ntokens)
            else:
                # Encoder-decoder: LossCompute applies generator → pass hidden states
                _, hidden = model.forward(
                    batch.src, batch.tgt, batch.src_mask, batch.tgt_mask, return_hidden=True
                )
                loss, loss_for_backward = loss_compute(hidden, batch.tgt_y, batch.ntokens)
        else:
            _, hidden = model.forward(
                batch.src, batch.tgt, batch.src_mask, batch.tgt_mask, return_hidden=True
            )
            loss, loss_for_backward = loss_compute(hidden, batch.tgt_y, batch.ntokens)

        if mode == 'train' and optimizer:
            if accum_iter > 1:
                loss_for_backward = loss_for_backward / accum_iter
            loss_for_backward.backward()
            accumulated_steps += 1

            if accumulated_steps % accum_iter == 0:
                if hasattr(loss_compute, 'grad_clip') and loss_compute.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), loss_compute.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_state.accum_step += 1
                accumulated_steps = 0

                if scheduler and hasattr(scheduler, 'step_per_batch') and scheduler.step_per_batch:
                    scheduler.step()

        batch_tokens = batch.ntokens if hasattr(batch, 'ntokens') else batch.src.numel()
        current_lr = optimizer.param_groups[0]['lr'] if optimizer else 0

        if mode == 'train' and train_state:
            batch_size = batch.src.size(0) if hasattr(batch, 'src') else batch.tgt.size(0)
            train_state.update(batch_size, batch_tokens, loss.item(), current_lr)

        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{current_lr:.2e}'})
        pbar.update(1)

    # Flush remaining accumulated gradients
    if mode == 'train' and optimizer and accumulated_steps > 0:
        if hasattr(loss_compute, 'grad_clip') and loss_compute.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), loss_compute.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_state.accum_step += 1

    pbar.close()

    if (
        mode == 'train'
        and scheduler
        and (not hasattr(scheduler, 'step_per_batch') or not scheduler.step_per_batch)
    ):
        scheduler.step()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss, train_state


class Trainer:
    """Lightweight trainer for transformer models with callback support."""

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

        model_type = getattr(getattr(model, 'config', None), 'model_type', None)
        if model_type not in ('encoder-decoder', 'encoder-only', 'decoder-only'):
            raise ValueError(
                'model_type must be one of: encoder-decoder, encoder-only, decoder-only'
            )

        self.model_type: Literal['encoder-decoder', 'encoder-only', 'decoder-only'] = model_type
        self.grad_accumulation_steps = grad_accumulation_steps
        self.metrics = TrainerMetrics()
        self.current_epoch = 0
        self.train_state = TrainState()
        self.console = Console()
        self.callbacks = callbacks or []
        self.stop_training = False

    def fit(self, epochs: int) -> TrainerMetrics:
        """Train for the specified number of epochs."""
        self.console.print(
            Panel('Training Starting...', title='[bold blue]Status[/]', border_style='blue')
        )
        for callback in self.callbacks:
            callback.on_train_begin(self)

        if self.fast_dev_run:
            epochs = 1

        for _ in range(epochs):
            epoch_start = time.time()
            self.train_state.epoch = self.current_epoch

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
            Panel('Training Complete!', title='[bold blue]Status[/]', border_style='blue')
        )
        return self._get_clean_metrics()

    def save_checkpoint(self, path: Path) -> None:
        """Save current training state to a checkpoint file."""
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
                'epoch': self.current_epoch - 1,
            },
        }
        torch.save(checkpoint, path)
        self.console.print(f'[bold blue]Checkpoint saved to {path}[/]')

    def load_checkpoint(self, path: Path | str, load_optimizer: bool = True) -> None:
        """Load training state from a checkpoint file."""
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Checkpoint not found at {path}')

        self.console.print(f'[bold green]Loading checkpoint from {path}[/]')
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        if (
            self.scheduler
            and 'scheduler_state_dict' in checkpoint
            and checkpoint['scheduler_state_dict']
        ):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'metrics' in checkpoint:
            self.metrics = TrainerMetrics.from_dict(checkpoint['metrics'])
            if self.metrics.epochs:
                self.current_epoch = int(self.metrics.epochs[-1]) + 1

        if 'train_state' in checkpoint:
            ts = checkpoint['train_state']
            self.train_state.step = ts.get('step', 0)
            self.train_state.accum_step = ts.get('accum_step', 0)
            self.train_state.samples = ts.get('samples', 0)
            self.train_state.tokens = ts.get('tokens', 0)

        self.current_epoch = checkpoint.get('epoch', self.current_epoch)
        if self.metrics.epochs:
            last = int(self.metrics.epochs[-1])
            if self.current_epoch <= last:
                self.current_epoch = last + 1

        self.console.print('[bold green]Successfully loaded checkpoint.[/]')

    def evaluate(self) -> float:
        """Evaluate the model on the validation set."""
        if self.val_dataloader is None:
            raise ValueError('Validation dataloader is required for evaluation')
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
        end_symbol: int | None = None,
    ) -> torch.Tensor:
        """Generate predictions using greedy decoding."""
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
                    return torch.argmax(output, dim=-1)

                case 'encoder-decoder':
                    if src is None or src_mask is None:
                        raise ValueError(
                            'Both src and src_mask must be provided for encoder-decoder models'
                        )
                    return greedy_decode(
                        model=self.model,
                        src=src.to(self.device),
                        src_mask=src_mask.to(self.device),
                        max_len=max_len,
                        start_symbol=start_symbol,
                        end_symbol=end_symbol,
                    )

                case _:
                    return greedy_decode(
                        model=self.model,
                        src=src,
                        src_mask=src_mask,
                        max_len=max_len,
                        start_symbol=start_symbol,
                        end_symbol=end_symbol,
                    )

    def _get_clean_metrics(self) -> TrainerMetrics:
        clean = TrainerMetrics()
        clean.train_losses = [
            float(l) if hasattr(l, 'item') else l
            for l in self.metrics.train_losses  # noqa: E741
        ]
        clean.val_losses = [float(l) if hasattr(l, 'item') else l for l in self.metrics.val_losses]  # noqa: E741
        clean.train_times = self.metrics.train_times
        clean.learning_rates = self.metrics.learning_rates
        clean.epochs = self.metrics.epochs
        return clean

    def _log_epoch_summary(
        self, epoch: int, train_loss: float, val_loss: float, epoch_time: float, lr: float
    ) -> None:
        summary = [
            f'Epoch: {epoch + 1}',
            f'Train Loss: {train_loss:.4f}',
            f'Val Loss: {val_loss:.4f}' if self.val_dataloader else 'Val Loss: N/A',
            f'LR: {lr:.2e}',
            f'Tokens: {self.train_state.tokens}',
            f'Time: {epoch_time:.2f}s',
        ]
        self.console.print(Panel('\t'.join(summary), title='Epoch Summary', border_style='blue'))
