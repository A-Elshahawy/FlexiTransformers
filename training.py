import time
from collections.abc import Callable

import torch

from .utils import subsequent_mask


class Batch:
    """
        Object for holding a batch of data with mask during training.

    Args:
        src (torch.Tensor): Source sequence.
        tgt (Optional[torch.Tensor]): Target sequence. Default: None.
        pad (int): Padding token index. Default: 2.

    """

    def __init__(self, src: torch.Tensor, tgt: torch.Tensor | None = None, pad: int = 2) -> None:
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt: torch.Tensor, pad: int) -> torch.Tensor:
        """
        Create a mask to hide padding and future words.

        Args:
            tgt (torch.Tensor): Target sequence.
            pad (int): Padding token index.

        Returns:
            torch.Tensor: Target mask.
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class TrainState:
    """
    Track number of steps, examples, and tokens processed.

    Attributes:
        step (int): Steps in the current epoch.
        accum_step (int): Number of gradient accumulation steps.
        samples (int): Total number of examples used.
        tokens (int): Total number of tokens processed.
    """

    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


def run_epoch(
    data_iter,
    model,
    loss_compute: Callable,
    optimizer,
    scheduler,
    mode: str = 'train',
    accum_iter: int = 1,
    train_state: TrainState | None = None,
):
    """
    Train or evaluate a single epoch.

    Args:
        data_iter: Data iterator.
        model: Model to train or evaluate.
        loss_compute (Callable): Loss computation function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        mode (str): Mode of operation ("train", "eval", "train+log"). Default: "train".
        accum_iter (int): Gradient accumulation steps. Default: 1.
        train_state (TrainState): Training state tracker. Default: TrainState().

    Returns:
        tuple[float, TrainState]: Average loss and updated training state.
    """
    train_state = train_state or TrainState()
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == 'train' or mode == 'train+log':
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == 'train' or mode == 'train+log'):
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start
            print(
                (
                    'Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f '
                    + '| Tokens / Sec: %7.1f | Learning Rate: %6.1e'
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


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
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))
