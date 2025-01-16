import torch
import torch.nn as nn
from utils import subsequent_mask


class LabelSmoothing(nn.Module):
    """
    Implement label smoothing.

    Args:
        size (int): Vocabulary size.
        padding_idx (int): Padding token index.
        smoothing (float): Smoothing value. Default: 0.0.
    """

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0) -> None:
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
    A simple loss compute and train function.

    Args:
        generator: Generator module.
        criterion: Loss criterion.
    """

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, norm: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.generator(x)
        sloss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        return sloss.data * norm, sloss


def greedy_decode(
    model: nn.Module, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, start_symbol: int
) -> torch.Tensor:
    """
    Perform greedy decoding.

    Args:
        model: Model to use for decoding.
        src (torch.Tensor): Source sequence.
        src_mask (torch.Tensor): Source mask.
        max_len (int): Maximum length of the output sequence.
        start_symbol (int): Start token index.

    Returns:
        torch.Tensor: Decoded sequence.
    """
    device = src.device
    memory = model.encode(src.to(device), src_mask.to(device))
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src).to(device)

    for _ in range(max_len - 1):
        mask = subsequent_mask(ys.size(1)).type_as(src.data).to(device)
        out = model.decode(memory, src_mask.to(device), ys, mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word).to(device)], dim=1)

    return ys
