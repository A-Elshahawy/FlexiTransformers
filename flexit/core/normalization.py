import torch
from torch import Tensor, nn


class LayerNorm(nn.Module):
    """Layer normalization with optional bias."""

    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)  # FIX: unbiased=False
        out = (x - mean) / (std + self.eps) * self.weight
        return out + self.bias if self.bias is not None else out


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def create_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    if norm_type == 'layernorm':
        return LayerNorm(dim, eps)
    elif norm_type == 'rmsnorm':
        return RMSNorm(dim, eps)
    raise ValueError(f'Unknown norm type: {norm_type}')
