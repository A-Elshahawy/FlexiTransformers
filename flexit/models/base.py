from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import Tensor, nn

from ..config import ModelConfig


class BaseModel(ABC, nn.Module):
    """Abstract base class for all transformer models."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model_type: str = config.model_type

    @abstractmethod
    def forward(self, *args: object, **kwargs: object) -> Tensor:  # type: ignore[override]
        ...

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __len__(self) -> int:
        return self.num_parameters()

    def __repr__(self) -> str:
        n = self.num_parameters()
        return (
            f'{self.__class__.__name__}('
            f'type={self.config.model_type}, '
            f'd_model={self.config.d_model}, '
            f'n_layers={self.config.n_layers}, '
            f'n_heads={self.config.n_heads}, '
            f'params={n:,})'
        )

    def save(self, path: str | Path) -> None:
        """Save model weights and config to a file."""
        torch.save({'config': self.config, 'state_dict': self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = 'cpu') -> 'BaseModel':
        """Load model from a file saved with :meth:`save`."""
        data = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(data['config'])
        model.load_state_dict(data['state_dict'])
        return model

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'BaseModel':
        return cls(config)
