import math

from torch import nn

from ..config import ModelConfig
from ..models.base import BaseModel
from ..models.decoder_only import DecoderOnlyModel
from ..models.encoder_decoder import EncoderDecoderModel
from ..models.encoder_only import EncoderOnlyModel

MODEL_REGISTRY: dict[str, type[DecoderOnlyModel | EncoderOnlyModel | EncoderDecoderModel]] = {
    'decoder-only': DecoderOnlyModel,
    'encoder-only': EncoderOnlyModel,
    'encoder-decoder': EncoderDecoderModel,
}


class TransformerFactory:
    """Factory for creating transformer models."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def create(self) -> BaseModel:
        model_cls = MODEL_REGISTRY.get(self.config.model_type)
        if model_cls is None:
            raise ValueError(f'Unknown model type: {self.config.model_type}')

        model = model_cls(self.config)
        self._init_weights(model)
        return model

    def _init_weights(self, model: BaseModel) -> None:
        """Initialize model weights."""
        n_layers = self.config.n_layers
        total_layers = n_layers if isinstance(n_layers, int) else sum(n_layers)
        scaled_std = 0.02 / math.sqrt(2 * total_layers)
        for name, param in model.named_parameters():
            if param.dim() > 1:
                if self.config.init_method == 'xavier':
                    nn.init.xavier_uniform_(param)
                elif self.config.init_method == 'kaiming':
                    nn.init.kaiming_uniform_(param)
                elif self.config.init_method == 'normal':
                    nn.init.normal_(param, std=self.config.init_std)
                elif self.config.init_method == 'scaled':
                    nn.init.normal_(param, mean=0.0, std=scaled_std)
            elif 'bias' in name:
                nn.init.zeros_(param)

    @classmethod
    def from_config(cls, config: ModelConfig) -> BaseModel:
        return cls(config).create()


def create_model(config: ModelConfig) -> BaseModel:
    """Convenience function."""
    return TransformerFactory.from_config(config)
