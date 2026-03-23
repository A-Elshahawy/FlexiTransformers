"""Batch handling for all transformer architectures."""

from typing import Literal

import torch

from ..utils.masks import subsequent_mask


class Batch:
    """
    Unified batch handling for all transformer architectures.
    Handles encoder-decoder, decoder-only, and encoder-only (BERT) models.
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
        self._validate_inputs(tgt, labels, model_type)
        self.model_type = model_type
        self.device = device
        self.pad = pad

        match model_type:
            case 'encoder-decoder':
                self.__init_encoder_decoder(src, tgt)
            case 'encoder-only':
                self.__init_encoder_only(src, labels)
            case 'decoder-only':
                self.__init_decoder_only(tgt if tgt is not None else src)
            case _:
                raise ValueError(f'Invalid model type: {model_type}')

    def _validate_inputs(
        self, tgt: torch.Tensor | None, labels: torch.Tensor | None, model_type: str
    ) -> None:
        if model_type == 'encoder-decoder' and tgt is None:
            raise ValueError('Target sequence required for encoder-decoder models')
        if model_type == 'encoder-only' and labels is None:
            raise ValueError('Labels required for encoder-only (BERT) models')

    def __init_encoder_only(self, src: torch.Tensor | None, labels: torch.Tensor | None) -> None:
        if src is None:
            raise ValueError('Source tensor cannot be None for encoder-only models')
        self.src = src
        self.labels = labels
        self.src_mask = (src != self.pad).unsqueeze(-2)
        self.ntokens = (self.src != self.pad).sum()

    def __init_decoder_only(self, sequence: torch.Tensor | None) -> None:
        if sequence is None:
            raise ValueError('Sequence cannot be None for decoder-only models')
        if sequence.size(1) < 2:
            raise ValueError(f'Sequence must have at least 2 tokens, got {sequence.size(1)}')
        self.tgt = sequence[:, :-1]
        self.tgt_y = sequence[:, 1:]
        self.tgt_mask = self.make_std_mask(self.tgt, self.pad)
        self.ntokens = (self.tgt_y != self.pad).data.sum()

    def __init_encoder_decoder(self, src: torch.Tensor | None, tgt: torch.Tensor | None) -> None:
        if src is None:
            raise ValueError('Source tensor (src) cannot be None for encoder-decoder models')
        self.src = src
        self.src_mask = (src != self.pad).unsqueeze(-2)
        if tgt is not None:
            if tgt.size(1) < 2:
                raise ValueError(f'Target must have at least 2 tokens, got {tgt.size(1)}')
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, self.pad)
            self.ntokens = (self.tgt_y != self.pad).data.sum()

    @staticmethod
    def make_std_mask(tgt: torch.Tensor, pad: int) -> torch.Tensor:
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

    def to(self, device: str) -> 'Batch':
        """Move batch to device."""
        for attr, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))
        return self
