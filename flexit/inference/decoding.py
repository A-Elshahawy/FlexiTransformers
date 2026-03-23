"""Greedy decoding strategies for transformer models."""

from abc import ABC

import torch
import torch.nn as nn

from ..utils.masks import subsequent_mask


class DecoderStrategy(nn.Module, ABC):
    """Base class for decoding strategies."""

    @staticmethod
    def decode(
        model: nn.Module,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int,
        start_symbol: int,
        end_symbol: int | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class EncoderDecoderStrategy(DecoderStrategy):
    """Greedy decoding for encoder-decoder (seq2seq) models."""

    @staticmethod
    def decode(
        model: nn.Module,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int,
        start_symbol: int,
        end_symbol: int | None = None,
    ) -> torch.Tensor:
        batch_size = src.size(0)
        device = src.device

        memory = model.encode(src, src_mask)  # type: ignore[operator]
        ys = torch.full((batch_size, 1), start_symbol, device=device).type_as(src)
        completed = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data).to(device)
            out = model.decode(ys, memory, tgt_mask, src_mask)  # type: ignore[operator]
            logits = model.generator(out[:, -1])  # type: ignore[operator]
            _, next_word = torch.max(logits, dim=1)
            next_word = torch.clamp(next_word, 0, logits.size(-1) - 1).unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)

            if end_symbol is not None:
                completed = completed | (next_word.squeeze(1) == end_symbol)
                if completed.all():
                    break

        return ys


class DecoderOnlyStrategy(DecoderStrategy):
    """Greedy decoding for decoder-only (GPT-style) models."""

    @staticmethod
    def decode(
        model: nn.Module,
        src: torch.Tensor | None,
        src_mask: torch.Tensor | None,
        max_len: int,
        start_symbol: int,
        end_symbol: int | None = None,
    ) -> torch.Tensor:
        device = src.device if src is not None else next(model.parameters()).device

        if src is None:
            ys = torch.full((1, 1), start_symbol, device=device, dtype=torch.long)
        else:
            ys = src.clone().to(device)

        batch_size = ys.size(0)
        completed = torch.zeros(batch_size, dtype=torch.bool, device=device)
        model.eval()

        with torch.no_grad():
            for _ in range(max_len - 1):
                tgt_mask = subsequent_mask(ys.size(1)).to(device)

                out = model(tgt=ys, tgt_mask=tgt_mask)

                # out is [batch, seq, vocab] — model already applied generator
                logits = out[:, -1, :] if out.dim() == 3 else out
                _, next_word = torch.max(logits, dim=1)
                next_word = torch.clamp(next_word, 0, logits.size(-1) - 1).unsqueeze(1)
                ys = torch.cat([ys, next_word], dim=1)

                if end_symbol is not None:
                    completed = completed | (next_word.squeeze(1) == end_symbol)
                    if completed.all():
                        break

        return ys


def greedy_decode(
    model: nn.Module,
    src: torch.Tensor | None,
    src_mask: torch.Tensor | None,
    max_len: int,
    start_symbol: int,
    end_symbol: int | None = None,
) -> torch.Tensor:
    """Dispatch to the appropriate decoding strategy based on model type."""
    strategies = {
        'encoder-decoder': EncoderDecoderStrategy,
        'decoder-only': DecoderOnlyStrategy,
    }
    model_type = getattr(model, 'model_type', 'encoder-decoder')
    strategy = strategies.get(model_type)
    if strategy is None:
        raise ValueError(f'Unsupported model type: {model_type}')
    return strategy.decode(model, src, src_mask, max_len, start_symbol, end_symbol)  # type: ignore[arg-type]
