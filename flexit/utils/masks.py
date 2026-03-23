"""Mask utilities for transformer models."""

import torch
from torch import Tensor


def create_padding_mask(seq: Tensor, pad_id: int = 0) -> Tensor:
    """
    Create padding mask from token sequence.

    Args:
        seq: [batch, seq_len] token ids
        pad_id: Padding token id (default: 0)

    Returns:
        mask: [batch, 1, 1, seq_len] - True for valid tokens, False for padding
    """
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len: int, device: torch.device | None = None) -> Tensor:
    """
    Create causal (autoregressive) mask.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        mask: [1, 1, seq_len, seq_len] - lower triangular (True for visible positions)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)


def create_combined_mask(seq: Tensor, pad_id: int = 0) -> Tensor:
    """
    Create combined causal + padding mask for decoder.

    Combines autoregressive masking (can't attend to future) with padding masking
    (can't attend to padding tokens).

    Args:
        seq: [batch, seq_len] token ids
        pad_id: Padding token id

    Returns:
        mask: [batch, 1, seq_len, seq_len] - True for valid positions
    """
    _, seq_len = seq.shape

    # Causal mask: [1, 1, seq_len, seq_len]
    causal = create_causal_mask(seq_len, seq.device)

    # Padding mask: [batch, 1, 1, seq_len]
    padding = create_padding_mask(seq, pad_id)

    # Combine: valid if both causal allows AND not padding
    # Broadcasting: [1, 1, seq_len, seq_len] & [batch, 1, 1, seq_len]
    # -> [batch, 1, seq_len, seq_len]
    return causal & padding


def create_look_ahead_mask(
    tgt_seq: Tensor,
    src_seq: Tensor | None = None,
    pad_id: int = 0,
) -> tuple[Tensor, Tensor | None]:
    """
    Create masks for encoder-decoder model.

    Args:
        tgt_seq: [batch, tgt_len] target sequence
        src_seq: [batch, src_len] source sequence (optional)
        pad_id: Padding token id

    Returns:
        tgt_mask: [batch, 1, tgt_len, tgt_len] target self-attention mask
        src_mask: [batch, 1, 1, src_len] source padding mask (None if src_seq is None)
    """
    # Target mask: causal + padding
    tgt_mask = create_combined_mask(tgt_seq, pad_id)

    # Source mask: padding only
    src_mask = None
    if src_seq is not None:
        src_mask = create_padding_mask(src_seq, pad_id)

    return tgt_mask, src_mask


def apply_mask(scores: Tensor, mask: Tensor | None, fill_value: float = -1e9) -> Tensor:
    """
    Apply mask to attention scores.

    Args:
        scores: [batch, n_heads, q_len, k_len] attention scores
        mask: [batch, 1, q_len, k_len] or broadcastable - True for valid, False to mask
        fill_value: Value to fill masked positions (default: -1e9)

    Returns:
        masked_scores: Same shape as scores
    """
    if mask is None:
        return scores

    # Convert boolean mask to float mask
    # True -> 0.0 (keep), False -> fill_value (mask out)
    return scores.masked_fill(~mask, fill_value)


def subsequent_mask(size: int, device: torch.device | None = None) -> Tensor:
    """
    Create subsequent (causal) mask. Alias for create_causal_mask.

    Args:
        size: Sequence length
        device: Device to create mask on

    Returns:
        mask: [1, size, size] - lower triangular
    """
    return torch.tril(torch.ones(1, size, size, device=device, dtype=torch.bool))
