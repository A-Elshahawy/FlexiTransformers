"""Sampling-based decoding strategies for transformer models.

Provides stochastic generation alternatives to greedy decoding:
  - temperature_sample   -- scale logits then draw from the distribution
  - top_k_sample         -- restrict to top-k tokens before sampling
  - top_p_sample         -- nucleus sampling (keep smallest set summing to >= p)
  - sample_decode        -- full autoregressive loop for decoder-only models
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.masks import subsequent_mask

# ---------------------------------------------------------------------------
# Logit-level sampling helpers
# ---------------------------------------------------------------------------


def temperature_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Sample a token index from logits with temperature scaling.

    Args:
        logits:      ``[batch, vocab_size]`` raw (unnormalised) logit scores.
        temperature: Softmax temperature. Values < 1 sharpen the distribution
                     (more deterministic); values > 1 flatten it (more random).

    Returns:
        ``[batch]`` sampled token indices.
    """
    if temperature <= 0.0:
        raise ValueError(f'temperature must be > 0, got {temperature}')
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def top_k_sample(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample from the top-k most likely tokens.

    Tokens outside the top-k receive -inf before softmax (zero probability).

    Args:
        logits:      ``[batch, vocab_size]`` raw logit scores.
        k:           Number of top tokens to keep. Must be >= 1.
        temperature: Softmax temperature applied after top-k filtering.

    Returns:
        ``[batch]`` sampled token indices.
    """
    if k < 1:
        raise ValueError(f'k must be >= 1, got {k}')
    return temperature_sample(_apply_top_k(logits, k), temperature)


def top_p_sample(
    logits: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Nucleus (top-p) sampling.

    Keeps the smallest set of tokens whose cumulative probability >= *p*,
    then samples from that set.

    Args:
        logits:      ``[batch, vocab_size]`` raw logit scores.
        p:           Cumulative probability threshold in (0, 1].
        temperature: Softmax temperature applied after nucleus filtering.

    Returns:
        ``[batch]`` sampled token indices.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f'p must be in (0, 1], got {p}')
    return temperature_sample(_apply_top_p(logits, p), temperature)


# ---------------------------------------------------------------------------
# Full autoregressive generation loop
# ---------------------------------------------------------------------------


def sample_decode(
    model: nn.Module,
    src: torch.Tensor | None,
    src_mask: torch.Tensor | None,
    max_len: int,
    start_symbol: int,
    end_symbol: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Autoregressive sampling loop for decoder-only models.

    Filters are applied in order: top-k -> top-p -> temperature.
    Omit both ``top_k`` and ``top_p`` for pure temperature sampling.

    Args:
        model:        A decoder-only ``nn.Module`` whose ``forward(tgt, tgt_mask)``
                      returns ``[batch, seq, vocab]`` logits.
        src:          Optional prompt ``[batch, prompt_len]``. When *None*,
                      generation starts from a single ``start_symbol`` token.
        src_mask:     Unused — kept for API symmetry with ``greedy_decode``.
        max_len:      Maximum total sequence length (prompt + generated).
        start_symbol: Token id used when *src* is None.
        end_symbol:   Optional EOS id; generation stops when all batch items
                      have emitted this token.
        temperature:  Softmax temperature (default 1.0 = unscaled).
        top_k:        If set, restrict sampling to the top-k logits.
        top_p:        If set, apply nucleus filtering at this probability mass.

    Returns:
        ``[batch, total_len]`` token id tensor (includes the prompt).
    """
    device = src.device if src is not None else next(model.parameters()).device

    ys = (
        src.clone().to(device)
        if src is not None
        else torch.full((1, 1), start_symbol, device=device, dtype=torch.long)
    )

    batch_size = ys.size(0)
    completed = torch.zeros(batch_size, dtype=torch.bool, device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_len - ys.size(1)):
            tgt_mask = subsequent_mask(ys.size(1)).to(device)

            out = model(tgt=ys, tgt_mask=tgt_mask)
            logits: torch.Tensor = out[:, -1, :] if out.dim() == 3 else out

            if top_k is not None:
                logits = _apply_top_k(logits, top_k)
            if top_p is not None:
                logits = _apply_top_p(logits, top_p)

            next_token = temperature_sample(logits, temperature).unsqueeze(1)
            ys = torch.cat([ys, next_token], dim=1)

            if end_symbol is not None:
                completed = completed | (next_token.squeeze(1) == end_symbol)
                if completed.all():
                    break

    return ys


# ---------------------------------------------------------------------------
# Internal filter helpers
# ---------------------------------------------------------------------------


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.size(-1))
    top_k_vals, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_vals[..., -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float('-inf'))


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_mask = (cumulative_probs - sorted_probs) > p
    sorted_logits = sorted_logits.masked_fill(remove_mask, float('-inf'))
    filtered = torch.full_like(logits, float('-inf'))
    filtered.scatter_(-1, sorted_idx, sorted_logits)
    return filtered
