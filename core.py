import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.

    Args:
        encoder (nn.Module): Encoder module.
        decoder (nn.Module): Decoder module.
        src_embed (nn.Module): Source embedding module.
        tgt_embed (nn.Module): Target embedding module.
        generator (nn.Module): Generator module.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        generator: nn.Module,
    ) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the encoder-decoder model.

        Args:
            src (torch.Tensor): Source sequence.
            tgt (torch.Tensor): Target sequence.
            src_mask (torch.Tensor): Source mask.
            tgt_mask (torch.Tensor): Target mask.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    Implements the generator (linear + softmax).

    Args:
        d_model (int): Model dimension.
        vocab (int): Vocabulary size.
    """

    def __init__(self, d_model: int, vocab: int) -> None:
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.proj(x), dim=-1)
