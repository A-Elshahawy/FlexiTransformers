from torch import Tensor, nn


class Generator(nn.Module):
    """Output projection: hidden states -> logits."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: [batch, seq_len, d_model] -> [batch, seq_len, vocab_size]"""
        return self.proj(x)
