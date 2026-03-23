from torch import Tensor, nn

from ..core.normalization import create_norm


class SublayerConnection(nn.Module):
    """Residual connection with normalization."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        pre_norm: bool = True,
        norm_type: str = 'layernorm',
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm = create_norm(norm_type, d_model, norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        if self.pre_norm:
            return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))
