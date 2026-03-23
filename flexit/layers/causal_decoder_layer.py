from torch import Tensor, nn

from ..attention.multi_head import MultiHeadAttention
from ..core.feedforward import create_feedforward
from .sublayer import SublayerConnection


class CausalDecoderLayer(nn.Module):
    """
    GPT-style decoder layer: Causal Self-Attention -> FFN
    (No cross-attention)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        pre_norm: bool = True,
        norm_type: str = 'layernorm',
        ff_activation: str = 'gelu',
        attn: MultiHeadAttention | None = None,
    ):
        super().__init__()
        self.self_attn = attn or MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = create_feedforward(d_model, d_ff, dropout, ff_activation)

        self.sublayer1 = SublayerConnection(d_model, dropout, pre_norm, norm_type)
        self.sublayer2 = SublayerConnection(d_model, dropout, pre_norm, norm_type)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        kv_cache: dict | None = None,
        position_offset: int = 0,
    ) -> Tensor:
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask, kv_cache, position_offset))
        x = self.sublayer2(x, self.ff)
        return x
