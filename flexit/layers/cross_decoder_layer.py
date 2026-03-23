from torch import Tensor, nn

from ..attention.multi_head import MultiHeadAttention
from ..core.feedforward import create_feedforward
from .sublayer import SublayerConnection


class CrossAttentionDecoderLayer(nn.Module):
    """
    Seq2Seq decoder layer: Self-Attention -> Cross-Attention -> FFN
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
        self_attn: MultiHeadAttention | None = None,
        cross_attn: MultiHeadAttention | None = None,
    ):
        super().__init__()
        self.self_attn = self_attn or MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = cross_attn or MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = create_feedforward(d_model, d_ff, dropout, ff_activation)

        self.sublayer1 = SublayerConnection(d_model, dropout, pre_norm, norm_type)
        self.sublayer2 = SublayerConnection(d_model, dropout, pre_norm, norm_type)
        self.sublayer3 = SublayerConnection(d_model, dropout, pre_norm, norm_type)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        kv_cache: dict | None = None,
        position_offset: int = 0,
    ) -> Tensor:
        # Self-attention
        x = self.sublayer1(
            x, lambda x: self.self_attn(x, x, x, tgt_mask, kv_cache, position_offset)
        )
        # Cross-attention
        x = self.sublayer2(x, lambda x: self.cross_attn(x, memory, memory, memory_mask))
        # FFN
        x = self.sublayer3(x, self.ff)
        return x
