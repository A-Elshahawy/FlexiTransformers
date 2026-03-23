from torch import Tensor, nn

from ..core.normalization import create_norm
from ..layers.cross_decoder_layer import CrossAttentionDecoderLayer


class CrossAttentionDecoder(nn.Module):
    """Stack of cross-attention decoder layers (Seq2Seq-style)."""

    def __init__(
        self,
        layer: CrossAttentionDecoderLayer,
        n_layers: int,
        d_model: int,
        pre_norm: bool = True,
        norm_type: str = 'layernorm',
    ):
        super().__init__()
        self.layers = nn.ModuleList([self._clone_layer(layer) for _ in range(n_layers)])
        self.norm = create_norm(norm_type, d_model) if pre_norm else nn.Identity()
        self.n_layers = n_layers

    def _clone_layer(self, layer: CrossAttentionDecoderLayer) -> CrossAttentionDecoderLayer:
        import copy

        return copy.deepcopy(layer)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        kv_cache: list[dict] | None = None,
        position_offset: int = 0,
    ) -> Tensor:
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache else None
            x = layer(x, memory, tgt_mask, memory_mask, layer_cache, position_offset)
        return self.norm(x)
