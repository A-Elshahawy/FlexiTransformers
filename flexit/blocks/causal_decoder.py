from torch import Tensor, nn

from ..core.normalization import create_norm
from ..layers.causal_decoder_layer import CausalDecoderLayer


class CausalDecoder(nn.Module):
    """Stack of causal decoder layers (GPT-style)."""

    def __init__(
        self,
        layer: CausalDecoderLayer,
        n_layers: int,
        d_model: int,
        pre_norm: bool = True,
        norm_type: str = 'layernorm',
    ):
        super().__init__()
        self.layers = nn.ModuleList([self._clone_layer(layer) for _ in range(n_layers)])
        self.norm = create_norm(norm_type, d_model) if pre_norm else nn.Identity()
        self.n_layers = n_layers

    def _clone_layer(self, layer: CausalDecoderLayer) -> CausalDecoderLayer:
        import copy

        return copy.deepcopy(layer)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        kv_cache: list[dict] | None = None,
        position_offset: int = 0,
    ) -> Tensor:
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache else None
            x = layer(x, mask, layer_cache, position_offset)
        return self.norm(x)

    def init_kv_cache(self) -> list[dict]:
        """Initialize empty KV cache for inference."""
        return [{} for _ in range(self.n_layers)]
