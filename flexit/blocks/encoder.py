from torch import Tensor, nn

from ..core.normalization import create_norm
from ..layers.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """Stack of encoder layers with final norm."""

    def __init__(
        self,
        layer: EncoderLayer,
        n_layers: int,
        d_model: int,
        pre_norm: bool = True,
        norm_type: str = 'layernorm',
    ):
        super().__init__()
        self.layers = nn.ModuleList([self._clone_layer(layer) for _ in range(n_layers)])
        # Final norm only for pre-norm architecture
        self.norm = create_norm(norm_type, d_model) if pre_norm else nn.Identity()
        self.pre_norm = pre_norm

    def _clone_layer(self, layer: EncoderLayer) -> EncoderLayer:
        import copy

        return copy.deepcopy(layer)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
