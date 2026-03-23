from torch import Tensor

from ..attention.multi_head import MultiHeadAttention
from ..attention.positional import create_pe
from ..blocks.causal_decoder import CausalDecoder
from ..config import ModelConfig
from ..core.embeddings import EmbeddingWithPE
from ..core.generator import Generator
from ..layers.causal_decoder_layer import CausalDecoderLayer
from .base import BaseModel


class DecoderOnlyModel(BaseModel):
    """
    GPT-style decoder-only transformer.

    Architecture:
        Embedding (+ PE if embedding-level) -> CausalDecoder -> Generator
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        vocab_size = config.vocab_size or config.tgt_vocab_size
        if vocab_size is None:
            raise ValueError('vocab_size or tgt_vocab_size required')

        # Create PE
        pe = create_pe(config)

        # Embedding (with PE if embedding-level)
        embed_pe = pe if pe and pe.injection_point == 'embedding' else None
        attn_pe = pe if pe and pe.injection_point != 'embedding' else None

        self.embed = EmbeddingWithPE(vocab_size, config.d_model, embed_pe)

        # Decoder
        attn = MultiHeadAttention(
            config.d_model,
            config.n_heads,
            config.attention_dropout,  # type: ignore
            config.attention_bias,
            attn_pe,
        )
        layer = CausalDecoderLayer(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
            config.pre_norm,
            config.norm_type,
            config.ff_activation,
            attn,
        )
        n_layers = config.n_layers if isinstance(config.n_layers, int) else config.n_layers[1]
        self.decoder = CausalDecoder(
            layer,
            n_layers,
            config.d_model,
            config.pre_norm,
            config.norm_type,
        )

        # Output projection
        self.generator = Generator(config.d_model, vocab_size)

        # Weight tying
        if config.tie_word_embeddings:
            self.generator.proj.weight = self.embed.embed.embed.weight

    def forward(  # type: ignore[override]
        self,
        tgt: Tensor,
        tgt_mask: Tensor | None = None,
        kv_cache: list[dict] | None = None,
        position_offset: int = 0,
        return_hidden: bool = False,
        # src / src_mask accepted for interface parity but unused
        src: Tensor | None = None,
        src_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            tgt: [batch, seq_len] token ids
            tgt_mask: [batch, 1, seq_len, seq_len] causal mask
            kv_cache: Optional KV cache for generation
            position_offset: Position offset for cached generation
            return_hidden: If True, also return hidden states

        Returns:
            logits: [batch, seq_len, vocab_size]
            hidden: [batch, seq_len, d_model] (if return_hidden)
        """
        x = self.embed(tgt)
        x = self.decoder(x, tgt_mask, kv_cache, position_offset)
        logits = self.generator(x)

        if return_hidden:
            return logits, x
        return logits

    def init_kv_cache(self) -> list[dict]:
        return self.decoder.init_kv_cache()
