# models/encoder_decoder.py
from torch import Tensor

from ..attention.multi_head import MultiHeadAttention
from ..attention.positional import create_pe
from ..blocks.cross_decoder import CrossAttentionDecoder
from ..blocks.encoder import Encoder
from ..config import ModelConfig
from ..core.embeddings import EmbeddingWithPE
from ..core.generator import Generator
from ..layers.cross_decoder_layer import CrossAttentionDecoderLayer
from ..layers.encoder_layer import EncoderLayer
from .base import BaseModel


class EncoderDecoderModel(BaseModel):
    """
    Standard encoder-decoder transformer (T5, BART style).

    Architecture:
        Encoder: Embedding -> Encoder Stack
        Decoder: Embedding -> Cross-Attention Decoder Stack -> Generator
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        src_vocab = config.src_vocab_size or config.vocab_size
        tgt_vocab = config.tgt_vocab_size or config.vocab_size

        if not src_vocab or not tgt_vocab:
            raise ValueError('src_vocab_size and tgt_vocab_size required')

        # Create PE
        pe = create_pe(config)
        embed_pe = pe if pe and pe.injection_point == 'embedding' else None
        attn_pe = pe if pe and pe.injection_point != 'embedding' else None

        # Embeddings
        self.src_embed = EmbeddingWithPE(src_vocab, config.d_model, embed_pe)
        self.tgt_embed = EmbeddingWithPE(tgt_vocab, config.d_model, embed_pe)

        # Layer counts
        if isinstance(config.n_layers, tuple):
            n_enc, n_dec = config.n_layers
        else:
            n_enc = n_dec = config.n_layers

        # Encoder
        enc_attn = MultiHeadAttention(
            config.d_model,
            config.n_heads,
            config.attention_dropout,  # type: ignore
            config.attention_bias,
            attn_pe,
        )
        enc_layer = EncoderLayer(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
            config.pre_norm,
            config.norm_type,
            config.ff_activation,
            enc_attn,
        )
        self.encoder = Encoder(
            enc_layer,
            n_enc,
            config.d_model,
            config.pre_norm,
            config.norm_type,
        )

        # Decoder
        dec_self_attn = MultiHeadAttention(
            config.d_model,
            config.n_heads,
            config.attention_dropout,  # type: ignore
            config.attention_bias,
            attn_pe,
        )
        dec_cross_attn = MultiHeadAttention(
            config.d_model,
            config.n_heads,
            config.attention_dropout,  # type: ignore
            config.attention_bias,
            None,  # No PE for cross-attention
        )
        dec_layer = CrossAttentionDecoderLayer(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
            config.pre_norm,
            config.norm_type,
            config.ff_activation,
            dec_self_attn,
            dec_cross_attn,
        )
        self.decoder = CrossAttentionDecoder(
            dec_layer,
            n_dec,
            config.d_model,
            config.pre_norm,
            config.norm_type,
        )

        # Generator
        self.generator = Generator(config.d_model, tgt_vocab)

        # Weight tying
        if config.tie_word_embeddings:
            self.generator.proj.weight = self.tgt_embed.embed.embed.weight

    def encode(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """Encode source sequence."""
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        kv_cache: list[dict] | None = None,
        position_offset: int = 0,
    ) -> Tensor:
        """Decode target sequence."""
        x = self.tgt_embed(tgt)
        x = self.decoder(x, memory, tgt_mask, memory_mask, kv_cache, position_offset)
        return x

    def forward(  # type: ignore[override]
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Full forward pass.

        Returns:
            logits: [batch, tgt_len, vocab_size]
            hidden: [batch, tgt_len, d_model] (if return_hidden)
        """
        memory = self.encode(src, src_mask)
        x = self.decode(tgt, memory, tgt_mask, src_mask)
        logits = self.generator(x)

        if return_hidden:
            return logits, x
        return logits
