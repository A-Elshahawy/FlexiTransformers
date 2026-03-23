import torch
from torch import Tensor, nn

from ..attention.multi_head import MultiHeadAttention
from ..attention.positional import create_pe
from ..blocks.encoder import Encoder
from ..config import ModelConfig
from ..core.embeddings import EmbeddingWithPE
from ..layers.encoder_layer import EncoderLayer
from .base import BaseModel


class EncoderOnlyModel(BaseModel):
    """
    BERT-style encoder-only transformer.

    Architecture:
        Embedding (+ PE) -> Encoder -> Classification Head
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        vocab_size = config.vocab_size or config.src_vocab_size
        if vocab_size is None:
            raise ValueError('vocab_size or src_vocab_size required')

        # Create PE
        pe = create_pe(config)

        embed_pe = pe if pe and pe.injection_point == 'embedding' else None
        attn_pe = pe if pe and pe.injection_point != 'embedding' else None

        self.embed = EmbeddingWithPE(vocab_size, config.d_model, embed_pe)

        # Encoder
        attn = MultiHeadAttention(
            config.d_model,
            config.n_heads,
            config.attention_dropout or 0.0,
            config.attention_bias,
            attn_pe,
        )
        layer = EncoderLayer(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
            config.pre_norm,
            config.norm_type,
            config.ff_activation,
            attn,
        )
        n_layers = config.n_layers if isinstance(config.n_layers, int) else config.n_layers[0]
        self.encoder = Encoder(
            layer,
            n_layers,
            config.d_model,
            config.pre_norm,
            config.norm_type,
        )

        # Classification head
        self.head: ClassificationHead | None
        if config.num_classes:
            self.head = ClassificationHead(config.d_model, config.num_classes, config.dropout)
        else:
            self.head = None

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        mask: Tensor | None = None,
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
            mask: [batch, 1, 1, seq_len] padding mask
            return_hidden: If True, also return hidden states

        Returns:
            logits: [batch, num_classes] or [batch, seq_len, d_model] if no head
            hidden: [batch, seq_len, d_model] (if return_hidden)
        """
        x = self.embed(input_ids)
        x = self.encoder(x, mask)

        logits = self.head(x) if self.head else x

        if return_hidden:
            return logits, x
        return logits


class ClassificationHead(nn.Module):
    """Classification head for encoder-only models."""

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """hidden_states: [batch, seq_len, d_model] -> [batch, num_classes]"""
        cls_token = hidden_states[:, 0]  # Take [CLS] token
        x = self.dropout(torch.tanh(self.dense(cls_token)))
        return self.classifier(x)
