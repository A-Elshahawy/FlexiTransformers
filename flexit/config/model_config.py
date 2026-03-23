"""Model configuration dataclass."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Unified model configuration for all transformer architectures."""

    # Architecture
    model_type: Literal['encoder-decoder', 'encoder-only', 'decoder-only']
    d_model: int = 512
    d_ff: int = 2048
    n_heads: int = 8
    n_layers: int | tuple[int, int] = 6  # (enc, dec) for encoder-decoder

    # Vocabulary
    vocab_size: int | None = None  # Unified vocab
    src_vocab_size: int | None = None  # Encoder vocab (if different)
    tgt_vocab_size: int | None = None  # Decoder vocab (if different)

    # Positional Encoding
    pe_type: Literal[
        'absolute', 'learned', 'rotary', 'alibi', 'relative', 'relative_bias', 'none'
    ] = 'rotary'
    max_seq_len: int = 2048
    rope_base: int = 10000
    rope_percentage: float = 1.0

    # Normalization
    norm_type: Literal['layernorm', 'rmsnorm'] = 'layernorm'
    norm_eps: float = 1e-6
    pre_norm: bool = True

    # Regularization
    dropout: float = 0.1
    attention_dropout: float | None = None  # Defaults to dropout
    ff_dropout: float | None = None  # Defaults to dropout

    # FFN
    ff_activation: Literal['relu', 'gelu', 'silu', 'geglu', 'swiglu'] = 'gelu'
    ff_bias: bool = True

    # Attention
    attention_bias: bool = True

    # Initialization
    init_method: Literal['xavier', 'kaiming', 'normal', 'scaled'] = 'xavier'
    init_std: float = 0.02

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Classification (encoder-only)
    num_classes: int | None = None
    pooling: Literal['cls', 'mean', 'max'] = 'cls'

    # Weight tying
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        """Post-initialization validation and defaults."""
        # Set attention/ff dropout to main dropout if not specified
        if self.attention_dropout is None:
            self.attention_dropout = self.dropout
        if self.ff_dropout is None:
            self.ff_dropout = self.dropout

        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        # Check d_model is divisible by n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f'd_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})'
            )

        # Validate vocab sizes
        if self.vocab_size is None and self.src_vocab_size is None:
            raise ValueError('Either vocab_size or src_vocab_size must be specified')

        # For encoder-only and decoder-only, vocab_size should be set
        if (
            self.model_type in ['encoder-only', 'decoder-only']
            and self.vocab_size is None
            and self.src_vocab_size is not None
        ):
            self.vocab_size = self.src_vocab_size

        # For encoder-decoder, handle different vocab scenarios
        if self.model_type == 'encoder-decoder':
            if self.vocab_size is not None:
                # Unified vocabulary
                if self.src_vocab_size is None:
                    self.src_vocab_size = self.vocab_size
                if self.tgt_vocab_size is None:
                    self.tgt_vocab_size = self.vocab_size
            else:
                # Separate vocabularies
                if self.src_vocab_size is None or self.tgt_vocab_size is None:
                    raise ValueError(
                        'For encoder-decoder with separate vocabs, both '
                        'src_vocab_size and tgt_vocab_size must be specified'
                    )

        # Validate n_layers
        if self.model_type == 'encoder-decoder':
            if isinstance(self.n_layers, int):
                self.n_layers = (self.n_layers, self.n_layers)
            elif len(self.n_layers) != 2:
                raise ValueError('n_layers for encoder-decoder must be int or tuple of 2 ints')
        else:
            if isinstance(self.n_layers, tuple):
                raise ValueError(f'n_layers for {self.model_type} must be int, not tuple')

        # Validate rope parameters
        if self.pe_type == 'rotary':
            if not 0 < self.rope_percentage <= 1.0:
                raise ValueError('rope_percentage must be in (0, 1]')
            if self.rope_base <= 0:
                raise ValueError('rope_base must be positive')

        # Validate dropout values
        for dropout_name in ['dropout', 'attention_dropout', 'ff_dropout']:
            val = getattr(self, dropout_name)
            if val is not None and not 0 <= val < 1:
                raise ValueError(f'{dropout_name} must be in [0, 1)')

        # Validate classification config
        if (
            self.model_type == 'encoder-only'
            and self.num_classes is not None
            and self.num_classes < 2
        ):
            raise ValueError('num_classes must be >= 2')

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create config from dictionary."""
        # Filter only valid fields
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def __repr__(self) -> str:
        """Pretty string representation."""
        fields = []
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if v is not None and v != self.__dataclass_fields__[k].default:
                fields.append(f'{k}={v}')
        return f'ModelConfig({", ".join(fields)})'
