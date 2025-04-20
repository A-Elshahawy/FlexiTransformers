from dataclasses import dataclass, fields
from typing import Any, Literal, cast

from typing_extensions import Self


@dataclass
class ModelConfig:
    """Configuration for transformer models.

    This dataclass holds configuration settings for various types of transformer models,
    including encoder-decoder, encoder-only, and decoder-only architectures.

    Attributes:
        src_vocab: Source vocabulary size.
            Required for encoder-decoder or encoder-only models.
        tgt_vocab: Target vocabulary size.
            Required for encoder-decoder or decoder-only models.
        num_classes: Number of classes for classification tasks (e.g., BERT-like).
        d_model: Model dimension. Default is 512.
        d_ff: Feed-forward dimension. Default is 2048.
        n_heads: Number of attention heads. Default is 8.
        dropout: Dropout probability. Default is 0.1.
        n_layers: Number of layers. Default is 6.
        mask_eps: Mask epsilon value for attention. Default is 1e-9.
        model_type: Model type. Options are:
            - 'encoder-decoder'
            - 'encoder-only'
            - 'decoder-only'
        pe_type: Positional encoding type. Options are:
            - 'absolute'
            - 'alibi'
            - 'relative'
            - 'rotary'
        pre_norm: Use pre-normalization. Default is True.
        device: Device for computation ('cpu' or 'gpu'). Default is 'cpu'.
        init_method: Weight initialization method. Options are:
            - 'xavier_uniform'
            - 'xavier_normal'
            - 'kaiming_uniform'
            - 'kaiming_normal'
            - 'orthogonal'
            - 'zero'
            - 'one'

    Examples:
        Create a ModelConfig for an encoder-decoder model:

        >>> config = ModelConfig(
        ...     src_vocab=32000,
        ...     tgt_vocab=32000,
        ...     d_model=512,
        ...     n_heads=8,
        ...     model_type='encoder-decoder',
        ...     pe_type='absolute',
        ... )

        Create a ModelConfig from a dictionary:

        >>> config_dict = {
        ...     'src_vocab': 32000,
        ...     'tgt_vocab': 32000,
        ...     'd_model': 512,
        ...     'n_heads': 8,
        ...     'model_type': 'encoder-decoder',
        ...     'pe_type': 'absolute',
        ... }
        >>> config = ModelConfig.from_dict(
        ...     config_dict
        ... )
    """

    src_vocab: int | None = None
    tgt_vocab: int | None = None
    num_classes: int | None = None
    cls_token_id: int | None = None
    d_model: int = 512
    d_ff: int = 2048
    n_heads: int = 8
    dropout: float = 0.1
    n_layers: tuple[int, int] | int = 6
    model_type: Literal['encoder-decoder', 'encoder-only', 'decoder-only'] = 'encoder-decoder'
    pe_type: Literal['absolute', 'alibi', 'relative', 'rotary'] = 'absolute'
    pre_norm: bool = True
    ff_activation: Literal[
        'relu', 'gelu', 'tanh', 'sigmoid', 'leaky_relu', 'silu', 'elu', 'selu'
    ] = 'relu'
    device: str = 'cpu'
    init_method: Literal[
        'xavier_uniform',
        'xavier_normal',
        'kaiming_uniform',
        'kaiming_normal',
        'orthogonal',
        'zero',
        'one',
    ] = 'xavier_uniform'

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> 'ModelConfig':
        """
        Create a ModelConfig instance from a dictionary.

        This method filters the input dictionary to include only valid configuration
        parameters and creates a new ModelConfig instance.

        Args:
            config_dict: Dictionary containing configuration parameters.
                Must contain valid ModelConfig field names.

        Returns:
            ModelConfig: A new instance initialized with the dictionary values.

        Example:
            Creating a config from a dictionary:

            >>> config_dict = {
            ...     'src_vocab': 32000,
            ...     'd_model': 512,
            ...     'model_type': 'encoder-decoder',
            ... }
            >>> config = ModelConfig.from_dict(
            ...     config_dict
            ... )
        """
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


class ConfigDescriptor:
    """
    Descriptor managing model configurations.

    This descriptor is used to manage model configurations in the `FlexiTransformer` class.
    It allows for easy configuration of the model using keyword arguments.

    Parameters
    ----------
    value : dict or ModelConfig
        The configuration dictionary or ModelConfig instance.

    Returns
    -------
    ModelConfig
        The validated ModelConfig instance.

    Raises
    ------
    AttributeError
        If the configuration is not initialized.
    TypeError
        If the configuration is not a dictionary or ModelConfig instance.

    Example
    -------
    >>> model = FlexiTransformer(
    ...     src_vocab=32000,
    ...     d_model=512,
    ... )
    >>> model.config
    ModelConfig(src_vocab=32000, d_model=512, ...)
    """

    def __init__(self) -> None:
        self.name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, instance: object, owner: type) -> ModelConfig | Self:
        if instance is None:
            return self
        if self.name is None:
            raise AttributeError('Descriptor name not set')
        value = instance.__dict__.get(self.name)
        if value is None:
            raise AttributeError(f'Config not initialized for {instance.__class__.__name__}')
        return cast(ModelConfig, value)

    def __set__(self, instance: object, value: dict[str, Any] | ModelConfig) -> None:
        if self.name is None:
            raise AttributeError('Descriptor name not set')
        if isinstance(value, dict):
            value = ModelConfig.from_dict(value)
        elif not isinstance(value, ModelConfig):
            raise TypeError('Config must be ModelConfig or dict')
        instance.__dict__[self.name] = value

    def __delete__(self, instance: object) -> None:
        if self.name is None:
            return
        instance.__dict__.pop(self.name, None)
