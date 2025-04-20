"""
Transformer Models

This module implements various transformer model variants,
including encoder-decoder, encoder-only, and decoder-only architectures.
"""

from typing import Any, Generic, TypeVar, cast

import torch
import torch.nn as nn

from .configs import ConfigDescriptor, ModelConfig
from .factory import TransformerFactory

T = TypeVar('T', bound=ModelConfig)


class BaseTransformer(Generic[T], nn.Module):
    """Base transformer that initializes configuration and model building.
    This class serves as a base implementation for transformer models, handling configuration
    initialization and model construction. It provides a generic interface for encoder-decoder,
    encoder-only, and decoder-only transformer architectures.

    Attributes:
        config (ModelConfig): Configuration object containing model parameters.
        factory (TransformerFactory): Factory class for creating transformer models.
        _base_model (nn.Module): The underlying transformer model instance.
        generator (nn.Module): Output generator module of the model.
        encoder (nn.Module): Encoder module if present in the model.
        decoder (nn.Module): Decoder module if present in the model.

        Example usage:
        >>> config = {
        ...     'model_type': 'encoder-decoder',
        ...     'src_vocab': 1000,
        ...     'tgt_vocab': 1000,
        ... }
        >>> transformer = (
        ...     BaseTransformer(
        ...         config
        ...     )
        ... )
        >>> output = (
        ...     transformer(
        ...         input_ids
        ...     )
        ... )
    Note:
        This class uses a ConfigDescriptor for managing model configuration and provides
        flexible initialization through either a configuration dictionary/object or
        individual parameter settings.
    """

    config = ConfigDescriptor()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize base transformer.

        Args:
            *args: Variable arguments.
            **kwargs: Keyword arguments for model configuration.
        """
        super().__init__()
        self._initialize_config(*args, **kwargs)

        self.factory = TransformerFactory(config=cast(ModelConfig, self.config))
        self._base_model = self.factory.create_model()

        # Transfer model attributes to self
        if hasattr(self._base_model, 'generator'):
            self.generator = self._base_model.generator
        elif isinstance(self._base_model, nn.Sequential):
            # For decoder-only models
            self.generator = self._base_model[-1]

        if hasattr(self._base_model, 'encoder'):
            self.encoder = self._base_model.encoder
        if hasattr(self._base_model, 'decoder'):
            self.decoder = self._base_model.decoder

        # Copy all methods and attributes from _base_model that don't exist in self
        for name in dir(self._base_model):
            if not name.startswith('_') and not hasattr(self, name):
                setattr(self, name, getattr(self._base_model, name))

    def _initialize_config(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize model configuration from arguments.
        This method sets up the model configuration either from a dictionary/ModelConfig object
        or from individual parameters. If model_type is not specified, it will be inferred
        from the presence of src_vocab and/or tgt_vocab parameters.

        Args:
            args: Variable length argument list. If single argument is provided,
                    it should be either a dictionary or ModelConfig object.
            kwargs: Arbitrary keyword arguments. Used to specify model parameters individually.
                    Common parameters include:
                    - model_type (str): Type of the model ('encoder-decoder', 'encoder-only',
                                        or 'decoder-only')
                    - src_vocab (int): Source vocabulary size
                    - tgt_vocab (int): Target vocabulary size
                    - d_model (int, optional): Model dimension. Defaults to 512
                    - d_ff (int, optional): Feed-forward dimension. Defaults to 2048
                    - n_heads (int, optional): Number of attention heads. Defaults to 8
                    - dropout (float, optional): Dropout rate. Defaults to 0.1
                    - n_layers (int, optional): Number of layers. Defaults to 6
                    - pre_norm (bool, optional): Whether to use pre-norm. Defaults to True
                    - device (str, optional): Device to use. Defaults to 'cpu'

        Raises:
            ValueError: If neither src_vocab nor tgt_vocab is specified in the configuration.

        Returns:
            None
        """

        if len(args) == 1 and isinstance(args[0], dict | ModelConfig):
            self.config = (
                args[0] if isinstance(args[0], ModelConfig) else ModelConfig.from_dict(args[0])
            )
        else:
            if 'model_type' not in kwargs:
                if 'src_vocab' in kwargs and 'tgt_vocab' in kwargs:
                    kwargs['model_type'] = 'encoder-decoder'
                elif 'src_vocab' in kwargs:
                    kwargs['model_type'] = 'encoder-only'
                elif 'tgt_vocab' in kwargs:
                    kwargs['model_type'] = 'decoder-only'
                else:
                    raise ValueError(
                        "Must specify at least 'src_vocab' \
                                        or 'tgt_vocab' in configuration."
                    )

            config_dict = {
                'd_model': 512,
                'd_ff': 2048,
                'n_heads': 8,
                'dropout': 0.1,
                'n_layers': 6,
                'pre_norm': True,
                'device': 'cpu',
            }
            config_dict.update(kwargs)
            self.config = ModelConfig.from_dict(config_dict)

    def encode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forwards the encode method call to the underlying model.
        This method attempts to call the 'encode' method on the underlying model instance.
        If the model does not have an encode method, it raises an AttributeError.

        Args:
            *args: Variable length argument list to pass to model's encode method
            **kwargs: Arbitrary keyword arguments to pass to model's encode method

        Returns:
            torch.Tensor: The encoded output from the model's encode method

        Raises:
            AttributeError: If the underlying model does not have an encode method
        """

        if hasattr(self._base_model, 'encode'):
            return self._base_model.encode(*args, **kwargs)
        raise AttributeError(f"'{self.__class__.__name__}' has no encode method")

    def decode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Decodes the model's output.
        This method attempts to call the decode method of the underlying model if it exists.
        If the model doesn't have a decode method, it raises an AttributeError.

        Args:
            *args: Variable length argument list to pass to model's decode method.
            **kwargs: Arbitrary keyword arguments to pass to model's decode method.

        Returns:
            The decoded output from the model's decode method.

        Raises:
            AttributeError: If the underlying model doesn't have a decode method.
        """

        if hasattr(self._base_model, 'decode'):
            return self._base_model.decode(*args, **kwargs)
        raise AttributeError(f"'{self.__class__.__name__}' has no decode method")

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass of the model wrapper.
        This method performs a forward pass by delegating to the underlying model's forward method.

        Args:
            *args: Variable length argument list to be passed to the underlying model.
            **kwargs: Arbitrary keyword arguments to be passed to the underlying model.

        Returns:
            torch.Tensor: The output tensor from the model's forward pass.
        """

        return self._base_model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Fallback method to get attributes from configuration or model.

        This method is called when an attribute is not found in the normal places. It first checks
        if the attribute exists in the configuration object, then checks the model object. If the
        attribute is not found in either place, it raises an AttributeError.

        Args:
            name (str): The name of the attribute to get.

        Returns:
            Any: The value of the attribute from either config or model.

        Raises:
            AttributeError: If the attribute is not found in either config or model.

        Example:
            >>> instance.some_attribute  # If not found directly, this method is called
            >>> # Checks self.config.some_attribute, then self.model.some_attribute
        """

        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.config, name):
                return getattr(self.config, name)
            if hasattr(self._base_model, name):
                return getattr(self._base_model, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        """
        String representation of the model showing architecture.
        Returns:
            str: Model name and architecture details
        """
        model_repr = f'{self.__class__.__name__}(\n'

        # Add config parameters
        model_repr += f'  (config): {vars(self.config)}\n'

        # Add underlying model architecture
        base_model_repr = self._base_model.__repr__()
        # Indent the base model representation
        base_model_repr = '  ' + base_model_repr.replace('\n', '\n  ')
        model_repr += f'  (base_model): {base_model_repr}\n'

        # Add other important attributes
        for name in ['encoder', 'decoder', 'generator']:
            if hasattr(self, name):
                attr = getattr(self, name)
                attr_repr = attr.__repr__()
                # Indent the attribute representation
                attr_repr = '  ' + attr_repr.replace('\n', '\n  ')
                model_repr += f'  ({name}): {attr_repr}\n'

        model_repr += ')'
        return model_repr

    def __len__(self) -> int:
        """Returns the total number of layers."""
        c = self.factory.config
        return c.n_layers if isinstance(c.n_layers, int) else sum(c.n_layers)

    def save(self, path: str) -> None:
        """
        Save the configuration and model state.

        Saves the model's configuration and state dictionary to a file at the specified path.

        Args:
            path (str): The file path where the model and configuration will be saved.

        Example:
            >>> model.save(
            ...     'model_checkpoint.pt'
            ... )
        """
        torch.save({'config': self.config, 'model_state': self._base_model.state_dict()}, path)

    def load(self, path: str) -> None:
        """Load configuration and model state from file.

        Args:
            path (str): Path to the checkpoint file containing model configuration and state.

        Returns:
            None

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the checkpoint file is corrupted or incompatible.
        """

        checkpoint = torch.load(path)
        self.config = checkpoint['config']
        self._base_model.load_state_dict(checkpoint['model_state'])


class FlexiTransformer(BaseTransformer):
    """
    This class provides a flexible implementation that can be configured for different
    transformer architectures including BERT-style encoder-only models, GPT-style decoder-only
    models, and full encoder-decoder transformer models.

    Args:
        src_vocab (int, optional): Size of source vocabulary for encoder. Required for encoder-only
            and encoder-decoder models.
        tgt_vocab (int, optional): Size of target vocabulary for decoder. Required for decoder-only
            and encoder-decoder models.
        n_layers (int, optional): Number of transformer layers. Defaults to model's default setting.
        n_heads (int, optional): Number of attention heads. Defaults to model's default setting.
        **kwargs: Additional keyword arguments passed to parent BaseTransformer class.
    Examples:
        Encoder-only (BERT-style):
        Decoder-only (GPT-style):
        Encoder-Decoder (Transformer-style):

    Note:
        This class uses a ConfigDescriptor for managing model configuration and provides
        flexible initialization through either a configuration dictionary/object or
        individual parameter settings.

    usage:
    >>> config = {
    ...     'model_type': 'encoder-decoder',
    ...     'src_vocab': 1000,
    ...     'tgt_vocab': 1000,
    ...     'd_model': 768,
    ...     'n_heads': 12,
    ...     'n_layers': 12,
    ...     'pe_type': 'absolute',
    ...     'init_method': 'xavier_uniform',
    ...     'pre_norm': True,
    ... }

    >>> transformer = FlexiTransformer(
    ...     **config
    ... )
    >>> output = (
    ...     transformer(
    ...         input_ids
    ...     )
    ... )

    >>> config = {
    ...     'model_type': 'encoder-only',
    ...     'src_vocab': 1000,
    ...     'd_model': 768,
    ...     'n_heads': 12,
    ...     'n_layers': 12,
    ...     'pe_type': 'absolute',
    ...     'init_method': 'xavier_uniform',
    ...     'pre_norm': True,
    ... }

    >>> BERT = FlexiTransformer(
    ...     **config
    ... )
    >>> output = (
    ...     transformer(
    ...         input_ids
    ...     )
    ... )

    >>> config = {
    ...     'model_type': 'decoder-only',
    ...     'tgt_vocab': 1000,
    ...     'd_model': 768,
    ...     'n_heads': 12,
    ...     'n_layers': 12,
    ...     'pe_type': 'absolute',
    ...     'init_method': 'xavier_uniform',
    ...     'pre_norm': True,
    ... }

    >>> GPT = FlexiTransformer(
    ...     **config
    ... )
    >>> output = (
    ...     transformer(
    ...         input_ids
    ...     )
    ... )
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the FlexiTransformer model with the given configuration.

        Args:
            **kwargs: The configuration options for the model.

        Returns:
            None
        """
        super().__init__(**kwargs)


class TransformerModel(FlexiTransformer):
    """A flexible implementation of the Transformer architecture.

    This class implements a configurable Transformer model that can be adapted for
    various sequence-to-sequence tasks. It extends the FlexiTransformer base class
    with specific configurations for a standard Transformer architecture.

    Args:
        src_vocab (int): Size of the source vocabulary.
        tgt_vocab (int): Size of the target vocabulary.
        d_model (int, optional): Dimension of the model's hidden states. Defaults to 512.
        n_heads (int, optional): Number of attention heads in multi-head attention layers.
            Defaults to 8.
        n_layers (int, optional): Number of encoder and decoder layers. Defaults to 6.
        pe_type (str, optional): Type of positional encoding to use ('absolute' or 'relative').
            Defaults to 'absolute'.
        init_method (str, optional): Weight initialization method. Defaults to 'xavier_uniform'.
        pre_norm (bool, optional): If True, uses pre-layer normalization architecture.
            If False, uses post-layer normalization. Defaults to True.
        mask_eps (float, optional): Mask epsilon value for attention. Defaults to 1e-9.
        device (str, optional): Device for computation ('cpu' or 'gpu'). Defaults to 'cpu'.

        **kwargs: Additional arguments passed to the parent FlexiTransformer class.

    Note:
        The model uses identical number of layers for both encoder and decoder parts.
        For different encoder/decoder depths, use the base FlexiTransformer class directly.
    """

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        pe_type: str = 'absolute',
        init_method: str = 'xavier_uniform',
        pre_norm: bool = True,
        mask_eps: float = 1e-9,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            model_type='encoder-decoder',
            d_model=d_model,
            n_heads=n_heads,
            pre_norm=pre_norm,
            pe_type=pe_type,
            init_method=init_method,
            n_layers=(n_layers, n_layers),
            **kwargs,
        )


class FlexiBERT(BaseTransformer):
    """
    A flexible BERT-style transformer implementation that can be configured for various tasks.
    This class extends BaseTransformer to provide a configurable BERT-like architecture
    with flexible positional encoding, normalization, and initialization options.

    Args:
        src_vocab (int): Size of source vocabulary.
        num_classes (int, optional): Number of output classes for classification. Defaults to 2.
        d_model (int, optional): Dimension of model embeddings. Defaults to 512.
        n_heads (int, optional): Number of attention heads. Defaults to 8.
        n_layers (int, optional): Number of transformer layers. Defaults to 6.
        pe_type (str, optional): Type of positional encoding to use. Defaults to 'alibi'.
        init_method (str, optional): Weight initialization method. Defaults to 'xavier_uniform'.
        pre_norm (bool, optional): If True, uses pre-norm architecture variant. Defaults to True.
        mask_eps (float, optional): Mask epsilon value for attention. Defaults to 1e-9.
        **kwargs: Additional keyword arguments passed to BaseTransformer.

    Methods:
        reconfigure_head(new_head: nn.Module) -> None:
            Replaces the classification head with a new module.

        __call__(*args, **kwargs):
            Forward pass through the model. Delegates to internal model instance.
    """

    def __init__(
        self,
        src_vocab: int,
        num_classes: int = 2,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        pe_type: str = 'alibi',
        init_method: str = 'xavier_uniform',
        pre_norm: bool = True,
        mask_eps: float = 1e-9,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the FlexiBERT model with the given configuration.

        Args:
            src_vocab (int): Size of the source vocabulary.
            num_classes (int, optional): Number of output classes for classification. Defaults to 2.
            d_model (int, optional): Dimension of model embeddings. Defaults to 512.
            n_heads (int, optional): Number of attention heads. Defaults to 8.
            n_layers (int, optional): Number of transformer layers. Defaults to 6.
            pe_type (str, optional): Type of positional encoding to use. Defaults to 'alibi'.
            init_method (str, optional): Weight initialization method. Defaults to 'xavier_uniform'.
            pre_norm (bool, optional): If True, uses pre-norm architecture variant.
            Defaults to True.
            mask_eps (float, optional): Mask epsilon value for attention. Defaults to 1e-9.
            **kwargs: Additional keyword arguments passed to BaseTransformer.
        """
        super().__init__(
            src_vocab=src_vocab,
            model_type='encoder-only',
            num_classes=num_classes,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            pre_norm=pre_norm,
            pe_type=pe_type,
            init_method=init_method,
            **kwargs,
        )

    def reconfigure_head(self, new_head: nn.Module) -> None:
        """Replace the classification head of the model with a new one.

        Args:
            new_head (nn.Module): New classification head module to replace the existing one.

        Returns:
            None

        Examples:
            >>> model.reconfigure_head(
            ...     new_head=nn.Linear(
            ...         512, 10
            ...     )
            ... )
        """

        self.model.head = new_head


class FlexiGPT(BaseTransformer):
    """
    A flexible GPT-style transformer implementation that can be configured for various tasks.
    This class implements a GPT-style decoder-only transformer that can be customized for
    different language modeling tasks. It inherits from BaseTransformer and provides a
    configurable architecture through various parameters.

    Args:
        tgt_vocab (int): Size of target vocabulary.
        d_model (int, optional): Dimension of model embeddings and hidden states. Defaults to 512.
        n_heads (int, optional): Number of attention heads in each layer. Defaults to 8.
        n_layers (int, optional): Number of transformer layers. Defaults to 6.
        pe_type (str, optional): Type of positional encoding to use ('rotary', 'absolute', etc).
            Defaults to 'rotary'.
        init_method (str, optional): Weight initialization method. Defaults to 'xavier_uniform'.
        pre_norm (bool, optional): Whether to use pre-layer normalization. Defaults to True.
        mask_eps (float, optional): Mask epsilon value for attention. Defaults to 1e-9.
        **kwargs: Additional keyword arguments passed to the base transformer.

    Example:
        >>> model = FlexiGPT(
        ...     tgt_vocab=50000,
        ...     d_model=768,
        ...     n_heads=12,
        ... )
        >>> output = model(
        ...     input_ids
        ... )
    """

    def __init__(
        self,
        tgt_vocab: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        pe_type: str = 'rotary',
        init_method: str = 'xavier_uniform',
        pre_norm: bool = True,
        mask_eps: float = 1e-9,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the FlexiGPT model with the given configuration.

        Args:
            tgt_vocab (int): The size of the target vocabulary.
            d_model (int, optional): The dimension of model embeddings and hidden states.
                Default is 512.
            n_heads (int, optional): The number of attention heads in each layer. Default is 8.
            n_layers (int, optional): The number of transformer layers. Default is 6.
            pe_type (str, optional): The type of positional encoding to use. Default is 'rotary'.
            init_method (str, optional): The weight initialization method.
            Default is 'xavier_uniform'.
            pre_norm (bool, optional): Whether to use pre-layer normalization. Default is True.
            **kwargs: Additional keyword arguments passed to the base transformer.'
        """
        super().__init__(
            tgt_vocab=tgt_vocab,
            model_type='decoder-only',
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            pe_type=pe_type,
            init_method=init_method,
            pre_norm=pre_norm,
            **kwargs,
        )
