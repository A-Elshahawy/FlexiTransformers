"""Training configuration dataclass."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainingConfig:
    """Configuration for training transformer models."""

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_epochs: int = 10
    max_steps: int | None = None

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_scheduler: Literal['constant', 'linear', 'cosine', 'warmup_cosine'] = 'warmup_cosine'
    warmup_steps: int = 4000
    warmup_ratio: float | None = None  # Alternative to warmup_steps

    # Loss
    label_smoothing: float = 0.1

    # Checkpointing
    save_steps: int | None = None
    save_epochs: int = 1
    save_total_limit: int | None = None
    checkpoint_dir: str = './checkpoints'

    # Logging
    logging_steps: int = 100
    log_dir: str = './logs'

    # Evaluation
    eval_steps: int | None = None
    eval_epochs: int | None = 1
    eval_batch_size: int | None = None  # Defaults to batch_size

    # Hardware
    device: str = 'cuda'
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False

    # Distributed
    local_rank: int = -1
    world_size: int = 1

    # Misc
    seed: int = 42
    dataloader_num_workers: int = 4
    pin_memory: bool = True

    def __post_init__(self) -> None:
        """Post-initialization validation and defaults."""
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size

        if self.fp16 and self.bf16:
            raise ValueError('Cannot use both fp16 and bf16')

        if self.warmup_ratio is not None:
            if self.max_steps is None:
                raise ValueError('warmup_ratio requires max_steps to be set')
            self.warmup_steps = int(self.max_steps * self.warmup_ratio)

        if self.max_steps is not None and self.max_epochs is not None:
            # Both are set, will stop at whichever comes first
            pass

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create config from dictionary."""
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
