"""
Configuration classes for STEP models.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class STEPTokenizerConfig:
    """Configuration for STEP tokenizer."""
    vocab_size: int = 50000
    max_length: int = 2048
    special_tokens: dict = None

    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = {
                'pad_token': '<pad>',
                'unk_token': '<unk>',
                'bos_token': '<bos>',
                'eos_token': '<eos>'
            }


@dataclass
class STEPEncoderConfig:
    """Configuration for STEP encoder."""
    vocab_size: int = 50000
    token_embed_dim: int = 256
    graph_node_dim: int = 128
    output_dim: int = 1024
    num_transformer_layers: int = 6
    num_graph_layers: int = 3
    dropout: float = 0.1


@dataclass
class STEPClassificationConfig:
    """Configuration for STEP classification model."""
    vocab_size: int = 50000
    num_classes: int = 10
    output_dim: int = 1024
    dropout: float = 0.1


@dataclass
class STEPPropertyPredictionConfig:
    """Configuration for STEP property prediction model."""
    vocab_size: int = 50000
    num_properties: int = 6
    output_dim: int = 1024
    dropout: float = 0.1


@dataclass
class STEPCaptioningConfig:
    """Configuration for STEP captioning model."""
    vocab_size: int = 50000
    decoder_vocab_size: int = 50000
    output_dim: int = 1024
    max_caption_length: int = 128


@dataclass
class STEPSimilarityConfig:
    """Configuration for STEP similarity model."""
    vocab_size: int = 50000
    embedding_dim: int = 512


@dataclass
class STEPQAConfig:
    """Configuration for STEP question answering model."""
    step_vocab_size: int = 50000
    text_vocab_size: int = 50000
    output_dim: int = 1024


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    save_every: int = 1
    eval_every: int = 1
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_dir: str = 'data'
    train_split: str = 'train'
    val_split: str = 'val'
    test_split: str = 'test'
    max_length: int = 2048
    use_topology: bool = True
    num_workers: int = 4


def get_config(task: str = 'classification', **kwargs):
    """
    Get configuration for a specific task.

    Args:
        task: Task name ('classification', 'property', 'captioning', 'similarity', 'qa')
        **kwargs: Additional config overrides

    Returns:
        Configuration object
    """
    config_map = {
        'classification': STEPClassificationConfig,
        'property': STEPPropertyPredictionConfig,
        'captioning': STEPCaptioningConfig,
        'similarity': STEPSimilarityConfig,
        'qa': STEPQAConfig
    }

    if task not in config_map:
        raise ValueError(f"Unknown task: {task}. Choose from {list(config_map.keys())}")

    config_cls = config_map[task]
    return config_cls(**kwargs)
