"""
Configuration classes for STEP models.
"""

from dataclasses import dataclass, field
from typing import List, Optional


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
    graph_input_dim: int = 48    # cadling TopologyGraph native dim
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


@dataclass
class STEPReserializationConfig:
    """Configuration for DFS reserialization of STEP files."""
    max_depth: int = 50
    float_precision: int = 6
    normalize_floats: bool = True
    renumber_ids: bool = True
    root_strategy: str = "both"      # "no_incoming" | "type_hierarchy" | "both"
    include_orphans: bool = True


@dataclass
class STEPAnnotationConfig:
    """Configuration for structural annotations."""
    include_file_summary: bool = True
    include_branch_annotations: bool = True
    max_types_shown: int = 5
    verbose: bool = False


@dataclass
class VAEConfig:
    """Configuration for the STEP Variational Autoencoder.

    Command types follow geotoken's vocabulary:
        SOL=0, LINE=1, ARC=2, CIRCLE=3, EXTRUDE=4, EOS=5
    """
    latent_dim: int = 256
    kl_weight: float = 1.0
    kl_warmup_epochs: int = 10
    encoder_vocab_size: int = 50000
    encoder_embed_dim: int = 256
    encoder_layers: int = 6
    decoder_layers: int = 6
    num_command_types: int = 6      # SOL, LINE, ARC, CIRCLE, EXTRUDE, EOS
    num_param_levels: int = 256
    max_seq_len: int = 60


@dataclass
class LatentGANConfig:
    """Configuration for the latent-space WGAN-GP."""
    latent_dim: int = 256
    gen_hidden_dims: List[int] = field(default_factory=lambda: [512, 512])
    disc_hidden_dims: List[int] = field(default_factory=lambda: [512, 512])
    gp_lambda: float = 10.0
    n_critic: int = 5
    lr_gen: float = 1e-4
    lr_disc: float = 1e-4


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion-based CAD denoiser."""
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    inference_steps: int = 200
    denoiser_layers: int = 12
    denoiser_heads: int = 12
    denoiser_hidden_dim: int = 1024
    latent_dim: int = 256


@dataclass
class ConditioningConfig:
    """Configuration for cross-attention conditioning modules."""
    text_encoder_name: str = "bert-base-uncased"
    image_encoder_name: str = "facebook/dinov2-base"
    conditioning_dim: int = 1024
    skip_cross_attention_blocks: int = 2
    freeze_encoder: bool = True
    num_adaptive_layers: int = 1


@dataclass
class StreamingCadlingConfig:
    """Configuration for streaming cadling data into ll_stepnet trainers.

    When passed to a streaming trainer's ``__init__``, the trainer will
    lazy-import ``cadling.data.streaming.CADStreamingDataset`` and build
    a streaming data pipeline from cadling data automatically.

    Attributes:
        dataset_id: HuggingFace dataset ID or local path to cadling data.
        split: Dataset split (``"train"``, ``"val"``, ``"test"``).
        streaming: Whether to use HuggingFace streaming mode.
        batch_size: Training batch size.
        shuffle: Whether to shuffle the stream.
        shuffle_buffer_size: Buffer size for streaming shuffle.
        max_samples: Maximum samples to load (None = all).
        max_commands: Maximum command sequence length (pad/truncate).
        compact_topology: Use 48-dim compact topology (cadling native).

        lazy_load_topology: Whether to load topology on-demand.
        topology_cache_size: Maximum topologies to cache in memory.
        preprocess_fn: Preprocessing function name ('geotoken', 'tokenize', None).
        prefetch_factor: Number of batches to prefetch in background.
        max_memory_mb: Maximum memory for cached data.
        chunk_size: Number of samples per processing chunk.
        include_graph_data: Whether to include graph/topology data.
        graph_feature_dim: Expected node feature dimension (48 cadling, 129 legacy).
        num_workers: Number of dataloader workers.
    """

    # Dataset source
    dataset_id: str = ""
    split: str = "train"
    streaming: bool = True

    # Batching
    batch_size: int = 8
    shuffle: bool = True
    shuffle_buffer_size: int = 10000
    max_samples: Optional[int] = None
    max_commands: int = 60

    # Lazy loading
    lazy_load_topology: bool = True
    topology_cache_size: int = 1000

    # Preprocessing
    preprocess_fn: Optional[str] = None  # 'geotoken' | 'tokenize' | None
    prefetch_factor: int = 2

    # Memory management
    max_memory_mb: int = 4096
    chunk_size: int = 1000

    # Graph data
    include_graph_data: bool = False
    graph_feature_dim: int = 48
    compact_topology: bool = True

    # Workers
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
        'qa': STEPQAConfig,
        'vae': VAEConfig,
        'latent_gan': LatentGANConfig,
        'diffusion': DiffusionConfig,
        'conditioning': ConditioningConfig,
    }

    if task not in config_map:
        raise ValueError(f"Unknown task: {task}. Choose from {list(config_map.keys())}")

    config_cls = config_map[task]
    return config_cls(**kwargs)
