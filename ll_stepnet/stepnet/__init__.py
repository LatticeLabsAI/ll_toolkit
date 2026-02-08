"""
LL-STEPNet: STEP/B-Rep Neural Network Package
Clean, separated architecture for processing CAD/STEP files.
"""

from .tokenizer import STEPTokenizer
from .features import STEPFeatureExtractor
from .topology import STEPTopologyBuilder
from .encoder import STEPEncoder, STEPTransformerEncoder, STEPTransformerDecoder, STEPGraphEncoder
from .tasks import (
    STEPForCaptioning,
    STEPForClassification,
    STEPForPropertyPrediction,
    STEPForSimilarity,
    STEPForQA
)
from .pretrain import (
    STEPForCausalLM,
    STEPForMaskedLM,
    STEPForHybridLM,
    mask_tokens
)
from .data import STEPDataset, STEPCollator, GeoTokenDataset, GeoTokenCollator, CadlingDataset, create_dataloader
from .trainer import STEPTrainer
from .data_requirements import (
    STEPLearningCurveGenerator,
    STEPScalingLawAnalyzer,
    power_law_loss,
    power_law_error,
    inverse_power_law_accuracy,
    chinchilla_optimal_tokens,
    plot_learning_curve_with_scaling_law,
    estimate_data_requirements,
    count_model_parameters,
    suggest_dataset_size
)
from .config import (
    STEPEncoderConfig,
    STEPClassificationConfig,
    STEPPropertyPredictionConfig,
    STEPCaptioningConfig,
    STEPSimilarityConfig,
    STEPQAConfig,
    STEPReserializationConfig,
    STEPAnnotationConfig,
    TrainingConfig,
    DataConfig,
    VAEConfig,
    LatentGANConfig,
    DiffusionConfig,
    ConditioningConfig,
    StreamingCadlingConfig,
    get_config,
)
from .vae import STEPVAE
from .output_heads import (
    CommandType,
    CommandTypeHead,
    ParameterHeads,
    CompositeHead,
    PARAMETER_MASKS,
)
from .latent_gan import (
    LatentGenerator,
    LatentDiscriminator,
    LatentGAN,
)
from .diffusion import (
    DDPMScheduler,
    SinusoidalTimestepEmbedding,
    CADDenoiser,
    StructuredDiffusion,
)
from .conditioning import (
    AdaptiveLayer,
    TextConditioner,
    ImageConditioner,
    MultiModalConditioner,
)
from .reserialization import (
    STEPEntityNode,
    STEPEntityGraph,
    STEPDFSSerializer,
    STEPReserializedOutput,
    reserialize_step,
)
from .annotations import (
    BranchAnnotation,
    StructuralSummary,
    STEPStructuralAnnotator,
    STEPAnnotatedOutput,
)
from .vqvae import (
    VectorQuantizer,
    DisentangledCodebooks,
    CodebookDecoder,
    VQVAEModel,
)
from .training import (
    VAETrainer,
    GANTrainer,
    DiffusionTrainer,
    StreamingVAETrainer,
    StreamingDiffusionTrainer,
    StreamingGANTrainer,
)
from .generation_pipeline import CADGenerationPipeline

__version__ = "0.1.0"

__all__ = [
    # Core components
    "STEPTokenizer",
    "STEPFeatureExtractor",
    "STEPTopologyBuilder",
    "STEPEncoder",
    "STEPTransformerEncoder",
    "STEPTransformerDecoder",
    "STEPGraphEncoder",

    # Task models
    "STEPForCaptioning",
    "STEPForClassification",
    "STEPForPropertyPrediction",
    "STEPForSimilarity",
    "STEPForQA",

    # Pre-training models (unsupervised)
    "STEPForCausalLM",
    "STEPForMaskedLM",
    "STEPForHybridLM",
    "mask_tokens",

    # Data utilities
    "STEPDataset",
    "STEPCollator",
    "GeoTokenDataset",
    "GeoTokenCollator",
    "CadlingDataset",
    "create_dataloader",

    # Training
    "STEPTrainer",

    # Data Requirements Analysis
    "STEPLearningCurveGenerator",
    "STEPScalingLawAnalyzer",
    "power_law_loss",
    "power_law_error",
    "inverse_power_law_accuracy",
    "chinchilla_optimal_tokens",
    "plot_learning_curve_with_scaling_law",
    "estimate_data_requirements",
    "count_model_parameters",
    "suggest_dataset_size",

    # Configuration
    "STEPEncoderConfig",
    "STEPClassificationConfig",
    "STEPPropertyPredictionConfig",
    "STEPCaptioningConfig",
    "STEPSimilarityConfig",
    "STEPQAConfig",
    "STEPReserializationConfig",
    "STEPAnnotationConfig",
    "TrainingConfig",
    "DataConfig",
    "VAEConfig",
    "LatentGANConfig",
    "DiffusionConfig",
    "ConditioningConfig",
    "StreamingCadlingConfig",
    "get_config",

    # VAE
    "STEPVAE",

    # Output Heads
    "CommandType",
    "CommandTypeHead",
    "ParameterHeads",
    "CompositeHead",
    "PARAMETER_MASKS",

    # Latent GAN
    "LatentGenerator",
    "LatentDiscriminator",
    "LatentGAN",

    # Diffusion
    "DDPMScheduler",
    "SinusoidalTimestepEmbedding",
    "CADDenoiser",
    "StructuredDiffusion",

    # Conditioning
    "AdaptiveLayer",
    "TextConditioner",
    "ImageConditioner",
    "MultiModalConditioner",

    # Reserialization
    "STEPEntityNode",
    "STEPEntityGraph",
    "STEPDFSSerializer",
    "STEPReserializedOutput",
    "reserialize_step",

    # Annotations
    "BranchAnnotation",
    "StructuralSummary",
    "STEPStructuralAnnotator",
    "STEPAnnotatedOutput",

    # VQ-VAE / CAD Generation
    "VectorQuantizer",
    "DisentangledCodebooks",
    "CodebookDecoder",
    "VQVAEModel",

    # Generative Training Infrastructure
    "VAETrainer",
    "GANTrainer",
    "DiffusionTrainer",
    "StreamingVAETrainer",
    "StreamingDiffusionTrainer",
    "StreamingGANTrainer",

    # End-to-end Generation Pipeline
    "CADGenerationPipeline",
]
