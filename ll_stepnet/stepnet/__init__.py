"""
LL-STEPNet: STEP/B-Rep Neural Network Package
Clean, separated architecture for processing CAD/STEP files.
"""

from .annotations import (
    BranchAnnotation,
    STEPAnnotatedOutput,
    STEPStructuralAnnotator,
    StructuralSummary,
)
from .conditioning import (
    AdaptiveLayer,
    ImageConditioner,
    MultiModalConditioner,
    TextConditioner,
)
from .config import (
    ConditioningConfig,
    DataConfig,
    DiffusionConfig,
    LatentGANConfig,
    STEPAnnotationConfig,
    STEPCaptioningConfig,
    STEPClassificationConfig,
    STEPEncoderConfig,
    STEPPropertyPredictionConfig,
    STEPQAConfig,
    STEPReserializationConfig,
    STEPSimilarityConfig,
    StreamingCadlingConfig,
    TrainingConfig,
    VAEConfig,
    get_config,
)
from .data import (
    CadlingDataset,
    GeoTokenCollator,
    GeoTokenDataset,
    STEPCollator,
    STEPDataset,
    create_dataloader,
)
from .data_requirements import (
    STEPLearningCurveGenerator,
    STEPScalingLawAnalyzer,
    chinchilla_optimal_tokens,
    count_model_parameters,
    estimate_data_requirements,
    inverse_power_law_accuracy,
    plot_learning_curve_with_scaling_law,
    power_law_error,
    power_law_loss,
    suggest_dataset_size,
)
from .diffusion import (
    CADDenoiser,
    DDPMScheduler,
    SinusoidalTimestepEmbedding,
    StructuredDiffusion,
)
from .encoder import (
    STEPEncoder,
    STEPGraphEncoder,
    STEPTransformerDecoder,
    STEPTransformerEncoder,
)
from .features import STEPFeatureExtractor
from .generation_pipeline import CADGenerationPipeline
from .latent_gan import (
    LatentDiscriminator,
    LatentGAN,
    LatentGenerator,
)
from .output_heads import (
    PARAMETER_MASKS,
    CommandType,
    CommandTypeHead,
    CompositeHead,
    ParameterHeads,
)
from .pretrain import STEPForCausalLM, STEPForHybridLM, STEPForMaskedLM, mask_tokens
from .reserialization import (
    STEPDFSSerializer,
    STEPEntityGraph,
    STEPEntityNode,
    STEPReserializedOutput,
    reserialize_step,
)
from .tasks import (
    STEPForCaptioning,
    STEPForClassification,
    STEPForPropertyPrediction,
    STEPForQA,
    STEPForSimilarity,
)
from .tokenizer import STEPTokenizer
from .topology import STEPTopologyBuilder
from .trainer import STEPTrainer
from .training import (
    DiffusionTrainer,
    GANTrainer,
    StreamingDiffusionTrainer,
    StreamingGANTrainer,
    StreamingVAETrainer,
    VAETrainer,
)
from .vae import STEPVAE
from .vqvae import (
    CodebookDecoder,
    DisentangledCodebooks,
    VectorQuantizer,
    VQVAEModel,
)

__version__ = "0.1.0"

__all__ = [
    "PARAMETER_MASKS",
    # VAE
    "STEPVAE",
    # Conditioning
    "AdaptiveLayer",
    # Annotations
    "BranchAnnotation",
    "CADDenoiser",
    # End-to-end Generation Pipeline
    "CADGenerationPipeline",
    "CadlingDataset",
    "CodebookDecoder",
    # Output Heads
    "CommandType",
    "CommandTypeHead",
    "CompositeHead",
    "ConditioningConfig",
    # Diffusion
    "DDPMScheduler",
    "DataConfig",
    "DiffusionConfig",
    "DiffusionTrainer",
    "DisentangledCodebooks",
    "GANTrainer",
    "GeoTokenCollator",
    "GeoTokenDataset",
    "ImageConditioner",
    "LatentDiscriminator",
    "LatentGAN",
    "LatentGANConfig",
    # Latent GAN
    "LatentGenerator",
    "MultiModalConditioner",
    "ParameterHeads",
    "STEPAnnotatedOutput",
    "STEPAnnotationConfig",
    "STEPCaptioningConfig",
    "STEPClassificationConfig",
    "STEPCollator",
    "STEPDFSSerializer",
    # Data utilities
    "STEPDataset",
    "STEPEncoder",
    # Configuration
    "STEPEncoderConfig",
    "STEPEntityGraph",
    # Reserialization
    "STEPEntityNode",
    "STEPFeatureExtractor",
    # Task models
    "STEPForCaptioning",
    # Pre-training models (unsupervised)
    "STEPForCausalLM",
    "STEPForClassification",
    "STEPForHybridLM",
    "STEPForMaskedLM",
    "STEPForPropertyPrediction",
    "STEPForQA",
    "STEPForSimilarity",
    "STEPGraphEncoder",
    # Data Requirements Analysis
    "STEPLearningCurveGenerator",
    "STEPPropertyPredictionConfig",
    "STEPQAConfig",
    "STEPReserializationConfig",
    "STEPReserializedOutput",
    "STEPScalingLawAnalyzer",
    "STEPSimilarityConfig",
    "STEPStructuralAnnotator",
    # Core components
    "STEPTokenizer",
    "STEPTopologyBuilder",
    # Training
    "STEPTrainer",
    "STEPTransformerDecoder",
    "STEPTransformerEncoder",
    "SinusoidalTimestepEmbedding",
    "StreamingCadlingConfig",
    "StreamingDiffusionTrainer",
    "StreamingGANTrainer",
    "StreamingVAETrainer",
    "StructuralSummary",
    "StructuredDiffusion",
    "TextConditioner",
    "TrainingConfig",
    "VAEConfig",
    # Generative Training Infrastructure
    "VAETrainer",
    "VQVAEModel",
    # VQ-VAE / CAD Generation
    "VectorQuantizer",
    "chinchilla_optimal_tokens",
    "count_model_parameters",
    "create_dataloader",
    "estimate_data_requirements",
    "get_config",
    "inverse_power_law_accuracy",
    "mask_tokens",
    "plot_learning_curve_with_scaling_law",
    "power_law_error",
    "power_law_loss",
    "reserialize_step",
    "suggest_dataset_size",
]
