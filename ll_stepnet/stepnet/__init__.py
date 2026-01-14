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
from .data import STEPDataset, STEPCollator, create_dataloader
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
    TrainingConfig,
    DataConfig,
    get_config
)

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
    "TrainingConfig",
    "DataConfig",
    "get_config",
]
