"""Geometric tokenizer with adaptive quantization.

Provides adaptive precision quantization for 3D geometric data,
allocating more bits to geometrically complex regions (high curvature,
dense features) and fewer bits to flat/simple regions.

Includes command sequence tokenization (DeepCAD-style construction
history), constraint tokenization (SketchGraphs-style), and graph
tokenization (B-Rep topology with dense features).
"""
from __future__ import annotations

from .config import (
    PrecisionTier,
    QuantizationConfig,
    NormalizationConfig,
    AdaptiveBitAllocationConfig,
    CommandTokenizationConfig,
    GraphTokenizationConfig,
)
from .tokenizer.geo_tokenizer import GeoTokenizer
from .tokenizer.token_types import (
    CoordinateToken,
    GeometryToken,
    TokenSequence,
    CommandToken,
    CommandType,
    ConstraintToken,
    ConstraintType,
    BooleanOpToken,
    BooleanOpType,
    SequenceConfig,
    GraphNodeToken,
    GraphEdgeToken,
    GraphStructureToken,
)
from .tokenizer.command_tokenizer import CommandSequenceTokenizer
from .tokenizer.graph_tokenizer import GraphTokenizer
from .tokenizer.vocabulary import (
    CADVocabulary,
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    SEP_TOKEN_ID,
    UNK_TOKEN_ID,
    NUM_SPECIAL_TOKENS,
)
from .quantization.adaptive import AdaptiveQuantizer
from .quantization.uniform import UniformQuantizer
from .quantization.feature_quantizer import (
    FeatureVectorQuantizer,
    FeatureQuantizationParams,
)
from .quantization.uv_grid_quantizer import (
    UVGridQuantizer,
    UVGridTokens,
    FaceUVGridTokens,
    EdgeUVGridTokens,
)

__all__ = [
    # Config
    "PrecisionTier",
    "QuantizationConfig",
    "NormalizationConfig",
    "AdaptiveBitAllocationConfig",
    "CommandTokenizationConfig",
    "GraphTokenizationConfig",
    # Geometric tokenizer
    "GeoTokenizer",
    "CoordinateToken",
    "GeometryToken",
    "TokenSequence",
    # Command tokenization
    "CommandSequenceTokenizer",
    "CADVocabulary",
    "CommandToken",
    "CommandType",
    "ConstraintToken",
    "ConstraintType",
    "BooleanOpToken",
    "BooleanOpType",
    "SequenceConfig",
    # Graph tokenization
    "GraphTokenizer",
    "GraphNodeToken",
    "GraphEdgeToken",
    "GraphStructureToken",
    # Special token IDs
    "PAD_TOKEN_ID",
    "BOS_TOKEN_ID",
    "EOS_TOKEN_ID",
    "SEP_TOKEN_ID",
    "UNK_TOKEN_ID",
    "NUM_SPECIAL_TOKENS",
    # Quantizers
    "AdaptiveQuantizer",
    "UniformQuantizer",
    "FeatureVectorQuantizer",
    "FeatureQuantizationParams",
    # UV-grid quantization
    "UVGridQuantizer",
    "UVGridTokens",
    "FaceUVGridTokens",
    "EdgeUVGridTokens",
]
