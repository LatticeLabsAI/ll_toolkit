"""ASR (Automatic Shape Recognition) model pipeline options.

This module provides configuration options for automatic shape recognition
and feature detection in CAD models. ASR is used to identify machining features,
geometric primitives, and design patterns.

Note: ASR in CAD context refers to Automatic Shape/Feature Recognition,
not Automatic Speech Recognition.

Classes:
    ASRFeatureType: Types of features to recognize
    ASRModelOptions: Configuration for ASR models
    ASRDetectionOptions: Detection thresholds and parameters
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict

from pydantic import BaseModel, Field


class ASRFeatureType(str, Enum):
    """Types of machining/geometric features to recognize."""

    # Machining features
    HOLE = "hole"  # Through holes, blind holes
    POCKET = "pocket"  # Rectangular/circular pockets
    SLOT = "slot"  # Slots and grooves
    BOSS = "boss"  # Raised features
    RIB = "rib"  # Thin wall features
    CHAMFER = "chamfer"  # Edge chamfers
    FILLET = "fillet"  # Edge fillets/rounds
    THREAD = "thread"  # Threaded features

    # Geometric primitives
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    CONE = "cone"
    TORUS = "torus"
    BOX = "box"
    PLANE = "plane"

    # Patterns
    LINEAR_PATTERN = "linear_pattern"
    CIRCULAR_PATTERN = "circular_pattern"
    MIRROR = "mirror"

    # All features
    ALL = "all"


class ASRDetectionOptions(BaseModel):
    """Detection thresholds and parameters for ASR.

    Attributes:
        min_confidence: Minimum confidence threshold (0-1)
        min_feature_size: Minimum feature size to detect (mm)
        max_feature_size: Maximum feature size to detect (mm)
        detect_patterns: Whether to detect pattern instances
        merge_similar: Merge similar/duplicate detections
        similarity_threshold: Threshold for merging (0-1)
    """

    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_feature_size: float = 0.1  # mm
    max_feature_size: Optional[float] = None
    detect_patterns: bool = True
    merge_similar: bool = True
    similarity_threshold: float = 0.9


class ASRModelOptions(BaseModel):
    """Configuration options for ASR models in pipelines.

    Attributes:
        feature_types: List of feature types to recognize
        detection_options: Detection threshold settings
        model_path: Path to ASR model weights
        use_gpu: Whether to use GPU acceleration
        batch_size: Batch size for inference
        enable_visualization: Generate feature visualizations
        output_format: Output format ("json", "dict", "graph")
    """

    feature_types: List[ASRFeatureType] = Field(
        default_factory=lambda: [ASRFeatureType.ALL]
    )
    detection_options: ASRDetectionOptions = Field(
        default_factory=ASRDetectionOptions
    )

    # Model configuration
    model_path: Optional[str] = None
    use_gpu: bool = True
    batch_size: int = 1

    # Output options
    enable_visualization: bool = False
    output_format: str = "dict"

    model_config = {"use_enum_values": True}


class RecognizedFeature(BaseModel):
    """A recognized feature from ASR.

    Attributes:
        feature_type: Type of feature
        confidence: Detection confidence (0-1)
        bounding_box: 3D bounding box
        parameters: Feature-specific parameters
        face_ids: Associated face IDs
        edge_ids: Associated edge IDs
        metadata: Additional metadata
    """

    feature_type: ASRFeatureType
    confidence: float
    bounding_box: Optional[List[float]] = None
    parameters: Dict[str, float] = Field(default_factory=dict)
    face_ids: List[int] = Field(default_factory=list)
    edge_ids: List[int] = Field(default_factory=list)
    metadata: Dict[str, any] = Field(default_factory=dict)


class ASRResult(BaseModel):
    """Result from ASR feature recognition.

    Attributes:
        features: List of recognized features
        num_features: Total number of features found
        processing_time_ms: Processing time in milliseconds
        model_version: ASR model version used
    """

    features: List[RecognizedFeature] = Field(default_factory=list)
    num_features: int = 0
    processing_time_ms: Optional[float] = None
    model_version: Optional[str] = None

    def get_features_by_type(
        self, feature_type: ASRFeatureType
    ) -> List[RecognizedFeature]:
        """Get all features of a specific type.

        Args:
            feature_type: Feature type to filter by

        Returns:
            List of matching features
        """
        return [f for f in self.features if f.feature_type == feature_type]

    def get_high_confidence_features(
        self, min_confidence: float = 0.8
    ) -> List[RecognizedFeature]:
        """Get features above confidence threshold.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            List of high-confidence features
        """
        return [f for f in self.features if f.confidence >= min_confidence]


def get_default_asr_options() -> ASRModelOptions:
    """Get default ASR options for feature recognition.

    Returns:
        ASRModelOptions with default settings
    """
    return ASRModelOptions(
        feature_types=[ASRFeatureType.ALL],
        detection_options=ASRDetectionOptions(),
    )


def get_machining_feature_options() -> ASRModelOptions:
    """Get ASR options optimized for machining feature detection.

    Returns:
        ASRModelOptions configured for machining features
    """
    return ASRModelOptions(
        feature_types=[
            ASRFeatureType.HOLE,
            ASRFeatureType.POCKET,
            ASRFeatureType.SLOT,
            ASRFeatureType.CHAMFER,
            ASRFeatureType.FILLET,
        ],
        detection_options=ASRDetectionOptions(
            min_confidence=0.7,
            min_feature_size=0.5,
            detect_patterns=True,
        ),
    )
