"""Experimental enrichment models for CADling.

This module provides experimental enrichment models for advanced CAD analysis
including PMI extraction, feature recognition, manufacturability assessment,
design intent inference, CAD-to-text generation, and geometric constraint extraction.
"""

# Phase 2: Vision-based models (Features 4-6)
from .feature_recognition_vlm_model import (
    FeatureRecognitionVlmModel,
    MachiningFeature,
)
from .manufacturability_assessment_model import (
    DFMIssue,
    DFMRule,
    IssueSeverity,
    ManufacturabilityAssessmentModel,
    ManufacturabilityReport,
    ManufacturingProcess,
)
from .pmi_extraction_model import PMIExtractionModel

# Phase 3: AI understanding models (Features 7-9)
from .cad_to_text_generation_model import CADDescription, CADToTextGenerationModel
from .design_intent_inference_model import (
    DesignIntent,
    DesignIntentInferenceModel,
    IntentCategory,
    LoadType,
)
from .geometric_constraint_model import (
    ConstraintGraph,
    ConstraintType,
    GeometricConstraint,
    GeometricConstraintModel,
)

__all__ = [
    # PMI Extraction (Feature 4)
    "PMIExtractionModel",
    # Feature Recognition (Feature 5)
    "FeatureRecognitionVlmModel",
    "MachiningFeature",
    # Manufacturability Assessment (Feature 6)
    "ManufacturabilityAssessmentModel",
    "ManufacturabilityReport",
    "DFMIssue",
    "DFMRule",
    "IssueSeverity",
    "ManufacturingProcess",
    # Design Intent Inference (Feature 7)
    "DesignIntentInferenceModel",
    "DesignIntent",
    "IntentCategory",
    "LoadType",
    # CAD-to-Text Generation (Feature 8)
    "CADToTextGenerationModel",
    "CADDescription",
    # Geometric Constraint Extraction (Feature 9)
    "GeometricConstraintModel",
    "GeometricConstraint",
    "ConstraintType",
    "ConstraintGraph",
]
