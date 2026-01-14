"""Enrichment models for CADling.

This module provides neural network models for enriching CADlingDocument items
with predictions, embeddings, and computed properties.

Models are integrated with ll_stepnet for CAD-specific understanding.

Classes:
    EnrichmentModel: Base class for all enrichment models
    CADPartClassifier: Part classification model
    CADPropertyPredictor: Physical property prediction model
    CADSimilarityEmbedder: Embedding generation for RAG
    VlmModel: Base class for vision-language models
    ApiVlmModel: API-based VLM (GPT-4V, Claude)
    InlineVlmModel: Local VLM (LLaVA, Qwen-VL)
"""

from cadling.models.base_model import EnrichmentModel
from cadling.models.vlm_model import (
    ApiVlmModel,
    ApiVlmOptions,
    InlineVlmModel,
    InlineVlmOptions,
    VlmAnnotation,
    VlmModel,
    VlmOptions,
    VlmResponse,
)

__all__ = [
    "EnrichmentModel",
    "VlmModel",
    "ApiVlmModel",
    "InlineVlmModel",
    "VlmOptions",
    "ApiVlmOptions",
    "InlineVlmOptions",
    "VlmResponse",
    "VlmAnnotation",
]
